from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import t as student_t

from app.backtest_runner import build_backtest_feature_cache
from backtest.metrics import ceq_annualized, realized_forward_returns
from data_contract.derived_series import derive_rv20_nasdaq100
from data_contract.fred_adapter import FredClient
from data_contract.nasdaq_client import NasdaqClient
from engine_types import TimeSeries

RESULTS_PATH = Path("/tmp/backtest-10y-12w-v5/backtest/backtest_results.jsonl")
STRICT_START = date(2015, 2, 20)
AS_OF_END = date(2024, 12, 27)
REQUIRED_SERIES = (
    "DGS10",
    "DGS2",
    "DGS1",
    "EFFR",
    "BAA10Y",
    "WALCL",
    "VXNCLS",
    "VIXCLS",
    "VXVCLS",
)
TAUS_FULL = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95], dtype=np.float64)
TAUS_CRPS = np.linspace(0.05, 0.95, 19, dtype=np.float64)
NU = 5.0
L2 = 1.0e-3
COVERAGE_WEIGHT = 10.0


@dataclass(frozen=True, slots=True)
class EvalRow:
    as_of: date
    quantiles: np.ndarray
    realized: float
    offense_final: float
    x_scaled: np.ndarray
    pi: np.ndarray


def _load_series() -> dict[str, TimeSeries]:
    fred = FredClient(api_key="", cache_root=Path("data/raw/fred"))
    nasdaq = NasdaqClient(cache_root=Path("data/raw/nasdaq"))
    series = {series_id: fred.get_series(series_id, AS_OF_END, "strict") for series_id in REQUIRED_SERIES}
    price_series = nasdaq.get_series("NASDAQXNDX", AS_OF_END)
    series["NASDAQXNDX"] = price_series
    series["RV20_NDX"] = derive_rv20_nasdaq100(price_series, AS_OF_END)
    return series


def _load_rows() -> list[EvalRow]:
    payloads = [json.loads(line) for line in RESULTS_PATH.read_text().splitlines() if line.strip()]
    series = _load_series()
    feature_cache = build_backtest_feature_cache(series, end=AS_OF_END)
    history_ordinals = np.asarray(feature_cache["history_week_ordinals"], dtype=np.int64)
    history_x_scaled = np.asarray(feature_cache["history_x_scaled"], dtype=np.float64)
    x_by_week = {int(history_ordinals[idx]): history_x_scaled[idx] for idx in range(history_ordinals.shape[0])}
    realized = realized_forward_returns(
        series["NASDAQXNDX"],
        [date.fromisoformat(row["as_of_date"]) for row in payloads],
    )
    rows: list[EvalRow] = []
    for idx, row in enumerate(payloads):
        as_of = date.fromisoformat(row["as_of_date"])
        x_scaled = x_by_week.get(as_of.toordinal())
        quantiles = np.array(
            [row["distribution"][key] for key in ("q05", "q10", "q25", "q50", "q75", "q90", "q95")],
            dtype=np.float64,
        )
        pi = np.asarray(row["state"]["post"], dtype=np.float64)
        y_true = float(realized[idx])
        if (
            row["vintage_mode"] != "strict"
            or as_of < STRICT_START
            or x_scaled is None
            or not np.isfinite(x_scaled).all()
            or not np.isfinite(quantiles).all()
            or not np.isfinite(pi).all()
            or not math.isfinite(y_true)
        ):
            continue
        rows.append(
            EvalRow(
                as_of=as_of,
                quantiles=quantiles,
                realized=y_true,
                offense_final=float(row["decision"]["offense_final"]),
                x_scaled=np.asarray(x_scaled, dtype=np.float64),
                pi=pi,
            ),
        )
    return rows


def _quantile_score(y_true: np.ndarray, quantiles: np.ndarray, taus: np.ndarray) -> float:
    delta = y_true[:, None] - quantiles
    losses = np.maximum(taus[None, :] * delta, (taus[None, :] - 1.0) * delta)
    return float(2.0 * np.mean(losses))


def _baseline_a_quantiles(history: np.ndarray, fallback: float) -> np.ndarray:
    if history.shape[0] < 2:
        return np.full(TAUS_FULL.shape, fallback, dtype=np.float64)
    return np.quantile(history, TAUS_FULL).astype(np.float64)


def _evaluate_baseline(rows: list[EvalRow]) -> dict[str, float]:
    y = np.array([row.realized for row in rows], dtype=np.float64)
    quantiles = np.stack([row.quantiles for row in rows])
    baseline_a = np.stack(
        [_baseline_a_quantiles(y[:idx], float(y[idx])) for idx in range(y.shape[0])],
    )
    q10_hits = y <= quantiles[:, 1]
    q90_hits = y <= quantiles[:, 5]
    offense = np.array([row.offense_final for row in rows], dtype=np.float64)
    return {
        "q10_error": abs(float(np.mean(q10_hits)) - 0.10),
        "q90_error": abs(float(np.mean(q90_hits)) - 0.90),
        "crps_improvement": float(
            1.0
            - _quantile_score(y, quantiles, TAUS_FULL) / _quantile_score(y, baseline_a, TAUS_FULL)
        ),
        "ceq_diff": float(
            ceq_annualized(offense / 100.0 * y) - ceq_annualized(0.5 * y),
        ),
    }


def _design(rows: list[EvalRow]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_scaled = np.stack([row.x_scaled for row in rows])
    pi = np.stack([row.pi for row in rows])
    y = np.array([row.realized for row in rows], dtype=np.float64)
    x_mu = np.column_stack([np.ones(y.shape[0], dtype=np.float64), x_scaled, pi])
    x_sigma = np.column_stack([np.ones(y.shape[0], dtype=np.float64), pi])
    return x_mu, x_sigma, y


def _predict_quantiles(
    x_mu: np.ndarray,
    x_sigma: np.ndarray,
    beta_mu: np.ndarray,
    beta_sigma: np.ndarray,
    taus: np.ndarray,
) -> np.ndarray:
    mu = x_mu @ beta_mu
    log_sigma = np.clip(x_sigma @ beta_sigma, -4.0, 2.0)
    sigma = np.exp(log_sigma)
    z = student_t.ppf(taus, df=NU).astype(np.float64)
    quantiles = mu[:, None] + sigma[:, None] * z[None, :]
    return np.maximum.accumulate(quantiles, axis=1)


def _objective(theta: np.ndarray, x_mu: np.ndarray, x_sigma: np.ndarray, y: np.ndarray) -> float:
    mu_dim = x_mu.shape[1]
    beta_mu = theta[:mu_dim]
    beta_sigma = theta[mu_dim:]
    quantiles = _predict_quantiles(x_mu, x_sigma, beta_mu, beta_sigma, TAUS_CRPS)
    crps_proxy = _quantile_score(y, quantiles, TAUS_CRPS)
    eval_quantiles = _predict_quantiles(x_mu, x_sigma, beta_mu, beta_sigma, TAUS_FULL)
    q10_cov = float(np.mean(y <= eval_quantiles[:, 1]))
    q90_cov = float(np.mean(y <= eval_quantiles[:, 5]))
    coverage_penalty = (q10_cov - 0.10) ** 2 + (q90_cov - 0.90) ** 2
    regularization = L2 * float(np.sum(theta * theta))
    return crps_proxy + COVERAGE_WEIGHT * coverage_penalty + regularization


def _fit_candidate(rows: list[EvalRow]) -> np.ndarray:
    x_mu, x_sigma, y = _design(rows)
    ridge_mu = 1.0e-3 * np.eye(x_mu.shape[1], dtype=np.float64)
    beta_mu0 = np.linalg.solve(x_mu.T @ x_mu + ridge_mu, x_mu.T @ y)
    residual = y - x_mu @ beta_mu0
    sigma_target = np.log(np.maximum(np.abs(residual), 1.0e-3))
    ridge_sigma = 1.0e-3 * np.eye(x_sigma.shape[1], dtype=np.float64)
    beta_sigma0 = np.linalg.solve(
        x_sigma.T @ x_sigma + ridge_sigma,
        x_sigma.T @ sigma_target,
    )
    theta0 = np.concatenate([beta_mu0, beta_sigma0]).astype(np.float64)
    result = minimize(
        _objective,
        theta0,
        args=(x_mu, x_sigma, y),
        method="Powell",
        options={"maxiter": 200, "xtol": 1.0e-4, "ftol": 1.0e-4},
    )
    if not result.success:
        raise RuntimeError(f"candidate optimization failed: {result.message}")
    return np.asarray(result.x, dtype=np.float64)


def _evaluate_candidate(rows: list[EvalRow], theta: np.ndarray) -> dict[str, float]:
    x_mu, x_sigma, y = _design(rows)
    mu_dim = x_mu.shape[1]
    quantiles = _predict_quantiles(
        x_mu,
        x_sigma,
        theta[:mu_dim],
        theta[mu_dim:],
        TAUS_FULL,
    )
    baseline_a = np.stack(
        [_baseline_a_quantiles(y[:idx], float(y[idx])) for idx in range(y.shape[0])],
    )
    q10_hits = y <= quantiles[:, 1]
    q90_hits = y <= quantiles[:, 5]
    offense = np.array([row.offense_final for row in rows], dtype=np.float64)
    return {
        "q10_error": abs(float(np.mean(q10_hits)) - 0.10),
        "q90_error": abs(float(np.mean(q90_hits)) - 0.90),
        "crps_improvement": float(
            1.0
            - _quantile_score(y, quantiles, TAUS_FULL) / _quantile_score(y, baseline_a, TAUS_FULL)
        ),
        "ceq_diff": float(
            ceq_annualized(offense / 100.0 * y) - ceq_annualized(0.5 * y),
        ),
    }


def main() -> None:
    rows = _load_rows()
    baseline = _evaluate_baseline(rows)
    theta = _fit_candidate(rows)
    candidate = _evaluate_candidate(rows, theta)
    payload = {
        "candidate_law": "state-conditional Student-t location-scale (mu=[1,x_scaled,pi], log_sigma=[1,pi], nu=5)",
        "objective": "mean quantile-score over taus 0.05..0.95 as CRPS proxy + 10*((cov10-0.10)^2 + (cov90-0.90)^2) + l2",
        "baseline": baseline,
        "candidate": candidate,
    }
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
