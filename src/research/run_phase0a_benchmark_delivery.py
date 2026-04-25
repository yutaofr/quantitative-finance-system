"""io: Phase 0B benchmark delivery runner for the Phase 0A lock."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import norm, spearmanr, t as student_t


WINDOWS: Final[dict[str, tuple[date, date]]] = {
    "Window_2017": (date(2017, 7, 7), date(2017, 12, 29)),
    "Window_2018": (date(2018, 7, 6), date(2018, 12, 28)),
    "Window_2020": (date(2020, 1, 3), date(2020, 6, 26)),
    "Window_2008_Benchmark": (date(2008, 1, 4), date(2008, 12, 26)),
}

HISTORY_START: Final[date] = date(1985, 10, 1)
FORECAST_HORIZON_WEEKS: Final[int] = 52
TRAIN_WINDOW_WEEKS: Final[int] = 416
TRAIN_EMBARGO_WEEKS: Final[int] = 53
CRPS_TAUS: Final[NDArray[np.float64]] = np.linspace(0.001, 0.999, 401, dtype=np.float64)
BLUP_THRESHOLD_MULTIPLIER: Final[float] = 10.0
MAX_OPTIMIZER_ITERS: Final[int] = 200
OPTIMIZER_TOL: Final[float] = 1.0e-6
MODEL_NAMES: Final[tuple[str, ...]] = (
    "EGARCH(1,1)-Normal",
    "EGARCH(1,1)-Student-t",
    "GJR-GARCH(1,1)-Normal",
    "GJR-GARCH(1,1)-Student-t",
)


@dataclass(frozen=True, slots=True)
class BenchmarkFit:
    """pure. Frozen fitted parameters for one benchmark model."""

    model_name: str
    params: NDArray[np.float64]
    objective_value: float
    optimizer_status: str
    converged: bool


def _load_weekly_forward_returns() -> pd.Series:
    """io: Load weekly Friday-close NDX forward 52-week returns from local parquet."""
    price_frame = pd.read_parquet(Path("data/raw/nasdaq/NASDAQXNDX/close.parquet"))
    price_frame["date"] = pd.to_datetime(price_frame["date"])
    weekly_prices = (
        price_frame.set_index("date")
        .sort_index()["close"]
        .resample("W-FRI")
        .last()
        .dropna()
        .astype(np.float64)
    )
    forward_returns = np.log(weekly_prices.shift(-FORECAST_HORIZON_WEEKS) / weekly_prices)
    forward_returns.name = "forward_52w_log_return"
    return forward_returns.dropna()


def _fit_bounds_normalized_student_t_df(nu: float) -> float:
    """pure. Expected absolute value of a variance-one Student-t shock."""
    if nu <= 2.0 or not np.isfinite(nu):
        return float("nan")
    return float(
        2.0
        * np.sqrt(nu - 2.0)
        * np.exp(gammaln((nu + 1.0) / 2.0) - gammaln(nu / 2.0))
        / ((nu - 1.0) * np.sqrt(np.pi))
    )


def _student_t_std_ppf(taus: NDArray[np.float64], nu: float) -> NDArray[np.float64]:
    """pure. Quantiles of a variance-one Student-t shock."""
    scale = np.sqrt((nu - 2.0) / nu)
    return student_t.ppf(taus, df=nu).astype(np.float64) * scale


def _pinball_crps_proxy(y_true: float, quantiles: NDArray[np.float64], taus: NDArray[np.float64]) -> float:
    """pure. Approximate CRPS by numerical integration of the pinball score."""
    delta = y_true - quantiles
    losses = np.maximum(taus * delta, (taus - 1.0) * delta)
    return float(2.0 * np.trapezoid(losses, taus))


def _normal_quantiles(mu: float, sigma: float, taus: NDArray[np.float64]) -> NDArray[np.float64]:
    return mu + sigma * norm.ppf(taus).astype(np.float64)


def _student_t_quantiles(mu: float, sigma: float, nu: float, taus: NDArray[np.float64]) -> NDArray[np.float64]:
    return mu + sigma * _student_t_std_ppf(taus, nu)


def _window_metrics(
    y_true: NDArray[np.float64],
    mu_hat: NDArray[np.float64],
    sigma_hat: NDArray[np.float64],
    crps_quantiles: NDArray[np.float64],
    *,
    sigma_blowup_count: int,
    pathology_count: int,
    corr_next: float,
    rank_next: float,
) -> dict[str, object]:
    e = y_true - mu_hat
    z = e / sigma_hat
    finite = np.isfinite(z) & np.isfinite(e) & np.isfinite(sigma_hat)
    z = z[finite]
    e = e[finite]
    sigma_hat = sigma_hat[finite]
    y_true = y_true[finite]
    if z.size > 1:
        lag1_acf = float(np.corrcoef(z[:-1], z[1:])[0, 1])
        lag1_abs_acf = float(np.corrcoef(np.abs(z[:-1]), np.abs(z[1:]))[0, 1])
    else:
        lag1_acf = float("nan")
        lag1_abs_acf = float("nan")
    if sigma_hat.size > 1:
        q = np.quantile(sigma_hat, [0.1, 0.5, 0.9, 0.99])
        sigma_p10, sigma_med, sigma_p90, sigma_p99 = [float(value) for value in q]
    else:
        sigma_p10 = sigma_med = sigma_p90 = sigma_p99 = float("nan")
    crps = float(np.mean(crps_quantiles[finite])) if finite.any() else float("nan")
    return {
        "mean_z": float(np.mean(z)) if z.size else float("nan"),
        "std_z": float(np.std(z, ddof=0)) if z.size else float("nan"),
        "median_abs_z": float(np.median(np.abs(z))) if z.size else float("nan"),
        "p90_abs_z": float(np.quantile(np.abs(z), 0.9)) if z.size else float("nan"),
        "lag1_acf_z": lag1_acf,
        "lag1_acf_abs_z": lag1_abs_acf,
        "sigma_med": sigma_med,
        "sigma_p10": sigma_p10,
        "sigma_p90": sigma_p90,
        "sigma_p99_over_med": float(sigma_p99 / sigma_med) if np.isfinite(sigma_p99) and np.isfinite(sigma_med) and sigma_med > 0.0 else float("nan"),
        "sigma_blowup": int(sigma_blowup_count),
        "pathology": int(pathology_count),
        "corr_next": float(corr_next),
        "rank_next": float(rank_next),
        "crps": crps,
        "n_obs": int(z.size),
    }


def _spearman(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    if a.size < 2 or b.size < 2:
        return float("nan")
    value = spearmanr(a, b, nan_policy="omit").correlation
    return float(value) if np.isfinite(value) else float("nan")


def _corr(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    if a.size < 2 or b.size < 2:
        return float("nan")
    value = np.corrcoef(a, b)[0, 1]
    return float(value) if np.isfinite(value) else float("nan")


def _gjr_transform(theta: NDArray[np.float64]) -> tuple[float, float, float, float, float]:
    mu = float(theta[0])
    omega = float(np.exp(theta[1]))
    alpha = float(0.5 / (1.0 + np.exp(-theta[2])))
    gamma = float(0.5 / (1.0 + np.exp(-theta[3])))
    beta_cap = max(1.0e-6, 0.999 - alpha - 0.5 * gamma)
    beta = float(beta_cap / (1.0 + np.exp(-theta[4])))
    return mu, omega, alpha, gamma, beta


def _egarch_transform(theta: NDArray[np.float64]) -> tuple[float, float, float, float, float]:
    mu = float(theta[0])
    omega = float(theta[1])
    alpha = float(0.5 * np.tanh(theta[2]))
    gamma = float(0.5 * np.tanh(theta[3]))
    beta = float(0.98 / (1.0 + np.exp(-theta[4])))
    return mu, omega, alpha, gamma, beta


def _gjr_filter(
    train: NDArray[np.float64],
    theta: NDArray[np.float64],
    *,
    with_student_t: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float | None]:
    mu, omega, alpha, gamma, beta = _gjr_transform(theta)
    residual = train - mu
    h = np.empty_like(residual)
    h[0] = max(float(np.var(residual, ddof=0)), 1.0e-8)
    for t in range(1, residual.size):
        asym = gamma * residual[t - 1] ** 2 if residual[t - 1] < 0.0 else 0.0
        h[t] = omega + alpha * residual[t - 1] ** 2 + asym + beta * h[t - 1]
        if not np.isfinite(h[t]) or h[t] <= 1.0e-12:
            raise FloatingPointError("invalid GJR variance recursion")
    nu = None
    if with_student_t:
        nu = float(2.1 + np.exp(theta[5]))
    return residual, h, nu


def _egarch_filter(
    train: NDArray[np.float64],
    theta: NDArray[np.float64],
    *,
    with_student_t: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float | None]:
    mu, omega, alpha, gamma, beta = _egarch_transform(theta)
    residual = train - mu
    log_h = np.empty_like(residual)
    log_h[0] = np.log(max(float(np.var(residual, ddof=0)), 1.0e-8))
    nu = None
    if with_student_t:
        nu = float(2.1 + np.exp(theta[5]))
        eabs = _fit_bounds_normalized_student_t_df(nu)
    else:
        eabs = float(np.sqrt(2.0 / np.pi))
    if not np.isfinite(eabs):
        raise FloatingPointError("invalid EGARCH absolute moment")
    for t in range(1, residual.size):
        z_prev = residual[t - 1] / np.exp(0.5 * log_h[t - 1])
        log_h[t] = omega + beta * log_h[t - 1] + alpha * (abs(z_prev) - eabs) + gamma * z_prev
        if not np.isfinite(log_h[t]) or log_h[t] < -20.0 or log_h[t] > 20.0:
            raise FloatingPointError("invalid EGARCH log variance recursion")
    return residual, np.exp(log_h), nu


def _gjr_nll(theta: NDArray[np.float64], train: NDArray[np.float64], *, with_student_t: bool) -> float:
    try:
        residual, h, nu = _gjr_filter(train, theta, with_student_t=with_student_t)
    except FloatingPointError:
        return 1.0e15
    if with_student_t:
        assert nu is not None
        z = residual / np.sqrt(h)
        scale = np.sqrt((nu - 2.0) / nu)
        standardized = z / scale
        logpdf = student_t.logpdf(standardized, df=nu) - np.log(scale) - 0.5 * np.log(h)
    else:
        z = residual / np.sqrt(h)
        logpdf = norm.logpdf(z) - 0.5 * np.log(h)
    if not np.isfinite(logpdf).all():
        return 1.0e15
    return float(-np.sum(logpdf))


def _egarch_nll(theta: NDArray[np.float64], train: NDArray[np.float64], *, with_student_t: bool) -> float:
    try:
        residual, h, nu = _egarch_filter(train, theta, with_student_t=with_student_t)
    except FloatingPointError:
        return 1.0e15
    z = residual / np.sqrt(h)
    if with_student_t:
        assert nu is not None
        scale = np.sqrt((nu - 2.0) / nu)
        standardized = z / scale
        logpdf = student_t.logpdf(standardized, df=nu) - np.log(scale) - 0.5 * np.log(h)
    else:
        logpdf = norm.logpdf(z) - 0.5 * np.log(h)
    if not np.isfinite(logpdf).all():
        return 1.0e15
    return float(-np.sum(logpdf))


def _fit_model(train: NDArray[np.float64], model_name: str, initial: NDArray[np.float64] | None = None) -> BenchmarkFit:
    if model_name == "GJR-GARCH(1,1)-Normal":
        objective = lambda theta: _gjr_nll(theta, train, with_student_t=False)
        if initial is None:
            initial = np.array([float(np.mean(train)), np.log(max(float(np.var(train, ddof=0)), 1.0e-6)), 0.0, 0.0, 0.0], dtype=np.float64)
    elif model_name == "GJR-GARCH(1,1)-Student-t":
        objective = lambda theta: _gjr_nll(theta, train, with_student_t=True)
        if initial is None:
            initial = np.array([float(np.mean(train)), np.log(max(float(np.var(train, ddof=0)), 1.0e-6)), 0.0, 0.0, 0.0, np.log(8.0 - 2.1)], dtype=np.float64)
    elif model_name == "EGARCH(1,1)-Normal":
        objective = lambda theta: _egarch_nll(theta, train, with_student_t=False)
        if initial is None:
            initial = np.array([float(np.mean(train)), -0.1, 0.0, 0.0, 0.0], dtype=np.float64)
    elif model_name == "EGARCH(1,1)-Student-t":
        objective = lambda theta: _egarch_nll(theta, train, with_student_t=True)
        if initial is None:
            initial = np.array([float(np.mean(train)), -0.1, 0.0, 0.0, 0.0, np.log(8.0 - 2.1)], dtype=np.float64)
    else:
        raise ValueError(model_name)
    result = minimize(
        objective,
        np.asarray(initial, dtype=np.float64),
        method="L-BFGS-B",
        options={"maxiter": MAX_OPTIMIZER_ITERS, "ftol": OPTIMIZER_TOL},
    )
    return BenchmarkFit(
        model_name=model_name,
        params=np.asarray(result.x, dtype=np.float64),
        objective_value=float(result.fun),
        optimizer_status=str(result.message),
        converged=bool(result.success),
    )


def _forecast_step(train: NDArray[np.float64], fit: BenchmarkFit) -> tuple[float, float, float | None]:
    if fit.model_name == "GJR-GARCH(1,1)-Normal":
        mu, omega, alpha, gamma, beta = _gjr_transform(fit.params)
        residual, h, _ = _gjr_filter(train, fit.params, with_student_t=False)
        last_eps = float(residual[-1])
        last_h = float(h[-1])
        h_next = omega + alpha * last_eps**2 + (gamma * last_eps**2 if last_eps < 0.0 else 0.0) + beta * last_h
        return mu, float(np.sqrt(h_next)), None
    if fit.model_name == "GJR-GARCH(1,1)-Student-t":
        mu, omega, alpha, gamma, beta = _gjr_transform(fit.params)
        residual, h, nu = _gjr_filter(train, fit.params, with_student_t=True)
        last_eps = float(residual[-1])
        last_h = float(h[-1])
        h_next = omega + alpha * last_eps**2 + (gamma * last_eps**2 if last_eps < 0.0 else 0.0) + beta * last_h
        assert nu is not None
        return mu, float(np.sqrt(h_next)), nu
    if fit.model_name == "EGARCH(1,1)-Normal":
        mu, omega, alpha, gamma, beta = _egarch_transform(fit.params)
        residual, h, _ = _egarch_filter(train, fit.params, with_student_t=False)
        z_last = float(residual[-1] / np.sqrt(h[-1]))
        logh_next = omega + beta * np.log(h[-1]) + alpha * (abs(z_last) - np.sqrt(2.0 / np.pi)) + gamma * z_last
        return mu, float(np.exp(0.5 * logh_next)), None
    if fit.model_name == "EGARCH(1,1)-Student-t":
        mu, omega, alpha, gamma, beta = _egarch_transform(fit.params)
        residual, h, nu = _egarch_filter(train, fit.params, with_student_t=True)
        z_last = float(residual[-1] / np.sqrt(h[-1]))
        assert nu is not None
        eabs = _fit_bounds_normalized_student_t_df(nu)
        logh_next = omega + beta * np.log(h[-1]) + alpha * (abs(z_last) - eabs) + gamma * z_last
        return mu, float(np.exp(0.5 * logh_next)), nu
    raise ValueError(fit.model_name)


def _evaluate_model(
    model_name: str,
    window: tuple[date, date],
    forward_returns: pd.Series,
) -> dict[str, object]:
    start, end = window
    eval_index = forward_returns.loc[pd.Timestamp(start) : pd.Timestamp(end)].dropna().index
    if eval_index.empty:
        return {"status": "FAILED_TO_RUN", "reason": "no evaluation weeks"}

    mu_list: list[float] = []
    sigma_list: list[float] = []
    y_list: list[float] = []
    z_list: list[float] = []
    crps_list: list[float] = []
    fit_failures = 0
    failure_reasons: list[str] = []
    blowup_count = 0
    last_fit: BenchmarkFit | None = None

    for as_of in eval_index:
        as_of_date = as_of.date()
        train_end = as_of_date - pd.Timedelta(weeks=TRAIN_EMBARGO_WEEKS)
        train = forward_returns.loc[:pd.Timestamp(train_end)].dropna().to_numpy(dtype=np.float64)
        if train.size < TRAIN_WINDOW_WEEKS:
            fit_failures += 1
            failure_reasons.append("insufficient_training_window")
            continue
        train = train[-TRAIN_WINDOW_WEEKS:]
        try:
            fit = _fit_model(train, model_name, initial=last_fit.params if last_fit is not None else None)
            if not fit.converged:
                fit_failures += 1
                failure_reasons.append(f"optimizer:{fit.optimizer_status}")
                last_fit = fit
                continue
            mu_hat, sigma_hat, nu = _forecast_step(train, fit)
        except (FloatingPointError, ArithmeticError, ValueError) as exc:
            fit_failures += 1
            failure_reasons.append(f"{type(exc).__name__}:{exc}")
            continue
        y_true = float(forward_returns.loc[as_of])
        if not np.isfinite(y_true) or not np.isfinite(mu_hat) or not np.isfinite(sigma_hat) or sigma_hat <= 0.0:
            fit_failures += 1
            failure_reasons.append("invalid_forecast_output")
            continue
        median_train_sigma = float(np.median(np.abs(train - np.mean(train))))
        if not np.isfinite(median_train_sigma) or median_train_sigma <= 0.0:
            median_train_sigma = float(np.std(train, ddof=0)) if np.isfinite(np.std(train, ddof=0)) else 1.0
        if sigma_hat > BLUP_THRESHOLD_MULTIPLIER * median_train_sigma:
            blowup_count += 1
        mu_list.append(mu_hat)
        sigma_list.append(sigma_hat)
        y_list.append(y_true)
        z_list.append((y_true - mu_hat) / sigma_hat)
        if model_name.endswith("Normal"):
            taus = CRPS_TAUS
            quantiles = _normal_quantiles(mu_hat, sigma_hat, taus)
        else:
            assert nu is not None
            taus = CRPS_TAUS
            quantiles = _student_t_quantiles(mu_hat, sigma_hat, nu, taus)
        crps_list.append(_pinball_crps_proxy(y_true, quantiles, taus))
        last_fit = fit

    if not y_list:
        return {
            "status": "FAILED_TO_RUN",
            "reason": "no valid fits",
            "fit_failures": fit_failures,
            "failure_reasons": sorted(set(failure_reasons)),
        }

    y_arr = np.asarray(y_list, dtype=np.float64)
    mu_arr = np.asarray(mu_list, dtype=np.float64)
    sigma_arr = np.asarray(sigma_list, dtype=np.float64)
    z_arr = np.asarray(z_list, dtype=np.float64)
    abs_resid = np.abs(y_arr - mu_arr)
    next_sigma = sigma_arr[:-1]
    next_abs = abs_resid[1:]
    same_corr = _corr(sigma_arr, abs_resid)
    next_corr = _corr(next_sigma, next_abs)
    same_rank = _spearman(sigma_arr, abs_resid)
    next_rank = _spearman(next_sigma, next_abs)
    metrics = _window_metrics(
        y_arr,
        mu_arr,
        sigma_arr,
        np.asarray(crps_list, dtype=np.float64),
        sigma_blowup_count=blowup_count,
        pathology_count=fit_failures,
        corr_next=next_corr,
        rank_next=next_rank,
    )
    metrics.update(
        {
            "corr_same": same_corr,
            "rank_same": same_rank,
            "optimizer_failures": fit_failures,
            "failure_reasons": sorted(set(failure_reasons)),
            "status": "PASS" if fit_failures == 0 and blowup_count == 0 else "FAILED_TO_RUN",
            "n_eval": int(y_arr.size),
        },
    )
    return metrics


def run(argv: list[str] | None = None) -> int:
    """io: Execute benchmark delivery and print a JSON summary."""
    parser = argparse.ArgumentParser(prog="run_phase0a_benchmark_delivery")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    args = parser.parse_args(argv)

    forward_returns = _load_weekly_forward_returns()
    model_results: dict[str, dict[str, object]] = {}
    for model_name in MODEL_NAMES:
        window_results: dict[str, dict[str, object]] = {}
        for window_name, window in WINDOWS.items():
            window_results[window_name] = _evaluate_model(model_name, window, forward_returns)
        model_results[model_name] = window_results

    payload = {
        "benchmark_family": "EGARCH/GJR-GARCH(1,1) fixed benchmark family",
        "windows": {name: [start.isoformat(), end.isoformat()] for name, (start, end) in WINDOWS.items()},
        "train_window_weeks": TRAIN_WINDOW_WEEKS,
        "train_embargo_weeks": TRAIN_EMBARGO_WEEKS,
        "forecast_horizon_weeks": FORECAST_HORIZON_WEEKS,
        "results": model_results,
    }
    if args.pretty:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(json.dumps(payload, sort_keys=True))
    return 0


def main() -> None:
    """io: process exit entrypoint."""
    raise SystemExit(run())


if __name__ == "__main__":
    main()
