"""io: run independent minimum falsification experiments for new research hypotheses."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm, rankdata, t as student_t

from research.run_phase0a_benchmark_delivery import (
    FORECAST_HORIZON_WEEKS,
    TRAIN_EMBARGO_WEEKS,
    TRAIN_WINDOW_WEEKS,
    _load_weekly_forward_returns,
)

HYPOTHESIS_IDS: Final[tuple[str, ...]] = (
    "A_DIRECT_DENSITY",
    "B_DIRECT_QUANTILES",
    "C_DECOUPLED_HEADS",
    "D_LATENT_STATE",
)
PILOT_WINDOWS: Final[dict[str, tuple[date, date]]] = {
    "Window_2017": (date(2017, 7, 7), date(2017, 12, 29)),
    "Window_2018": (date(2018, 7, 6), date(2018, 12, 28)),
    "Window_2020": (date(2020, 1, 3), date(2020, 6, 26)),
}
CORE_METRIC_KEYS: Final[tuple[str, ...]] = (
    "mean_z",
    "std_z",
    "corr_next",
    "rank_next",
    "lag1_acf_z",
    "sigma_blowup",
    "pathology",
    "crps",
)
TAUS: Final[NDArray[np.float64]] = np.array([0.10, 0.25, 0.50, 0.75, 0.90], dtype=np.float64)
CRPS_TAUS: Final[NDArray[np.float64]] = np.linspace(0.01, 0.99, 199, dtype=np.float64)
OUTPUT_DIR: Final[Path] = Path("artifacts/research/new_hypotheses")
MATERIAL_IMPROVEMENT_MIN: Final[float] = 0.03
RIDGE_ALPHA: Final[float] = 1.0
MIN_SIGMA: Final[float] = 1.0e-4
MIN_TRAIN_OBS: Final[int] = 120
MIN_CORR_OBS: Final[int] = 2
MIN_STATE_OBS: Final[int] = 20
DEGENERATE_STD_GUARD: Final[float] = 1.0e-12
SIGMA_BLOWUP_MULT: Final[float] = 10.0
STD_SCALE_TOLERANCE: Final[float] = 0.35
DF_GRID: Final[tuple[float, ...]] = (5.0, 8.0, 12.0, 30.0)

BASELINE_SUMMARY: Final[dict[str, dict[str, dict[str, float]]]] = {
    "T5": {
        "Window_2017": {
            "corr_next": 0.629802,
            "rank_next": 0.776923,
            "std_z": 2.010464,
            "crps": 0.086197,
        },
        "Window_2018": {
            "corr_next": 0.386938,
            "rank_next": 0.535385,
            "std_z": 1.976233,
            "crps": 0.067559,
        },
        "Window_2020": {
            "corr_next": -0.033011,
            "rank_next": 0.020000,
            "std_z": 2.380050,
            "crps": 0.054682,
        },
    },
    "EGARCH_NORMAL": {
        "Window_2017": {
            "corr_next": 0.530067,
            "rank_next": 0.504615,
            "std_z": 1.702306,
            "crps": 0.061159,
        },
        "Window_2018": {
            "corr_next": -0.485068,
            "rank_next": -0.580769,
            "std_z": 2.177269,
            "crps": 0.095610,
        },
        "Window_2020": {
            "corr_next": 0.412949,
            "rank_next": 0.362308,
            "std_z": 0.810571,
            "crps": 0.160771,
        },
    },
    "OLD_FAILED_OPTION_2A_B": {
        "Window_2017": {
            "corr_next": 0.015043,
            "rank_next": 0.051538,
            "std_z": 1.087309,
            "crps": 0.086197,
        },
        "Window_2018": {
            "corr_next": -0.158647,
            "rank_next": 0.077692,
            "std_z": 1.189256,
            "crps": 0.067559,
        },
        "Window_2020": {
            "corr_next": -0.287052,
            "rank_next": -0.263478,
            "std_z": 1.026910,
            "crps": 0.054682,
        },
    },
}


@dataclass(frozen=True, slots=True)
class ForecastSeries:
    """pure. Forecast arrays for one hypothesis and window."""

    dates: tuple[date, ...]
    y: NDArray[np.float64]
    mu: NDArray[np.float64]
    sigma: NDArray[np.float64]
    crps: NDArray[np.float64]
    pathology: int


@dataclass(frozen=True, slots=True)
class FeatureTable:
    """pure. Embargo-safe features aligned to target forward returns."""

    dates: tuple[date, ...]
    y: NDArray[np.float64]
    x: NDArray[np.float64]


def _trapezoid(y_values: NDArray[np.float64], x_values: NDArray[np.float64]) -> float:
    """pure. Trapezoid integration compatible with NumPy 1.x and 2.x."""
    widths = np.diff(x_values)
    heights = (y_values[:-1] + y_values[1:]) * 0.5
    return float(np.sum(widths * heights))


def _pinball_crps(
    y_true: float,
    quantiles: NDArray[np.float64],
    taus: NDArray[np.float64],
) -> float:
    """pure. Proper-score proxy from integrated pinball loss."""
    delta = y_true - quantiles
    losses = np.maximum(taus * delta, (taus - 1.0) * delta)
    return float(2.0 * _trapezoid(losses, taus))


def _normal_crps(y_true: float, mu: float, sigma: float) -> float:
    """pure. Normal CRPS proxy from quantile integration."""
    quantiles = mu + sigma * norm.ppf(CRPS_TAUS).astype(np.float64)
    return _pinball_crps(y_true, quantiles, CRPS_TAUS)


def _student_t_crps(y_true: float, mu: float, sigma: float, df: float) -> float:
    """pure. Student-t CRPS proxy from quantile integration."""
    scale = float(np.sqrt((df - 2.0) / df))
    quantiles = mu + sigma * student_t.ppf(CRPS_TAUS, df=df).astype(np.float64) * scale
    return _pinball_crps(y_true, quantiles, CRPS_TAUS)


def _corr(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """pure. Pearson correlation with degenerate guard."""
    finite = np.isfinite(a) & np.isfinite(b)
    if int(finite.sum()) < MIN_CORR_OBS:
        return float("nan")
    x = a[finite]
    y = b[finite]
    if (
        float(np.std(x, ddof=0)) <= DEGENERATE_STD_GUARD
        or float(np.std(y, ddof=0)) <= DEGENERATE_STD_GUARD
    ):
        return float("nan")
    value = float(np.corrcoef(x, y)[0, 1])
    return value if np.isfinite(value) else float("nan")


def _rank_corr(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """pure. Spearman rank correlation with degenerate guard."""
    return _corr(rankdata(a).astype(np.float64), rankdata(b).astype(np.float64))


def _ridge_fit(x: NDArray[np.float64], y: NDArray[np.float64], alpha: float) -> NDArray[np.float64]:
    """pure. Ridge fit with unpenalized intercept."""
    penalty = alpha * np.eye(x.shape[1], dtype=np.float64)
    penalty[0, 0] = 0.0
    return np.linalg.solve(x.T @ x + penalty, x.T @ y).astype(np.float64)


def _build_feature_table(forward_returns: pd.Series) -> FeatureTable:
    """pure. Build low-DOF embargo-safe features from the target series itself."""
    y_series = forward_returns.dropna().astype(np.float64)
    lag = y_series.shift(TRAIN_EMBARGO_WEEKS)
    rolling_mean = y_series.shift(TRAIN_EMBARGO_WEEKS).rolling(52, min_periods=26).mean()
    rolling_std = y_series.shift(TRAIN_EMBARGO_WEEKS).rolling(52, min_periods=26).std(ddof=0)
    frame = pd.DataFrame(
        {
            "y": y_series,
            "lag": lag,
            "rolling_mean": rolling_mean,
            "rolling_std": rolling_std,
        },
    ).dropna()
    dates = tuple(ts.date() for ts in frame.index)
    x = np.column_stack(
        [
            np.ones(len(frame), dtype=np.float64),
            frame["lag"].to_numpy(dtype=np.float64),
            frame["rolling_mean"].to_numpy(dtype=np.float64),
            frame["rolling_std"].to_numpy(dtype=np.float64),
        ],
    )
    return FeatureTable(dates=dates, y=frame["y"].to_numpy(dtype=np.float64), x=x)


def _training_indices(table: FeatureTable, as_of: date) -> NDArray[np.int64]:
    """pure. Return fixed-window training indices ending before the embargo boundary."""
    cutoff = pd.Timestamp(as_of) - pd.Timedelta(weeks=TRAIN_EMBARGO_WEEKS)
    eligible = np.asarray(
        [idx for idx, day in enumerate(table.dates) if pd.Timestamp(day) <= cutoff],
        dtype=np.int64,
    )
    if eligible.size <= TRAIN_WINDOW_WEEKS:
        return eligible
    return eligible[-TRAIN_WINDOW_WEEKS:]


def _eval_indices(table: FeatureTable, window: tuple[date, date]) -> NDArray[np.int64]:
    """pure. Return target indices in one pilot window."""
    start, end = window
    return np.asarray(
        [idx for idx, day in enumerate(table.dates) if start <= day <= end],
        dtype=np.int64,
    )


def _fit_density_params(
    x_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
    x_eval: NDArray[np.float64],
) -> tuple[float, float, float]:
    """pure. Fit direct location-scale-t parameters for one evaluation row."""
    beta_mu = _ridge_fit(x_train, y_train, RIDGE_ALPHA)
    mu_train = x_train @ beta_mu
    residual = y_train - mu_train
    sigma_floor = max(float(np.std(residual, ddof=0)) * 0.05, MIN_SIGMA)
    log_abs = np.log(np.maximum(np.abs(residual), sigma_floor))
    beta_log_sigma = _ridge_fit(x_train, log_abs, RIDGE_ALPHA)
    sigma_train = np.maximum(np.exp(x_train @ beta_log_sigma), MIN_SIGMA)
    z_train = residual / sigma_train
    best_df = min(
        DF_GRID,
        key=lambda df: float(-np.sum(student_t.logpdf(z_train / np.sqrt((df - 2.0) / df), df=df))),
    )
    mu_hat = float(x_eval @ beta_mu)
    sigma_hat = float(max(np.exp(float(x_eval @ beta_log_sigma)), MIN_SIGMA))
    return mu_hat, sigma_hat, float(best_df)


def _fit_quantile_betas(
    x_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
) -> NDArray[np.float64]:
    """pure. Fit independent linear quantiles and enforce monotonicity at prediction time."""
    betas: list[NDArray[np.float64]] = []
    initial = _ridge_fit(x_train, y_train, RIDGE_ALPHA)
    for tau in TAUS:
        tau_value = float(tau)

        def objective(beta: NDArray[np.float64], *, tau_bound: float = tau_value) -> float:
            residual = y_train - x_train @ beta
            loss = np.maximum(tau_bound * residual, (tau_bound - 1.0) * residual)
            return float(np.sum(loss) + 0.5 * RIDGE_ALPHA * np.sum(beta[1:] * beta[1:]))

        result = minimize(objective, initial, method="BFGS", options={"maxiter": 250})
        beta = np.asarray(result.x if result.success else initial, dtype=np.float64)
        betas.append(beta)
        initial = beta
    return np.vstack(betas).astype(np.float64)


def _quantile_forecast(
    betas: NDArray[np.float64],
    x_eval: NDArray[np.float64],
) -> tuple[float, float, NDArray[np.float64]]:
    """pure. Convert direct quantile surface into comparable median and scale diagnostics."""
    q_vals = np.maximum.accumulate(betas @ x_eval)
    q50 = float(q_vals[2])
    iqr = max(float(q_vals[3] - q_vals[1]), MIN_SIGMA)
    sigma = iqr / float(norm.ppf(0.75) - norm.ppf(0.25))
    return q50, max(sigma, MIN_SIGMA), q_vals


def _decoupled_forecast(
    x_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
    x_eval: NDArray[np.float64],
) -> tuple[float, float]:
    """pure. Fit separate ordinal direction and cardinal scale heads."""
    ranks = (rankdata(y_train).astype(np.float64) - 0.5) / float(y_train.size)
    ordinal_target = norm.ppf(np.clip(ranks, 0.01, 0.99)).astype(np.float64)
    beta_rank = _ridge_fit(x_train, ordinal_target, RIDGE_ALPHA)
    scale_target = np.log(np.maximum(np.abs(y_train - np.median(y_train)), MIN_SIGMA))
    beta_scale = _ridge_fit(x_train, scale_target, RIDGE_ALPHA)
    score_eval = float(x_eval @ beta_rank)
    sigma_hat = float(max(np.exp(float(x_eval @ beta_scale)), MIN_SIGMA))
    direction = float(np.tanh(score_eval))
    mu_hat = float(np.median(y_train) + direction * sigma_hat)
    return mu_hat, sigma_hat


def _latent_state_forecast(
    x_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
    x_eval: NDArray[np.float64],
) -> tuple[float, float]:
    """pure. Low-complexity two-state model keyed by embargoed rolling volatility."""
    train_state_feature = x_train[:, 3] if x_train.size else np.asarray([], dtype=np.float64)
    threshold = float(np.median(train_state_feature))
    states = train_state_feature > threshold
    state = bool(float(x_eval[3]) > threshold)
    state_y = y_train[states == state]
    if state_y.size < MIN_STATE_OBS:
        state_y = y_train
    mu_hat = float(np.mean(state_y))
    sigma_hat = float(max(np.std(state_y, ddof=0), MIN_SIGMA))
    return mu_hat, sigma_hat


def _series_metrics(series: ForecastSeries) -> dict[str, float | int]:
    """pure. Compute unified diagnostics for one forecast series."""
    finite = (
        np.isfinite(series.y)
        & np.isfinite(series.mu)
        & np.isfinite(series.sigma)
        & (series.sigma > 0.0)
    )
    y = series.y[finite]
    mu = series.mu[finite]
    sigma = series.sigma[finite]
    e = y - mu
    z = e / sigma
    sigma_med = float(np.median(sigma)) if sigma.size else float("nan")
    sigma_blowup = (
        int(np.sum(sigma > SIGMA_BLOWUP_MULT * sigma_med))
        if sigma.size and sigma_med > 0.0
        else 0
    )
    corr_next = _corr(sigma[:-1], np.abs(e[1:])) if sigma.size > 1 else float("nan")
    rank_next = _rank_corr(sigma[:-1], np.abs(e[1:])) if sigma.size > 1 else float("nan")
    lag1_acf = _corr(z[:-1], z[1:]) if z.size > 1 else float("nan")
    return {
        "mean_z": float(np.mean(z)) if z.size else float("nan"),
        "std_z": float(np.std(z, ddof=0)) if z.size else float("nan"),
        "corr_next": corr_next if np.isfinite(corr_next) else -1.0,
        "rank_next": rank_next if np.isfinite(rank_next) else -1.0,
        "lag1_acf_z": lag1_acf if np.isfinite(lag1_acf) else 0.0,
        "sigma_blowup": sigma_blowup,
        "pathology": int(series.pathology),
        "crps": float(np.mean(series.crps[finite])) if np.any(finite) else float("nan"),
        "n_obs": int(z.size),
    }


def _window_metric_payload_is_complete(payload: Mapping[str, object]) -> bool:
    """pure. Check the unified output contract for one window metric payload."""
    return all(key in payload for key in CORE_METRIC_KEYS)


def _run_model_for_window(
    table: FeatureTable,
    hypothesis_id: str,
    window: tuple[date, date],
) -> ForecastSeries:
    """pure. Run one hypothesis on one fixed pilot window."""
    dates: list[date] = []
    y_values: list[float] = []
    mu_values: list[float] = []
    sigma_values: list[float] = []
    crps_values: list[float] = []
    pathology = 0

    for eval_idx in _eval_indices(table, window):
        as_of = table.dates[int(eval_idx)]
        train_idx = _training_indices(table, as_of)
        if train_idx.size < MIN_TRAIN_OBS:
            pathology += 1
            continue
        x_train = table.x[train_idx]
        y_train = table.y[train_idx]
        x_eval = table.x[int(eval_idx)]
        try:
            if hypothesis_id == "A_DIRECT_DENSITY":
                mu_hat, sigma_hat, df = _fit_density_params(x_train, y_train, x_eval)
                crps = _student_t_crps(float(table.y[int(eval_idx)]), mu_hat, sigma_hat, df)
            elif hypothesis_id == "B_DIRECT_QUANTILES":
                betas = _fit_quantile_betas(x_train, y_train)
                mu_hat, sigma_hat, q_vals = _quantile_forecast(betas, x_eval)
                crps = _pinball_crps(float(table.y[int(eval_idx)]), q_vals, TAUS)
            elif hypothesis_id == "C_DECOUPLED_HEADS":
                mu_hat, sigma_hat = _decoupled_forecast(x_train, y_train, x_eval)
                crps = _normal_crps(float(table.y[int(eval_idx)]), mu_hat, sigma_hat)
            elif hypothesis_id == "D_LATENT_STATE":
                mu_hat, sigma_hat = _latent_state_forecast(x_train, y_train, x_eval)
                crps = _normal_crps(float(table.y[int(eval_idx)]), mu_hat, sigma_hat)
            else:
                raise ValueError(hypothesis_id)
        except (FloatingPointError, ArithmeticError, ValueError, np.linalg.LinAlgError):
            pathology += 1
            continue
        if (
            not np.isfinite(mu_hat)
            or not np.isfinite(sigma_hat)
            or sigma_hat <= 0.0
            or not np.isfinite(crps)
        ):
            pathology += 1
            continue
        dates.append(as_of)
        y_values.append(float(table.y[int(eval_idx)]))
        mu_values.append(mu_hat)
        sigma_values.append(sigma_hat)
        crps_values.append(crps)

    return ForecastSeries(
        dates=tuple(dates),
        y=np.asarray(y_values, dtype=np.float64),
        mu=np.asarray(mu_values, dtype=np.float64),
        sigma=np.asarray(sigma_values, dtype=np.float64),
        crps=np.asarray(crps_values, dtype=np.float64),
        pathology=pathology,
    )


def _aggregate_candidate(
    window_metrics: Mapping[str, Mapping[str, float | int]],
) -> dict[str, float | bool]:
    """pure. Aggregate one hypothesis against preregistered cross-hypothesis gates."""
    t5_direction_margin = min(
        float(window_metrics[name]["corr_next"]) - BASELINE_SUMMARY["T5"][name]["corr_next"]
        for name in PILOT_WINDOWS
    )
    egarch_scale_margin = min(
        BASELINE_SUMMARY["EGARCH_NORMAL"][name]["std_z"]
        + STD_SCALE_TOLERANCE
        - float(window_metrics[name]["std_z"])
        for name in PILOT_WINDOWS
    )
    old_crps = np.mean(
        [BASELINE_SUMMARY["OLD_FAILED_OPTION_2A_B"][name]["crps"] for name in PILOT_WINDOWS],
    )
    candidate_crps = np.mean([float(window_metrics[name]["crps"]) for name in PILOT_WINDOWS])
    material_improvement_score = float((old_crps - candidate_crps) / old_crps)
    no_blowup = all(int(window_metrics[name]["sigma_blowup"]) == 0 for name in PILOT_WINDOWS)
    no_pathology = all(int(window_metrics[name]["pathology"]) == 0 for name in PILOT_WINDOWS)
    return {
        "t5_direction_margin_min": float(t5_direction_margin),
        "egarch_scale_margin_min": float(egarch_scale_margin),
        "mean_crps": float(candidate_crps),
        "material_improvement_score": material_improvement_score,
        "no_blowup": no_blowup,
        "no_pathology": no_pathology,
    }


def _single_line_decision(aggregate: Mapping[str, float | bool]) -> tuple[str, bool]:
    """pure. Apply single-line and unified continuation gates."""
    protocol_pass = bool(
        aggregate["t5_direction_margin_min"] >= 0.0
        and aggregate["egarch_scale_margin_min"] >= 0.0
        and aggregate["no_blowup"]
        and aggregate["no_pathology"]
        and aggregate["material_improvement_score"] >= MATERIAL_IMPROVEMENT_MIN
    )
    if protocol_pass:
        return "WORTH_CONTINUING", True
    if (
        bool(aggregate["no_pathology"])
        and bool(aggregate["no_blowup"])
        and float(aggregate["material_improvement_score"]) > 0.0
    ):
        return "PROMISING_BUT_INSUFFICIENT", False
    return "FAILED", False


def _apply_final_decision(candidates: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    """pure. Select exactly one winner, terminate, or mark implementation limits."""
    invalid = [
        name
        for name, payload in candidates.items()
        if payload.get("single_line_decision") == "INVALID_IMPLEMENTATION"
    ]
    if invalid:
        return {
            "overall_decision": "INCONCLUSIVE_DUE_TO_IMPLEMENTATION_LIMITS",
            "winner": None,
            "invalid": invalid,
        }
    worth: list[tuple[str, float]] = []
    for name, payload in candidates.items():
        aggregate = payload.get("aggregate")
        if (
            isinstance(aggregate, Mapping)
            and payload.get("single_line_decision") == "WORTH_CONTINUING"
            and payload.get("protocol_pass") is True
        ):
            worth.append((name, float(aggregate["material_improvement_score"])))
    if not worth:
        return {"overall_decision": "NO_MODEL_WORTH_CONTINUING", "winner": None, "invalid": []}
    worth.sort(key=lambda item: item[1], reverse=True)
    return {"overall_decision": "SELECT_ONE_CONTINUE", "winner": worth[0][0], "invalid": []}


def _json_default(value: object) -> object:
    """pure. JSON encoder for numpy scalar values."""
    if isinstance(value, np.floating | np.integer):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _markdown_report(payload: Mapping[str, object]) -> str:
    """pure. Render a compact result table."""
    lines = [
        "# New Hypotheses Parallel Results",
        "",
        "> Generated by `src/research/run_new_hypotheses_parallel_experiment.py`.",
        "",
        "## Single-Line Decisions",
        "",
        "| hypothesis | decision | protocol_pass | mean_crps | material_improvement |",
        "|---|---|---:|---:|---:|",
    ]
    candidates = payload["candidates"]
    if not isinstance(candidates, Mapping):
        raise TypeError("candidates payload must be a mapping")
    for hypothesis_id in HYPOTHESIS_IDS:
        candidate = candidates[hypothesis_id]
        if not isinstance(candidate, Mapping):
            raise TypeError("candidate payload must be a mapping")
        aggregate = candidate["aggregate"]
        if not isinstance(aggregate, Mapping):
            raise TypeError("aggregate payload must be a mapping")
        lines.append(
            "| "
            f"{hypothesis_id} | {candidate['single_line_decision']} | "
            f"{candidate['protocol_pass']} | "
            f"{float(aggregate['mean_crps']):.6f} | "
            f"{float(aggregate['material_improvement_score']):.6f} |",
        )
    final = payload["final_decision"]
    if not isinstance(final, Mapping):
        raise TypeError("final decision payload must be a mapping")
    lines.extend(
        [
            "",
            "## Final Decision",
            "",
            f"- overall_decision: `{final['overall_decision']}`",
            f"- winner: `{final['winner']}`",
        ],
    )
    return "\n".join(lines) + "\n"


def run(argv: Sequence[str] | None = None) -> int:
    """io: Execute all new-hypothesis minimum experiments and write artifacts."""
    parser = argparse.ArgumentParser(prog="run_new_hypotheses_parallel_experiment")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args(list(argv) if argv is not None else None)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table = _build_feature_table(_load_weekly_forward_returns())
    candidates: dict[str, dict[str, object]] = {}
    for hypothesis_id in HYPOTHESIS_IDS:
        windows: dict[str, dict[str, float | int]] = {}
        for window_name, window in PILOT_WINDOWS.items():
            series = _run_model_for_window(table, hypothesis_id, window)
            windows[window_name] = _series_metrics(series)
        if not all(_window_metric_payload_is_complete(metrics) for metrics in windows.values()):
            candidates[hypothesis_id] = {
                "windows": windows,
                "aggregate": {},
                "single_line_decision": "INVALID_IMPLEMENTATION",
                "protocol_pass": False,
            }
            continue
        aggregate = _aggregate_candidate(windows)
        decision, protocol_pass = _single_line_decision(aggregate)
        candidates[hypothesis_id] = {
            "windows": windows,
            "aggregate": aggregate,
            "single_line_decision": decision,
            "protocol_pass": protocol_pass,
        }
    payload: dict[str, object] = {
        "hypotheses": HYPOTHESIS_IDS,
        "windows": {
            name: [start.isoformat(), end.isoformat()]
            for name, (start, end) in PILOT_WINDOWS.items()
        },
        "train_window_weeks": TRAIN_WINDOW_WEEKS,
        "train_embargo_weeks": TRAIN_EMBARGO_WEEKS,
        "forecast_horizon_weeks": FORECAST_HORIZON_WEEKS,
        "material_improvement_min": MATERIAL_IMPROVEMENT_MIN,
        "baselines": BASELINE_SUMMARY,
        "candidates": candidates,
    }
    payload["final_decision"] = _apply_final_decision(candidates)
    (output_dir / "parallel_results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    (output_dir / "parallel_results.md").write_text(_markdown_report(payload), encoding="utf-8")
    return 0


def main() -> None:
    """io: process exit entrypoint."""
    raise SystemExit(run())


if __name__ == "__main__":
    main()
