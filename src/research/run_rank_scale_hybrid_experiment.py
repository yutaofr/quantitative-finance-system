"""io: run the preregistered rank-scale hybrid sigma experiment."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path
import sys
from typing import Any, Final, Protocol

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from scipy.stats import norm, rankdata

PILOT_WINDOWS: Final[dict[str, tuple[date, date]]] = {
    "Window_2017": (date(2017, 7, 7), date(2017, 12, 29)),
    "Window_2018": (date(2018, 7, 6), date(2018, 12, 28)),
    "Window_2020": (date(2020, 1, 3), date(2020, 6, 26)),
}
EGARCH_NORMAL: Final[str] = "EGARCH(1,1)-Normal"
OUTPUT_DIR: Final[Path] = Path("artifacts/research/rank_scale_hybrid")
PREREG_DOC: Final[Path] = Path("docs/rank_scale_hybrid/02_preregistration.md")
CRPS_TAUS: Final[NDArray[np.float64]] = np.linspace(0.001, 0.999, 401, dtype=np.float64)
COMPRESSION_ALPHA: Final[float] = 0.50
MIN_CORR_OBS: Final[int] = 2
STD_Z_MAX: Final[float] = 1.5


@dataclass(frozen=True, slots=True)
class ForecastSeries:
    """pure. Aligned forecast arrays for one model and window."""

    dates: tuple[date, ...]
    y: NDArray[np.float64]
    mu: NDArray[np.float64]
    sigma: NDArray[np.float64]
    e: NDArray[np.float64]
    crps: NDArray[np.float64]
    pathology: int


class BenchmarkFitLike(Protocol):
    """pure. Structural type for benchmark fit objects used by this runner."""

    params: NDArray[np.float64]
    converged: bool


def _json_default(obj: object) -> object:
    """pure. JSON encoder for numpy scalar values."""
    if isinstance(obj, np.floating | np.integer):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def _normal_crps_proxy(y_true: float, mu: float, sigma: float) -> float:
    """pure. Approximate normal CRPS by numerical integration of pinball scores."""
    quantiles = mu + sigma * norm.ppf(CRPS_TAUS).astype(np.float64)
    delta = y_true - quantiles
    losses = np.maximum(CRPS_TAUS * delta, (CRPS_TAUS - 1.0) * delta)
    return float(2.0 * _trapezoid(losses, CRPS_TAUS))


def _trapezoid(y_values: NDArray[np.float64], x_values: NDArray[np.float64]) -> float:
    """pure. Local trapezoid integration compatible with NumPy 1.x and 2.x."""
    widths = np.diff(x_values)
    heights = (y_values[:-1] + y_values[1:]) * 0.5
    return float(np.sum(widths * heights))


def _rank_percentiles(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """pure. Convert values to inclusive empirical rank percentiles."""
    if values.size <= 1:
        return np.zeros_like(values, dtype=np.float64)
    ranks = rankdata(values, method="average").astype(np.float64)
    return (ranks - 1.0) / float(values.size - 1)


def _corr(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """pure. Pearson correlation with finite guard."""
    if a.size < MIN_CORR_OBS or b.size < MIN_CORR_OBS:
        return float("nan")
    value = float(np.corrcoef(a, b)[0, 1])
    return value if np.isfinite(value) else float("nan")


def _rank_corr(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """pure. Spearman rank correlation with finite guard."""
    if a.size < MIN_CORR_OBS or b.size < MIN_CORR_OBS:
        return float("nan")
    return _corr(rankdata(a).astype(np.float64), rankdata(b).astype(np.float64))


def _acf1(values: NDArray[np.float64]) -> float:
    """pure. Lag-1 autocorrelation with finite guard."""
    if values.size < MIN_CORR_OBS:
        return float("nan")
    return _corr(values[:-1], values[1:])


def _protocol_decision(metrics: dict[str, float | int]) -> str:
    """pure. Apply preregistered rank-scale hybrid decision labels."""
    direction_ok = bool(metrics["corr_next"] > 0.0 and metrics["rank_next"] > 0.0)
    scale_ok = bool(metrics["std_z"] < STD_Z_MAX)
    safety_ok = bool(metrics["sigma_blowup"] == 0 and metrics["pathology"] == 0)
    if direction_ok and scale_ok and safety_ok:
        return "SUCCESS"
    if direction_ok and safety_ok and not scale_ok:
        return "PARTIAL_FAIL_SCALE"
    if scale_ok and safety_ok and not direction_ok:
        return "PARTIAL_FAIL_DIRECTION"
    return "FULL_FAIL"


def _metrics(series: ForecastSeries) -> dict[str, float | int | str]:
    """pure. Compute the preregistered diagnostic metrics for one forecast series."""
    finite = (
        np.isfinite(series.y)
        & np.isfinite(series.mu)
        & np.isfinite(series.sigma)
        & np.isfinite(series.e)
        & (series.sigma > 0.0)
    )
    y = series.y[finite]
    mu = series.mu[finite]
    sigma = series.sigma[finite]
    e = y - mu
    z = e / sigma
    abs_resid = np.abs(e)
    sigma_blowup = int(np.sum(sigma > 2.0 * float(np.median(sigma)))) if sigma.size else 0
    metric_values: dict[str, float | int] = {
        "mean_z": float(np.mean(z)) if z.size else float("nan"),
        "std_z": float(np.std(z, ddof=0)) if z.size else float("nan"),
        "corr_next": _corr(sigma[:-1], abs_resid[1:]) if sigma.size > 1 else float("nan"),
        "rank_next": _rank_corr(sigma[:-1], abs_resid[1:]) if sigma.size > 1 else float("nan"),
        "lag1_acf_z": _acf1(z),
        "sigma_blowup": sigma_blowup,
        "pathology": int(series.pathology),
        "crps": float(np.mean(series.crps[finite])) if np.any(finite) else float("nan"),
        "n_obs": int(z.size),
    }
    return {**metric_values, "decision": _protocol_decision(metric_values)}


def _t5_series_by_window() -> dict[str, ForecastSeries]:
    """io: Run recovered T5 and return three pilot-window forecast arrays."""
    from research import t5_recovered_source as t5  # noqa: PLC0415

    context = t5.build_har_context(max(end for _start, end in PILOT_WINDOWS.values()))
    result: dict[str, ForecastSeries] = {}
    for window_name, (start, end) in PILOT_WINDOWS.items():
        arr = t5._eval_original_t5_window(context, start, end)
        dates = tuple(week for week in context.base.frame.feature_dates if start <= week <= end)
        mu = arr.y - arr.e
        crps = np.asarray(
            [t5._quantile_score(float(arr.y[i]), arr.q[i]) for i in range(arr.y.size)],
            dtype=np.float64,
        )
        result[window_name] = ForecastSeries(
            dates=dates,
            y=arr.y,
            mu=mu.astype(np.float64),
            sigma=arr.sigma,
            e=arr.e,
            crps=crps,
            pathology=0,
        )
    return result


def _egarch_normal_series(window: tuple[date, date], forward_returns: pd.Series) -> ForecastSeries:
    """io: Run EGARCH-Normal benchmark and return forecast arrays for one pilot window."""
    from research import run_phase0a_benchmark_delivery as benchmark  # noqa: PLC0415

    start, end = window
    eval_index = forward_returns.loc[pd.Timestamp(start) : pd.Timestamp(end)].dropna().index
    dates: list[date] = []
    y_list: list[float] = []
    mu_list: list[float] = []
    sigma_list: list[float] = []
    crps_list: list[float] = []
    pathology = 0
    last_fit: BenchmarkFitLike | None = None
    for as_of in eval_index:
        as_of_date = as_of.date()
        train_end = as_of_date - pd.Timedelta(weeks=benchmark.TRAIN_EMBARGO_WEEKS)
        train = forward_returns.loc[: pd.Timestamp(train_end)].dropna().to_numpy(dtype=np.float64)
        if train.size < benchmark.TRAIN_WINDOW_WEEKS:
            pathology += 1
            continue
        train = train[-benchmark.TRAIN_WINDOW_WEEKS :]
        try:
            fit = benchmark._fit_model(
                train,
                EGARCH_NORMAL,
                initial=last_fit.params if last_fit is not None else None,
            )
            last_fit = fit
            if not fit.converged:
                pathology += 1
                continue
            mu_hat, sigma_hat, _nu = benchmark._forecast_step(train, fit)
        except (FloatingPointError, ArithmeticError, ValueError, np.linalg.LinAlgError):
            pathology += 1
            continue
        y_true = float(forward_returns.loc[as_of])
        if not np.isfinite(y_true) or not np.isfinite(mu_hat) or not np.isfinite(sigma_hat):
            pathology += 1
            continue
        if sigma_hat <= 0.0:
            pathology += 1
            continue
        dates.append(as_of_date)
        y_list.append(y_true)
        mu_list.append(mu_hat)
        sigma_list.append(sigma_hat)
        crps_list.append(_normal_crps_proxy(y_true, mu_hat, sigma_hat))
    y = np.asarray(y_list, dtype=np.float64)
    mu = np.asarray(mu_list, dtype=np.float64)
    sigma = np.asarray(sigma_list, dtype=np.float64)
    return ForecastSeries(
        dates=tuple(dates),
        y=y,
        mu=mu,
        sigma=sigma,
        e=y - mu,
        crps=np.asarray(crps_list, dtype=np.float64),
        pathology=pathology,
    )


def _align(left: ForecastSeries, right: ForecastSeries) -> tuple[ForecastSeries, ForecastSeries]:
    """pure. Align two forecast series by date."""
    left_pos = {day: idx for idx, day in enumerate(left.dates)}
    right_pos = {day: idx for idx, day in enumerate(right.dates)}
    common = tuple(day for day in left.dates if day in right_pos)
    left_indices = np.asarray([left_pos[day] for day in common], dtype=np.int64)
    right_indices = np.asarray([right_pos[day] for day in common], dtype=np.int64)
    return _take(left, common, left_indices), _take(right, common, right_indices)


def _take(
    series: ForecastSeries,
    dates: tuple[date, ...],
    indices: NDArray[np.int64],
) -> ForecastSeries:
    """pure. Take a date-aligned subset from a forecast series."""
    return ForecastSeries(
        dates=dates,
        y=series.y[indices],
        mu=series.mu[indices],
        sigma=series.sigma[indices],
        e=series.e[indices],
        crps=series.crps[indices],
        pathology=series.pathology,
    )


def _hybrid_quantile_map(
    t5_series: ForecastSeries,
    egarch_series: ForecastSeries,
) -> ForecastSeries:
    """pure. Zero-parameter same-quantile mapping from T5 ranks to EGARCH scale."""
    rank_pct = _rank_percentiles(t5_series.sigma)
    hybrid_sigma = np.quantile(egarch_series.sigma, rank_pct).astype(np.float64)
    crps = np.asarray(
        [
            _normal_crps_proxy(
                float(egarch_series.y[i]),
                float(egarch_series.mu[i]),
                float(hybrid_sigma[i]),
            )
            for i in range(hybrid_sigma.size)
        ],
        dtype=np.float64,
    )
    return ForecastSeries(
        dates=egarch_series.dates,
        y=egarch_series.y,
        mu=egarch_series.mu,
        sigma=hybrid_sigma,
        e=egarch_series.y - egarch_series.mu,
        crps=crps,
        pathology=egarch_series.pathology,
    )


def _hybrid_compressed_quantile_map(
    t5_series: ForecastSeries,
    egarch_series: ForecastSeries,
) -> ForecastSeries:
    """pure. One-parameter monotone compression around EGARCH median scale."""
    rank_pct = _rank_percentiles(t5_series.sigma)
    mapped_sigma = np.quantile(egarch_series.sigma, rank_pct).astype(np.float64)
    median_sigma = float(np.median(egarch_series.sigma))
    log_sigma = np.log(median_sigma) + COMPRESSION_ALPHA * (
        np.log(np.maximum(mapped_sigma, 1.0e-8)) - np.log(median_sigma)
    )
    hybrid_sigma = np.exp(log_sigma).astype(np.float64)
    crps = np.asarray(
        [
            _normal_crps_proxy(
                float(egarch_series.y[i]),
                float(egarch_series.mu[i]),
                float(hybrid_sigma[i]),
            )
            for i in range(hybrid_sigma.size)
        ],
        dtype=np.float64,
    )
    return ForecastSeries(
        dates=egarch_series.dates,
        y=egarch_series.y,
        mu=egarch_series.mu,
        sigma=hybrid_sigma,
        e=egarch_series.y - egarch_series.mu,
        crps=crps,
        pathology=egarch_series.pathology,
    )


def _all_windows_pass(
    table: dict[str, dict[str, dict[str, float | int | str]]],
    model_name: str,
) -> bool:
    """pure. Return whether one model passed all three preregistered windows."""
    return all(
        table[window_name][model_name]["decision"] == "SUCCESS" for window_name in PILOT_WINDOWS
    )


def _markdown_table(
    rows: Sequence[tuple[str, str, dict[str, float | int | str]]],
) -> str:
    """pure. Render the diagnostic table in Markdown."""
    headers = (
        "window",
        "model",
        "mean(z)",
        "std(z)",
        "corr_next",
        "rank_next",
        "lag1_acf(z)",
        "sigma_blowup",
        "pathology",
        "CRPS",
        "decision",
    )
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for window_name, model_name, metrics in rows:
        lines.append(
            "| "
            + " | ".join(
                (
                    window_name,
                    model_name,
                    _fmt(metrics["mean_z"]),
                    _fmt(metrics["std_z"]),
                    _fmt(metrics["corr_next"]),
                    _fmt(metrics["rank_next"]),
                    _fmt(metrics["lag1_acf_z"]),
                    str(metrics["sigma_blowup"]),
                    str(metrics["pathology"]),
                    _fmt(metrics["crps"]),
                    str(metrics["decision"]),
                ),
            )
            + " |",
        )
    return "\n".join(lines)


def _fmt(value: float | int | str) -> str:
    """pure. Format table cells deterministically."""
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    if np.isfinite(value):
        return f"{value:.6f}"
    return "nan"


def _write_report(payload: dict[str, Any]) -> None:
    """io: Write the Markdown experiment result report."""
    rows: list[tuple[str, str, dict[str, float | int | str]]] = []
    for window_name in PILOT_WINDOWS:
        rows.extend(
            (window_name, model_name, payload["diagnostics"][window_name][model_name])
            for model_name in (
                "T5_resid_persistence_M4",
                EGARCH_NORMAL,
                "hybrid_quantile_map_zero_param",
                "hybrid_compressed_quantile_map_alpha_0_50",
            )
        )
    table = _markdown_table(rows)
    final_decision = payload["final_decision"]
    compressed_decision = final_decision["hybrid_compressed_quantile_map_alpha_0_50"]
    report = f"""# Rank-Scale Hybrid Three-Window Diagnostics

> Generated by `src/research/run_rank_scale_hybrid_experiment.py`.
> These results belong only to the research sidecar.

## 1. Experiment Config

- Hypothesis: T5 rank ordering plus EGARCH(1,1)-Normal scale calibration.
- Primary candidate: `hybrid_quantile_map_zero_param`
- Low-degree comparator: `hybrid_compressed_quantile_map_alpha_0_50`
- Preregistration: `{PREREG_DOC.as_posix()}`
- Windows: 2017, 2018, and 2020 pilot windows.

## 2. Diagnostics

{table}

## 3. Protocol Decision

- `hybrid_quantile_map_zero_param`: `{final_decision["hybrid_quantile_map_zero_param"]}`
- `hybrid_compressed_quantile_map_alpha_0_50`: `{compressed_decision}`

## 4. Boundary

This experiment does not reopen trigger audit, crisis regime classification,
override, hard switch, or MoE. If no hybrid candidate reaches three-window
`SUCCESS`, this new rank-scale hybrid hypothesis terminates under preregistered
rules.
"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "rank_scale_hybrid_results.md").write_text(report, encoding="utf-8")


def run(argv: list[str] | None = None) -> int:
    """io: Execute the rank-scale hybrid experiment and write JSON/Markdown artifacts."""
    from research import run_phase0a_benchmark_delivery as benchmark  # noqa: PLC0415

    parser = argparse.ArgumentParser(prog="run_rank_scale_hybrid_experiment")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON to stdout.")
    args = parser.parse_args(argv)

    t5_by_window = _t5_series_by_window()
    forward_returns = benchmark._load_weekly_forward_returns()
    diagnostics: dict[str, dict[str, dict[str, float | int | str]]] = {}
    for window_name, window in PILOT_WINDOWS.items():
        egarch_series = _egarch_normal_series(window, forward_returns)
        aligned_t5, aligned_egarch = _align(t5_by_window[window_name], egarch_series)
        hybrid_qmap = _hybrid_quantile_map(aligned_t5, aligned_egarch)
        hybrid_compressed = _hybrid_compressed_quantile_map(aligned_t5, aligned_egarch)
        diagnostics[window_name] = {
            "T5_resid_persistence_M4": _metrics(aligned_t5),
            EGARCH_NORMAL: _metrics(aligned_egarch),
            "hybrid_quantile_map_zero_param": _metrics(hybrid_qmap),
            "hybrid_compressed_quantile_map_alpha_0_50": _metrics(hybrid_compressed),
        }

    payload: dict[str, Any] = {
        "hypothesis": "T5 rank ordering plus EGARCH-Normal scale calibration",
        "preregistration": PREREG_DOC.as_posix(),
        "windows": {
            name: [start.isoformat(), end.isoformat()]
            for name, (start, end) in PILOT_WINDOWS.items()
        },
        "candidate_models": [
            "hybrid_quantile_map_zero_param",
            "hybrid_compressed_quantile_map_alpha_0_50",
        ],
        "diagnostics": diagnostics,
        "final_decision": {
            "hybrid_quantile_map_zero_param": "SUCCESS"
            if _all_windows_pass(diagnostics, "hybrid_quantile_map_zero_param")
            else "FAIL",
            "hybrid_compressed_quantile_map_alpha_0_50": "SUCCESS"
            if _all_windows_pass(diagnostics, "hybrid_compressed_quantile_map_alpha_0_50")
            else "FAIL",
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "rank_scale_hybrid_results.json").write_text(
        json.dumps(payload, default=_json_default, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(payload)
    if args.pretty:
        sys.stdout.write(json.dumps(payload, default=_json_default, indent=2, sort_keys=True))
    else:
        sys.stdout.write(json.dumps(payload, default=_json_default, sort_keys=True))
    sys.stdout.write("\n")
    return 0


def main() -> None:
    """io: process exit entrypoint."""
    raise SystemExit(run())


if __name__ == "__main__":
    main()
