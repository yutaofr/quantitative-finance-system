"""Strict-PIT acceptance metric calculations for SRD v8.7 section 16."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
from numpy.typing import NDArray

from engine_types import TimeSeries, WeeklyOutput

FORECAST_HORIZON_WEEKS = 52
ANNUAL_WEEKS = 52.0
NEUTRAL_WEIGHT = 0.50
TAUS = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95], dtype=np.float64)
MIN_BASELINE_HISTORY = 2


@dataclass(frozen=True, slots=True)
class BacktestMetricSeries:
    """pure. Per-week strict-PIT metric components before gate aggregation."""

    q10_hits: NDArray[np.float64]
    q90_hits: NDArray[np.float64]
    crps_production: NDArray[np.float64]
    crps_baseline_a: NDArray[np.float64]
    production_returns: NDArray[np.float64]
    baseline_b_returns: NDArray[np.float64]


def _value_in_week(series: TimeSeries, as_of: date) -> float:
    target = np.datetime64(as_of, "D")
    week_start = np.datetime64(as_of - timedelta(days=6), "D")
    timestamps = series.timestamps.astype("datetime64[D]")
    matches = np.flatnonzero((timestamps >= week_start) & (timestamps <= target))
    if matches.size == 0:
        return float("nan")
    return float(series.values[int(matches[-1])])


def realized_forward_returns(
    target_series: TimeSeries,
    as_of_dates: Sequence[date],
) -> NDArray[np.float64]:
    """pure. Align realized 52-week log returns to inference weeks."""
    realized: list[float] = []
    for as_of in as_of_dates:
        current = _value_in_week(target_series, as_of)
        future = _value_in_week(target_series, as_of + timedelta(weeks=FORECAST_HORIZON_WEEKS))
        if not np.isfinite(current) or not np.isfinite(future) or current <= 0.0 or future <= 0.0:
            realized.append(float("nan"))
        else:
            realized.append(float(np.log(future / current)))
    return np.asarray(realized, dtype=np.float64)


def _output_quantiles(output: WeeklyOutput) -> NDArray[np.float64]:
    return np.array(
        [
            output.distribution.q05,
            output.distribution.q10,
            output.distribution.q25,
            output.distribution.q50,
            output.distribution.q75,
            output.distribution.q90,
            output.distribution.q95,
        ],
        dtype=np.float64,
    )


def _quantile_score(y_true: float, quantiles: NDArray[np.float64]) -> float:
    losses = np.maximum(TAUS * (y_true - quantiles), (TAUS - 1.0) * (y_true - quantiles))
    return float(2.0 * np.mean(losses))


def _baseline_a_quantiles(history: NDArray[np.float64], fallback: float) -> NDArray[np.float64]:
    finite = history[np.isfinite(history)]
    if finite.shape[0] < MIN_BASELINE_HISTORY:
        return np.full(TAUS.shape, fallback, dtype=np.float64)
    return np.quantile(finite, TAUS).astype(np.float64)


def strict_metric_series(
    outputs: Sequence[WeeklyOutput],
    realized_52w_returns: Sequence[float],
    *,
    effective_strict_start: date,
) -> BacktestMetricSeries:
    """pure. Build per-week acceptance metric components on strict finite weeks only."""
    if len(outputs) != len(realized_52w_returns):
        msg = "outputs and realized returns must have the same length"
        raise ValueError(msg)
    y_all = np.asarray(realized_52w_returns, dtype=np.float64)
    q10_hits: list[float] = []
    q90_hits: list[float] = []
    crps_production: list[float] = []
    crps_baseline_a: list[float] = []
    production_returns: list[float] = []
    baseline_b_returns: list[float] = []

    for idx, output in enumerate(outputs):
        y_true = float(y_all[idx])
        if (
            output.vintage_mode != "strict"
            or output.as_of_date < effective_strict_start
            or not np.isfinite(y_true)
        ):
            continue
        quantiles = _output_quantiles(output)
        if not np.isfinite(quantiles).all():
            continue
        baseline_a = _baseline_a_quantiles(y_all[:idx], y_true)
        q10_hits.append(float(y_true <= output.distribution.q10))
        q90_hits.append(float(y_true <= output.distribution.q90))
        crps_production.append(_quantile_score(y_true, quantiles))
        crps_baseline_a.append(_quantile_score(y_true, baseline_a))
        production_weight = output.decision.offense_final / 100.0
        production_returns.append(float(production_weight * y_true))
        baseline_b_returns.append(float(NEUTRAL_WEIGHT * y_true))

    return BacktestMetricSeries(
        q10_hits=np.asarray(q10_hits, dtype=np.float64),
        q90_hits=np.asarray(q90_hits, dtype=np.float64),
        crps_production=np.asarray(crps_production, dtype=np.float64),
        crps_baseline_a=np.asarray(crps_baseline_a, dtype=np.float64),
        production_returns=np.asarray(production_returns, dtype=np.float64),
        baseline_b_returns=np.asarray(baseline_b_returns, dtype=np.float64),
    )


def stationary_bootstrap_indices(
    n_observations: int,
    *,
    block_length: int,
    rng: np.random.Generator,
) -> NDArray[np.int64]:
    """pure. Draw stationary bootstrap indices with expected block_length."""
    if n_observations <= 0 or block_length <= 0:
        msg = "bootstrap dimensions must be positive"
        raise ValueError(msg)
    restart_probability = 1.0 / float(block_length)
    out = np.empty(n_observations, dtype=np.int64)
    current = int(rng.integers(0, n_observations))
    for idx in range(n_observations):
        if idx > 0 and float(rng.random()) < restart_probability:
            current = int(rng.integers(0, n_observations))
        out[idx] = current
        current = (current + 1) % n_observations
    return out


def ceq_annualized(returns: NDArray[np.float64]) -> float:
    """pure. Compute annualized certainty-equivalent return for weekly returns."""
    finite = returns[np.isfinite(returns)]
    if finite.size == 0:
        return float("nan")
    variance = float(np.var(finite, ddof=1)) if finite.size > 1 else 0.0
    return float(ANNUAL_WEEKS * (float(np.mean(finite)) - 0.5 * variance))


def max_drawdown(returns: NDArray[np.float64]) -> float:
    """pure. Compute positive max drawdown depth from weekly log returns."""
    finite = returns[np.isfinite(returns)]
    if finite.size == 0:
        return float("nan")
    wealth = np.exp(np.cumsum(finite))
    running_peak = np.maximum.accumulate(wealth)
    drawdowns = 1.0 - wealth / running_peak
    return float(np.max(drawdowns))


def crps_improvement_ratio(
    production_crps: NDArray[np.float64],
    baseline_crps: NDArray[np.float64],
) -> float:
    """pure. Return mean relative CRPS improvement versus Baseline_A."""
    finite = np.isfinite(production_crps) & np.isfinite(baseline_crps) & (baseline_crps > 0.0)
    if not finite.any():
        return float("nan")
    return float(1.0 - np.mean(production_crps[finite]) / np.mean(baseline_crps[finite]))


def bootstrap_crps_improvement_p05(
    series: BacktestMetricSeries,
    *,
    block_length: int,
    replications: int,
    rng: np.random.Generator,
) -> float:
    """pure. Bootstrap 5th percentile of CRPS relative improvement."""
    if series.crps_production.size == 0:
        return float("nan")
    samples = np.empty(replications, dtype=np.float64)
    for idx in range(replications):
        sample_idx = stationary_bootstrap_indices(
            series.crps_production.size,
            block_length=block_length,
            rng=rng,
        )
        samples[idx] = crps_improvement_ratio(
            series.crps_production[sample_idx],
            series.crps_baseline_a[sample_idx],
        )
    if not np.isfinite(samples).any():
        return float("nan")
    return float(np.nanquantile(samples, 0.05))


def bootstrap_ceq_diff_p05(
    series: BacktestMetricSeries,
    *,
    block_length: int,
    replications: int,
    rng: np.random.Generator,
) -> float:
    """pure. Bootstrap 5th percentile of production CEQ minus Baseline_B CEQ."""
    if series.production_returns.size == 0:
        return float("nan")
    samples = np.empty(replications, dtype=np.float64)
    for idx in range(replications):
        sample_idx = stationary_bootstrap_indices(
            series.production_returns.size,
            block_length=block_length,
            rng=rng,
        )
        samples[idx] = ceq_annualized(series.production_returns[sample_idx]) - ceq_annualized(
            series.baseline_b_returns[sample_idx],
        )
    if not np.isfinite(samples).any():
        return float("nan")
    return float(np.nanquantile(samples, 0.05))
