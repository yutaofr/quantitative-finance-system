"""Pure walk-forward orchestration over weekly PIT history."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import numpy as np

from config_types import FrozenConfig
from engine_types import TimeSeries, WeeklyOutput
from inference.weekly import TrainingArtifacts

FitTrainingArtifacts = Callable[
    [date, Mapping[str, TimeSeries], FrozenConfig, Mapping[date, Any] | None],
    TrainingArtifacts,
]
InferWeekly = Callable[
    [date, FrozenConfig, Mapping[str, TimeSeries], TrainingArtifacts, Mapping[date, Any] | None],
    WeeklyOutput,
]


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """pure. Ordered walk-forward weekly outputs."""

    outputs: tuple[WeeklyOutput, ...]
    realized_52w_returns: tuple[float, ...] = ()


def _truncate_series(series: TimeSeries, as_of: date) -> TimeSeries:
    timestamps = series.timestamps.astype("datetime64[D]")
    mask = timestamps <= np.datetime64(as_of, "D")
    return TimeSeries(
        series_id=series.series_id,
        timestamps=series.timestamps[mask],
        values=series.values[mask],
        is_pseudo_pit=series.is_pseudo_pit,
    )


def _slice_series_history(
    series: Mapping[str, TimeSeries],
    as_of: date,
) -> dict[str, TimeSeries]:
    return {series_id: _truncate_series(ts, as_of) for series_id, ts in series.items()}


def _weekly_dates(start: date, end: date) -> tuple[date, ...]:
    if end < start:
        msg = "walk-forward end must be on or after start"
        raise ValueError(msg)
    dates: list[date] = []
    current = start
    while current <= end:
        dates.append(current)
        current += timedelta(days=7)
    return tuple(dates)


def run_walkforward(  # noqa: PLR0913
    start: date,
    end: date,
    series: Mapping[str, TimeSeries],
    cfg: FrozenConfig,
    *,
    fit_training_artifacts: FitTrainingArtifacts,
    infer_weekly: InferWeekly,
    feature_cache: Mapping[date, Any] | None = None,
) -> BacktestResult:
    """pure. Run weekly walk-forward inference on PIT-truncated history only."""
    outputs: list[WeeklyOutput] = []
    for as_of in _weekly_dates(start, end):
        history = _slice_series_history(series, as_of)
        training_artifacts = fit_training_artifacts(as_of, history, cfg, feature_cache)
        outputs.append(infer_weekly(as_of, cfg, history, training_artifacts, feature_cache))
    return BacktestResult(outputs=tuple(outputs))
