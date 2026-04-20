"""Imperative shell runner for walk-forward backtests."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from app.output_serializer import serialize_weekly_output
from backtest.walkforward import BacktestResult, run_walkforward
from config_types import FrozenConfig
from engine_types import TimeSeries, VintageMode, WeeklyOutput
from inference.weekly import TrainingArtifacts

EXIT_OK = 0

FetchSeries = Callable[[date, VintageMode], Mapping[str, TimeSeries]]
FitTrainingArtifacts = Callable[[date, Mapping[str, TimeSeries], FrozenConfig], TrainingArtifacts]
InferWeekly = Callable[
    [date, FrozenConfig, Mapping[str, TimeSeries], TrainingArtifacts],
    WeeklyOutput,
]
WriteBacktestResult = Callable[[BacktestResult, Path], None]


@dataclass(frozen=True, slots=True)
class BacktestRunnerDeps:
    """io: shell dependencies for one walk-forward backtest."""

    fetch_series: FetchSeries
    fit_training_artifacts: FitTrainingArtifacts
    infer_weekly: InferWeekly
    write_result: WriteBacktestResult


def write_backtest_jsonl(result: BacktestResult, path: Path) -> None:
    """io: Persist weekly backtest outputs as deterministic JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = b"".join(serialize_weekly_output(output) for output in result.outputs)
    path.write_bytes(payload)


def run_backtest_job(
    *,
    start: date,
    end: date,
    cfg: FrozenConfig,
    output_path: Path,
    deps: BacktestRunnerDeps,
) -> int:
    """io: execute walk-forward backtest over PIT-truncated history."""
    vintage_mode: VintageMode = "strict" if end >= cfg.strict_pit_start else "pseudo"
    series = deps.fetch_series(end, vintage_mode)
    result = run_walkforward(
        start,
        end,
        series,
        cfg,
        fit_training_artifacts=deps.fit_training_artifacts,
        infer_weekly=deps.infer_weekly,
    )
    deps.write_result(result, output_path)
    return EXIT_OK
