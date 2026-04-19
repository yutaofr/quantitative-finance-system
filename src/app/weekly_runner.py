"""Imperative weekly shell runner with deterministic degradation behavior."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from app.cli import ExitCode
from config_types import FrozenConfig
from engine_types import TimeSeries, VintageMode, WeeklyOutput
from errors import HMMConvergenceError, QuantileSolverError
from inference.weekly import TrainingArtifacts, blocked_weekly_output, degraded_weekly_output

FetchSeries = Callable[[date, VintageMode], Mapping[str, TimeSeries]]
LoadTrainingArtifacts = Callable[[], TrainingArtifacts]
InferWeekly = Callable[
    [date, FrozenConfig, Mapping[str, TimeSeries], TrainingArtifacts],
    WeeklyOutput,
]
WriteOutput = Callable[[WeeklyOutput, Path], None]


@dataclass(frozen=True, slots=True)
class WeeklyRunnerDeps:
    """io: shell dependencies for one weekly production run."""

    fetch_series: FetchSeries
    load_training_artifacts: LoadTrainingArtifacts
    infer_weekly: InferWeekly
    write_output: WriteOutput


def run_weekly_job(
    *,
    as_of: date,
    vintage_mode: VintageMode,
    cfg: FrozenConfig,
    output_path: Path,
    deps: WeeklyRunnerDeps,
) -> int:
    """io: execute one shell weekly run with deterministic fallback handling."""
    try:
        series = deps.fetch_series(as_of, vintage_mode)
        training_artifacts = deps.load_training_artifacts()
        output = deps.infer_weekly(as_of, cfg, series, training_artifacts)
    except HMMConvergenceError:
        output = degraded_weekly_output(
            as_of,
            vintage_mode=vintage_mode,
            hmm_status="em_nonconverge",
        )
        deps.write_output(output, output_path)
        return int(ExitCode.OK)
    except QuantileSolverError:
        output = blocked_weekly_output(
            as_of,
            vintage_mode=vintage_mode,
            quantile_solver_status="failed",
        )
        deps.write_output(output, output_path)
        return int(ExitCode.BLOCKED)

    deps.write_output(output, output_path)
    if output.mode == "BLOCKED":
        return int(ExitCode.BLOCKED)
    return int(ExitCode.OK)
