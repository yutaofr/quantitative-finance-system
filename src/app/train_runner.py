"""Imperative shell runner for training artifact generation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

from config_types import FrozenConfig
from engine_types import TimeSeries, VintageMode
from inference.weekly import TrainingArtifacts

EXIT_OK = 0

FetchSeries = Callable[[date, VintageMode], Mapping[str, TimeSeries]]
FitTrainingArtifacts = Callable[
    [date, Mapping[str, TimeSeries], FrozenConfig, int],
    TrainingArtifacts,
]
WriteTrainingArtifacts = Callable[[TrainingArtifacts, Path], None]


@dataclass(frozen=True, slots=True)
class TrainRunnerDeps:
    """io: shell dependencies for one training run."""

    fetch_series: FetchSeries
    fit_training_artifacts: FitTrainingArtifacts
    write_training_artifacts: WriteTrainingArtifacts


def parse_window_weeks(raw: str) -> int:
    """io: Parse CLI training window notation such as '312w'."""
    suffix = raw[-1:]
    value = raw[:-1] if suffix == "w" else raw
    weeks = int(value)
    if weeks <= 0:
        msg = "training window must be positive"
        raise ValueError(msg)
    return weeks


def run_train_job(  # noqa: PLR0913
    *,
    as_of: date,
    vintage_mode: VintageMode,
    cfg: FrozenConfig,
    training_root: Path,
    window_weeks: int,
    deps: TrainRunnerDeps,
) -> int:
    """io: execute one training run and persist artifacts."""
    series = deps.fetch_series(as_of, vintage_mode)
    artifacts = deps.fit_training_artifacts(as_of, series, cfg, window_weeks)
    deps.write_training_artifacts(artifacts, training_root)
    return EXIT_OK


def deterministic_training_rng(cfg: FrozenConfig, as_of: date) -> np.random.Generator:
    """io: Build a deterministic per-as_of RNG for shell orchestration."""
    return np.random.default_rng(cfg.random_seed + as_of.toordinal())
