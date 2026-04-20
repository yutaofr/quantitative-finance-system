"""Default imperative dependencies for production CLI commands."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from pathlib import Path

from app.config_loader import AdapterSecrets
from app.output_serializer import write_weekly_output
from app.training_artifacts import load_training_artifacts
from app.weekly_runner import WeeklyRunnerDeps
from config_types import FrozenConfig
from data_contract.fred_adapter import FredClient
from engine_types import TimeSeries, VintageMode, WeeklyOutput
from inference.weekly import TrainingArtifacts, run_weekly

REQUIRED_SERIES = (
    "DGS10",
    "DGS2",
    "DGS1",
    "EFFR",
    "BAA10Y",
    "WALCL",
    "VXNCLS",
    "RV20_NDX",
    "VIXCLS",
    "VXVCLS",
)


def build_weekly_runner_deps(
    secrets: AdapterSecrets,
    *,
    cache_root: Path = Path("data/raw/fred"),
    training_root: Path = Path("artifacts/training"),
) -> WeeklyRunnerDeps:
    """io: Build real weekly shell dependencies from adapter secrets and paths."""
    client = FredClient(api_key=secrets.fred_api_key, cache_root=cache_root)

    def fetch_series(as_of: date, vintage_mode: VintageMode) -> dict[str, TimeSeries]:
        return {
            series_id: client.get_series(series_id, as_of, vintage_mode)
            for series_id in REQUIRED_SERIES
        }

    def load_artifacts() -> TrainingArtifacts:
        return load_training_artifacts(training_root)

    def infer(
        as_of: date,
        cfg: FrozenConfig,
        series: Mapping[str, TimeSeries],
        training_artifacts: TrainingArtifacts,
    ) -> WeeklyOutput:
        vintage_mode: VintageMode = "strict" if as_of >= cfg.strict_pit_start else "pseudo"
        return run_weekly(as_of, vintage_mode, series, training_artifacts)

    return WeeklyRunnerDeps(
        fetch_series=fetch_series,
        load_training_artifacts=load_artifacts,
        infer_weekly=infer,
        write_output=write_weekly_output,
    )
