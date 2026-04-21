"""Default imperative dependencies for production CLI commands."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
import os
from pathlib import Path
from typing import Any

from app.backtest_runner import BacktestRunnerDeps, write_backtest_jsonl
from app.config_loader import AdapterSecrets
from app.output_serializer import write_weekly_output
from app.train_runner import TrainRunnerDeps, deterministic_training_rng
from app.training_artifacts import load_training_artifacts, write_training_artifacts
from app.weekly_runner import WeeklyRunnerDeps
from backtest.walkforward import BacktestResult
from config_types import FrozenConfig
from data_contract.derived_series import derive_rv20_nasdaq100
from data_contract.fred_adapter import FredClient
from data_contract.nasdaq_client import NasdaqClient
from engine_types import TimeSeries, VintageMode, WeeklyOutput
from inference.train import build_training_artifacts
from inference.weekly import TrainingArtifacts, run_weekly

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
DERIVED_PRICE_SERIES = "NASDAQXNDX"


def build_weekly_runner_deps(
    secrets: AdapterSecrets,
    *,
    cache_root: Path = Path("data/raw/fred"),
    nasdaq_cache_root: Path = Path("data/raw/nasdaq"),
    training_root: Path = Path("artifacts/training"),
) -> WeeklyRunnerDeps:
    """io: Build real weekly shell dependencies from adapter secrets and paths."""
    client = FredClient(api_key=secrets.fred_api_key, cache_root=cache_root)
    nasdaq_client = NasdaqClient(cache_root=nasdaq_cache_root)

    def fetch_series(as_of: date, vintage_mode: VintageMode) -> dict[str, TimeSeries]:
        fetched = {
            series_id: client.get_series(series_id, as_of, vintage_mode)
            for series_id in REQUIRED_SERIES
        }
        price_series = nasdaq_client.get_series(DERIVED_PRICE_SERIES, as_of)
        fetched["RV20_NDX"] = derive_rv20_nasdaq100(price_series, as_of)
        return fetched

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


def _fetch_training_series(
    client: FredClient,
    nasdaq_client: NasdaqClient,
    as_of: date,
    vintage_mode: VintageMode,
) -> dict[str, TimeSeries]:
    fetched = {
        series_id: client.get_series(series_id, as_of, vintage_mode)
        for series_id in REQUIRED_SERIES
    }
    price_series = nasdaq_client.get_series(DERIVED_PRICE_SERIES, as_of)
    fetched[DERIVED_PRICE_SERIES] = price_series
    fetched["RV20_NDX"] = derive_rv20_nasdaq100(price_series, as_of)
    return fetched


def fit_backtest_training_artifacts(
    as_of: date,
    series: Mapping[str, TimeSeries],
    cfg: FrozenConfig,
    feature_cache: Any = None,
) -> TrainingArtifacts:
    """pure. Build deterministic backtest training artifacts for one week."""
    return build_training_artifacts(
        as_of,
        series,
        cfg,
        rng=deterministic_training_rng(cfg, as_of),
        feature_cache=feature_cache,
    )


def infer_backtest_weekly(
    as_of: date,
    cfg: FrozenConfig,
    series: Mapping[str, TimeSeries],
    training_artifacts: TrainingArtifacts,
    feature_cache: Any = None,
) -> WeeklyOutput:
    """pure. Run one deterministic backtest weekly inference step."""
    vintage_mode: VintageMode = "strict" if as_of >= cfg.strict_pit_start else "pseudo"
    return run_weekly(
        as_of,
        vintage_mode,
        series,
        training_artifacts,
        feature_cache=feature_cache,
    )


def build_train_runner_deps(
    secrets: AdapterSecrets,
    *,
    cache_root: Path = Path("data/raw/fred"),
    nasdaq_cache_root: Path = Path("data/raw/nasdaq"),
) -> TrainRunnerDeps:
    """io: Build real training shell dependencies from adapter secrets."""
    client = FredClient(api_key=secrets.fred_api_key, cache_root=cache_root)
    nasdaq_client = NasdaqClient(cache_root=nasdaq_cache_root)

    def fetch_series(as_of: date, vintage_mode: VintageMode) -> dict[str, TimeSeries]:
        return _fetch_training_series(client, nasdaq_client, as_of, vintage_mode)

    def fit_training_artifacts(
        as_of: date,
        series: Mapping[str, TimeSeries],
        cfg: FrozenConfig,
        min_training_weeks: int,
    ) -> TrainingArtifacts:
        return build_training_artifacts(
            as_of,
            series,
            cfg,
            rng=deterministic_training_rng(cfg, as_of),
            min_training_weeks=min_training_weeks,
        )

    return TrainRunnerDeps(
        fetch_series=fetch_series,
        fit_training_artifacts=fit_training_artifacts,
        write_training_artifacts=write_training_artifacts,
    )


def build_backtest_runner_deps(
    secrets: AdapterSecrets,
    *,
    cache_root: Path = Path("data/raw/fred"),
    nasdaq_cache_root: Path = Path("data/raw/nasdaq"),
) -> BacktestRunnerDeps:
    """io: Build real walk-forward backtest dependencies from adapter secrets."""
    train_deps = build_train_runner_deps(
        secrets, cache_root=cache_root, nasdaq_cache_root=nasdaq_cache_root
    )

    def write_result(result: BacktestResult, path: Path) -> None:
        write_backtest_jsonl(result, path)

    return BacktestRunnerDeps(
        fetch_series=train_deps.fetch_series,
        fit_training_artifacts=fit_backtest_training_artifacts,
        infer_weekly=infer_backtest_weekly,
        write_result=write_result,
        max_workers=max(1, min(os.cpu_count() or 1, 4)),
    )
