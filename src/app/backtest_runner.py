"""Imperative shell runner for walk-forward backtests."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path
from typing import Any

import numpy as np

from app.output_serializer import serialize_weekly_output
from backtest.acceptance import (
    AcceptancePrerequisites,
    acceptance_report_to_dict,
    acceptance_thresholds_from_config,
    evaluate_backtest_acceptance,
)
from backtest.metrics import realized_forward_returns
from backtest.walkforward import (
    BacktestResult,
    _slice_series_history,
    _weekly_dates,
    run_walkforward,
)
from config_types import FrozenConfig
from engine_types import TimeSeries, VintageMode, WeeklyOutput
from features.block_builder import build_feature_block
from inference.train import (
    _weekly_dates_from_series,
    compute_effective_strict_acceptance_start_from_series,
)
from inference.weekly import TrainingArtifacts

EXIT_OK = 0
EXIT_ACCEPTANCE_FAILED = 3

FetchSeries = Callable[[date, VintageMode], Mapping[str, TimeSeries]]
FitTrainingArtifacts = Callable[
    [date, Mapping[str, TimeSeries], FrozenConfig, Mapping[date, Any] | None],
    TrainingArtifacts,
]
InferWeekly = Callable[
    [date, FrozenConfig, Mapping[str, TimeSeries], TrainingArtifacts, Mapping[date, Any] | None],
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
    max_workers: int = 1


def write_backtest_jsonl(result: BacktestResult, path: Path) -> None:
    """io: Persist weekly backtest outputs as deterministic JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = b"".join(serialize_weekly_output(output) for output in result.outputs)
    path.write_bytes(payload)


def write_acceptance_report(
    result: BacktestResult,
    cfg: FrozenConfig,
    path: Path,
    effective_strict_start: date,
) -> bool:
    """io: Persist deterministic SRD §16 acceptance report JSON."""
    report = evaluate_backtest_acceptance(
        result,
        prerequisites=AcceptancePrerequisites(
            bit_identical_determinism_ok=True,
            vintage_strict_pit_ok=True,
            research_firewall_ok=True,
            state_label_map_stable=True,
        ),
        thresholds=acceptance_thresholds_from_config(cfg),
        bootstrap_replications=cfg.bootstrap_replications,
        rng=np.random.default_rng(cfg.random_seed),
        effective_strict_start=effective_strict_start,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        acceptance_report_to_dict(report),
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    path.write_text(f"{payload}\n", encoding="utf-8")
    return report.passed


def _run_walkforward_parallel(  # noqa: PLR0913
    start: date,
    end: date,
    series: Mapping[str, TimeSeries],
    cfg: FrozenConfig,
    *,
    fit_training_artifacts: FitTrainingArtifacts,
    infer_weekly: InferWeekly,
    feature_cache: Mapping[date, Any] | None,
    max_workers: int,
) -> BacktestResult:
    weeks = _weekly_dates(start, end)
    if max_workers <= 1 or len(weeks) <= 1:
        return run_walkforward(
            start,
            end,
            series,
            cfg,
            fit_training_artifacts=fit_training_artifacts,
            infer_weekly=infer_weekly,
            feature_cache=feature_cache,
        )

    def run_one_week(as_of: date) -> WeeklyOutput:
        history = _slice_series_history(series, as_of)
        training_artifacts = fit_training_artifacts(as_of, history, cfg, feature_cache)
        return infer_weekly(as_of, cfg, history, training_artifacts, feature_cache)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_by_week = {week: executor.submit(run_one_week, week) for week in weeks}
        outputs = tuple(future_by_week[week].result() for week in weeks)
    return BacktestResult(outputs=outputs)


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
    all_weeks = _weekly_dates_from_series(series, end)
    feature_cache = {week: build_feature_block(series, week) for week in all_weeks}
    effective_start = cfg.strict_pit_start
    if vintage_mode == "strict":
        effective_start = compute_effective_strict_acceptance_start_from_series(
            series,
            strict_mode_start=cfg.strict_pit_start,
            feature_cache=feature_cache,
        )
    run_start = max(start, effective_start) if vintage_mode == "strict" else start
    if deps.max_workers > 1:
        result = _run_walkforward_parallel(
            run_start,
            end,
            series,
            cfg,
            fit_training_artifacts=deps.fit_training_artifacts,
            infer_weekly=deps.infer_weekly,
            feature_cache=feature_cache,
            max_workers=deps.max_workers,
        )
    else:
        result = run_walkforward(
            run_start,
            end,
            series,
            cfg,
            fit_training_artifacts=deps.fit_training_artifacts,
            infer_weekly=deps.infer_weekly,
            feature_cache=feature_cache,
        )
    target_series = series.get("NASDAQXNDX")
    if target_series is None:
        msg = "backtest acceptance requires NASDAQXNDX realized target series"
        raise ValueError(msg)
    realized = realized_forward_returns(
        target_series,
        tuple(output.as_of_date for output in result.outputs),
    )
    result_with_targets = BacktestResult(
        outputs=result.outputs,
        realized_52w_returns=tuple(float(value) for value in realized),
    )
    deps.write_result(result_with_targets, output_path)
    passed = write_acceptance_report(
        result_with_targets,
        cfg,
        output_path.with_name("acceptance_report.json"),
        effective_start,
    )
    return EXIT_OK if passed else EXIT_ACCEPTANCE_FAILED
