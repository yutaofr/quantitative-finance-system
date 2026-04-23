"""Imperative shell runner for walk-forward backtests."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import date, timedelta
import json
import multiprocessing
from pathlib import Path
from typing import Any, cast

import numpy as np

from app.challenger_artifacts import (
    challenger_fit_artifact_to_dict,
    challenger_output_to_dict,
    write_challenger_fit_artifact,
    write_challenger_output,
    write_challenger_report,
)
from app.output_serializer import serialize_weekly_output
from backtest.acceptance import (
    AcceptancePrerequisites,
    acceptance_report_to_dict,
    acceptance_thresholds_from_config,
    evaluate_backtest_acceptance,
)
from backtest.metrics import ceq_annualized, crps_improvement_ratio, realized_forward_returns
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
    _forward_52w_return,
    _scale_feature_history,
    compute_effective_strict_acceptance_start_from_series,
)
from inference.weekly import TrainingArtifacts
from law.student_t_location_scale import (
    StudentTFitResult,
    fit_student_t_location_scale,
    predict_student_t_quantiles,
)

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

_WORKER_STATE: dict[str, object] = {}
BLOCKS_KEY = "blocks"
HISTORY_WEEK_ORDINALS_KEY = "history_week_ordinals"
HISTORY_X_RAW_KEY = "history_x_raw"
HISTORY_X_SCALED_KEY = "history_x_scaled"
TRAINING_WEEK_ORDINALS_KEY = "training_week_ordinals"
TRAINING_WEEKS_KEY = "training_weeks"
TRAINING_X_RAW_KEY = "training_x_raw"
TRAINING_X_SCALED_KEY = "training_x_scaled"
TRAINING_Y_KEY = "training_y_52w"
FRIDAY = 4
CHALLENGER_MIN_TRAIN_ROWS = 52
CHALLENGER_EMBARGO_WEEKS = 53
CHALLENGER_REFIT_EVERY_WEEKS = 13
TAUS = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95], dtype=np.float64)
MIN_BASELINE_HISTORY = 2


@dataclass(frozen=True, slots=True)
class BacktestRunnerDeps:
    """io: shell dependencies for one walk-forward backtest."""

    fetch_series: FetchSeries
    fit_training_artifacts: FitTrainingArtifacts
    infer_weekly: InferWeekly
    write_result: WriteBacktestResult
    max_workers: int = 1


@dataclass(frozen=True, slots=True)
class BacktestRuntimeMetadata:
    """io: Deterministic runtime metadata for one backtest invocation."""

    requested_start: str
    effective_strict_start: str
    actual_start: str
    end: str
    vintage_mode: VintageMode
    max_workers: int


@dataclass(frozen=True, slots=True)
class _ChallengerRow:
    as_of: date
    realized_52w: float
    offense_final: float
    quantiles_production: np.ndarray
    post: np.ndarray
    x_scaled: np.ndarray


@dataclass(frozen=True, slots=True)
class _ChallengerPrediction:
    row: _ChallengerRow
    global_index: int
    status: str
    quantiles: np.ndarray | None
    fit_status: str
    optimization_failed: bool


def write_backtest_jsonl(result: BacktestResult, path: Path) -> None:
    """io: Persist weekly backtest outputs as deterministic JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = b"".join(serialize_weekly_output(output) for output in result.outputs)
    path.write_bytes(payload)


def write_backtest_metadata(path: Path, metadata: BacktestRuntimeMetadata) -> None:
    """io: Persist deterministic backtest runtime metadata."""
    payload = json.dumps(
        asdict(metadata),
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{payload}\n", encoding="utf-8")


def _quantile_score(y_true: float, quantiles: np.ndarray) -> float:
    losses = np.maximum(TAUS * (y_true - quantiles), (TAUS - 1.0) * (y_true - quantiles))
    return float(2.0 * np.mean(losses))


def build_backtest_feature_cache(
    series: Mapping[str, TimeSeries],
    *,
    end: date,
) -> Mapping[str, object]:
    """io: Precompute immutable feature/history/training caches for backtest workers."""
    all_weeks = _backtest_weeks_from_series(series, end=end)
    blocks = {week: build_feature_block(series, week) for week in all_weeks}

    finite_history_weeks = tuple(week for week in all_weeks if not blocks[week][1].any())
    if finite_history_weeks:
        history_x_raw = np.vstack(
            [blocks[week][0] for week in finite_history_weeks],
        ).astype(np.float64)
        history_x_scaled = _scale_feature_history(history_x_raw)
        history_week_ordinals = np.array(
            [week.toordinal() for week in finite_history_weeks],
            dtype=np.int64,
        )
    else:
        history_x_raw = np.empty((0, 10), dtype=np.float64)
        history_x_scaled = np.empty((0, 10), dtype=np.float64)
        history_week_ordinals = np.empty(0, dtype=np.int64)

    training_weeks = tuple(
        week
        for week in finite_history_weeks
        if np.isfinite(_forward_52w_return(series["NASDAQXNDX"], week))
    )
    if training_weeks:
        training_x_raw = np.vstack([blocks[week][0] for week in training_weeks]).astype(np.float64)
        training_x_scaled = _scale_feature_history(training_x_raw)
        training_y_52w = np.asarray(
            [_forward_52w_return(series["NASDAQXNDX"], week) for week in training_weeks],
            dtype=np.float64,
        )
        training_week_ordinals = np.array(
            [week.toordinal() for week in training_weeks],
            dtype=np.int64,
        )
    else:
        training_x_raw = np.empty((0, 10), dtype=np.float64)
        training_x_scaled = np.empty((0, 10), dtype=np.float64)
        training_y_52w = np.empty(0, dtype=np.float64)
        training_week_ordinals = np.empty(0, dtype=np.int64)

    return {
        BLOCKS_KEY: blocks,
        HISTORY_WEEK_ORDINALS_KEY: history_week_ordinals,
        HISTORY_X_RAW_KEY: history_x_raw,
        HISTORY_X_SCALED_KEY: history_x_scaled,
        TRAINING_WEEK_ORDINALS_KEY: training_week_ordinals,
        TRAINING_WEEKS_KEY: training_weeks,
        TRAINING_X_RAW_KEY: training_x_raw,
        TRAINING_X_SCALED_KEY: training_x_scaled,
        TRAINING_Y_KEY: training_y_52w,
    }


def _backtest_weeks_from_series(
    series: Mapping[str, TimeSeries],
    *,
    end: date,
) -> tuple[date, ...]:
    timestamps = series["DGS10"].timestamps.astype("datetime64[D]")
    if timestamps.shape[0] == 0:
        msg = "backtest feature cache requires at least one DGS10 timestamp"
        raise ValueError(msg)
    first_observation = date.fromisoformat(str(timestamps[0]))
    first_friday = first_observation + timedelta(days=(FRIDAY - first_observation.weekday()) % 7)
    if first_friday > end:
        return ()
    return _weekly_dates(first_friday, end)


def _challenger_root(output_path: Path) -> Path:
    return output_path.parent / "challenger"


def _challenger_week_dir(root: Path, as_of: date) -> Path:
    return root / f"as_of={as_of.isoformat()}"


def _history_x_scaled_by_week(feature_cache: Mapping[str, object]) -> dict[int, np.ndarray]:
    ordinals = np.asarray(feature_cache[HISTORY_WEEK_ORDINALS_KEY], dtype=np.int64)
    x_scaled = np.asarray(feature_cache[HISTORY_X_SCALED_KEY], dtype=np.float64)
    return {int(ordinals[idx]): x_scaled[idx].copy() for idx in range(ordinals.shape[0])}


def _build_challenger_rows(
    result: BacktestResult,
    feature_cache: Mapping[str, object],
) -> list[tuple[int, _ChallengerRow]]:
    x_by_week = _history_x_scaled_by_week(feature_cache)
    rows: list[tuple[int, _ChallengerRow]] = []
    for idx, output in enumerate(result.outputs):
        if idx >= len(result.realized_52w_returns):
            continue
        x_scaled = x_by_week.get(output.as_of_date.toordinal())
        realized = float(result.realized_52w_returns[idx])
        if (
            output.vintage_mode != "strict"
            or x_scaled is None
            or not np.isfinite(realized)
            or not np.isfinite(output.state.post).all()
        ):
            continue
        rows.append(
            (
                idx,
                _ChallengerRow(
                    as_of=output.as_of_date,
                    realized_52w=realized,
                    offense_final=float(output.decision.offense_final),
                    quantiles_production=np.array(
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
                    ),
                    post=np.asarray(output.state.post, dtype=np.float64).copy(),
                    x_scaled=np.asarray(x_scaled, dtype=np.float64).copy(),
                ),
            ),
        )
    return rows


def _write_pending_challenger_fit_artifact(
    challenger_root: Path,
    as_of: date,
    train_end: date,
    fit_result: StudentTFitResult,
) -> None:
    week_dir = _challenger_week_dir(challenger_root, as_of)
    write_challenger_fit_artifact(
        week_dir / "challenger_fit_artifact.json",
        challenger_fit_artifact_to_dict(as_of=as_of, train_end=train_end, fit_result=fit_result),
    )


def _write_challenger_week_output(
    challenger_root: Path,
    prediction: _ChallengerPrediction,
) -> None:
    week_dir = _challenger_week_dir(challenger_root, prediction.row.as_of)
    write_challenger_output(
        week_dir / "challenger_output.json",
        challenger_output_to_dict(
            as_of=prediction.row.as_of,
            status=prediction.status,
            quantiles=prediction.quantiles,
            fit_status=prediction.fit_status,
            optimization_failed=prediction.optimization_failed,
            source_offense_final=prediction.row.offense_final,
        ),
    )


def _collect_challenger_metrics(
    valid_predictions: list[_ChallengerPrediction],
    realized_all: np.ndarray,
) -> dict[str, dict[str, float]]:
    if not valid_predictions:
        nan_metrics = {
            "q10_error": float("nan"),
            "q90_error": float("nan"),
            "crps_improvement": float("nan"),
            "ceq_diff": float("nan"),
        }
        return {"production": dict(nan_metrics), "challenger": dict(nan_metrics)}
    production_q10_hits: list[float] = []
    production_q90_hits: list[float] = []
    challenger_q10_hits: list[float] = []
    challenger_q90_hits: list[float] = []
    production_crps: list[float] = []
    challenger_crps: list[float] = []
    baseline_crps: list[float] = []
    production_returns: list[float] = []
    challenger_returns: list[float] = []
    baseline_returns: list[float] = []
    for item in valid_predictions:
        challenger_quantiles = cast(np.ndarray, item.quantiles)
        y_true = float(item.row.realized_52w)
        baseline = (
            np.full(TAUS.shape, y_true, dtype=np.float64)
            if item.global_index < MIN_BASELINE_HISTORY
            else np.quantile(realized_all[: item.global_index], TAUS).astype(np.float64)
        )
        production_q10_hits.append(float(y_true <= item.row.quantiles_production[1]))
        production_q90_hits.append(float(y_true <= item.row.quantiles_production[5]))
        challenger_q10_hits.append(float(y_true <= challenger_quantiles[1]))
        challenger_q90_hits.append(float(y_true <= challenger_quantiles[5]))
        production_crps.append(_quantile_score(y_true, item.row.quantiles_production))
        challenger_crps.append(_quantile_score(y_true, challenger_quantiles))
        baseline_crps.append(_quantile_score(y_true, baseline))
        production_weight = item.row.offense_final / 100.0
        production_returns.append(float(production_weight * y_true))
        challenger_returns.append(float(production_weight * y_true))
        baseline_returns.append(float(0.5 * y_true))
    return {
        "production": {
            "q10_error": abs(float(np.mean(production_q10_hits)) - 0.10),
            "q90_error": abs(float(np.mean(production_q90_hits)) - 0.90),
            "crps_improvement": crps_improvement_ratio(
                np.asarray(production_crps, dtype=np.float64),
                np.asarray(baseline_crps, dtype=np.float64),
            ),
            "ceq_diff": float(
                ceq_annualized(np.asarray(production_returns, dtype=np.float64))
                - ceq_annualized(np.asarray(baseline_returns, dtype=np.float64))
            ),
        },
        "challenger": {
            "q10_error": abs(float(np.mean(challenger_q10_hits)) - 0.10),
            "q90_error": abs(float(np.mean(challenger_q90_hits)) - 0.90),
            "crps_improvement": crps_improvement_ratio(
                np.asarray(challenger_crps, dtype=np.float64),
                np.asarray(baseline_crps, dtype=np.float64),
            ),
            "ceq_diff": float(
                ceq_annualized(np.asarray(challenger_returns, dtype=np.float64))
                - ceq_annualized(np.asarray(baseline_returns, dtype=np.float64))
            ),
        },
    }


def _run_challenger_shadow(
    result: BacktestResult,
    output_path: Path,
    feature_cache: Mapping[str, object],
    *,
    effective_start: date,
) -> dict[str, object]:
    challenger_root = _challenger_root(output_path)
    metadata = cast(
        dict[str, object],
        json.loads(output_path.with_name("backtest_metadata.json").read_text(encoding="utf-8")),
    )
    rows = _build_challenger_rows(result, feature_cache)
    predictions: list[_ChallengerPrediction] = []
    next_refit_as_of: date | None = None
    current_fit: StudentTFitResult | None = None
    current_theta: np.ndarray | None = None
    for row_idx, (global_index, row) in enumerate(rows):
        train_cutoff = row.as_of - timedelta(weeks=CHALLENGER_EMBARGO_WEEKS)
        eligible = [item for item in rows[:row_idx] if item[1].as_of <= train_cutoff]
        if len(eligible) < CHALLENGER_MIN_TRAIN_ROWS:
            prediction = _ChallengerPrediction(
                row=row,
                global_index=global_index,
                status="not_ready",
                quantiles=None,
                fit_status="not_ready",
                optimization_failed=False,
            )
            predictions.append(prediction)
            _write_challenger_week_output(challenger_root, prediction)
            continue
        if current_fit is None or next_refit_as_of is None or row.as_of >= next_refit_as_of:
            train_rows = [item[1] for item in eligible]
            fit_result = fit_student_t_location_scale(
                np.vstack([item.x_scaled for item in train_rows]).astype(np.float64),
                np.vstack([item.post for item in train_rows]).astype(np.float64),
                np.asarray([item.realized_52w for item in train_rows], dtype=np.float64),
                theta0=current_theta,
                maxiter=60 if current_fit is None else 25,
            )
            current_fit = fit_result
            current_theta = np.concatenate(
                [fit_result.params.beta_mu, fit_result.params.beta_sigma],
            ).astype(np.float64)
            next_refit_as_of = row.as_of + timedelta(weeks=CHALLENGER_REFIT_EVERY_WEEKS)
            _write_pending_challenger_fit_artifact(
                challenger_root,
                row.as_of,
                train_cutoff,
                fit_result,
            )
        quantiles = predict_student_t_quantiles(row.x_scaled, row.post, current_fit.params)
        prediction = _ChallengerPrediction(
            row=row,
            global_index=global_index,
            status="ok",
            quantiles=np.asarray(quantiles, dtype=np.float64),
            fit_status=current_fit.optimizer_status,
            optimization_failed=current_fit.optimization_failed,
        )
        predictions.append(prediction)
        _write_challenger_week_output(challenger_root, prediction)

    valid_predictions = [item for item in predictions if item.quantiles is not None]
    realized_all = np.asarray(result.realized_52w_returns, dtype=np.float64)
    comparison = {
        "window": {
            "requested_start": str(metadata["requested_start"]),
            "actual_start": str(metadata["actual_start"]),
            "end": str(metadata["end"]),
        },
        **_collect_challenger_metrics(valid_predictions, realized_all),
        "counts": {
            "valid_predictions": len(valid_predictions),
            "fit_artifact_weeks": sum(
                int(
                    (
                        _challenger_week_dir(challenger_root, prediction.row.as_of)
                        / "challenger_fit_artifact.json"
                    ).exists(),
                )
                for prediction in predictions
            ),
            "effective_strict_start": effective_start.isoformat(),
        },
    }
    write_challenger_report(challenger_root / "challenger_comparison_report.json", comparison)
    return comparison


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


def _is_process_safe_callable(func: object) -> bool:
    module = getattr(func, "__module__", "")
    qualname = getattr(func, "__qualname__", "")
    return callable(func) and "<locals>" not in qualname and module not in {"", "__main__"}


def _uses_prefetched_backtest_cache(feature_cache: Mapping[str, object] | None) -> bool:
    if not isinstance(feature_cache, Mapping):
        return False
    required_keys = {
        BLOCKS_KEY,
        HISTORY_WEEK_ORDINALS_KEY,
        HISTORY_X_RAW_KEY,
        HISTORY_X_SCALED_KEY,
        TRAINING_WEEK_ORDINALS_KEY,
        TRAINING_WEEKS_KEY,
        TRAINING_X_RAW_KEY,
        TRAINING_X_SCALED_KEY,
        TRAINING_Y_KEY,
    }
    return required_keys.issubset(feature_cache.keys())


def _configure_process_worker(
    series: Mapping[str, TimeSeries],
    cfg: FrozenConfig,
    fit_training_artifacts: FitTrainingArtifacts,
    infer_weekly: InferWeekly,
    feature_cache: Mapping[date, Any] | None,
) -> None:
    _WORKER_STATE["series"] = series
    _WORKER_STATE["cfg"] = cfg
    _WORKER_STATE["fit"] = fit_training_artifacts
    _WORKER_STATE["infer"] = infer_weekly
    _WORKER_STATE["feature_cache"] = feature_cache


def _clear_process_worker() -> None:
    _WORKER_STATE.clear()


def _run_backtest_week(as_of: date) -> WeeklyOutput:
    series = _WORKER_STATE.get("series")
    cfg = _WORKER_STATE.get("cfg")
    fit_training_artifacts = _WORKER_STATE.get("fit")
    infer_weekly = _WORKER_STATE.get("infer")
    feature_cache = _WORKER_STATE.get("feature_cache")
    if (
        not isinstance(series, Mapping)
        or not isinstance(cfg, FrozenConfig)
        or not callable(fit_training_artifacts)
        or not callable(infer_weekly)
    ):
        msg = "backtest worker state is not initialized"
        raise RuntimeError(msg)
    history = (
        series
        if _uses_prefetched_backtest_cache(cast(Mapping[str, object] | None, feature_cache))
        else _slice_series_history(series, as_of)
    )
    training_artifacts = fit_training_artifacts(as_of, history, cfg, feature_cache)
    return cast(
        WeeklyOutput,
        infer_weekly(as_of, cfg, history, training_artifacts, feature_cache),
    )


def _run_walkforward_process_pool(  # noqa: PLR0913
    start: date,
    end: date,
    series: Mapping[str, TimeSeries],
    cfg: FrozenConfig,
    *,
    fit_training_artifacts: FitTrainingArtifacts,
    infer_weekly: InferWeekly,
    feature_cache: Mapping[date, Any] | None,
    max_workers: int,
    effective_strict_start: date | None = None,
    strict_start_policy: str = "allow",
) -> BacktestResult:
    weeks = _weekly_dates(start, end)
    if (
        max_workers <= 1
        or len(weeks) <= 1
        or not _is_process_safe_callable(fit_training_artifacts)
        or not _is_process_safe_callable(infer_weekly)
    ):
        return run_walkforward(
            start,
            end,
            series,
            cfg,
            fit_training_artifacts=fit_training_artifacts,
            infer_weekly=infer_weekly,
            feature_cache=feature_cache,
            effective_strict_start=effective_strict_start,
            strict_start_policy=cast(Any, strict_start_policy),
        )
    context = cast(multiprocessing.context.BaseContext, multiprocessing.get_context("spawn"))
    try:
        with ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=context,
            initializer=_configure_process_worker,
            initargs=(
                series,
                cfg,
                fit_training_artifacts,
                infer_weekly,
                feature_cache,
            ),
        ) as executor:
            outputs = tuple(executor.map(_run_backtest_week, weeks))
    finally:
        _clear_process_worker()
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
    feature_cache = build_backtest_feature_cache(series, end=end)
    effective_start = cfg.strict_pit_start
    if vintage_mode == "strict":
        effective_start = compute_effective_strict_acceptance_start_from_series(
            series,
            strict_mode_start=cfg.strict_pit_start,
            feature_cache=feature_cache,
        )
    strict_start_policy = "clip_diagnostic" if vintage_mode == "strict" else "allow"
    run_start = effective_start if vintage_mode == "strict" and start < effective_start else start
    write_backtest_metadata(
        output_path.with_name("backtest_metadata.json"),
        BacktestRuntimeMetadata(
            requested_start=start.isoformat(),
            effective_strict_start=effective_start.isoformat(),
            actual_start=run_start.isoformat(),
            end=end.isoformat(),
            vintage_mode=vintage_mode,
            max_workers=deps.max_workers,
        ),
    )
    if deps.max_workers > 1:
        result = _run_walkforward_process_pool(
            run_start,
            end,
            series,
            cfg,
            fit_training_artifacts=deps.fit_training_artifacts,
            infer_weekly=deps.infer_weekly,
            feature_cache=cast(Mapping[date, Any], feature_cache),
            max_workers=deps.max_workers,
            effective_strict_start=effective_start,
            strict_start_policy=cast(Any, strict_start_policy),
        )
    elif _uses_prefetched_backtest_cache(cast(Mapping[str, object] | None, feature_cache)):
        outputs = _run_backtest_week_direct(
            _weekly_dates(run_start, end),
            series,
            cfg,
            deps,
            cast(Mapping[date, Any], feature_cache),
        )
        result = BacktestResult(outputs=outputs)
    else:
        result = run_walkforward(
            run_start,
            end,
            series,
            cfg,
            fit_training_artifacts=deps.fit_training_artifacts,
            infer_weekly=deps.infer_weekly,
            feature_cache=cast(Mapping[date, Any], feature_cache),
            effective_strict_start=effective_start,
            strict_start_policy=cast(Any, strict_start_policy),
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
    _run_challenger_shadow(
        result_with_targets,
        output_path,
        feature_cache,
        effective_start=effective_start,
    )
    passed = write_acceptance_report(
        result_with_targets,
        cfg,
        output_path.with_name("acceptance_report.json"),
        effective_start,
    )
    return EXIT_OK if passed else EXIT_ACCEPTANCE_FAILED


def _run_backtest_week_direct(
    weeks: tuple[date, ...],
    series: Mapping[str, TimeSeries],
    cfg: FrozenConfig,
    deps: BacktestRunnerDeps,
    feature_cache: Mapping[date, Any],
) -> tuple[WeeklyOutput, ...]:
    outputs: list[WeeklyOutput] = []
    for as_of in weeks:
        training_artifacts = deps.fit_training_artifacts(as_of, series, cfg, feature_cache)
        outputs.append(deps.infer_weekly(as_of, cfg, series, training_artifacts, feature_cache))
    return tuple(outputs)
