"""Pure weekly output assembly and fallback helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from config_types import SRD_VERSION
from decision.cycle_position import cycle_position
from decision.hysteresis import apply_band
from decision.offense_abs import OffenseThresholds, offense_raw, stance_from_offense
from decision.utility import UtilityZStats, excess_return, utility
from engine_types import (
    DecisionOutput,
    DiagnosticsOutput,
    DistributionOutput,
    Stance,
    TimeSeries,
    VintageMode,
    WeeklyOutput,
    WeeklyState,
)
from errors import HMMConvergenceError
from features.block_builder import build_feature_block
from features.pca import robust_pca_2d
from features.scaling import robust_zscore_expanding, soft_squash_clip
from law.linear_quantiles import QRCoefs, predict_interior_with_status
from law.quantile_moments import moments_from_quantiles
from law.tail_extrapolation import extrapolate_tails
from state.ti_hmm_single import HMMModel, degraded_hmm_posterior, infer_hmm

STATE_COUNT = 3
BLOCKED_OFFENSE = 50.0
FULL_TAUS = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
MISSING_DEGRADED = 0.10
MISSING_BLOCKED = 0.20
DEFAULT_PREVIOUS_OFFENSE = 50.0
FEATURE_COUNT = 10
HMM_OBS_MIN_ROWS = 2
FRIDAY = 4
BLOCKS_KEY = "blocks"
HISTORY_WEEK_ORDINALS_KEY = "history_week_ordinals"
HISTORY_X_RAW_KEY = "history_x_raw"
HISTORY_X_SCALED_KEY = "history_x_scaled"


@dataclass(frozen=True, slots=True)
class TrainingArtifacts:
    """pure. Frozen training-era artifacts consumed at inference time."""

    utility_zstats: UtilityZStats | None
    offense_thresholds: OffenseThresholds | None
    train_distributions: Mapping[str, NDArray[np.float64]]
    state_label_map: Mapping[int, Stance]
    qr_coefs: QRCoefs | None = None
    hmm_model: HMMModel | None = None


def blocked_weekly_output(  # noqa: PLR0913
    as_of: date,
    *,
    vintage_mode: VintageMode,
    missing_rate: float = 0.0,
    quantile_solver_status: str = "failed",
    tail_extrapolation_status: str = "ok",
    hmm_status: str = "ok",
) -> WeeklyOutput:
    """pure. Build SRD §10 BLOCKED fallback output."""
    return WeeklyOutput(
        as_of_date=as_of,
        srd_version=SRD_VERSION,
        mode="BLOCKED",
        vintage_mode=vintage_mode,
        state=WeeklyState(
            post=np.array([0.25, 0.50, 0.25], dtype=np.float64),
            state_name="NEUTRAL",
            dwell_weeks=0,
            hazard_covariate=0.0,
        ),
        distribution=DistributionOutput(
            q05=0.0,
            q10=0.0,
            q25=0.0,
            q50=0.0,
            q75=0.0,
            q90=0.0,
            q95=0.0,
            q05_ci_low=0.0,
            q05_ci_high=0.0,
            q95_ci_low=0.0,
            q95_ci_high=0.0,
            mu_hat=0.0,
            sigma_hat=0.0,
            p_loss=0.0,
            es20=0.0,
        ),
        decision=DecisionOutput(
            excess_return=0.0,
            utility=0.0,
            offense_raw=BLOCKED_OFFENSE,
            offense_final=BLOCKED_OFFENSE,
            stance="NEUTRAL",
            cycle_position=0.0,
        ),
        diagnostics=DiagnosticsOutput(
            missing_rate=missing_rate,
            quantile_solver_status=quantile_solver_status,
            tail_extrapolation_status=tail_extrapolation_status,
            hmm_status=hmm_status,
            coverage_q10_trailing_104w=0.0,
            coverage_q90_trailing_104w=0.0,
        ),
    )


def degraded_weekly_output(  # noqa: PLR0913
    as_of: date,
    *,
    vintage_mode: VintageMode,
    missing_rate: float = 0.0,
    quantile_solver_status: str = "ok",
    tail_extrapolation_status: str = "ok",
    hmm_status: str = "degenerate",
) -> WeeklyOutput:
    """pure. Build SRD §10 DEGRADED fallback output."""
    blocked = blocked_weekly_output(
        as_of,
        vintage_mode=vintage_mode,
        missing_rate=missing_rate,
        quantile_solver_status=quantile_solver_status,
        tail_extrapolation_status=tail_extrapolation_status,
        hmm_status=hmm_status,
    )
    return WeeklyOutput(
        as_of_date=blocked.as_of_date,
        srd_version=blocked.srd_version,
        mode="DEGRADED",
        vintage_mode=blocked.vintage_mode,
        state=blocked.state,
        distribution=blocked.distribution,
        decision=blocked.decision,
        diagnostics=blocked.diagnostics,
    )


def _require_artifacts(
    artifacts: TrainingArtifacts,
) -> tuple[UtilityZStats, OffenseThresholds, QRCoefs]:
    if (
        artifacts.utility_zstats is None
        or artifacts.offense_thresholds is None
        or artifacts.qr_coefs is None
    ):
        msg = "training artifacts are incomplete"
        raise ValueError(msg)
    return artifacts.utility_zstats, artifacts.offense_thresholds, artifacts.qr_coefs


def _scale_feature_history(raw_history: NDArray[np.float64]) -> NDArray[np.float64]:
    values = np.asarray(raw_history, dtype=np.float64)
    scaled = np.empty_like(values, dtype=np.float64)
    for idx in range(values.shape[1]):
        scaled[:, idx] = soft_squash_clip(robust_zscore_expanding(values[:, idx]))
    return scaled


def _weekly_dates_from_series(series: Mapping[str, TimeSeries], as_of: date) -> tuple[date, ...]:
    timestamps = series["DGS10"].timestamps.astype("datetime64[D]")
    filtered = timestamps[timestamps <= np.datetime64(as_of, "D")]
    return tuple(
        week
        for week in (date.fromisoformat(str(value)) for value in filtered.tolist())
        if week.weekday() == FRIDAY
    )


def _feature_history(
    series: Mapping[str, TimeSeries],
    as_of: date,
    feature_cache: Any = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    if isinstance(feature_cache, Mapping) and HISTORY_WEEK_ORDINALS_KEY in feature_cache:
        week_ordinals = np.asarray(feature_cache[HISTORY_WEEK_ORDINALS_KEY], dtype=np.int64)
        count = int(np.searchsorted(week_ordinals, as_of.toordinal(), side="right"))
        if count == 0:
            return (
                np.empty((0, FEATURE_COUNT), dtype=np.float64),
                np.empty((0, FEATURE_COUNT), dtype=np.float64),
                np.empty((0, FEATURE_COUNT), dtype=np.bool_),
            )
        raw_history = np.asarray(feature_cache[HISTORY_X_RAW_KEY], dtype=np.float64)[:count].copy()
        scaled_history = np.asarray(feature_cache[HISTORY_X_SCALED_KEY], dtype=np.float64)[
            :count
        ].copy()
        return (
            raw_history,
            scaled_history,
            np.zeros((count, FEATURE_COUNT), dtype=np.bool_),
        )
    rows: list[NDArray[np.float64]] = []
    masks: list[NDArray[np.bool_]] = []
    for week in _weekly_dates_from_series(series, as_of):
        if isinstance(feature_cache, Mapping) and BLOCKS_KEY in feature_cache:
            raw, missing = cast(
                Mapping[date, tuple[NDArray[np.float64], NDArray[np.bool_]]],
                feature_cache[BLOCKS_KEY],
            )[week]
        elif feature_cache is not None and week in feature_cache:
            raw, missing = feature_cache[week]
        else:
            raw, missing = build_feature_block(series, week)
        rows.append(raw)
        masks.append(missing)
    if not rows:
        return (
            np.empty((0, FEATURE_COUNT), dtype=np.float64),
            np.empty((0, FEATURE_COUNT), dtype=np.float64),
            np.empty((0, FEATURE_COUNT), dtype=np.bool_),
        )
    raw_history = np.vstack(rows)
    missing_history = np.vstack(masks)
    finite_mask = ~np.any(missing_history, axis=1)
    finite_history = raw_history[finite_mask]
    if finite_history.shape[0] == 0:
        return raw_history, np.empty((0, FEATURE_COUNT), dtype=np.float64), missing_history
    return raw_history, _scale_feature_history(finite_history), missing_history


def _hmm_inputs(
    x_scaled_history: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if x_scaled_history.shape[0] < HMM_OBS_MIN_ROWS:
        msg = "not enough finite feature rows for HMM inference"
        raise HMMConvergenceError(msg)
    pc = robust_pca_2d(x_scaled_history)
    delta = pc[1:] - pc[:-1]
    y_obs = np.column_stack(
        [
            pc[1:, 0],
            pc[1:, 1],
            delta[:, 0],
            delta[:, 1],
            x_scaled_history[1:, 7],
            x_scaled_history[1:, 8],
        ],
    )
    h = x_scaled_history[1:, 4] + 0.5 * x_scaled_history[1:, 8]
    return y_obs, h


def _distribution_output(full_quantiles: NDArray[np.float64]) -> DistributionOutput:
    moments = moments_from_quantiles(FULL_TAUS, full_quantiles)
    return DistributionOutput(
        q05=float(full_quantiles[0]),
        q10=float(full_quantiles[1]),
        q25=float(full_quantiles[2]),
        q50=float(full_quantiles[3]),
        q75=float(full_quantiles[4]),
        q90=float(full_quantiles[5]),
        q95=float(full_quantiles[6]),
        q05_ci_low=float(full_quantiles[0]),
        q05_ci_high=float(full_quantiles[0]),
        q95_ci_low=float(full_quantiles[6]),
        q95_ci_high=float(full_quantiles[6]),
        mu_hat=moments["mu_hat"],
        sigma_hat=moments["sigma_hat"],
        p_loss=moments["p_loss"],
        es20=moments["es20"],
    )


def run_weekly(  # noqa: PLR0913
    as_of: date,
    vintage_mode: VintageMode,
    series: Mapping[str, TimeSeries],
    training_artifacts: TrainingArtifacts,
    *,
    previous_offense_final: float = DEFAULT_PREVIOUS_OFFENSE,
    feature_cache: Any = None,
) -> WeeklyOutput:
    """pure. Assemble one SRD §11 WeeklyOutput from injected PIT data and artifacts."""
    zstats, thresholds, qr_coefs = _require_artifacts(training_artifacts)
    if isinstance(feature_cache, Mapping) and BLOCKS_KEY in feature_cache:
        raw, missing_mask = cast(
            Mapping[date, tuple[NDArray[np.float64], NDArray[np.bool_]]],
            feature_cache[BLOCKS_KEY],
        )[as_of]
    elif feature_cache is not None and as_of in feature_cache:
        raw, missing_mask = feature_cache[as_of]
    else:
        raw, missing_mask = build_feature_block(series, as_of)
    missing_rate = float(np.mean(missing_mask))
    if missing_rate > MISSING_BLOCKED:
        return blocked_weekly_output(as_of, vintage_mode=vintage_mode, missing_rate=missing_rate)

    _raw_history, x_scaled_history, _missing_history = _feature_history(
        series,
        as_of,
        feature_cache,
    )
    if x_scaled_history.shape[0] == 0:
        return blocked_weekly_output(as_of, vintage_mode=vintage_mode, missing_rate=missing_rate)
    x_scaled = x_scaled_history[-1]
    hmm = degraded_hmm_posterior()
    dwell_weeks = 0
    hazard_covariate = 0.0
    if training_artifacts.hmm_model is not None:
        try:
            y_obs, h_history = _hmm_inputs(x_scaled_history)
            hmm_result = infer_hmm(training_artifacts.hmm_model, y_obs, h_history)
            hmm = hmm_result.posterior
            dwell_weeks = hmm_result.dwell_weeks
            hazard_covariate = hmm_result.hazard_covariate
        except (HMMConvergenceError, ValueError):
            hmm = degraded_hmm_posterior()
    post = hmm.post
    interior, quantile_solver_status = predict_interior_with_status(qr_coefs, x_scaled, post)
    full_quantiles, tail_status = extrapolate_tails(interior)
    distribution = _distribution_output(full_quantiles)
    dgs1 = float(raw[2])
    er = excess_return(distribution.mu_hat, dgs1)
    score = utility(
        er,
        distribution.es20,
        distribution.p_loss,
        zstats,
    )
    raw_offense = offense_raw(score, thresholds)
    final_offense = apply_band(raw_offense, previous_offense_final)
    stance = stance_from_offense(final_offense)
    cycle = cycle_position(
        raw[4],
        raw[8],
        raw[0],
        training_artifacts.train_distributions,
    )
    mode: Literal["NORMAL", "DEGRADED", "BLOCKED"]
    mode = (
        "DEGRADED"
        if (
            missing_rate >= MISSING_DEGRADED
            or hmm.model_status == "DEGRADED"
            or tail_status == "fallback"
        )
        else "NORMAL"
    )
    return WeeklyOutput(
        as_of_date=as_of,
        srd_version=SRD_VERSION,
        mode=mode,
        vintage_mode=vintage_mode,
        state=WeeklyState(
            post=post,
            state_name=hmm.state_name,
            dwell_weeks=dwell_weeks,
            hazard_covariate=hazard_covariate,
        ),
        distribution=distribution,
        decision=DecisionOutput(
            excess_return=er,
            utility=score,
            offense_raw=raw_offense,
            offense_final=final_offense,
            stance=stance,
            cycle_position=cycle,
        ),
        diagnostics=DiagnosticsOutput(
            missing_rate=missing_rate,
            quantile_solver_status=quantile_solver_status,
            tail_extrapolation_status=tail_status,
            hmm_status="degenerate" if hmm.model_status == "DEGRADED" else "ok",
            coverage_q10_trailing_104w=0.0,
            coverage_q90_trailing_104w=0.0,
        ),
    )
