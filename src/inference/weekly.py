"""Pure weekly output assembly and fallback helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import Literal

import numpy as np
from numpy.typing import NDArray

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
from features.block_builder import build_feature_block
from features.scaling import robust_zscore_expanding, soft_squash_clip
from law.linear_quantiles import QRCoefs, predict_interior
from law.quantile_moments import moments_from_quantiles
from law.tail_extrapolation import extrapolate_tails
from state.ti_hmm_single import HMMModel, degraded_hmm_posterior

SRD_VERSION: Literal["8.7"] = "8.7"
STATE_COUNT = 3
BLOCKED_OFFENSE = 50.0
FULL_TAUS = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
MISSING_BLOCKED = 0.20
DEFAULT_PREVIOUS_OFFENSE = 50.0


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


def _scale_current_features(raw: NDArray[np.float64]) -> NDArray[np.float64]:
    z = robust_zscore_expanding(np.asarray(raw, dtype=np.float64))
    return soft_squash_clip(z)


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


def run_weekly(
    as_of: date,
    vintage_mode: VintageMode,
    series: Mapping[str, TimeSeries],
    training_artifacts: TrainingArtifacts,
    *,
    previous_offense_final: float = DEFAULT_PREVIOUS_OFFENSE,
) -> WeeklyOutput:
    """pure. Assemble one SRD §11 WeeklyOutput from injected PIT data and artifacts."""
    zstats, thresholds, qr_coefs = _require_artifacts(training_artifacts)
    raw, missing_mask = build_feature_block(series, as_of)
    missing_rate = float(np.mean(missing_mask))
    if missing_rate > MISSING_BLOCKED:
        return blocked_weekly_output(as_of, vintage_mode=vintage_mode, missing_rate=missing_rate)

    x_scaled = _scale_current_features(raw)
    hmm = degraded_hmm_posterior()
    post = hmm.post
    interior = predict_interior(qr_coefs, x_scaled, post)
    full_quantiles, tail_status = extrapolate_tails(interior)
    distribution = _distribution_output(full_quantiles)
    dgs1 = float(series["DGS1"].values[-1])
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
    mode = "DEGRADED" if hmm.model_status == "DEGRADED" or tail_status == "fallback" else "NORMAL"
    return WeeklyOutput(
        as_of_date=as_of,
        srd_version=SRD_VERSION,
        mode=mode,
        vintage_mode=vintage_mode,
        state=WeeklyState(
            post=post,
            state_name=hmm.state_name,
            dwell_weeks=0,
            hazard_covariate=0.0,
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
            quantile_solver_status=qr_coefs.solver_status,
            tail_extrapolation_status=tail_status,
            hmm_status="degenerate" if hmm.model_status == "DEGRADED" else "ok",
            coverage_q10_trailing_104w=0.0,
            coverage_q90_trailing_104w=0.0,
        ),
    )
