"""Pure weekly output assembly and fallback helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from decision.offense_abs import OffenseThresholds
from decision.utility import UtilityZStats
from engine_types import (
    DecisionOutput,
    DiagnosticsOutput,
    DistributionOutput,
    Stance,
    VintageMode,
    WeeklyOutput,
    WeeklyState,
)
from law.linear_quantiles import QRCoefs
from state.ti_hmm_single import HMMModel

SRD_VERSION: Literal["8.7"] = "8.7"
STATE_COUNT = 3
BLOCKED_OFFENSE = 50.0


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
