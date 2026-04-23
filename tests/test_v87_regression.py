from __future__ import annotations

from datetime import date
import hashlib

import numpy as np

from app.output_serializer import serialize_weekly_output
from engine_types import (
    DecisionOutput,
    DiagnosticsOutput,
    DistributionOutput,
    WeeklyOutput,
    WeeklyState,
)

EXPECTED_WEEKLY_HASH = "a4a2d0108717f8db710302eeba772aca90c2c1321ea97a82237a1a9ddd4e9549"


def test_v87_weekly_output_snapshot_remains_byte_identical() -> None:
    output = WeeklyOutput(
        as_of_date=date(2024, 12, 27),
        srd_version="8.7.1",
        mode="NORMAL",
        vintage_mode="strict",
        state=WeeklyState(
            post=np.array([0.25, 0.50, 0.25], dtype=np.float64),
            state_name="NEUTRAL",
            dwell_weeks=4,
            hazard_covariate=0.15,
        ),
        distribution=DistributionOutput(
            q05=-0.12,
            q10=-0.08,
            q25=-0.02,
            q50=0.04,
            q75=0.09,
            q90=0.15,
            q95=0.18,
            q05_ci_low=-0.20,
            q05_ci_high=-0.10,
            q95_ci_low=0.10,
            q95_ci_high=0.25,
            mu_hat=0.05,
            sigma_hat=0.12,
            p_loss=0.42,
            es20=0.09,
        ),
        decision=DecisionOutput(
            excess_return=0.03,
            utility=0.8,
            offense_raw=56.0,
            offense_final=56.0,
            stance="NEUTRAL",
            cycle_position=48.5,
        ),
        diagnostics=DiagnosticsOutput(
            missing_rate=0.0,
            quantile_solver_status="ok",
            tail_extrapolation_status="ok",
            hmm_status="ok",
            coverage_q10_trailing_104w=0.88,
            coverage_q90_trailing_104w=0.91,
        ),
    )

    digest = hashlib.sha256(serialize_weekly_output(output)).hexdigest()

    assert digest == EXPECTED_WEEKLY_HASH
