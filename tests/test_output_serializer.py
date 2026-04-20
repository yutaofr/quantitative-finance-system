from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np

from app.output_serializer import serialize_weekly_output, to_serializable_dict, write_weekly_output
from engine_types import (
    DecisionOutput,
    DiagnosticsOutput,
    DistributionOutput,
    WeeklyOutput,
    WeeklyState,
)


def _sample_output() -> WeeklyOutput:
    return WeeklyOutput(
        as_of_date=date(2024, 12, 27),
        srd_version="8.7",
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


def test_to_serializable_dict_matches_srd_output_contract() -> None:
    payload = to_serializable_dict(_sample_output())

    assert tuple(payload) == (
        "as_of_date",
        "srd_version",
        "mode",
        "vintage_mode",
        "state",
        "distribution",
        "decision",
        "diagnostics",
    )
    assert tuple(payload["state"]) == ("post", "state_name", "dwell_weeks", "hazard_covariate")
    assert tuple(payload["decision"]) == (
        "excess_return",
        "utility",
        "offense_raw",
        "offense_final",
        "stance",
        "cycle_position",
    )
    assert payload["state"]["post"] == [0.25, 0.5, 0.25]
    assert payload["decision"]["stance"] == "NEUTRAL"


def test_serialize_weekly_output_is_byte_stable() -> None:
    first = serialize_weekly_output(_sample_output())
    second = serialize_weekly_output(_sample_output())

    assert first == second
    assert b" " not in first
    assert first.endswith(b"\n")


def test_write_weekly_output_persists_deterministic_bytes(tmp_path: Path) -> None:
    target = tmp_path / "production_output.json"

    write_weekly_output(_sample_output(), target)
    first = target.read_bytes()
    write_weekly_output(_sample_output(), target)
    second = target.read_bytes()

    assert first == second
