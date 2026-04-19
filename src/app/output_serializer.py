"""Deterministic shell serialization for WeeklyOutput artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from engine_types import WeeklyOutput


def to_serializable_dict(output: WeeklyOutput) -> dict[str, Any]:
    """io: convert a frozen WeeklyOutput into the exact SRD section 11 JSON payload."""
    return {
        "as_of_date": output.as_of_date.isoformat(),
        "decision": {
            "cycle_position": output.decision.cycle_position,
            "excess_return": output.decision.excess_return,
            "offense_final": output.decision.offense_final,
            "offense_raw": output.decision.offense_raw,
            "stance": output.decision.stance,
            "utility": output.decision.utility,
        },
        "diagnostics": {
            "coverage_q10_trailing_104w": output.diagnostics.coverage_q10_trailing_104w,
            "coverage_q90_trailing_104w": output.diagnostics.coverage_q90_trailing_104w,
            "hmm_status": output.diagnostics.hmm_status,
            "missing_rate": output.diagnostics.missing_rate,
            "quantile_solver_status": output.diagnostics.quantile_solver_status,
            "tail_extrapolation_status": output.diagnostics.tail_extrapolation_status,
        },
        "distribution": {
            "es20": output.distribution.es20,
            "mu_hat": output.distribution.mu_hat,
            "p_loss": output.distribution.p_loss,
            "q05": output.distribution.q05,
            "q05_ci_high": output.distribution.q05_ci_high,
            "q05_ci_low": output.distribution.q05_ci_low,
            "q10": output.distribution.q10,
            "q25": output.distribution.q25,
            "q50": output.distribution.q50,
            "q75": output.distribution.q75,
            "q90": output.distribution.q90,
            "q95": output.distribution.q95,
            "q95_ci_high": output.distribution.q95_ci_high,
            "q95_ci_low": output.distribution.q95_ci_low,
            "sigma_hat": output.distribution.sigma_hat,
        },
        "mode": output.mode,
        "srd_version": output.srd_version,
        "state": {
            "dwell_weeks": output.state.dwell_weeks,
            "hazard_covariate": output.state.hazard_covariate,
            "post": [float(value) for value in output.state.post.tolist()],
            "state_name": output.state.state_name,
        },
        "vintage_mode": output.vintage_mode,
    }


def serialize_weekly_output(output: WeeklyOutput) -> bytes:
    """io: render a WeeklyOutput into deterministic UTF-8 JSON bytes."""
    payload: dict[str, Any] = to_serializable_dict(output)
    text = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return (text + "\n").encode("utf-8")


def write_weekly_output(output: WeeklyOutput, path: Path) -> None:
    """io: write deterministic weekly output bytes to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(serialize_weekly_output(output))
