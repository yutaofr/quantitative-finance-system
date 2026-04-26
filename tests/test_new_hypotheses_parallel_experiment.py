from __future__ import annotations

import pytest

from research.run_new_hypotheses_parallel_experiment import (
    CORE_METRIC_KEYS,
    HYPOTHESIS_IDS,
    _apply_final_decision,
    _window_metric_payload_is_complete,
)

pytestmark = pytest.mark.synthetic


def test_parallel_hypothesis_set_is_independent_and_complete() -> None:
    assert HYPOTHESIS_IDS == (
        "A_DIRECT_DENSITY",
        "B_DIRECT_QUANTILES",
        "C_DECOUPLED_HEADS",
        "D_LATENT_STATE",
    )


def test_window_metric_payload_requires_unified_core_keys() -> None:
    payload = {
        "mean_z": 0.0,
        "std_z": 1.0,
        "corr_next": 0.1,
        "rank_next": 0.1,
        "lag1_acf_z": 0.0,
        "sigma_blowup": 0,
        "pathology": 0,
        "crps": 0.05,
        "n_obs": 26,
    }

    assert set(CORE_METRIC_KEYS).issubset(payload)
    assert _window_metric_payload_is_complete(payload)

    incomplete = dict(payload)
    del incomplete["crps"]
    assert not _window_metric_payload_is_complete(incomplete)


def test_final_decision_selects_single_winner_only_when_protocol_passes() -> None:
    candidates = {
        "A_DIRECT_DENSITY": {
            "single_line_decision": "WORTH_CONTINUING",
            "aggregate": {"material_improvement_score": 0.06},
            "protocol_pass": True,
        },
        "B_DIRECT_QUANTILES": {
            "single_line_decision": "WORTH_CONTINUING",
            "aggregate": {"material_improvement_score": 0.02},
            "protocol_pass": True,
        },
        "C_DECOUPLED_HEADS": {
            "single_line_decision": "PROMISING_BUT_INSUFFICIENT",
            "aggregate": {"material_improvement_score": 0.10},
            "protocol_pass": False,
        },
    }

    decision = _apply_final_decision(candidates)

    assert decision["overall_decision"] == "SELECT_ONE_CONTINUE"
    assert decision["winner"] == "A_DIRECT_DENSITY"


def test_final_decision_terminates_when_no_candidate_passes() -> None:
    candidates = {
        "A_DIRECT_DENSITY": {
            "single_line_decision": "FAILED",
            "aggregate": {"material_improvement_score": -0.01},
            "protocol_pass": False,
        },
        "B_DIRECT_QUANTILES": {
            "single_line_decision": "PROMISING_BUT_INSUFFICIENT",
            "aggregate": {"material_improvement_score": 0.01},
            "protocol_pass": False,
        },
    }

    decision = _apply_final_decision(candidates)

    assert decision["overall_decision"] == "NO_MODEL_WORTH_CONTINUING"
    assert decision["winner"] is None
