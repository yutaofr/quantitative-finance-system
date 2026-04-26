from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from research.run_cycle_evaluation_protocol import (
    EvaluationSeries,
    _classification_metrics,
    _cycle_states_from_scores,
    _final_decision,
    _primary_labels_from_market_frame,
)

pytestmark = pytest.mark.synthetic


def test_primary_labels_reward_forward_return_and_penalize_risk() -> None:
    forward_return = np.array([-0.20, -0.05, 0.05, 0.20], dtype=np.float64)
    forward_risk = np.array([0.30, 0.20, 0.10, 0.05], dtype=np.float64)
    forward_drawdown = np.array([0.35, 0.18, 0.08, 0.02], dtype=np.float64)

    labels = _primary_labels_from_market_frame(forward_return, forward_risk, forward_drawdown)

    assert labels.composite_score[0] < labels.composite_score[-1]
    assert labels.states[0] == "CONTRACTION"
    assert labels.states[-1] == "EXPANSION"


def test_cycle_states_from_scores_uses_registered_three_state_order() -> None:
    states = _cycle_states_from_scores(np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64))

    assert states == ("CONTRACTION", "CONTRACTION", "SLOWDOWN", "EXPANSION", "EXPANSION")


def test_classification_metrics_are_balanced_and_macro_averaged() -> None:
    truth = ("EXPANSION", "EXPANSION", "SLOWDOWN", "CONTRACTION", "CONTRACTION")
    pred = ("EXPANSION", "SLOWDOWN", "SLOWDOWN", "CONTRACTION", "EXPANSION")

    metrics = _classification_metrics(truth, pred)

    assert metrics["balanced_accuracy"] == pytest.approx((0.5 + 1.0 + 0.5) / 3.0)
    assert metrics["macro_f1"] > 0.0
    assert metrics["confusion_matrix"]["CONTRACTION"]["EXPANSION"] == 1


def test_final_decision_requires_current_system_to_survive_cross_window_gates() -> None:
    days = (date(2020, 1, 3), date(2020, 1, 10), date(2020, 1, 17))
    weak = EvaluationSeries(
        name="T5_DERIVED_CYCLE_PROXY",
        legal_status="evaluated_existing_object",
        dates=days,
        score=np.array([1.0, -1.0, 1.0], dtype=np.float64),
    )
    payload = {
        "objects": {
            "T5_DERIVED_CYCLE_PROXY": {
                "series": weak,
                "decision": {"layer_1": "PASS", "layer_2": "FAIL", "layer_3": "FAIL"},
            },
            "EGARCH_DERIVED_CYCLE_PROXY": {
                "series": weak,
                "decision": {"layer_1": "PASS", "layer_2": "FAIL", "layer_3": "FAIL"},
            },
        },
    }

    decision = _final_decision(payload)

    assert decision["category"] == "CURRENT_SYSTEM_HAS_NO_CYCLE_CAPABILITY"
    assert decision["allow_new_cycle_model"] is False
