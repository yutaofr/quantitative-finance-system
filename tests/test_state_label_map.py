from __future__ import annotations

import pytest

from state.state_label_map import build_label_map, label_map_json_bytes


def test_build_label_map_orders_states_by_forward_return() -> None:
    label_map = build_label_map({2: 0.08, 0: -0.12, 1: 0.01})

    assert label_map == {
        0: "DEFENSIVE",
        1: "NEUTRAL",
        2: "OFFENSIVE",
    }


def test_build_label_map_is_deterministic_for_permuted_input_order() -> None:
    first = build_label_map({2: 0.08, 0: -0.12, 1: 0.01})
    second = build_label_map({1: 0.01, 2: 0.08, 0: -0.12})

    assert first == second
    assert label_map_json_bytes(first) == label_map_json_bytes(second)


def test_build_label_map_ties_break_by_state_index() -> None:
    label_map = build_label_map({2: 0.01, 0: 0.01, 1: 0.08})

    assert label_map == {
        0: "DEFENSIVE",
        2: "NEUTRAL",
        1: "OFFENSIVE",
    }


def test_build_label_map_requires_three_finite_states() -> None:
    with pytest.raises(ValueError, match="exactly 3"):
        build_label_map({0: 0.01, 1: 0.02})

    with pytest.raises(ValueError, match="finite"):
        build_label_map({0: 0.01, 1: float("nan"), 2: 0.03})
