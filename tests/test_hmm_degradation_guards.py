from __future__ import annotations

import numpy as np

from engine_types import Stance
from state.ti_hmm_single import (
    has_degenerate_state_occupancy,
    has_invalid_posterior,
    has_label_order_flip,
    should_degrade_hmm,
)


def test_has_invalid_posterior_detects_nan_inf_and_bad_rows() -> None:
    assert has_invalid_posterior(np.array([[0.2, np.nan, 0.8]], dtype=np.float64))
    assert has_invalid_posterior(np.array([[0.2, np.inf, 0.8]], dtype=np.float64))
    assert has_invalid_posterior(np.array([[0.2, 0.2, 0.2]], dtype=np.float64))
    assert not has_invalid_posterior(np.array([[0.2, 0.3, 0.5]], dtype=np.float64))


def test_has_degenerate_state_occupancy_detects_26_week_starvation() -> None:
    posterior = np.tile(np.array([[0.995, 0.004, 0.001]], dtype=np.float64), (26, 1))

    assert has_degenerate_state_occupancy(posterior)


def test_has_degenerate_state_occupancy_ignores_short_windows() -> None:
    posterior = np.tile(np.array([[0.995, 0.004, 0.001]], dtype=np.float64), (25, 1))

    assert not has_degenerate_state_occupancy(posterior)


def test_has_label_order_flip_detects_changed_semantic_mapping() -> None:
    persisted: dict[int, Stance] = {0: "DEFENSIVE", 1: "NEUTRAL", 2: "OFFENSIVE"}
    refit_returns = {0: 0.10, 1: 0.02, 2: -0.05}

    assert has_label_order_flip(persisted, refit_returns)
    assert not has_label_order_flip(persisted, {0: -0.05, 1: 0.02, 2: 0.10})


def test_should_degrade_hmm_combines_all_srd_section_7_3_guards() -> None:
    posterior = np.array([[0.2, 0.3, 0.5]], dtype=np.float64)
    persisted: dict[int, Stance] = {0: "DEFENSIVE", 1: "NEUTRAL", 2: "OFFENSIVE"}

    assert not should_degrade_hmm(posterior, persisted_label_map=persisted)
    assert should_degrade_hmm(
        posterior,
        persisted_label_map=persisted,
        refit_forward_returns_by_state={0: 0.10, 1: 0.02, 2: -0.05},
    )
