from __future__ import annotations

import numpy as np

from state.ti_hmm_single import log_forward_filter, logsumexp_stable


def test_log_forward_filter_matches_manual_logsumexp_recurrence() -> None:
    log_initial = np.log(np.array([0.6, 0.3, 0.1], dtype=np.float64))
    transition = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7],
        ],
        dtype=np.float64,
    )
    log_transition = np.log(transition)[None, :, :]
    log_emission = np.log(
        np.array(
            [
                [0.5, 0.4, 0.1],
                [0.2, 0.7, 0.1],
            ],
            dtype=np.float64,
        ),
    )

    result = log_forward_filter(log_initial, log_transition, log_emission)

    first_joint = log_initial + log_emission[0]
    first_norm = logsumexp_stable(first_joint)
    first_filtered = first_joint - first_norm
    second_joint = np.array(
        [
            logsumexp_stable(first_filtered + log_transition[0, :, state]) + log_emission[1, state]
            for state in range(3)
        ],
        dtype=np.float64,
    )
    second_norm = logsumexp_stable(second_joint)

    assert np.allclose(result.log_alpha[0], first_filtered)
    assert np.allclose(result.log_alpha[1], second_joint - second_norm)
    assert np.isclose(result.log_likelihood, first_norm + second_norm)


def test_log_forward_filter_stays_finite_for_extreme_negative_likelihoods() -> None:
    log_initial = np.log(np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float64))
    transition = np.full((4, 3, 3), 1.0 / 3.0, dtype=np.float64)
    log_transition = np.log(transition)
    log_emission = np.full((5, 3), -10_000.0, dtype=np.float64)

    result = log_forward_filter(log_initial, log_transition, log_emission)

    assert np.isfinite(result.log_alpha).all()
    assert np.isfinite(result.log_likelihood)
    assert np.allclose(np.exp(result.log_alpha).sum(axis=1), np.ones(5, dtype=np.float64))
