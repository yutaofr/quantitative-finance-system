from __future__ import annotations

import numpy as np

from state.ti_hmm_single import e_step_smooth, log_forward_filter


def _example_logs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    log_initial = np.log(np.array([0.5, 0.3, 0.2], dtype=np.float64))
    transition = np.array(
        [
            [
                [0.8, 0.1, 0.1],
                [0.2, 0.7, 0.1],
                [0.2, 0.2, 0.6],
            ],
            [
                [0.7, 0.2, 0.1],
                [0.1, 0.8, 0.1],
                [0.2, 0.3, 0.5],
            ],
            [
                [0.6, 0.3, 0.1],
                [0.2, 0.6, 0.2],
                [0.1, 0.2, 0.7],
            ],
        ],
        dtype=np.float64,
    )
    emission = np.array(
        [
            [0.6, 0.3, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.3, 0.6],
            [0.1, 0.2, 0.7],
        ],
        dtype=np.float64,
    )
    return log_initial, np.log(transition), np.log(emission)


def test_e_step_smooth_returns_normalized_gamma_and_xi() -> None:
    log_initial, log_transition, log_emission = _example_logs()

    result = e_step_smooth(
        log_initial,
        log_transition,
        log_emission,
        usable_mask=np.ones(4, dtype=np.bool_),
    )

    assert result.gamma.shape == (4, 3)
    assert result.xi.shape == (3, 3, 3)
    assert np.allclose(result.gamma.sum(axis=1), np.ones(4, dtype=np.float64))
    assert np.allclose(result.xi.sum(axis=(1, 2)), np.ones(3, dtype=np.float64))
    assert result.parameter_mask.tolist() == [True, True, True, True]


def test_e_step_smooth_uses_forward_posterior_for_right_censored_tail() -> None:
    log_initial, log_transition, log_emission = _example_logs()
    usable_mask = np.array([True, True, False, False], dtype=np.bool_)

    filtering = log_forward_filter(log_initial, log_transition, log_emission)
    result = e_step_smooth(log_initial, log_transition, log_emission, usable_mask=usable_mask)

    assert np.allclose(result.gamma[2:], np.exp(filtering.log_alpha[2:]))
    assert result.parameter_mask.tolist() == [True, True, False, False]
    assert result.xi.shape == (1, 3, 3)
    assert np.allclose(result.xi.sum(axis=(1, 2)), np.ones(1, dtype=np.float64))


def test_e_step_smooth_keeps_all_outputs_finite_for_extreme_log_emissions() -> None:
    log_initial = np.log(np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float64))
    log_transition = np.log(np.full((3, 3, 3), 1.0 / 3.0, dtype=np.float64))
    log_emission = np.full((4, 3), -10_000.0, dtype=np.float64)

    result = e_step_smooth(
        log_initial,
        log_transition,
        log_emission,
        usable_mask=np.ones(4, dtype=np.bool_),
    )

    assert np.isfinite(result.gamma).all()
    assert np.isfinite(result.xi).all()
    assert np.allclose(result.gamma.sum(axis=1), np.ones(4, dtype=np.float64))
