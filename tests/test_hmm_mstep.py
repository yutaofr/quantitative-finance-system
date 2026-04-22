from __future__ import annotations

import numpy as np
import pytest

from errors import HMMConvergenceError
from state.ti_hmm_single import (
    _transition_objective,
    _transition_objective_grad,
    _TransitionObjectiveData,
    fit_transition_coefs,
    transition_matrix_t,
    update_emission_parameters,
)


def test_update_emission_parameters_uses_only_parameter_mask_rows() -> None:
    y_obs = np.array(
        [
            [0.0, 0.0],
            [2.0, 2.0],
            [100.0, 100.0],
        ],
        dtype=np.float64,
    )
    gamma = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    parameter_mask = np.array([True, True, False], dtype=np.bool_)

    means, covs = update_emission_parameters(y_obs, gamma, parameter_mask)

    assert means.shape == (3, 2)
    assert covs.shape == (3, 2, 2)
    assert np.all(means[0] < 2.0)
    assert np.all(means[1] > 0.0)
    assert not np.any(means == 100.0)
    for cov in covs:
        np.linalg.cholesky(cov)


def test_update_emission_parameters_raises_when_state_has_no_usable_weight() -> None:
    y_obs = np.eye(3, dtype=np.float64)
    gamma = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    with pytest.raises(HMMConvergenceError, match="positive mass"):
        update_emission_parameters(y_obs, gamma, np.ones(3, dtype=np.bool_))


def test_update_emission_parameters_ignores_underflow_from_tiny_state_weights() -> None:
    y_obs = np.array(
        [
            [1.0e-200, 2.0e-200],
            [2.0e-200, 3.0e-200],
            [3.0e-200, 4.0e-200],
        ],
        dtype=np.float64,
    )
    gamma = np.array(
        [
            [1.0e-200, 0.5, 0.5],
            [1.0e-200, 0.5, 0.5],
            [1.0e-200, 0.5, 0.5],
        ],
        dtype=np.float64,
    )

    means, covs = update_emission_parameters(y_obs, gamma, np.ones(3, dtype=np.bool_))

    assert means.shape == (3, 2)
    assert covs.shape == (3, 2, 2)
    assert np.isfinite(means[1:]).all()


def test_update_emission_parameters_hard_case_tiny_gamma_returns_finite_covariance() -> None:
    y_obs = np.array(
        [
            [1.0e-200, 2.0e-200],
            [2.0e-200, -3.0e-200],
            [-4.0e-200, 5.0e-200],
            [3.0e-200, -1.0e-200],
        ],
        dtype=np.float64,
    )
    gamma = np.array(
        [
            [1.0e-220, 2.0e-220, 3.0e-220],
            [2.0e-220, 3.0e-220, 4.0e-220],
            [3.0e-220, 4.0e-220, 5.0e-220],
            [4.0e-220, 5.0e-220, 6.0e-220],
        ],
        dtype=np.float64,
    )
    parameter_mask = np.ones(4, dtype=np.bool_)

    old_settings = np.seterr(all="raise")
    try:
        means, covs = update_emission_parameters(y_obs, gamma, parameter_mask)
    finally:
        np.seterr(**old_settings)

    assert np.isfinite(means).all()
    assert np.isfinite(covs).all()
    for cov in covs:
        assert np.allclose(cov, cov.T, rtol=0.0, atol=1.0e-18)
        np.linalg.cholesky(cov)


def test_fit_transition_coefs_prefers_staying_when_xi_stays() -> None:
    xi = np.zeros((8, 3, 3), dtype=np.float64)
    for time_idx in range(xi.shape[0]):
        for state_idx in range(3):
            xi[time_idx, state_idx, state_idx] = 0.95 / 3.0
            for target_idx in range(3):
                if target_idx != state_idx:
                    xi[time_idx, state_idx, target_idx] = 0.025 / 3.0
    dwell = np.ones((8, 3), dtype=np.float64)
    h = np.zeros(8, dtype=np.float64)

    coefs = fit_transition_coefs(xi, dwell, h, max_iter=200)
    transition = transition_matrix_t(coefs, dwell=np.ones(3, dtype=np.float64), h_t=0.0)

    assert coefs.shape == (3, 3)
    assert np.all(np.diag(transition) > 0.8)


def test_fit_transition_coefs_raises_on_optimizer_failure() -> None:
    xi = np.full((2, 3, 3), 1.0 / 9.0, dtype=np.float64)
    dwell = np.ones((2, 3), dtype=np.float64)
    h = np.zeros(2, dtype=np.float64)

    with pytest.raises(HMMConvergenceError, match="transition"):
        fit_transition_coefs(xi, dwell, h, max_iter=0)


def test_transition_objective_grad_matches_finite_difference() -> None:
    beta = np.array([0.2, -0.1, 0.3], dtype=np.float64)
    data = _TransitionObjectiveData(
        stay_weight=np.array([0.6, 0.5, 0.4, 0.7], dtype=np.float64),
        leave_weight=np.array([0.4, 0.5, 0.6, 0.3], dtype=np.float64),
        dwell=np.array([1.0, 2.0, 1.5, 3.0], dtype=np.float64),
        h=np.array([-0.2, 0.1, 0.3, -0.1], dtype=np.float64),
        l2_penalty=1.0e-3,
    )
    eps = 1.0e-6
    analytic = _transition_objective_grad(beta, data)
    numeric = np.empty_like(beta)
    for idx in range(beta.shape[0]):
        step = np.zeros_like(beta)
        step[idx] = eps
        plus = _transition_objective(beta + step, data)
        minus = _transition_objective(beta - step, data)
        numeric[idx] = (plus - minus) / (2.0 * eps)

    assert np.allclose(analytic, numeric, rtol=1.0e-5, atol=1.0e-6)


def test_transition_objective_grad_remains_finite_for_extreme_logits() -> None:
    beta = np.array([25.0, -4.0, 3.5], dtype=np.float64)
    data = _TransitionObjectiveData(
        stay_weight=np.array([1.0, 1.0e-250, 2.0], dtype=np.float64),
        leave_weight=np.array([1.0e-250, 3.0, 1.0], dtype=np.float64),
        dwell=np.array([1.0e6, -1.0e6, 2.5e5], dtype=np.float64),
        h=np.array([1.0e-200, -1.0e-200, 5.0e-201], dtype=np.float64),
        l2_penalty=1.0e-3,
    )
    old_settings = np.seterr(all="raise")
    try:
        gradient = _transition_objective_grad(beta, data)
    finally:
        np.seterr(**old_settings)

    assert np.isfinite(gradient).all()


def test_fit_transition_coefs_returns_finite_coefficients_and_valid_transition_rows() -> None:
    xi = np.full((6, 3, 3), 1.0 / 9.0, dtype=np.float64)
    dwell = np.full((6, 3), 2.0, dtype=np.float64)
    h = np.linspace(-0.5, 0.5, 6, dtype=np.float64)

    coefs = fit_transition_coefs(xi, dwell, h, max_iter=100)
    transition = transition_matrix_t(coefs, dwell=np.ones(3, dtype=np.float64), h_t=0.1)

    assert np.isfinite(coefs).all()
    assert np.isfinite(transition).all()
    assert np.allclose(transition.sum(axis=1), np.ones(3, dtype=np.float64))


def test_fit_transition_coefs_hard_case_completes_without_floating_pointer_error() -> None:
    xi = np.array(
        [
            [[0.999999, 0.0000005, 0.0000005], [0.0000005, 0.999999, 0.0000005], [0.0000005, 0.0000005, 0.999999]],
            [[0.999999, 0.0000005, 0.0000005], [0.0000005, 0.999999, 0.0000005], [0.0000005, 0.0000005, 0.999999]],
            [[0.999999, 0.0000005, 0.0000005], [0.0000005, 0.999999, 0.0000005], [0.0000005, 0.0000005, 0.999999]],
        ],
        dtype=np.float64,
    )
    dwell = np.array(
        [
            [1.0e6, 1.0e6, 1.0e6],
            [2.0e6, 2.0e6, 2.0e6],
            [3.0e6, 3.0e6, 3.0e6],
        ],
        dtype=np.float64,
    )
    h = np.array([1.0e-200, -1.0e-200, 5.0e-201], dtype=np.float64)
    old_settings = np.seterr(all="raise")
    try:
        coefs = fit_transition_coefs(xi, dwell, h, max_iter=100)
    finally:
        np.seterr(**old_settings)

    assert np.isfinite(coefs).all()
