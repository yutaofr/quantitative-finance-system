from __future__ import annotations

import numpy as np
import pytest

from errors import HMMConvergenceError
from state.ti_hmm_single import (
    _TransitionObjectiveData,
    _transition_objective_grad,
    COV_EPSILON,
    gaussian_log_likelihood,
    logsumexp3,
    logsumexp_axis3,
    logsumexp_stable,
    shrink_emission_covariance,
    transition_matrix_t,
)


def test_logsumexp_stable_handles_large_inputs_without_overflow() -> None:
    values = np.array([1000.0, 1001.0, 999.0], dtype=np.float64)

    result = logsumexp_stable(values)

    assert np.isfinite(result)
    assert np.isclose(result, 1001.0 + np.log1p(np.exp(-1.0) + np.exp(-2.0)))


def test_logsumexp3_matches_stable_reference() -> None:
    values = np.array([-7.0, -2.5, -12.0], dtype=np.float64)
    expected = float(np.log(np.sum(np.exp(values))))

    result = float(logsumexp3(values))

    assert np.isclose(result, expected, rtol=0.0, atol=1e-12)


def test_logsumexp_axis3_matches_row_and_column_reductions() -> None:
    matrix = np.array(
        [
            [-5.0, -2.0, -1.0],
            [-4.0, -3.5, -2.5],
            [-1.5, -7.0, -3.0],
        ],
        dtype=np.float64,
    )
    row_expected = np.array([np.log(np.sum(np.exp(row))) for row in matrix], dtype=np.float64)
    col_expected = np.array(
        [np.log(np.sum(np.exp(matrix[:, idx]))) for idx in range(matrix.shape[1])],
        dtype=np.float64,
    )

    row_result = logsumexp_axis3(matrix, axis=1)
    col_result = logsumexp_axis3(matrix, axis=0)

    assert np.allclose(row_result, row_expected, rtol=0.0, atol=1e-12)
    assert np.allclose(col_result, col_expected, rtol=0.0, atol=1e-12)


def test_transition_matrix_t_clips_extreme_logits_and_normalizes_rows() -> None:
    coefs = np.array(
        [
            [1000.0, 1000.0, 1000.0],
            [-1000.0, -1000.0, -1000.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    transition = transition_matrix_t(coefs, dwell=np.array([100.0, 100.0, 100.0]), h_t=100.0)

    assert transition.shape == (3, 3)
    assert np.isfinite(transition).all()
    assert np.allclose(transition.sum(axis=1), np.ones(3, dtype=np.float64))
    assert 0.0 < transition[0, 0] < 1.0
    assert 0.0 < transition[1, 1] < 1.0


def test_transition_objective_grad_zeroes_clipped_probability_contributions() -> None:
    beta = np.array([0.0, 1.0, 1.0], dtype=np.float64)
    data = _TransitionObjectiveData(
        stay_weight=np.array([4.0, 2.0, 3.0], dtype=np.float64),
        leave_weight=np.array([1.0, 5.0, 2.0], dtype=np.float64),
        dwell=np.array([1000.0, -1000.0, 0.0], dtype=np.float64),
        h=np.array([1000.0, -1000.0, 0.0], dtype=np.float64),
        l2_penalty=0.0,
    )

    gradient = _transition_objective_grad(beta, data)
    unclipped_data = _TransitionObjectiveData(
        stay_weight=np.array([0.0, 0.0, 3.0], dtype=np.float64),
        leave_weight=np.array([0.0, 0.0, 2.0], dtype=np.float64),
        dwell=data.dwell,
        h=data.h,
        l2_penalty=0.0,
    )
    expected = _transition_objective_grad(beta, unclipped_data)

    assert np.allclose(gradient, expected, rtol=0.0, atol=1.0e-12)


def test_shrink_emission_covariance_adds_diagonal_regularization() -> None:
    observations = np.ones((4, 2), dtype=np.float64)
    weights = np.ones(4, dtype=np.float64)

    cov = shrink_emission_covariance(observations, weights)

    assert np.all(np.diag(cov) > 0.0)
    np.linalg.cholesky(cov)


def test_shrink_emission_covariance_ignores_underflow_from_tiny_weights() -> None:
    observations = np.array(
        [
            [1.0e-200, 2.0e-200],
            [2.0e-200, 3.0e-200],
            [3.0e-200, 4.0e-200],
        ],
        dtype=np.float64,
    )
    weights = np.array([1.0e-200, 2.0e-200, 3.0e-200], dtype=np.float64)

    cov = shrink_emission_covariance(observations, weights)

    assert cov.shape == (2, 2)
    assert np.isfinite(cov).all()


def test_shrink_emission_covariance_remains_finite_and_psd_for_tiny_weights() -> None:
    observations = np.array(
        [
            [1.0e-200, -2.0e-200],
            [2.0e-200, 3.0e-200],
            [-3.0e-200, 4.0e-200],
        ],
        dtype=np.float64,
    )
    weights = np.array([1.0e-220, 2.0e-220, 3.0e-220], dtype=np.float64)

    old_settings = np.seterr(all="raise")
    try:
        cov = shrink_emission_covariance(observations, weights)
    finally:
        np.seterr(**old_settings)

    assert np.isfinite(cov).all()
    assert np.allclose(cov, cov.T, rtol=0.0, atol=1.0e-18)
    assert np.all(np.diag(cov) >= COV_EPSILON)


def test_gaussian_log_likelihood_uses_cholesky_and_rejects_singular_covariance() -> None:
    observations = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    mean = np.array([0.0, 0.0], dtype=np.float64)
    singular_cov = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float64)

    with pytest.raises(HMMConvergenceError, match="Cholesky"):
        gaussian_log_likelihood(observations, mean, singular_cov)


def test_gaussian_log_likelihood_returns_finite_values_for_regularized_covariance() -> None:
    observations = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    mean = np.array([0.0, 0.0], dtype=np.float64)
    cov = np.array([[1.0, 0.2], [0.2, 1.5]], dtype=np.float64)

    log_like = gaussian_log_likelihood(observations, mean, cov)

    assert log_like.shape == (2,)
    assert np.isfinite(log_like).all()


def test_gaussian_log_likelihood_ignores_benign_underflow_in_mahalanobis_sum() -> None:
    observations = np.array(
        [
            [1.0e-220, -2.0e-220],
            [-3.0e-220, 4.0e-220],
        ],
        dtype=np.float64,
    )
    mean = np.zeros(2, dtype=np.float64)
    cov = np.array(
        [
            [1.0e40, 0.0],
            [0.0, 1.0e40],
        ],
        dtype=np.float64,
    )

    old_settings = np.seterr(all="raise")
    try:
        log_like = gaussian_log_likelihood(observations, mean, cov)
    finally:
        np.seterr(**old_settings)

    assert log_like.shape == (2,)
    assert np.isfinite(log_like).all()
