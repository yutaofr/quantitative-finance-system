from __future__ import annotations

import numpy as np
import pytest

from errors import HMMConvergenceError
from state.ti_hmm_single import (
    gaussian_log_likelihood,
    logsumexp_stable,
    shrink_emission_covariance,
    transition_matrix_t,
)


def test_logsumexp_stable_handles_large_inputs_without_overflow() -> None:
    values = np.array([1000.0, 1001.0, 999.0], dtype=np.float64)

    result = logsumexp_stable(values)

    assert np.isfinite(result)
    assert np.isclose(result, 1001.0 + np.log1p(np.exp(-1.0) + np.exp(-2.0)))


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


def test_shrink_emission_covariance_adds_diagonal_regularization() -> None:
    observations = np.ones((4, 2), dtype=np.float64)
    weights = np.ones(4, dtype=np.float64)

    cov = shrink_emission_covariance(observations, weights)

    assert np.all(np.diag(cov) > 0.0)
    np.linalg.cholesky(cov)


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
