from __future__ import annotations

import numpy as np
import pytest

from errors import HMMConvergenceError
from state.ti_hmm_single import degraded_hmm_posterior, fit_hmm


def _obs() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.1, 0.0, 0.0, 1.0, 0.5],
            [0.2, 0.3, 0.2, 0.2, 1.1, 0.4],
            [0.4, 0.5, 0.2, 0.2, 1.2, 0.3],
        ],
        dtype=np.float64,
    )


def test_degraded_hmm_posterior_matches_srd_section_7_3() -> None:
    posterior = degraded_hmm_posterior()

    assert np.allclose(posterior.post, np.array([0.25, 0.50, 0.25], dtype=np.float64))
    assert posterior.state_name == "NEUTRAL"
    assert posterior.model_status == "DEGRADED"


def test_fit_hmm_requires_injected_generator() -> None:
    with pytest.raises(TypeError, match="Generator"):
        fit_hmm(_obs(), np.array([0.1, 0.2, 0.3], dtype=np.float64), rng=None)  # type: ignore[arg-type]


def test_fit_hmm_raises_domain_error_when_max_iter_is_exhausted() -> None:
    with pytest.raises(HMMConvergenceError, match="did not converge"):
        fit_hmm(
            _obs(),
            np.array([0.1, 0.2, 0.3], dtype=np.float64),
            rng=np.random.default_rng(1),
            max_iter=0,
        )


def test_fit_hmm_rejects_non_finite_observations() -> None:
    y_obs = _obs()
    y_obs[0, 0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        fit_hmm(y_obs, np.array([0.1, 0.2, 0.3], dtype=np.float64), rng=np.random.default_rng(1))


def test_fit_hmm_returns_best_converged_model_with_label_map() -> None:
    y_obs = np.array(
        [
            [-3.0, -3.0, 0.0, 0.0, 0.1, 0.1],
            [-2.8, -3.1, 0.0, 0.0, 0.2, 0.1],
            [0.0, 0.1, 0.0, 0.0, 1.0, 0.5],
            [0.2, -0.1, 0.0, 0.0, 1.1, 0.4],
            [3.0, 3.0, 0.0, 0.0, 2.0, 0.9],
            [3.2, 2.9, 0.0, 0.0, 2.1, 0.8],
        ],
        dtype=np.float64,
    )
    forward_returns = np.array([-0.2, -0.1, 0.02, 0.03, 0.12, 0.15], dtype=np.float64)

    model = fit_hmm(
        y_obs,
        h=np.zeros(6, dtype=np.float64),
        rng=np.random.default_rng(4),
        forward_52w_returns=forward_returns,
        restarts=2,
        max_iter=20,
        tolerance=1.0e-4,
    )

    assert model.transition_coefs.shape == (3, 3)
    assert model.emission_mean.shape == (3, 6)
    assert model.emission_cov.shape == (3, 6, 6)
    assert set(model.label_map.values()) == {"DEFENSIVE", "NEUTRAL", "OFFENSIVE"}


def test_fit_hmm_requires_forward_returns_for_label_identification() -> None:
    with pytest.raises(HMMConvergenceError, match="forward_52w_returns"):
        fit_hmm(
            _obs(),
            np.array([0.1, 0.2, 0.3], dtype=np.float64),
            rng=np.random.default_rng(1),
            max_iter=1,
        )
