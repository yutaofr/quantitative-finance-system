from __future__ import annotations

import numpy as np

from state.ti_hmm_single import (
    HMMModel,
    infer_hmm,
    infer_hmm_posterior_path,
    log_forward_filter,
    logsumexp_stable,
)


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


def test_infer_hmm_posterior_path_is_finite_and_row_normalized() -> None:
    model = HMMModel(
        transition_coefs=np.zeros((3, 3), dtype=np.float64),
        emission_mean=np.zeros((3, 6), dtype=np.float64),
        emission_cov=np.stack([np.eye(6, dtype=np.float64) for _ in range(3)]),
        label_map={0: "DEFENSIVE", 1: "NEUTRAL", 2: "OFFENSIVE"},
        log_likelihood=0.0,
    )
    y_obs = np.array(
        [
            [0.1, -0.2, 0.0, 0.2, 0.0, -0.1],
            [0.0, 0.1, -0.1, 0.0, 0.1, 0.0],
            [-0.2, 0.0, 0.1, -0.1, 0.0, 0.2],
        ],
        dtype=np.float64,
    )
    h = np.array([0.0, 0.1, -0.1], dtype=np.float64)

    posterior_path = infer_hmm_posterior_path(model, y_obs, h)

    assert posterior_path.shape == (3, 3)
    assert np.isfinite(posterior_path).all()
    assert np.allclose(posterior_path.sum(axis=1), np.ones(3, dtype=np.float64), atol=1e-8)


def test_infer_hmm_posterior_path_ignores_benign_underflow_in_gaussian_log_likelihood() -> None:
    covariance = np.stack(
        [
            np.diag(np.array([1.0e40] * 6, dtype=np.float64)),
            np.diag(np.array([1.0e40] * 6, dtype=np.float64)),
            np.diag(np.array([1.0e40] * 6, dtype=np.float64)),
        ],
    )
    model = HMMModel(
        transition_coefs=np.zeros((3, 3), dtype=np.float64),
        emission_mean=np.zeros((3, 6), dtype=np.float64),
        emission_cov=covariance,
        label_map={0: "DEFENSIVE", 1: "NEUTRAL", 2: "OFFENSIVE"},
        log_likelihood=0.0,
    )
    y_obs = np.array(
        [
            [1.0e-220, -2.0e-220, 3.0e-220, -4.0e-220, 5.0e-220, -6.0e-220],
            [-2.0e-220, 3.0e-220, -4.0e-220, 5.0e-220, -6.0e-220, 7.0e-220],
        ],
        dtype=np.float64,
    )
    h = np.array([0.0, 1.0e-220], dtype=np.float64)

    old_settings = np.seterr(all="raise")
    try:
        posterior_path = infer_hmm_posterior_path(model, y_obs, h)
    finally:
        np.seterr(**old_settings)

    assert posterior_path.shape == (2, 3)
    assert np.isfinite(posterior_path).all()
    assert np.allclose(posterior_path.sum(axis=1), np.ones(2, dtype=np.float64), atol=1e-8)


def test_infer_hmm_posterior_path_ignores_benign_underflow_in_exp_recovery() -> None:
    covariance = np.stack(
        [
            np.diag(np.array([1.0e40] * 6, dtype=np.float64)),
            np.diag(np.array([1.0e40] * 6, dtype=np.float64)),
            np.diag(np.array([1.0e40] * 6, dtype=np.float64)),
        ],
    )
    model = HMMModel(
        transition_coefs=np.zeros((3, 3), dtype=np.float64),
        emission_mean=np.zeros((3, 6), dtype=np.float64),
        emission_cov=covariance,
        label_map={0: "DEFENSIVE", 1: "NEUTRAL", 2: "OFFENSIVE"},
        log_likelihood=0.0,
    )
    y_obs = np.array(
        [
            [1.0e-220, -2.0e-220, 3.0e-220, -4.0e-220, 5.0e-220, -6.0e-220],
            [-2.0e-220, 3.0e-220, -4.0e-220, 5.0e-220, -6.0e-220, 7.0e-220],
        ],
        dtype=np.float64,
    )
    h = np.array([0.0, 1.0e-220], dtype=np.float64)

    old_settings = np.seterr(all="raise")
    try:
        posterior_path = infer_hmm_posterior_path(model, y_obs, h)
        result = infer_hmm(model, y_obs, h)
    finally:
        np.seterr(**old_settings)

    assert posterior_path.shape == (2, 3)
    assert np.isfinite(posterior_path).all()
    assert np.allclose(posterior_path.sum(axis=1), np.ones(2, dtype=np.float64), atol=1e-8)
    assert np.isfinite(result.posterior.post).all()
