"""Single production TI-HMM interface and guardrails from SRD v8.7 section 7."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray

from engine_types import Stance
from errors import HMMConvergenceError

STATE_COUNT = 3
MATRIX_NDIM = 2
OBS_DIM = 6
DEFAULT_MAX_ITER = 200
DEFAULT_TOLERANCE = 1.0e-6
LOGIT_CLIP_BOUND = 50.0
COV_EPSILON = 1.0e-8
TRANSITION_PROB_EPSILON = 1.0e-12
GAUSSIAN_LOG_NORM = np.log(2.0 * np.pi)


@dataclass(frozen=True, slots=True)
class HMMModel:
    """pure. Fitted TI-HMM parameters with frozen state semantics."""

    transition_coefs: NDArray[np.float64]
    emission_mean: NDArray[np.float64]
    emission_cov: NDArray[np.float64]
    label_map: Mapping[int, Stance]


@dataclass(frozen=True, slots=True)
class HMMPosterior:
    """pure. Current HMM posterior and model status."""

    post: NDArray[np.float64]
    state_name: Stance
    model_status: str


def degraded_hmm_posterior() -> HMMPosterior:
    """pure. Return SRD §7.3 fallback posterior."""
    return HMMPosterior(
        post=np.array([0.25, 0.50, 0.25], dtype=np.float64),
        state_name="NEUTRAL",
        model_status="DEGRADED",
    )


def logsumexp_stable(
    values: NDArray[np.float64],
    *,
    axis: int | None = None,
) -> np.float64 | NDArray[np.float64]:
    """pure. Compute log(sum(exp(values))) using max-shift stabilization."""
    arr = np.asarray(values, dtype=np.float64)
    if not np.isfinite(arr).all():
        msg = "logsumexp inputs must be finite"
        raise ValueError(msg)
    max_value = np.max(arr, axis=axis, keepdims=True)
    shifted_sum = np.sum(np.exp(arr - max_value), axis=axis, keepdims=True)
    result = max_value + np.log(shifted_sum)
    if axis is None:
        return np.float64(np.squeeze(result))
    return cast(NDArray[np.float64], np.squeeze(result, axis=axis))


def _sigmoid_clipped(logit: NDArray[np.float64] | float) -> NDArray[np.float64]:
    clipped = np.clip(logit, -LOGIT_CLIP_BOUND, LOGIT_CLIP_BOUND)
    prob = 1.0 / (1.0 + np.exp(-clipped))
    return cast(
        NDArray[np.float64],
        np.clip(prob, TRANSITION_PROB_EPSILON, 1.0 - TRANSITION_PROB_EPSILON),
    )


def transition_matrix_t(
    transition_coefs: NDArray[np.float64],
    dwell: NDArray[np.float64],
    h_t: float,
) -> NDArray[np.float64]:
    """pure. Build one SRD §7.1 time-inhomogeneous transition matrix."""
    coefs = np.asarray(transition_coefs, dtype=np.float64)
    dwell_values = np.asarray(dwell, dtype=np.float64)
    if coefs.shape != (STATE_COUNT, STATE_COUNT):
        msg = "transition_coefs must have shape (3, 3)"
        raise ValueError(msg)
    if dwell_values.shape != (STATE_COUNT,):
        msg = "dwell must have shape (3,)"
        raise ValueError(msg)
    if not np.isfinite(coefs).all() or not np.isfinite(dwell_values).all() or not np.isfinite(h_t):
        msg = "transition inputs must be finite"
        raise ValueError(msg)

    logits = coefs[:, 0] + coefs[:, 1] * dwell_values + coefs[:, 2] * h_t
    leave_prob = _sigmoid_clipped(logits)
    transition = np.empty((STATE_COUNT, STATE_COUNT), dtype=np.float64)
    off_diag_prob = leave_prob / float(STATE_COUNT - 1)
    for row in range(STATE_COUNT):
        transition[row, :] = off_diag_prob[row]
        transition[row, row] = 1.0 - leave_prob[row]
    return transition


def shrink_emission_covariance(
    observations: NDArray[np.float64],
    weights: NDArray[np.float64],
    *,
    epsilon: float = COV_EPSILON,
) -> NDArray[np.float64]:
    """pure. Estimate full Gaussian emission covariance with diagonal shrinkage."""
    obs = np.asarray(observations, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if obs.ndim != MATRIX_NDIM:
        msg = "observations must be a 2D matrix"
        raise ValueError(msg)
    if w.ndim != 1 or w.shape[0] != obs.shape[0]:
        msg = "weights must be a 1D array aligned to observations"
        raise ValueError(msg)
    if not np.isfinite(obs).all() or not np.isfinite(w).all() or np.any(w < 0.0):
        msg = "covariance inputs must be finite with nonnegative weights"
        raise ValueError(msg)
    weight_sum = float(np.sum(w))
    if weight_sum <= 0.0:
        msg = "weights must have positive total mass"
        raise HMMConvergenceError(msg)

    mean = np.sum(obs * w[:, None], axis=0) / weight_sum
    centered = obs - mean
    sample_cov = (centered * w[:, None]).T @ centered / weight_sum
    target = np.diag(np.diag(sample_cov))
    diff = sample_cov - target
    shrink_den = float(np.sum(diff * diff))
    if shrink_den == 0.0:
        shrunk = target
    else:
        outer = np.einsum("ni,nj->nij", centered, centered)
        noise = outer - sample_cov
        shrink_num = float(np.sum((noise * w[:, None, None]) ** 2)) / (weight_sum * weight_sum)
        shrinkage = float(np.clip(shrink_num / shrink_den, 0.0, 1.0))
        shrunk = (1.0 - shrinkage) * sample_cov + shrinkage * target
    return shrunk + epsilon * np.eye(obs.shape[1], dtype=np.float64)


def gaussian_log_likelihood(
    observations: NDArray[np.float64],
    mean: NDArray[np.float64],
    covariance: NDArray[np.float64],
) -> NDArray[np.float64]:
    """pure. Compute multivariate Gaussian log-likelihood via Cholesky solves."""
    obs = np.asarray(observations, dtype=np.float64)
    mu = np.asarray(mean, dtype=np.float64)
    cov = np.asarray(covariance, dtype=np.float64)
    if obs.ndim != MATRIX_NDIM or mu.ndim != 1 or cov.ndim != MATRIX_NDIM:
        msg = "Gaussian inputs must be observation matrix, mean vector, covariance matrix"
        raise ValueError(msg)
    if obs.shape[1] != mu.shape[0] or cov.shape != (mu.shape[0], mu.shape[0]):
        msg = "Gaussian input dimensions are inconsistent"
        raise ValueError(msg)
    if not np.isfinite(obs).all() or not np.isfinite(mu).all() or not np.isfinite(cov).all():
        msg = "Gaussian inputs must be finite"
        raise ValueError(msg)
    try:
        chol = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError as exc:
        msg = "Cholesky decomposition failed for emission covariance"
        raise HMMConvergenceError(msg) from exc

    centered = (obs - mu).T
    solved = np.linalg.solve(chol, centered)
    mahalanobis = np.sum(solved * solved, axis=0)
    log_det = 2.0 * float(np.sum(np.log(np.diag(chol))))
    dim = float(mu.shape[0])
    return cast(NDArray[np.float64], -0.5 * (dim * GAUSSIAN_LOG_NORM + log_det + mahalanobis))


def _require_generator(rng: object) -> np.random.Generator:
    if not isinstance(rng, np.random.Generator):
        msg = "fit_hmm requires an injected np.random.Generator"
        raise TypeError(msg)
    return rng


def _validate_fit_inputs(y_obs: NDArray[np.float64], h: NDArray[np.float64]) -> None:
    if y_obs.ndim != MATRIX_NDIM or y_obs.shape[1] != OBS_DIM:
        msg = "y_obs must have shape (n_weeks, 6)"
        raise ValueError(msg)
    if h.ndim != 1 or h.shape[0] != y_obs.shape[0]:
        msg = "h must be a 1D array aligned to y_obs rows"
        raise ValueError(msg)
    if not np.isfinite(y_obs).all() or not np.isfinite(h).all():
        msg = "HMM inputs must be finite"
        raise ValueError(msg)


def fit_hmm(
    y_obs: NDArray[np.float64],
    h: NDArray[np.float64],
    rng: np.random.Generator,
    *,
    max_iter: int = DEFAULT_MAX_ITER,
    tolerance: float = DEFAULT_TOLERANCE,
) -> HMMModel:
    """pure. Fit TI-HMM deterministically for given inputs and injected rng."""
    _require_generator(rng)
    _validate_fit_inputs(y_obs, h)
    if max_iter <= 0:
        msg = "HMM EM did not converge within max_iter"
        raise HMMConvergenceError(msg)
    if tolerance <= 0.0 or not np.isfinite(tolerance):
        msg = "HMM tolerance must be finite and positive"
        raise ValueError(msg)

    raise NotImplementedError("SRD §7 right-censor-aware TI-HMM EM is not implemented yet.")
