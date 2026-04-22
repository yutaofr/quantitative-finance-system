"""Single production TI-HMM interface and guardrails from SRD v8.7 section 7."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize  # type: ignore[import-untyped]

from engine_types import Stance
from errors import HMMConvergenceError
from state.state_label_map import build_label_map

STATE_COUNT = 3
MATRIX_NDIM = 2
TENSOR_NDIM = 3
OBS_DIM = 6
DEFAULT_MAX_ITER = 200
DEFAULT_TOLERANCE = 1.0e-6
DEFAULT_RESTARTS = 50
DEGENERATE_OCCUPANCY_WINDOW = 26
DEGENERATE_OCCUPANCY_THRESHOLD = 0.01
POSTERIOR_SUM_TOLERANCE = 1.0e-8
LOGIT_CLIP_BOUND = 50.0
COV_EPSILON = 1.0e-8
TRANSITION_PROB_EPSILON = 1.0e-12
GAUSSIAN_LOG_NORM = np.log(2.0 * np.pi)
TRANSITION_L2_PENALTY = 1.0e-3
DEFAULT_OPTIMIZER_MAX_ITER = 200


@dataclass(frozen=True, slots=True)
class HMMModel:
    """pure. Fitted TI-HMM parameters with frozen state semantics."""

    transition_coefs: NDArray[np.float64]
    emission_mean: NDArray[np.float64]
    emission_cov: NDArray[np.float64]
    label_map: Mapping[int, Stance]
    log_likelihood: float


@dataclass(frozen=True, slots=True)
class HMMPosterior:
    """pure. Current HMM posterior and model status."""

    post: NDArray[np.float64]
    state_name: Stance
    model_status: str


@dataclass(frozen=True, slots=True)
class HMMInferenceResult:
    """pure. Filtered HMM posterior plus dwell/hazard diagnostics."""

    posterior: HMMPosterior
    dwell_weeks: int
    hazard_covariate: float


@dataclass(frozen=True, slots=True)
class HMMFilterResult:
    """pure. Log-space filtered posterior path and total log-likelihood."""

    log_alpha: NDArray[np.float64]
    log_likelihood: float


@dataclass(frozen=True, slots=True)
class HMMSmoothingResult:
    """pure. E-step posterior weights for usable history and right-censored tail."""

    gamma: NDArray[np.float64]
    xi: NDArray[np.float64]
    parameter_mask: NDArray[np.bool_]
    log_likelihood: float


@dataclass(frozen=True, slots=True)
class _TransitionObjectiveData:
    stay_weight: NDArray[np.float64]
    leave_weight: NDArray[np.float64]
    dwell: NDArray[np.float64]
    h: NDArray[np.float64]
    l2_penalty: float


@dataclass(frozen=True, slots=True)
class _FitData:
    y_obs: NDArray[np.float64]
    h: NDArray[np.float64]
    forward_returns: NDArray[np.float64]
    parameter_mask: NDArray[np.bool_]
    log_initial: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class _FitConfig:
    max_iter: int
    tolerance: float
    transition_max_iter: int


def degraded_hmm_posterior() -> HMMPosterior:
    """pure. Return SRD §7.3 fallback posterior."""
    return HMMPosterior(
        post=np.array([0.25, 0.50, 0.25], dtype=np.float64),
        state_name="NEUTRAL",
        model_status="DEGRADED",
    )


def has_invalid_posterior(posterior: NDArray[np.float64]) -> bool:
    """pure. Return whether posterior violates SRD §7.3 finite/simplex requirements."""
    post = np.asarray(posterior, dtype=np.float64)
    if post.ndim != MATRIX_NDIM or post.shape[1] != STATE_COUNT:
        return True
    if not np.isfinite(post).all() or np.any(post < 0.0):
        return True
    row_sums = np.sum(post, axis=1)
    return bool(np.any(np.abs(row_sums - 1.0) > POSTERIOR_SUM_TOLERANCE))


def has_degenerate_state_occupancy(
    posterior: NDArray[np.float64],
    *,
    window: int = DEGENERATE_OCCUPANCY_WINDOW,
    threshold: float = DEGENERATE_OCCUPANCY_THRESHOLD,
) -> bool:
    """pure. Detect SRD §7.3 state starvation over a consecutive rolling window."""
    post = np.asarray(posterior, dtype=np.float64)
    if has_invalid_posterior(post):
        return True
    if post.shape[0] < window:
        return False
    for end_idx in range(window, post.shape[0] + 1):
        means = np.mean(post[end_idx - window : end_idx], axis=0)
        if np.any(means < threshold):
            return True
    return False


def has_label_order_flip(
    persisted_label_map: Mapping[int, Stance],
    refit_forward_returns_by_state: Mapping[int, float],
) -> bool:
    """pure. Detect SRD §7.3 semantic label-order flip after rolling refit."""
    return build_label_map(refit_forward_returns_by_state) != dict(persisted_label_map)


def should_degrade_hmm(
    posterior: NDArray[np.float64],
    *,
    persisted_label_map: Mapping[int, Stance],
    refit_forward_returns_by_state: Mapping[int, float] | None = None,
) -> bool:
    """pure. Combine SRD §7.3 HMM degradation guards."""
    if has_invalid_posterior(posterior):
        return True
    if has_degenerate_state_occupancy(posterior):
        return True
    if refit_forward_returns_by_state is not None:
        return has_label_order_flip(persisted_label_map, refit_forward_returns_by_state)
    return False


def logsumexp3(values: NDArray[np.float64]) -> np.float64:
    """pure. Stable log-sum-exp for one length-3 log-domain vector."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape != (STATE_COUNT,):
        msg = "logsumexp3 expects shape (3,)"
        raise ValueError(msg)
    if not np.isfinite(arr).all():
        msg = "logsumexp inputs must be finite"
        raise ValueError(msg)
    vmax = float(max(arr[0], arr[1], arr[2]))
    with np.errstate(under="ignore"):
        total = float(np.exp(arr[0] - vmax) + np.exp(arr[1] - vmax) + np.exp(arr[2] - vmax))
    return np.float64(vmax + np.log(total))


def logsumexp_axis3(matrix: NDArray[np.float64], axis: int) -> NDArray[np.float64]:
    """pure. Stable log-sum-exp along one axis for 3-state tensors."""
    arr = np.asarray(matrix, dtype=np.float64)
    if axis == 1:
        if arr.ndim != MATRIX_NDIM or arr.shape[1] != STATE_COUNT:
            msg = "logsumexp_axis3 axis=1 expects shape (n, 3)"
            raise ValueError(msg)
        if not np.isfinite(arr).all():
            msg = "logsumexp inputs must be finite"
            raise ValueError(msg)
        vmax = np.maximum(np.maximum(arr[:, 0], arr[:, 1]), arr[:, 2])
        with np.errstate(under="ignore"):
            total = np.exp(arr[:, 0] - vmax) + np.exp(arr[:, 1] - vmax) + np.exp(arr[:, 2] - vmax)
        return cast(NDArray[np.float64], vmax + np.log(total))
    if axis == 0:
        if arr.ndim != MATRIX_NDIM or arr.shape[0] != STATE_COUNT:
            msg = "logsumexp_axis3 axis=0 expects shape (3, n)"
            raise ValueError(msg)
        if not np.isfinite(arr).all():
            msg = "logsumexp inputs must be finite"
            raise ValueError(msg)
        vmax = np.maximum(np.maximum(arr[0, :], arr[1, :]), arr[2, :])
        with np.errstate(under="ignore"):
            total = np.exp(arr[0, :] - vmax) + np.exp(arr[1, :] - vmax) + np.exp(arr[2, :] - vmax)
        return cast(NDArray[np.float64], vmax + np.log(total))
    msg = "logsumexp_axis3 supports only axis=0 or axis=1"
    raise ValueError(msg)


def logsumexp_stable(
    values: NDArray[np.float64],
    *,
    axis: int | None = None,
) -> np.float64 | NDArray[np.float64]:
    """pure. Compute stable log(sum(exp(values))) for production-required shapes."""
    arr = np.asarray(values, dtype=np.float64)
    if not np.isfinite(arr).all():
        msg = "logsumexp inputs must be finite"
        raise ValueError(msg)
    if axis is not None:
        return logsumexp_axis3(arr, axis)
    flat = arr.reshape(-1)
    if flat.size == STATE_COUNT:
        return logsumexp3(flat)
    vmax = float(np.max(flat))
    with np.errstate(under="ignore"):
        total = float(np.sum(np.exp(flat - vmax)))
    return np.float64(vmax + np.log(total))


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

    with np.errstate(under="ignore"):
        mean = np.sum(obs * w[:, None], axis=0) / weight_sum
    centered = obs - mean
    with np.errstate(under="ignore"):
        sqrt_w = np.sqrt(w)
        weighted_centered = centered * sqrt_w[:, None]
        sample_cov = weighted_centered.T @ weighted_centered / weight_sum
    target = np.diag(np.diag(sample_cov))
    diff = sample_cov - target
    with np.errstate(under="ignore"):
        shrink_den = float(np.sum(diff * diff))
    if shrink_den == 0.0:
        shrunk = target
    else:
        outer = np.einsum("ni,nj->nij", centered, centered)
        noise = outer - sample_cov
        with np.errstate(under="ignore"):
            noise_sq = np.sum(noise * noise, axis=(1, 2))
            w_sq = w * w
            shrink_num = float(np.sum(w_sq * noise_sq)) / (weight_sum * weight_sum)
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
    with np.errstate(under="ignore"):
        mahalanobis = np.sum(solved * solved, axis=0)
    log_det = 2.0 * float(np.sum(np.log(np.diag(chol))))
    dim = float(mu.shape[0])
    return cast(NDArray[np.float64], -0.5 * (dim * GAUSSIAN_LOG_NORM + log_det + mahalanobis))


def log_forward_filter(
    log_initial: NDArray[np.float64],
    log_transition: NDArray[np.float64],
    log_emission: NDArray[np.float64],
) -> HMMFilterResult:
    """pure. Run normalized HMM forward filtering strictly in log-space."""
    initial = np.asarray(log_initial, dtype=np.float64)
    transition = np.asarray(log_transition, dtype=np.float64)
    emission = np.asarray(log_emission, dtype=np.float64)
    if initial.shape != (STATE_COUNT,):
        msg = "log_initial must have shape (3,)"
        raise ValueError(msg)
    if emission.ndim != MATRIX_NDIM or emission.shape[1] != STATE_COUNT:
        msg = "log_emission must have shape (n_weeks, 3)"
        raise ValueError(msg)
    if transition.shape != (emission.shape[0] - 1, STATE_COUNT, STATE_COUNT):
        msg = "log_transition must have shape (n_weeks - 1, 3, 3)"
        raise ValueError(msg)
    if (
        not np.isfinite(initial).all()
        or not np.isfinite(transition).all()
        or not np.isfinite(emission).all()
    ):
        msg = "forward filter inputs must be finite log-probabilities"
        raise ValueError(msg)

    log_alpha = np.empty_like(emission, dtype=np.float64)
    first_joint = initial + emission[0]
    first_norm = float(logsumexp_stable(first_joint))
    log_alpha[0] = first_joint - first_norm
    log_likelihood = first_norm

    for time_idx in range(1, emission.shape[0]):
        previous = log_alpha[time_idx - 1]
        trans_t = transition[time_idx - 1]
        predicted = np.empty(STATE_COUNT, dtype=np.float64)
        predicted[0] = logsumexp3(
            np.array(
                [
                    previous[0] + trans_t[0, 0],
                    previous[1] + trans_t[1, 0],
                    previous[2] + trans_t[2, 0],
                ],
                dtype=np.float64,
            ),
        )
        predicted[1] = logsumexp3(
            np.array(
                [
                    previous[0] + trans_t[0, 1],
                    previous[1] + trans_t[1, 1],
                    previous[2] + trans_t[2, 1],
                ],
                dtype=np.float64,
            ),
        )
        predicted[2] = logsumexp3(
            np.array(
                [
                    previous[0] + trans_t[0, 2],
                    previous[1] + trans_t[1, 2],
                    previous[2] + trans_t[2, 2],
                ],
                dtype=np.float64,
            ),
        )
        joint = predicted + emission[time_idx]
        norm = float(logsumexp_stable(joint))
        log_alpha[time_idx] = joint - norm
        log_likelihood += norm

    return HMMFilterResult(log_alpha=log_alpha, log_likelihood=log_likelihood)


def _validate_usable_mask(usable_mask: NDArray[np.bool_], n_obs: int) -> NDArray[np.bool_]:
    usable = np.asarray(usable_mask, dtype=np.bool_)
    if usable.shape != (n_obs,):
        msg = "usable_mask must align to observation weeks"
        raise ValueError(msg)
    if not usable.any():
        msg = "usable_mask must include at least one usable week"
        raise HMMConvergenceError(msg)
    usable_indices = np.flatnonzero(usable)
    usable_end = int(usable_indices[-1]) + 1
    if np.any(~usable[:usable_end]):
        msg = "usable_mask must be a contiguous historical prefix"
        raise ValueError(msg)
    return usable


def _log_backward_smooth(
    log_transition: NDArray[np.float64],
    log_emission: NDArray[np.float64],
    usable_end: int,
) -> NDArray[np.float64]:
    log_beta = np.zeros((usable_end, STATE_COUNT), dtype=np.float64)
    for time_idx in range(usable_end - 2, -1, -1):
        trans_t = log_transition[time_idx]
        rhs = log_emission[time_idx + 1] + log_beta[time_idx + 1]
        log_beta[time_idx, 0] = logsumexp3(
            np.array(
                [
                    trans_t[0, 0] + rhs[0],
                    trans_t[0, 1] + rhs[1],
                    trans_t[0, 2] + rhs[2],
                ],
                dtype=np.float64,
            ),
        )
        log_beta[time_idx, 1] = logsumexp3(
            np.array(
                [
                    trans_t[1, 0] + rhs[0],
                    trans_t[1, 1] + rhs[1],
                    trans_t[1, 2] + rhs[2],
                ],
                dtype=np.float64,
            ),
        )
        log_beta[time_idx, 2] = logsumexp3(
            np.array(
                [
                    trans_t[2, 0] + rhs[0],
                    trans_t[2, 1] + rhs[1],
                    trans_t[2, 2] + rhs[2],
                ],
                dtype=np.float64,
            ),
        )
    return log_beta


def e_step_smooth(
    log_initial: NDArray[np.float64],
    log_transition: NDArray[np.float64],
    log_emission: NDArray[np.float64],
    *,
    usable_mask: NDArray[np.bool_],
) -> HMMSmoothingResult:
    """pure. Compute log-space E-step posteriors with right-censored tail handling."""
    filtering = log_forward_filter(log_initial, log_transition, log_emission)
    usable = _validate_usable_mask(usable_mask, log_emission.shape[0])
    usable_end = int(np.flatnonzero(usable)[-1]) + 1

    with np.errstate(under="ignore"):
        gamma = np.exp(filtering.log_alpha)
    xi_count = max(usable_end - 1, 0)
    xi = np.empty((xi_count, STATE_COUNT, STATE_COUNT), dtype=np.float64)
    if xi_count > 0:
        log_beta = _log_backward_smooth(log_transition, log_emission, usable_end)
        log_gamma = filtering.log_alpha[:usable_end] + log_beta
        log_gamma -= cast(NDArray[np.float64], logsumexp_stable(log_gamma, axis=1))[:, None]
        with np.errstate(under="ignore"):
            gamma[:usable_end] = np.exp(log_gamma)

        for time_idx in range(xi_count):
            log_xi_t = (
                filtering.log_alpha[time_idx, :, None]
                + log_transition[time_idx]
                + log_emission[time_idx + 1, None, :]
                + log_beta[time_idx + 1, None, :]
            )
            log_xi_t -= float(logsumexp_stable(log_xi_t.reshape(-1)))
            with np.errstate(under="ignore"):
                xi[time_idx] = np.exp(log_xi_t)

    return HMMSmoothingResult(
        gamma=gamma,
        xi=xi,
        parameter_mask=usable.copy(),
        log_likelihood=filtering.log_likelihood,
    )


def update_emission_parameters(
    y_obs: NDArray[np.float64],
    gamma: NDArray[np.float64],
    parameter_mask: NDArray[np.bool_],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """pure. M-step update for emission means/covariances on usable history only."""
    obs = np.asarray(y_obs, dtype=np.float64)
    weights = np.asarray(gamma, dtype=np.float64)
    mask = np.asarray(parameter_mask, dtype=np.bool_)
    if obs.ndim != MATRIX_NDIM:
        msg = "y_obs must be a 2D matrix"
        raise ValueError(msg)
    if weights.shape != (obs.shape[0], STATE_COUNT):
        msg = "gamma must have shape (n_weeks, 3)"
        raise ValueError(msg)
    if mask.shape != (obs.shape[0],):
        msg = "parameter_mask must align to y_obs rows"
        raise ValueError(msg)
    if not np.isfinite(obs).all() or not np.isfinite(weights).all() or np.any(weights < 0.0):
        msg = "emission update inputs must be finite with nonnegative gamma"
        raise ValueError(msg)

    usable_obs = obs[mask]
    usable_gamma = weights[mask]
    means = np.empty((STATE_COUNT, obs.shape[1]), dtype=np.float64)
    covs = np.empty((STATE_COUNT, obs.shape[1], obs.shape[1]), dtype=np.float64)
    for state_idx in range(STATE_COUNT):
        state_weights = usable_gamma[:, state_idx]
        weight_sum = float(np.sum(state_weights))
        if weight_sum <= 0.0:
            msg = "each state must have positive mass for emission update"
            raise HMMConvergenceError(msg)
        with np.errstate(under="ignore"):
            weighted_sum = np.sum(usable_obs * state_weights[:, None], axis=0)
        means[state_idx] = weighted_sum / weight_sum
        covs[state_idx] = shrink_emission_covariance(usable_obs, state_weights)
    return means, covs


def _transition_objective(
    beta: NDArray[np.float64],
    data: _TransitionObjectiveData,
) -> float:
    with np.errstate(under="ignore"):
        logits = beta[0] + beta[1] * data.dwell + beta[2] * data.h
    leave_prob = _sigmoid_clipped(logits)
    log_leave = np.log(leave_prob)
    log_stay = np.log1p(-leave_prob)
    log_offdiag_share = np.log(float(STATE_COUNT - 1))
    with np.errstate(under="ignore"):
        nll = -float(
            np.sum(
                data.stay_weight * log_stay
                + data.leave_weight * (log_leave - log_offdiag_share),
            ),
        )
        regularization = data.l2_penalty * float(np.sum(beta * beta))
    return nll + regularization


def _transition_objective_grad(
    beta: NDArray[np.float64],
    data: _TransitionObjectiveData,
) -> NDArray[np.float64]:
    with np.errstate(under="ignore"):
        logits = beta[0] + beta[1] * data.dwell + beta[2] * data.h
    clipped_logits = np.clip(logits, -LOGIT_CLIP_BOUND, LOGIT_CLIP_BOUND)
    with np.errstate(under="ignore"):
        raw_prob = 1.0 / (1.0 + np.exp(-clipped_logits))
    in_logit_range = (logits >= -LOGIT_CLIP_BOUND) & (logits <= LOGIT_CLIP_BOUND)
    in_prob_range = (raw_prob > TRANSITION_PROB_EPSILON) & (
        raw_prob < 1.0 - TRANSITION_PROB_EPSILON
    )
    active = in_logit_range & in_prob_range
    grad = 2.0 * data.l2_penalty * beta
    if not np.any(active):
        return grad

    active_idx = np.flatnonzero(active)
    stay_weight = data.stay_weight[active_idx]
    leave_weight = data.leave_weight[active_idx]
    active_prob = raw_prob[active_idx]
    dwell = data.dwell[active_idx]
    h = data.h[active_idx]
    # d/deta [-s log(1-p) - l log(p)] = (s + l) * p - l on the unclipped interior.
    with np.errstate(under="ignore"):
        factor = (stay_weight + leave_weight) * active_prob - leave_weight

    grad[0] += float(np.sum(factor))
    with np.errstate(under="ignore"):
        grad[1] += float(np.dot(factor, dwell))
        grad[2] += float(np.dot(factor, h))
    return grad


def fit_transition_coefs(
    xi: NDArray[np.float64],
    dwell: NDArray[np.float64],
    h: NDArray[np.float64],
    *,
    l2_penalty: float = TRANSITION_L2_PENALTY,
    max_iter: int = DEFAULT_OPTIMIZER_MAX_ITER,
) -> NDArray[np.float64]:
    """pure. M-step logistic fit for SRD §7.1 time-inhomogeneous transition hazards."""
    xi_values = np.asarray(xi, dtype=np.float64)
    dwell_values = np.asarray(dwell, dtype=np.float64)
    h_values = np.asarray(h, dtype=np.float64)
    if xi_values.ndim != TENSOR_NDIM or xi_values.shape[1:] != (STATE_COUNT, STATE_COUNT):
        msg = "xi must have shape (n_transitions, 3, 3)"
        raise ValueError(msg)
    if dwell_values.shape != (xi_values.shape[0], STATE_COUNT):
        msg = "dwell must have shape (n_transitions, 3)"
        raise ValueError(msg)
    if h_values.shape != (xi_values.shape[0],):
        msg = "h must have shape (n_transitions,)"
        raise ValueError(msg)
    if (
        not np.isfinite(xi_values).all()
        or not np.isfinite(dwell_values).all()
        or not np.isfinite(h_values).all()
        or np.any(xi_values < 0.0)
    ):
        msg = "transition fit inputs must be finite with nonnegative xi"
        raise ValueError(msg)
    if max_iter <= 0:
        msg = "transition optimizer failed before starting"
        raise HMMConvergenceError(msg)

    coefs = np.empty((STATE_COUNT, STATE_COUNT), dtype=np.float64)
    for state_idx in range(STATE_COUNT):
        stay_weight = xi_values[:, state_idx, state_idx]
        leave_weight = np.sum(xi_values[:, state_idx, :], axis=1) - stay_weight
        objective_data = _TransitionObjectiveData(
            stay_weight=stay_weight,
            leave_weight=leave_weight,
            dwell=dwell_values[:, state_idx],
            h=h_values,
            l2_penalty=l2_penalty,
        )

        result = minimize(
            lambda beta, data=objective_data: _transition_objective(beta, data),
            jac=lambda beta, data=objective_data: _transition_objective_grad(
                np.asarray(beta, dtype=np.float64),
                data,
            ),
            x0=np.zeros(STATE_COUNT, dtype=np.float64),
            method="L-BFGS-B",
            options={"maxiter": max_iter},
        )
        if not result.success or not np.isfinite(result.fun):
            msg = f"transition optimizer failed for state {state_idx}"
            raise HMMConvergenceError(msg)
        coefs[state_idx] = np.asarray(result.x, dtype=np.float64)
    return coefs


def _soft_assign_to_means(
    y_obs: NDArray[np.float64],
    means: NDArray[np.float64],
) -> NDArray[np.float64]:
    sq_dist = np.sum((y_obs[:, None, :] - means[None, :, :]) ** 2, axis=2)
    logits = -sq_dist
    logits -= np.max(logits, axis=1, keepdims=True)
    weights = np.exp(logits)
    return cast(NDArray[np.float64], weights / np.sum(weights, axis=1, keepdims=True))


def _log_emission_matrix(
    y_obs: NDArray[np.float64],
    emission_mean: NDArray[np.float64],
    emission_cov: NDArray[np.float64],
) -> NDArray[np.float64]:
    values = np.empty((y_obs.shape[0], STATE_COUNT), dtype=np.float64)
    for state_idx in range(STATE_COUNT):
        values[:, state_idx] = gaussian_log_likelihood(
            y_obs,
            emission_mean[state_idx],
            emission_cov[state_idx],
        )
    return values


def _dwell_from_gamma(gamma: NDArray[np.float64]) -> NDArray[np.float64]:
    path = np.argmax(gamma, axis=1)
    dwell = np.empty((max(gamma.shape[0] - 1, 0), STATE_COUNT), dtype=np.float64)
    durations = np.ones(STATE_COUNT, dtype=np.float64)
    for time_idx in range(dwell.shape[0]):
        dwell[time_idx] = durations
        current_state = int(path[time_idx])
        next_state = int(path[time_idx + 1])
        if next_state == current_state:
            durations[next_state] += 1.0
        else:
            durations[next_state] = 1.0
    return dwell


def _log_transition_matrices(
    transition_coefs: NDArray[np.float64],
    dwell: NDArray[np.float64],
    h: NDArray[np.float64],
) -> NDArray[np.float64]:
    matrices = np.empty((dwell.shape[0], STATE_COUNT, STATE_COUNT), dtype=np.float64)
    for time_idx in range(dwell.shape[0]):
        transition = transition_matrix_t(transition_coefs, dwell[time_idx], float(h[time_idx]))
        matrices[time_idx] = np.log(transition)
    return matrices


def infer_hmm(
    model: HMMModel,
    y_obs_history: NDArray[np.float64],
    h_history: NDArray[np.float64],
) -> HMMInferenceResult:
    """pure. Filter current TI-HMM posterior from fitted parameters and observations."""
    if set(model.label_map) != set(range(STATE_COUNT)):
        msg = "HMM label_map must contain all hidden states"
        raise HMMConvergenceError(msg)
    y_obs = np.asarray(y_obs_history, dtype=np.float64)
    h = np.asarray(h_history, dtype=np.float64)
    _validate_fit_inputs(y_obs, h)
    log_initial = np.log(np.full(STATE_COUNT, 1.0 / float(STATE_COUNT), dtype=np.float64))
    log_emission = _log_emission_matrix(y_obs, model.emission_mean, model.emission_cov)
    dwell = np.ones((max(y_obs.shape[0] - 1, 0), STATE_COUNT), dtype=np.float64)
    log_transition = _log_transition_matrices(model.transition_coefs, dwell, h[: dwell.shape[0]])
    filtering = log_forward_filter(log_initial, log_transition, log_emission)
    with np.errstate(under="ignore"):
        post = np.exp(filtering.log_alpha[-1])
    post_sum = float(np.sum(post))
    if not np.isfinite(post_sum) or post_sum <= 0.0:
        msg = "posterior exp underflow produced invalid mass"
        raise HMMConvergenceError(msg)
    post = post / post_sum
    state_idx = int(np.argmax(post))
    path = np.argmax(filtering.log_alpha, axis=1)
    dwell_weeks = 1
    for idx in range(path.shape[0] - 2, -1, -1):
        if int(path[idx]) != state_idx:
            break
        dwell_weeks += 1
    return HMMInferenceResult(
        posterior=HMMPosterior(
            post=post,
            state_name=model.label_map[state_idx],
            model_status="ok",
        ),
        dwell_weeks=dwell_weeks,
        hazard_covariate=float(h[-1]),
    )


def infer_hmm_posterior_path(
    model: HMMModel,
    y_obs_history: NDArray[np.float64],
    h_history: NDArray[np.float64],
) -> NDArray[np.float64]:
    """pure. Return the filtered posterior path for every prefix of the observation history."""
    if set(model.label_map) != set(range(STATE_COUNT)):
        msg = "HMM label_map must contain all hidden states"
        raise HMMConvergenceError(msg)
    y_obs = np.asarray(y_obs_history, dtype=np.float64)
    h = np.asarray(h_history, dtype=np.float64)
    _validate_fit_inputs(y_obs, h)
    log_initial = np.log(np.full(STATE_COUNT, 1.0 / float(STATE_COUNT), dtype=np.float64))
    log_emission = _log_emission_matrix(y_obs, model.emission_mean, model.emission_cov)
    dwell = np.ones((max(y_obs.shape[0] - 1, 0), STATE_COUNT), dtype=np.float64)
    log_transition = _log_transition_matrices(model.transition_coefs, dwell, h[: dwell.shape[0]])
    filtering = log_forward_filter(log_initial, log_transition, log_emission)
    with np.errstate(under="ignore"):
        post_path = np.exp(filtering.log_alpha)
    row_sums = np.sum(post_path, axis=1, keepdims=True)
    if not np.isfinite(row_sums).all() or np.any(row_sums <= 0.0):
        msg = "posterior path exp underflow produced invalid mass"
        raise HMMConvergenceError(msg)
    return cast(NDArray[np.float64], post_path / row_sums)


def _validate_forward_returns(
    forward_52w_returns: NDArray[np.float64] | None,
    n_obs: int,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    if forward_52w_returns is None:
        msg = "forward_52w_returns is required for SRD §7.2 label identification"
        raise HMMConvergenceError(msg)
    returns = np.asarray(forward_52w_returns, dtype=np.float64)
    if returns.shape != (n_obs,):
        msg = "forward_52w_returns must align to y_obs rows"
        raise ValueError(msg)
    usable = np.isfinite(returns)
    _validate_usable_mask(usable, n_obs)
    return returns, usable


def _label_map_from_gamma_returns(
    gamma: NDArray[np.float64],
    forward_returns: NDArray[np.float64],
    parameter_mask: NDArray[np.bool_],
) -> Mapping[int, Stance]:
    averages: dict[int, float] = {}
    for state_idx in range(STATE_COUNT):
        weights = gamma[parameter_mask, state_idx]
        total = float(np.sum(weights))
        if total <= 0.0:
            msg = "each state must have positive mass for label identification"
            raise HMMConvergenceError(msg)
        with np.errstate(under="ignore"):
            weighted_return = np.sum(weights * forward_returns[parameter_mask]) / total
        averages[state_idx] = float(weighted_return)
    return build_label_map(averages)


def _run_hmm_restart(
    data: _FitData,
    generator: np.random.Generator,
    config: _FitConfig,
) -> HMMModel:
    means = initialize_emission_means_kmeans_pp(
        data.y_obs,
        generator,
        usable_mask=data.parameter_mask,
    )
    gamma = _soft_assign_to_means(data.y_obs, means)
    means, covs = update_emission_parameters(data.y_obs, gamma, data.parameter_mask)
    transition_coefs = np.zeros((STATE_COUNT, STATE_COUNT), dtype=np.float64)
    previous_log_likelihood = -np.inf

    for _iteration in range(config.max_iter):
        log_emission = _log_emission_matrix(data.y_obs, means, covs)
        dwell = _dwell_from_gamma(gamma)
        log_transition = _log_transition_matrices(transition_coefs, dwell, data.h)
        smoothing = e_step_smooth(
            data.log_initial,
            log_transition,
            log_emission,
            usable_mask=data.parameter_mask,
        )
        improvement = smoothing.log_likelihood - previous_log_likelihood
        gamma = smoothing.gamma
        means, covs = update_emission_parameters(data.y_obs, gamma, data.parameter_mask)
        if smoothing.xi.shape[0] > 0:
            transition_coefs = fit_transition_coefs(
                smoothing.xi,
                dwell[: smoothing.xi.shape[0]],
                data.h[: smoothing.xi.shape[0]],
                max_iter=config.transition_max_iter,
            )
        if np.isfinite(previous_log_likelihood) and abs(improvement) < config.tolerance:
            label_map = _label_map_from_gamma_returns(
                gamma,
                data.forward_returns,
                data.parameter_mask,
            )
            return HMMModel(
                transition_coefs=transition_coefs,
                emission_mean=means,
                emission_cov=covs,
                label_map=label_map,
                log_likelihood=smoothing.log_likelihood,
            )
        previous_log_likelihood = smoothing.log_likelihood

    msg = "HMM EM did not converge within max_iter"
    raise HMMConvergenceError(msg)


def initialize_emission_means_kmeans_pp(
    y_obs: NDArray[np.float64],
    rng: np.random.Generator,
    *,
    state_count: int = STATE_COUNT,
    usable_mask: NDArray[np.bool_] | None = None,
) -> NDArray[np.float64]:
    """pure. Choose deterministic K-Means++ emission seeds for injected rng."""
    generator = _require_generator(rng)
    obs = np.asarray(y_obs, dtype=np.float64)
    if obs.ndim != MATRIX_NDIM:
        msg = "y_obs must be a 2D matrix"
        raise ValueError(msg)
    if not np.isfinite(obs).all():
        msg = "y_obs must be finite"
        raise ValueError(msg)
    if usable_mask is None:
        usable = np.ones(obs.shape[0], dtype=np.bool_)
    else:
        usable = np.asarray(usable_mask, dtype=np.bool_)
        if usable.shape != (obs.shape[0],):
            msg = "usable_mask must align to y_obs rows"
            raise ValueError(msg)
    candidates = obs[usable]
    if candidates.shape[0] < state_count:
        msg = "not enough usable observations for K-Means++ initialization"
        raise HMMConvergenceError(msg)

    means = np.empty((state_count, obs.shape[1]), dtype=np.float64)
    first_idx = int(generator.integers(0, candidates.shape[0]))
    means[0] = candidates[first_idx]
    min_sq_dist = np.sum((candidates - means[0]) ** 2, axis=1)
    for state_idx in range(1, state_count):
        total = float(np.sum(min_sq_dist))
        if total <= 0.0:
            unused = np.flatnonzero(min_sq_dist == 0.0)
            chosen_idx = int(unused[min(state_idx, unused.shape[0] - 1)])
        else:
            probs = min_sq_dist / total
            chosen_idx = int(generator.choice(candidates.shape[0], p=probs))
        means[state_idx] = candidates[chosen_idx]
        new_sq_dist = np.sum((candidates - means[state_idx]) ** 2, axis=1)
        min_sq_dist = np.minimum(min_sq_dist, new_sq_dist)
    return means


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


def fit_hmm(  # noqa: PLR0913
    y_obs: NDArray[np.float64],
    h: NDArray[np.float64],
    rng: np.random.Generator,
    *,
    max_iter: int = DEFAULT_MAX_ITER,
    tolerance: float = DEFAULT_TOLERANCE,
    restarts: int = DEFAULT_RESTARTS,
    forward_52w_returns: NDArray[np.float64] | None = None,
    transition_max_iter: int = DEFAULT_OPTIMIZER_MAX_ITER,
) -> HMMModel:
    """pure. Fit TI-HMM deterministically for given inputs and injected rng."""
    generator = _require_generator(rng)
    _validate_fit_inputs(y_obs, h)
    if max_iter <= 0:
        msg = "HMM EM did not converge within max_iter"
        raise HMMConvergenceError(msg)
    if tolerance <= 0.0 or not np.isfinite(tolerance):
        msg = "HMM tolerance must be finite and positive"
        raise ValueError(msg)
    if restarts <= 0:
        msg = "HMM restarts must be positive"
        raise ValueError(msg)
    forward_returns, parameter_mask = _validate_forward_returns(forward_52w_returns, y_obs.shape[0])
    log_initial = np.log(np.full(STATE_COUNT, 1.0 / float(STATE_COUNT), dtype=np.float64))
    data = _FitData(
        y_obs=y_obs,
        h=h,
        forward_returns=forward_returns,
        parameter_mask=parameter_mask,
        log_initial=log_initial,
    )
    config = _FitConfig(
        max_iter=max_iter,
        tolerance=tolerance,
        transition_max_iter=transition_max_iter,
    )

    best_model: HMMModel | None = None
    best_log_likelihood = -np.inf
    last_error: Exception | None = None
    for _ in range(restarts):
        try:
            candidate = _run_hmm_restart(data, generator, config)
            if candidate.log_likelihood > best_log_likelihood:
                best_model = candidate
                best_log_likelihood = candidate.log_likelihood
        except (HMMConvergenceError, ValueError) as exc:
            last_error = exc
            continue

    if best_model is None:
        if last_error is not None:
            msg = f"HMM EM did not converge in any restart: {last_error}"
            raise HMMConvergenceError(msg) from last_error
        msg = "HMM EM did not converge in any restart"
        raise HMMConvergenceError(msg)
    return best_model
