"""Single production TI-HMM interface and guardrails from SRD v8.7 section 7."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from engine_types import Stance
from errors import HMMConvergenceError

STATE_COUNT = 3
MATRIX_NDIM = 2
OBS_DIM = 6
DEFAULT_MAX_ITER = 200
DEFAULT_TOLERANCE = 1.0e-6


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
