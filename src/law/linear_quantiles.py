"""Linear non-crossing quantile regression from SRD v8.7 section 8.1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import cvxpy as cp
import numpy as np
from numpy.typing import NDArray

from errors import QuantileSolverError

_CP = cast(Any, cp)
INTERIOR_TAUS = (0.10, 0.25, 0.50, 0.75, 0.90)
QUANTILE_COUNT = 5
POSTERIOR_DIM = 3
MATRIX_NDIM = 2
DEFAULT_L2_ALPHA = 2.0
DEFAULT_MIN_GAP = 1.0e-4
OPTIMAL_STATUSES = {"optimal", "optimal_inaccurate"}
SRD_OK_STATUS = "ok"
SRD_REARRANGED_STATUS = "rearranged"


@dataclass(frozen=True, slots=True)
class QRCoefs:
    """pure. Coefficients for five linear conditional quantiles."""

    a: NDArray[np.float64]
    b: NDArray[np.float64]
    c: NDArray[np.float64]
    solver_status: str


def _validate_training_inputs(
    x_scaled: NDArray[np.float64],
    pi: NDArray[np.float64],
    y_52w: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    x = np.asarray(x_scaled, dtype=np.float64)
    post = np.asarray(pi, dtype=np.float64)
    y = np.asarray(y_52w, dtype=np.float64)
    if x.ndim != MATRIX_NDIM:
        msg = "x_scaled must be a 2D matrix"
        raise ValueError(msg)
    if post.shape != (x.shape[0], POSTERIOR_DIM):
        msg = "pi must have shape (n_obs, 3)"
        raise ValueError(msg)
    if y.shape != (x.shape[0],):
        msg = "y_52w must align to x_scaled rows"
        raise ValueError(msg)
    if not np.isfinite(x).all() or not np.isfinite(post).all() or not np.isfinite(y).all():
        msg = "quantile regression inputs must be finite"
        raise ValueError(msg)
    return x, post, y


def _pinball_loss(residual: Any, tau: float) -> Any:
    return _CP.sum(_CP.maximum(tau * residual, (tau - 1.0) * residual))


def fit_linear_quantiles(  # noqa: PLR0913
    x_scaled: NDArray[np.float64],
    pi: NDArray[np.float64],
    y_52w: NDArray[np.float64],
    *,
    l2_alpha: float = DEFAULT_L2_ALPHA,
    min_gap: float = DEFAULT_MIN_GAP,
    solver: str = "ECOS",
) -> QRCoefs:
    """pure. Jointly fit all SRD §8.1 interior quantiles with non-crossing constraints."""
    x, post, y = _validate_training_inputs(x_scaled, pi, y_52w)
    n_features = x.shape[1]
    a = cp.Variable(QUANTILE_COUNT)
    b = cp.Variable((QUANTILE_COUNT, n_features))
    c = cp.Variable((QUANTILE_COUNT, POSTERIOR_DIM))
    predictions = x @ b.T + post @ c.T + _CP.reshape(a, (1, QUANTILE_COUNT), order="C")

    objective_terms: list[Any] = []
    for idx, tau in enumerate(INTERIOR_TAUS):
        objective_terms.append(_pinball_loss(y - predictions[:, idx], tau))
    constraints = [
        predictions[:, idx + 1] - predictions[:, idx] >= min_gap
        for idx in range(QUANTILE_COUNT - 1)
    ]
    penalty = l2_alpha * (_CP.sum_squares(b) + _CP.sum_squares(c))
    problem = cp.Problem(cp.Minimize(_CP.sum(objective_terms) + penalty), constraints)
    try:
        cast(Any, problem).solve(solver=solver)
    except cp.error.SolverError as exc:
        msg = f"quantile solver failed: {exc}"
        raise QuantileSolverError(msg) from exc
    if problem.status not in OPTIMAL_STATUSES:
        msg = f"quantile solver returned status {problem.status}"
        raise QuantileSolverError(msg)
    if a.value is None or b.value is None or c.value is None:
        msg = "quantile solver did not return coefficients"
        raise QuantileSolverError(msg)

    return QRCoefs(
        a=np.asarray(a.value, dtype=np.float64),
        b=np.asarray(b.value, dtype=np.float64),
        c=np.asarray(c.value, dtype=np.float64),
        solver_status=SRD_OK_STATUS,
    )


def _rearrange(values: NDArray[np.float64], min_gap: float) -> NDArray[np.float64]:
    out = np.sort(np.asarray(values, dtype=np.float64))
    for idx in range(1, out.shape[0]):
        out[idx] = max(out[idx], out[idx - 1] + min_gap)
    return out


def predict_interior(
    coefs: QRCoefs,
    x_t: NDArray[np.float64],
    pi_t: NDArray[np.float64],
    *,
    min_gap: float = DEFAULT_MIN_GAP,
) -> NDArray[np.float64]:
    """pure. Predict and rearrange interior quantiles to preserve ordering."""
    predictions, _status = predict_interior_with_status(coefs, x_t, pi_t, min_gap=min_gap)
    return predictions


def predict_interior_with_status(
    coefs: QRCoefs,
    x_t: NDArray[np.float64],
    pi_t: NDArray[np.float64],
    *,
    min_gap: float = DEFAULT_MIN_GAP,
) -> tuple[NDArray[np.float64], str]:
    """pure. Predict interior quantiles and expose the SRD §11 solver status."""
    x = np.asarray(x_t, dtype=np.float64)
    post = np.asarray(pi_t, dtype=np.float64)
    if coefs.a.shape != (QUANTILE_COUNT,):
        msg = "coefs.a must have shape (5,)"
        raise ValueError(msg)
    if coefs.b.shape[0] != QUANTILE_COUNT or coefs.b.shape[1] != x.shape[0]:
        msg = "coefs.b shape is inconsistent with x_t"
        raise ValueError(msg)
    if coefs.c.shape != (QUANTILE_COUNT, POSTERIOR_DIM) or post.shape != (POSTERIOR_DIM,):
        msg = "posterior coefficient shape is inconsistent with pi_t"
        raise ValueError(msg)
    raw = coefs.a + coefs.b @ x + coefs.c @ post
    if np.all(np.diff(raw) >= min_gap):
        return raw, coefs.solver_status
    return _rearrange(raw, min_gap), SRD_REARRANGED_STATUS
