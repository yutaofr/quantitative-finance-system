"""Pure joint panel quantile regression from SRD v8.8 section P5."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, cast

import cvxpy as cp
import numpy as np
from numpy.typing import NDArray

from errors import QuantileSolverError

_CP = cast(Any, cp)
INTERIOR_TAUS = (0.10, 0.25, 0.50, 0.75, 0.90)
QUANTILE_COUNT = 5
MACRO_DIM = 7
STATE_DIM = 3
MICRO_DIM = 3
TOTAL_DIM = MACRO_DIM + STATE_DIM + MICRO_DIM
MATRIX_NDIM = 2
DEFAULT_L2_ALPHA = 2.0
DEFAULT_MIN_GAP = 1.0e-4
OPTIMAL_STATUSES = {"optimal", "optimal_inaccurate"}
STATUS_OK = "ok"
STATUS_FALLBACK = "per_asset_fallback"


@dataclass(frozen=True, slots=True)
class _AssetIndependentQR:
    alpha: NDArray[np.float64]
    beta: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class PanelQRCoefs:
    """pure. Joint panel quantile regression coefficients."""

    asset_ids: tuple[str, ...]
    alpha: Mapping[str, NDArray[np.float64]]
    b: NDArray[np.float64]
    c: NDArray[np.float64]
    delta: NDArray[np.float64]
    solver_status: str
    model_status: str
    fallback_asset_coefs: Mapping[str, _AssetIndependentQR] | None = None


def _pinball_loss(residual: Any, tau: float) -> Any:
    return _CP.maximum(tau * residual, (tau - 1.0) * residual)


def _rearrange(values: NDArray[np.float64], min_gap: float) -> NDArray[np.float64]:
    out = np.sort(np.asarray(values, dtype=np.float64))
    for idx in range(1, out.shape[0]):
        out[idx] = max(out[idx], out[idx - 1] + min_gap)
    return out


def _validate_core_inputs(
    x_macro: NDArray[np.float64],
    pi: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    macro = np.asarray(x_macro, dtype=np.float64)
    post = np.asarray(pi, dtype=np.float64)
    if macro.ndim != MATRIX_NDIM or macro.shape[1] != MACRO_DIM:
        msg = "x_macro must have shape (n_obs, 7)"
        raise ValueError(msg)
    if post.shape != (macro.shape[0], STATE_DIM):
        msg = "pi must have shape (n_obs, 3)"
        raise ValueError(msg)
    if not np.isfinite(macro).all() or not np.isfinite(post).all():
        msg = "x_macro and pi must be finite"
        raise ValueError(msg)
    return macro, post


def _masked_values(
    values: NDArray[np.float64],
    mask: NDArray[np.bool_],
    *,
    name: str,
) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64).copy()
    available = np.asarray(mask, dtype=np.bool_)
    if array.shape[0] != available.shape[0]:
        msg = f"{name} must align to asset availability rows"
        raise ValueError(msg)
    if array.ndim == 1:
        if np.any(available & ~np.isfinite(array)):
            msg = f"{name} must be finite on available rows"
            raise ValueError(msg)
        array[~available] = 0.0
        return array
    if array.ndim != MATRIX_NDIM:
        msg = f"{name} must be a vector or matrix"
        raise ValueError(msg)
    row_finite = np.isfinite(array).all(axis=1)
    if np.any(available & ~row_finite):
        msg = f"{name} must be finite on available rows"
        raise ValueError(msg)
    array[~available] = 0.0
    return array


def _validate_asset_inputs(
    asset_id: str,
    x_micro: NDArray[np.float64],
    y_52w: NDArray[np.float64],
    mask: NDArray[np.bool_],
    n_obs: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    micro = np.asarray(x_micro, dtype=np.float64)
    target = np.asarray(y_52w, dtype=np.float64)
    available = np.asarray(mask, dtype=np.bool_)
    if micro.shape != (n_obs, MICRO_DIM):
        msg = f"{asset_id} x_micro must have shape (n_obs, 3)"
        raise ValueError(msg)
    if target.shape != (n_obs,):
        msg = f"{asset_id} y_52w must have shape (n_obs,)"
        raise ValueError(msg)
    if available.shape != (n_obs,):
        msg = f"{asset_id} mask must have shape (n_obs,)"
        raise ValueError(msg)
    if not available.any():
        msg = f"{asset_id} has no available training rows"
        raise ValueError(msg)
    return (
        _masked_values(micro, available, name=f"{asset_id} x_micro"),
        _masked_values(target, available, name=f"{asset_id} y_52w"),
        available,
    )


def _solve_shared_panel(  # noqa: PLR0913
    x_macro: NDArray[np.float64],
    x_micro: Mapping[str, NDArray[np.float64]],
    pi: NDArray[np.float64],
    y_52w: Mapping[str, NDArray[np.float64]],
    mask: Mapping[str, NDArray[np.bool_]],
    *,
    l2_alpha_macro: float,
    l2_alpha_micro: float,
    min_gap: float,
    solver: str,
) -> PanelQRCoefs:
    asset_ids = tuple(x_micro)
    n_obs = x_macro.shape[0]
    alpha = cp.Variable((len(asset_ids), QUANTILE_COUNT))
    b = cp.Variable((QUANTILE_COUNT, MACRO_DIM))
    c = cp.Variable((QUANTILE_COUNT, STATE_DIM))
    delta = cp.Variable((QUANTILE_COUNT, MICRO_DIM))

    objective_terms: list[Any] = []
    constraints: list[Any] = []
    alpha_by_asset: dict[str, NDArray[np.float64]] = {}
    for asset_index, asset_id in enumerate(asset_ids):
        micro, target, available = _validate_asset_inputs(
            asset_id,
            x_micro[asset_id],
            y_52w[asset_id],
            mask[asset_id],
            n_obs,
        )
        weights = available.astype(np.float64)
        prediction = (
            x_macro @ b.T
            + pi @ c.T
            + micro @ delta.T
            + _CP.reshape(alpha[asset_index], (1, QUANTILE_COUNT), order="C")
        )
        for quantile_index, tau in enumerate(INTERIOR_TAUS):
            residual = target - prediction[:, quantile_index]
            objective_terms.append(
                _CP.sum(_CP.multiply(weights, _pinball_loss(residual, tau))),
            )
        constraints.extend(
            [
                _CP.multiply(weights, prediction[:, idx + 1] - prediction[:, idx])
                >= min_gap * weights
                for idx in range(QUANTILE_COUNT - 1)
            ],
        )
    penalty = l2_alpha_macro * (_CP.sum_squares(b) + _CP.sum_squares(c))
    penalty += l2_alpha_micro * _CP.sum_squares(delta)
    problem = cp.Problem(cp.Minimize(_CP.sum(objective_terms) + penalty), constraints)
    try:
        with np.errstate(under="warn"):
            cast(Any, problem).solve(solver=solver)
    except cp.error.SolverError as exc:
        msg = f"panel quantile solver failed: {exc}"
        raise QuantileSolverError(msg) from exc
    if problem.status not in OPTIMAL_STATUSES:
        msg = f"panel quantile solver returned status {problem.status}"
        raise QuantileSolverError(msg)
    if alpha.value is None or b.value is None or c.value is None or delta.value is None:
        msg = "panel quantile solver did not return coefficients"
        raise QuantileSolverError(msg)
    alpha_values = np.asarray(alpha.value, dtype=np.float64)
    for asset_index, asset_id in enumerate(asset_ids):
        alpha_by_asset[asset_id] = alpha_values[asset_index]
    return PanelQRCoefs(
        asset_ids=asset_ids,
        alpha=cast(Mapping[str, NDArray[np.float64]], MappingProxyType(alpha_by_asset)),
        b=np.asarray(b.value, dtype=np.float64),
        c=np.asarray(c.value, dtype=np.float64),
        delta=np.asarray(delta.value, dtype=np.float64),
        solver_status=STATUS_OK,
        model_status="NORMAL",
    )


def _solve_independent_asset(  # noqa: PLR0913
    x_macro: NDArray[np.float64],
    x_micro: NDArray[np.float64],
    pi: NDArray[np.float64],
    y_52w: NDArray[np.float64],
    mask: NDArray[np.bool_],
    *,
    l2_alpha_macro: float,
    l2_alpha_micro: float,
    min_gap: float,
    solver: str,
) -> _AssetIndependentQR:
    available = np.asarray(mask, dtype=np.bool_)
    design = np.column_stack(
        [x_macro[available], pi[available], x_micro[available]],
    ).astype(np.float64)
    target = np.asarray(y_52w[available], dtype=np.float64)
    if design.shape[0] == 0:
        msg = "asset fallback has no available rows"
        raise QuantileSolverError(msg)
    alpha = cp.Variable(QUANTILE_COUNT)
    beta = cp.Variable((QUANTILE_COUNT, TOTAL_DIM))
    prediction = design @ beta.T + _CP.reshape(alpha, (1, QUANTILE_COUNT), order="C")
    objective_terms: list[Any] = []
    for quantile_index, tau in enumerate(INTERIOR_TAUS):
        residual = target - prediction[:, quantile_index]
        objective_terms.append(_CP.sum(_pinball_loss(residual, tau)))
    penalty = l2_alpha_macro * (
        _CP.sum_squares(beta[:, :MACRO_DIM])
        + _CP.sum_squares(beta[:, MACRO_DIM : MACRO_DIM + STATE_DIM])
    )
    penalty += l2_alpha_micro * _CP.sum_squares(beta[:, MACRO_DIM + STATE_DIM :])
    constraints = [
        prediction[:, idx + 1] - prediction[:, idx] >= min_gap
        for idx in range(QUANTILE_COUNT - 1)
    ]
    problem = cp.Problem(cp.Minimize(_CP.sum(objective_terms) + penalty), constraints)
    try:
        with np.errstate(under="warn"):
            cast(Any, problem).solve(solver=solver)
    except cp.error.SolverError as exc:
        msg = f"per-asset fallback solver failed: {exc}"
        raise QuantileSolverError(msg) from exc
    if problem.status not in OPTIMAL_STATUSES or alpha.value is None or beta.value is None:
        msg = "per-asset fallback solver did not converge"
        raise QuantileSolverError(msg)
    return _AssetIndependentQR(
        alpha=np.asarray(alpha.value, dtype=np.float64),
        beta=np.asarray(beta.value, dtype=np.float64),
    )


def _fallback_panel(  # noqa: PLR0913
    x_macro: NDArray[np.float64],
    x_micro: Mapping[str, NDArray[np.float64]],
    pi: NDArray[np.float64],
    y_52w: Mapping[str, NDArray[np.float64]],
    mask: Mapping[str, NDArray[np.bool_]],
    *,
    l2_alpha_macro: float,
    l2_alpha_micro: float,
    min_gap: float,
    solver: str,
) -> PanelQRCoefs:
    asset_ids = tuple(x_micro)
    fallback_asset_coefs: dict[str, _AssetIndependentQR] = {}
    alpha_by_asset: dict[str, NDArray[np.float64]] = {}
    for asset_id in asset_ids:
        micro, target, available = _validate_asset_inputs(
            asset_id,
            x_micro[asset_id],
            y_52w[asset_id],
            mask[asset_id],
            x_macro.shape[0],
        )
        fitted = _solve_independent_asset(
            x_macro,
            micro,
            pi,
            target,
            available,
            l2_alpha_macro=l2_alpha_macro,
            l2_alpha_micro=l2_alpha_micro,
            min_gap=min_gap,
            solver=solver,
        )
        fallback_asset_coefs[asset_id] = fitted
        alpha_by_asset[asset_id] = fitted.alpha
    return PanelQRCoefs(
        asset_ids=asset_ids,
        alpha=cast(Mapping[str, NDArray[np.float64]], MappingProxyType(alpha_by_asset)),
        b=np.zeros((QUANTILE_COUNT, MACRO_DIM), dtype=np.float64),
        c=np.zeros((QUANTILE_COUNT, STATE_DIM), dtype=np.float64),
        delta=np.zeros((QUANTILE_COUNT, MICRO_DIM), dtype=np.float64),
        solver_status=STATUS_FALLBACK,
        model_status="DEGRADED",
        fallback_asset_coefs=cast(
            Mapping[str, _AssetIndependentQR],
            MappingProxyType(fallback_asset_coefs),
        ),
    )


def fit_panel_quantiles(  # noqa: PLR0913
    x_macro: NDArray[np.float64],
    x_micro: Mapping[str, NDArray[np.float64]],
    pi: NDArray[np.float64],
    y_52w: Mapping[str, NDArray[np.float64]],
    mask: Mapping[str, NDArray[np.bool_]],
    *,
    l2_alpha_macro: float = DEFAULT_L2_ALPHA,
    l2_alpha_micro: float = DEFAULT_L2_ALPHA,
    min_gap: float = DEFAULT_MIN_GAP,
    solver: str = "ECOS",
) -> PanelQRCoefs:
    """pure. Fit joint panel quantiles with masked objective and per-asset fallback."""
    macro, post = _validate_core_inputs(x_macro, pi)
    if tuple(x_micro) != tuple(y_52w) or tuple(x_micro) != tuple(mask):
        msg = "x_micro, y_52w, and mask must have identical asset keys and order"
        raise ValueError(msg)
    try:
        return _solve_shared_panel(
            macro,
            x_micro,
            post,
            y_52w,
            mask,
            l2_alpha_macro=l2_alpha_macro,
            l2_alpha_micro=l2_alpha_micro,
            min_gap=min_gap,
            solver=solver,
        )
    except QuantileSolverError:
        return _fallback_panel(
            macro,
            x_micro,
            post,
            y_52w,
            mask,
            l2_alpha_macro=l2_alpha_macro,
            l2_alpha_micro=l2_alpha_micro,
            min_gap=min_gap,
            solver=solver,
        )


def predict_panel_interior_with_status(  # noqa: PLR0913
    coefs: PanelQRCoefs,
    asset_id: str,
    x_macro_t: NDArray[np.float64],
    x_micro_t: NDArray[np.float64],
    pi_t: NDArray[np.float64],
    *,
    min_gap: float = DEFAULT_MIN_GAP,
) -> tuple[NDArray[np.float64], str]:
    """pure. Predict one asset's interior quantiles and expose the solver status."""
    macro = np.asarray(x_macro_t, dtype=np.float64)
    micro = np.asarray(x_micro_t, dtype=np.float64)
    post = np.asarray(pi_t, dtype=np.float64)
    if macro.shape != (MACRO_DIM,) or micro.shape != (MICRO_DIM,) or post.shape != (STATE_DIM,):
        msg = "predict_panel_interior_with_status expects shapes (7,), (3,), and (3,)"
        raise ValueError(msg)
    if asset_id not in coefs.alpha:
        msg = f"unknown asset_id {asset_id}"
        raise KeyError(msg)
    if coefs.fallback_asset_coefs is not None:
        fitted = coefs.fallback_asset_coefs[asset_id]
        design = np.concatenate([macro, post, micro]).astype(np.float64)
        raw = fitted.alpha + fitted.beta @ design
    else:
        raw = coefs.alpha[asset_id] + coefs.b @ macro + coefs.c @ post + coefs.delta @ micro
    if np.all(np.diff(raw) >= min_gap):
        return raw.astype(np.float64), coefs.solver_status
    return _rearrange(raw, min_gap), "rearranged"


def predict_panel_interior(  # noqa: PLR0913
    coefs: PanelQRCoefs,
    asset_id: str,
    x_macro_t: NDArray[np.float64],
    x_micro_t: NDArray[np.float64],
    pi_t: NDArray[np.float64],
    *,
    min_gap: float = DEFAULT_MIN_GAP,
) -> NDArray[np.float64]:
    """pure. Predict one asset's interior quantiles with non-crossing enforcement."""
    prediction, _status = predict_panel_interior_with_status(
        coefs,
        asset_id,
        x_macro_t,
        x_micro_t,
        pi_t,
        min_gap=min_gap,
    )
    return prediction
