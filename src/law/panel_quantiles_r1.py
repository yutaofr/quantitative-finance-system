"""Pure v8.8R1 panel quantile regression — q50 + spreads parameterization.

Design:
  - Predicts q50 + 4 non-negative spreads per asset
  - q75 = q50 + Δup1, q90 = q75 + Δup2
  - q25 = q50 - Δdn1, q10 = q25 - Δdn2
  - Non-crossing guaranteed by spread >= 0 constraint (no rearrangement needed)
  - Partial pooling: b_shared + delta_b per asset, d_shared + delta_d per asset
  - No HMM π input — macro + micro only
"""

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
MICRO_DIM = 3
MATRIX_NDIM = 2
N_SPREADS = 4
DEFAULT_L2_ALPHA = 2.0
DEFAULT_L2_DELTA = 4.0
DEFAULT_MIN_SPREAD = 1.0e-4
OPTIMAL_STATUSES = {"optimal", "optimal_inaccurate"}
STATUS_OK = "ok"
STATUS_FALLBACK = "per_asset_fallback"


@dataclass(frozen=True, slots=True)
class R1PanelQRCoefs:
    """pure. v8.8R1 panel QR coefficients with q50+spreads parameterization."""

    asset_ids: tuple[str, ...]
    # q50 coefficients
    alpha_q50: Mapping[str, float]
    b_shared_q50: NDArray[np.float64]       # (MACRO_DIM,)
    d_shared_q50: NDArray[np.float64]       # (MICRO_DIM,)
    delta_b_q50: Mapping[str, NDArray[np.float64]]  # per-asset (MACRO_DIM,)
    delta_d_q50: Mapping[str, NDArray[np.float64]]  # per-asset (MICRO_DIM,)
    # spread coefficients (4 spreads: dn2, dn1, up1, up2) — per asset
    alpha_spreads: Mapping[str, NDArray[np.float64]]  # per-asset (4,)
    b_shared_spreads: NDArray[np.float64]   # (4, MACRO_DIM)
    d_shared_spreads: NDArray[np.float64]   # (4, MICRO_DIM)
    delta_b_spreads: Mapping[str, NDArray[np.float64]]  # per-asset (4, MACRO_DIM)
    delta_d_spreads: Mapping[str, NDArray[np.float64]]  # per-asset (4, MICRO_DIM)
    solver_status: str
    model_status: str


def _pinball_loss(residual: Any, tau: float) -> Any:
    """pure. Asymmetric pinball loss."""
    return _CP.maximum(tau * residual, (tau - 1.0) * residual)


def _validate_inputs(
    x_macro: NDArray[np.float64],
    x_micro: Mapping[str, NDArray[np.float64]],
    y_52w: Mapping[str, NDArray[np.float64]],
    mask: Mapping[str, NDArray[np.bool_]],
) -> NDArray[np.float64]:
    """pure. Validate and return coerced macro array."""
    macro = np.asarray(x_macro, dtype=np.float64)
    if macro.ndim != MATRIX_NDIM or macro.shape[1] != MACRO_DIM:
        msg = "x_macro must have shape (n_obs, 7)"
        raise ValueError(msg)
    if not np.isfinite(macro).all():
        msg = "x_macro must be finite"
        raise ValueError(msg)
    if tuple(x_micro) != tuple(y_52w) or tuple(x_micro) != tuple(mask):
        msg = "x_micro, y_52w, mask must have identical asset keys"
        raise ValueError(msg)
    return macro


def fit_r1_panel_quantiles(  # noqa: PLR0913, PLR0915
    x_macro: NDArray[np.float64],
    x_micro: Mapping[str, NDArray[np.float64]],
    y_52w: Mapping[str, NDArray[np.float64]],
    mask: Mapping[str, NDArray[np.bool_]],
    *,
    l2_alpha_macro: float = DEFAULT_L2_ALPHA,
    l2_alpha_micro: float = DEFAULT_L2_ALPHA,
    l2_delta_macro: float = DEFAULT_L2_DELTA,
    l2_delta_micro: float = DEFAULT_L2_DELTA,
    min_spread: float = DEFAULT_MIN_SPREAD,
    solver: str = "ECOS",
) -> R1PanelQRCoefs:
    """pure. Fit v8.8R1 panel quantiles: q50 + spreads, base+delta pooling, no HMM."""
    macro = _validate_inputs(x_macro, x_micro, y_52w, mask)
    asset_ids = tuple(x_micro)
    n_obs = macro.shape[0]
    n_assets = len(asset_ids)

    # ── Decision variables (all 2D or 1D to avoid CVXPY 3D issues) ──
    # q50: intercept per asset, shared macro/micro + per-asset deltas
    alpha_q50 = cp.Variable(n_assets)
    b_shared_q50 = cp.Variable(MACRO_DIM)
    d_shared_q50 = cp.Variable(MICRO_DIM)
    delta_b_q50 = cp.Variable((n_assets, MACRO_DIM))
    delta_d_q50 = cp.Variable((n_assets, MICRO_DIM))

    # spreads: 4 spreads (dn2, dn1, up1, up2)
    # Per-asset intercepts
    alpha_sp = cp.Variable((n_assets, N_SPREADS))
    # Shared spread coefficients
    b_shared_sp = cp.Variable((N_SPREADS, MACRO_DIM))
    d_shared_sp = cp.Variable((N_SPREADS, MICRO_DIM))
    # Per-asset deltas for spreads — flattened to 2D: (n_assets, N_SPREADS * MACRO_DIM)
    delta_b_sp_flat = cp.Variable((n_assets, N_SPREADS * MACRO_DIM))
    delta_d_sp_flat = cp.Variable((n_assets, N_SPREADS * MICRO_DIM))

    objective_terms: list[Any] = []
    constraints: list[Any] = []

    for ai, asset_id in enumerate(asset_ids):
        micro_a = np.asarray(x_micro[asset_id], dtype=np.float64)
        target_a = np.asarray(y_52w[asset_id], dtype=np.float64)
        avail_a = np.asarray(mask[asset_id], dtype=np.bool_)

        if micro_a.shape != (n_obs, MICRO_DIM):
            msg = f"{asset_id} x_micro shape mismatch"
            raise ValueError(msg)

        weights = avail_a.astype(np.float64)

        # q50 prediction for this asset
        pred_q50 = (
            macro @ (b_shared_q50 + delta_b_q50[ai])
            + micro_a @ (d_shared_q50 + delta_d_q50[ai])
            + alpha_q50[ai]
        )

        # spread predictions
        pred_spreads = []
        for si in range(N_SPREADS):
            # Index into flattened delta arrays
            b_offset = si * MACRO_DIM
            d_offset = si * MICRO_DIM
            sp = (
                macro @ (b_shared_sp[si] + delta_b_sp_flat[ai, b_offset:b_offset + MACRO_DIM])
                + micro_a @ (d_shared_sp[si] + delta_d_sp_flat[ai, d_offset:d_offset + MICRO_DIM])
                + alpha_sp[ai, si]
            )
            pred_spreads.append(sp)
            # Non-negativity: spread >= min_spread on available rows
            constraints.append(_CP.multiply(weights, sp) >= min_spread * weights)

        # Reconstruct quantiles: q10, q25, q50, q75, q90
        # spreads: [dn2, dn1, up1, up2]
        pred_q25 = pred_q50 - pred_spreads[1]  # dn1
        pred_q10 = pred_q25 - pred_spreads[0]  # dn2
        pred_q75 = pred_q50 + pred_spreads[2]  # up1
        pred_q90 = pred_q75 + pred_spreads[3]  # up2

        predictions = [pred_q10, pred_q25, pred_q50, pred_q75, pred_q90]

        for qi, tau in enumerate(INTERIOR_TAUS):
            residual = target_a - predictions[qi]
            objective_terms.append(
                _CP.sum(_CP.multiply(weights, _pinball_loss(residual, tau))),
            )

    # ── Regularization ──
    penalty = l2_alpha_macro * (
        _CP.sum_squares(b_shared_q50)
        + _CP.sum_squares(b_shared_sp)
    )
    penalty += l2_alpha_micro * (
        _CP.sum_squares(d_shared_q50)
        + _CP.sum_squares(d_shared_sp)
    )
    # Delta penalties (partial pooling shrinkage)
    penalty += l2_delta_macro * (
        _CP.sum_squares(delta_b_q50)
        + _CP.sum_squares(delta_b_sp_flat)
    )
    penalty += l2_delta_micro * (
        _CP.sum_squares(delta_d_q50)
        + _CP.sum_squares(delta_d_sp_flat)
    )

    problem = cp.Problem(cp.Minimize(_CP.sum(objective_terms) + penalty), constraints)
    try:
        with np.errstate(under="warn"):
            cast(Any, problem).solve(solver=solver)
    except cp.error.SolverError as exc:
        msg = f"R1 panel solver failed: {exc}"
        raise QuantileSolverError(msg) from exc

    if problem.status not in OPTIMAL_STATUSES:
        msg = f"R1 panel solver status: {problem.status}"
        raise QuantileSolverError(msg)

    # Extract values
    alpha_q50_dict: dict[str, float] = {}
    delta_b_q50_dict: dict[str, NDArray[np.float64]] = {}
    delta_d_q50_dict: dict[str, NDArray[np.float64]] = {}
    alpha_sp_dict: dict[str, NDArray[np.float64]] = {}
    delta_b_sp_dict: dict[str, NDArray[np.float64]] = {}
    delta_d_sp_dict: dict[str, NDArray[np.float64]] = {}

    for ai, asset_id in enumerate(asset_ids):
        alpha_q50_dict[asset_id] = float(alpha_q50.value[ai])
        delta_b_q50_dict[asset_id] = np.asarray(delta_b_q50.value[ai], dtype=np.float64)
        delta_d_q50_dict[asset_id] = np.asarray(delta_d_q50.value[ai], dtype=np.float64)
        alpha_sp_dict[asset_id] = np.asarray(alpha_sp.value[ai], dtype=np.float64)
        # Reshape flat deltas back to (N_SPREADS, DIM)
        delta_b_sp_dict[asset_id] = np.asarray(
            delta_b_sp_flat.value[ai], dtype=np.float64,
        ).reshape(N_SPREADS, MACRO_DIM)
        delta_d_sp_dict[asset_id] = np.asarray(
            delta_d_sp_flat.value[ai], dtype=np.float64,
        ).reshape(N_SPREADS, MICRO_DIM)

    return R1PanelQRCoefs(
        asset_ids=asset_ids,
        alpha_q50=MappingProxyType(alpha_q50_dict),
        b_shared_q50=np.asarray(b_shared_q50.value, dtype=np.float64),
        d_shared_q50=np.asarray(d_shared_q50.value, dtype=np.float64),
        delta_b_q50=MappingProxyType(delta_b_q50_dict),
        delta_d_q50=MappingProxyType(delta_d_q50_dict),
        alpha_spreads=MappingProxyType(alpha_sp_dict),
        b_shared_spreads=np.asarray(b_shared_sp.value, dtype=np.float64),
        d_shared_spreads=np.asarray(d_shared_sp.value, dtype=np.float64),
        delta_b_spreads=MappingProxyType(delta_b_sp_dict),
        delta_d_spreads=MappingProxyType(delta_d_sp_dict),
        solver_status=STATUS_OK,
        model_status="NORMAL",
    )


def predict_r1_interior(
    coefs: R1PanelQRCoefs,
    asset_id: str,
    x_macro_t: NDArray[np.float64],
    x_micro_t: NDArray[np.float64],
) -> tuple[NDArray[np.float64], str]:
    """pure. Predict interior quantiles for one asset at one time step."""
    macro = np.asarray(x_macro_t, dtype=np.float64)
    micro = np.asarray(x_micro_t, dtype=np.float64)
    if macro.shape != (MACRO_DIM,) or micro.shape != (MICRO_DIM,):
        msg = f"predict_r1_interior expects shapes ({MACRO_DIM},), ({MICRO_DIM},)"
        raise ValueError(msg)
    if asset_id not in coefs.alpha_q50:
        msg = f"unknown asset_id {asset_id}"
        raise KeyError(msg)

    with np.errstate(under="ignore"):
        # q50
        b_q50 = coefs.b_shared_q50 + coefs.delta_b_q50[asset_id]
        d_q50 = coefs.d_shared_q50 + coefs.delta_d_q50[asset_id]
        q50 = float(coefs.alpha_q50[asset_id] + b_q50 @ macro + d_q50 @ micro)

        # spreads
        sp_alpha = coefs.alpha_spreads[asset_id]
        sp_b = coefs.b_shared_spreads + coefs.delta_b_spreads[asset_id]
        sp_d = coefs.d_shared_spreads + coefs.delta_d_spreads[asset_id]

        raw_spreads = sp_alpha + sp_b @ macro + sp_d @ micro

    # Clip spreads to minimum (should already be non-negative from solver)
    spreads = np.maximum(raw_spreads, DEFAULT_MIN_SPREAD)

    dn2, dn1, up1, up2 = float(spreads[0]), float(spreads[1]), float(spreads[2]), float(spreads[3])
    q25 = q50 - dn1
    q10 = q25 - dn2
    q75 = q50 + up1
    q90 = q75 + up2

    interior = np.array([q10, q25, q50, q75, q90], dtype=np.float64)
    if not np.isfinite(interior).all():
        msg = f"predict_r1_interior produced non-finite values for {asset_id}"
        raise QuantileSolverError(msg)
    return interior, coefs.solver_status
