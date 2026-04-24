"""Pure v8.8R1 panel quantile regression — q50 + spreads parameterization.

Design:
  - Predicts q50 + 4 non-negative spreads per asset
  - q75 = q50 + Δup1, q90 = q75 + Δup2
  - q25 = q50 - Δdn1, q10 = q25 - Δdn2
  - Non-crossing guaranteed by spread >= 0 constraint (no rearrangement needed)
  - Partial pooling: b_shared + delta_b per asset, d_shared + delta_d per asset
  - No HMM π input — macro + micro only

v8.8R1-rectified:
  - q50 equation: signed X_macro, X_micro  (unchanged)
  - spread equations: abs(X_macro), abs(X_micro)  ← absolute rectification
    Rationale: rectified inputs eliminate feature polarity conflict under
    the spread >= 0 non-negativity constraint. Solver can now learn positive
    coefficients that scale with feature magnitude regardless of sign direction.
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
DEFAULT_L2_ALPHA_SPREAD = 0.1
DEFAULT_L2_DELTA = 0.5
DEFAULT_MIN_SPREAD = 1.0e-4
OPTIMAL_STATUSES = {"optimal", "optimal_inaccurate"}
STATUS_OK = "ok"
STATUS_FALLBACK = "per_asset_fallback"
SPREAD_FEATURE_TRANSFORM = "abs_rectified"


@dataclass(frozen=True, slots=True)
class R1PanelQRCoefs:
    """pure. v8.8R1 panel QR coefficients with q50+spreads parameterization."""

    asset_ids: tuple[str, ...]
    # q50 coefficients (use signed features)
    alpha_q50: Mapping[str, float]
    b_shared_q50: NDArray[np.float64]       # (MACRO_DIM,)
    d_shared_q50: NDArray[np.float64]       # (MICRO_DIM,)
    delta_b_q50: Mapping[str, NDArray[np.float64]]  # per-asset (MACRO_DIM,)
    delta_d_q50: Mapping[str, NDArray[np.float64]]  # per-asset (MICRO_DIM,)
    # spread coefficients (use rectified |features|)
    alpha_spreads: Mapping[str, NDArray[np.float64]]  # per-asset (4,)
    b_shared_spreads: NDArray[np.float64]   # (4, MACRO_DIM)  — fitted on |X_macro|
    d_shared_spreads: NDArray[np.float64]   # (4, MICRO_DIM)  — fitted on |X_micro|
    delta_b_spreads: Mapping[str, NDArray[np.float64]]  # per-asset (4, MACRO_DIM)
    delta_d_spreads: Mapping[str, NDArray[np.float64]]  # per-asset (4, MICRO_DIM)
    solver_status: str
    model_status: str
    # Polarity audit metadata
    spread_feature_transform: str           # "abs_rectified" or "signed"
    feature_polarity_conflict_tested: bool
    b_shared_sp_l1_mean: float              # mean(|b_shared_sp|) — slope activity
    d_shared_sp_l1_mean: float              # mean(|d_shared_sp|) — slope activity


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
    l2_alpha_spread_macro: float = DEFAULT_L2_ALPHA_SPREAD,
    l2_alpha_spread_micro: float = DEFAULT_L2_ALPHA_SPREAD,
    l2_delta_macro: float = DEFAULT_L2_DELTA,
    l2_delta_micro: float = DEFAULT_L2_DELTA,
    min_spread: float = DEFAULT_MIN_SPREAD,
    solver: str = "ECOS",
) -> R1PanelQRCoefs:
    """pure. Fit v8.8R1 panel quantiles with abs-rectified spread inputs."""
    macro = _validate_inputs(x_macro, x_micro, y_52w, mask)
    asset_ids = tuple(x_micro)
    n_obs = macro.shape[0]
    n_assets = len(asset_ids)

    # ── Rectified inputs for spread equations ──────────────────────────────
    # q50 uses signed macro; spreads use |macro| to eliminate polarity conflict
    macro_abs = np.abs(macro)

    # ── Decision variables (all 2D or 1D to avoid CVXPY 3D issues) ────────
    # q50: signed features
    alpha_q50 = cp.Variable(n_assets)
    b_shared_q50 = cp.Variable(MACRO_DIM)
    d_shared_q50 = cp.Variable(MICRO_DIM)
    delta_b_q50 = cp.Variable((n_assets, MACRO_DIM))
    delta_d_q50 = cp.Variable((n_assets, MICRO_DIM))

    # spreads: rectified |features|
    alpha_sp = cp.Variable((n_assets, N_SPREADS))
    b_shared_sp = cp.Variable((N_SPREADS, MACRO_DIM))
    d_shared_sp = cp.Variable((N_SPREADS, MICRO_DIM))
    # Per-asset deltas for spreads — flattened to 2D: (n_assets, N_SPREADS * DIM)
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
        # Rectified micro for spread equations
        micro_abs_a = np.abs(micro_a)

        # q50 prediction — signed features
        pred_q50 = (
            macro @ (b_shared_q50 + delta_b_q50[ai])
            + micro_a @ (d_shared_q50 + delta_d_q50[ai])
            + alpha_q50[ai]
        )

        # spread predictions — rectified |features|
        pred_spreads = []
        for si in range(N_SPREADS):
            b_offset = si * MACRO_DIM
            d_offset = si * MICRO_DIM
            sp = (
                macro_abs @ (b_shared_sp[si] + delta_b_sp_flat[ai, b_offset:b_offset + MACRO_DIM])
                + micro_abs_a @ (d_shared_sp[si] + delta_d_sp_flat[ai, d_offset:d_offset + MICRO_DIM])
                + alpha_sp[ai, si]
            )
            pred_spreads.append(sp)
            # Non-negativity: constraint on the full prediction (intercept + Xb)
            constraints.append(_CP.multiply(weights, sp) >= min_spread * weights)

        # Reconstruct quantiles: spreads order = [dn2, dn1, up1, up2]
        pred_q25 = pred_q50 - pred_spreads[1]   # dn1
        pred_q10 = pred_q25 - pred_spreads[0]   # dn2
        pred_q75 = pred_q50 + pred_spreads[2]   # up1
        pred_q90 = pred_q75 + pred_spreads[3]   # up2

        predictions = [pred_q10, pred_q25, pred_q50, pred_q75, pred_q90]

        for qi, tau in enumerate(INTERIOR_TAUS):
            residual = target_a - predictions[qi]
            objective_terms.append(
                _CP.sum(_CP.multiply(weights, _pinball_loss(residual, tau))),
            )

    # ── Regularization (q50 shared and spread shared separated) ───────────
    # q50 shared (signed features) — stronger prior
    penalty = l2_alpha_macro * _CP.sum_squares(b_shared_q50)
    penalty += l2_alpha_micro * _CP.sum_squares(d_shared_q50)
    # spread shared (rectified features) — lighter prior, slope can grow freely
    penalty += l2_alpha_spread_macro * _CP.sum_squares(b_shared_sp)
    penalty += l2_alpha_spread_micro * _CP.sum_squares(d_shared_sp)
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
        delta_b_sp_dict[asset_id] = np.asarray(
            delta_b_sp_flat.value[ai], dtype=np.float64,
        ).reshape(N_SPREADS, MACRO_DIM)
        delta_d_sp_dict[asset_id] = np.asarray(
            delta_d_sp_flat.value[ai], dtype=np.float64,
        ).reshape(N_SPREADS, MICRO_DIM)

    b_sp_val = np.asarray(b_shared_sp.value, dtype=np.float64)
    d_sp_val = np.asarray(d_shared_sp.value, dtype=np.float64)
    b_l1_mean = float(np.mean(np.abs(b_sp_val)))
    d_l1_mean = float(np.mean(np.abs(d_sp_val)))

    return R1PanelQRCoefs(
        asset_ids=asset_ids,
        alpha_q50=MappingProxyType(alpha_q50_dict),
        b_shared_q50=np.asarray(b_shared_q50.value, dtype=np.float64),
        d_shared_q50=np.asarray(d_shared_q50.value, dtype=np.float64),
        delta_b_q50=MappingProxyType(delta_b_q50_dict),
        delta_d_q50=MappingProxyType(delta_d_q50_dict),
        alpha_spreads=MappingProxyType(alpha_sp_dict),
        b_shared_spreads=b_sp_val,
        d_shared_spreads=d_sp_val,
        delta_b_spreads=MappingProxyType(delta_b_sp_dict),
        delta_d_spreads=MappingProxyType(delta_d_sp_dict),
        solver_status=STATUS_OK,
        model_status="NORMAL",
        spread_feature_transform=SPREAD_FEATURE_TRANSFORM,
        feature_polarity_conflict_tested=True,
        b_shared_sp_l1_mean=b_l1_mean,
        d_shared_sp_l1_mean=d_l1_mean,
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
        # q50 — signed features
        b_q50 = coefs.b_shared_q50 + coefs.delta_b_q50[asset_id]
        d_q50 = coefs.d_shared_q50 + coefs.delta_d_q50[asset_id]
        q50 = float(coefs.alpha_q50[asset_id] + b_q50 @ macro + d_q50 @ micro)

        # spreads — rectified |features| (consistent with training)
        macro_abs = np.abs(macro)
        micro_abs = np.abs(micro)
        sp_alpha = coefs.alpha_spreads[asset_id]
        sp_b = coefs.b_shared_spreads + coefs.delta_b_spreads[asset_id]
        sp_d = coefs.d_shared_spreads + coefs.delta_d_spreads[asset_id]
        raw_spreads = sp_alpha + sp_b @ macro_abs + sp_d @ micro_abs

    # Clip to minimum
    spreads = np.maximum(raw_spreads, DEFAULT_MIN_SPREAD)

    dn2, dn1, up1, up2 = (
        float(spreads[0]), float(spreads[1]), float(spreads[2]), float(spreads[3]),
    )
    q25 = q50 - dn1
    q10 = q25 - dn2
    q75 = q50 + up1
    q90 = q75 + up2

    interior = np.array([q10, q25, q50, q75, q90], dtype=np.float64)
    if not np.isfinite(interior).all():
        msg = f"predict_r1_interior produced non-finite values for {asset_id}"
        raise QuantileSolverError(msg)
    return interior, coefs.solver_status
