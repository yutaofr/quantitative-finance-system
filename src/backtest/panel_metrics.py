"""Pure panel evaluator metrics from SRD v8.8 sections P2.5 and P7."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, timedelta

import numpy as np
from numpy.typing import NDArray

FRIDAY = 4
MIN_RV20_WARMUP_WEEKS = 4
DELTA_13W = 13
DELTA_26W = 26
MATRIX_NDIM = 2
TAUS = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95], dtype=np.float64)
Q10_INDEX = 1
Q90_INDEX = 5
LEVEL_TOLERANCE = 1.0e-12


def _next_friday(anchor: date) -> date:
    offset = (FRIDAY - anchor.weekday()) % 7
    return anchor + timedelta(days=offset)


def _count_fridays(start: date, end: date) -> int:
    if end < start:
        return 0
    return ((end.toordinal() - start.toordinal()) // 7) + 1


def _macro_ready_week(macro_feature_registry: Mapping[str, date]) -> date:
    required_series = {
        "DGS10",
        "DGS2",
        "DGS1",
        "EFFR",
        "BAA10Y",
        "WALCL",
        "VIXCLS",
        "VXVCLS",
    }
    missing = sorted(required_series.difference(macro_feature_registry))
    if missing:
        msg = f"macro_feature_registry missing required series: {', '.join(missing)}"
        raise ValueError(msg)
    spread_ready = max(macro_feature_registry["DGS10"], macro_feature_registry["DGS2"])
    raw_anchor = max(
        macro_feature_registry["DGS10"],
        macro_feature_registry["DGS2"],
        spread_ready + timedelta(weeks=DELTA_13W),
        macro_feature_registry["DGS1"],
        macro_feature_registry["EFFR"] + timedelta(weeks=DELTA_13W),
        macro_feature_registry["BAA10Y"] + timedelta(weeks=DELTA_13W),
        macro_feature_registry["WALCL"] + timedelta(weeks=DELTA_26W),
        macro_feature_registry["VIXCLS"],
        macro_feature_registry["VXVCLS"],
    )
    return _next_friday(raw_anchor)


def _first_available_friday(asset_spec: object) -> date:
    first_available = getattr(asset_spec, "first_available_friday", None)
    if not isinstance(first_available, date):
        msg = "asset_registry values must expose first_available_friday: date"
        raise TypeError(msg)
    return first_available


def _asset_ready_week(asset_registry: Mapping[str, object]) -> date:
    if not asset_registry:
        msg = "asset_registry must contain at least one panel asset"
        raise ValueError(msg)
    warm = max(
        _first_available_friday(spec) + timedelta(weeks=MIN_RV20_WARMUP_WEEKS)
        for spec in asset_registry.values()
    )
    return _next_friday(warm)


def compute_panel_effective_start(
    asset_registry: Mapping[str, object],
    macro_feature_registry: Mapping[str, date],
    *,
    min_training_weeks: int = 312,
    embargo_weeks: int = 53,
    weekly_calendar: str = "Friday",
) -> date:
    """pure. Scan week-by-week to find the earliest valid panel acceptance date."""
    if weekly_calendar != "Friday":
        msg = "compute_panel_effective_start currently supports Friday anchors only"
        raise ValueError(msg)
    if min_training_weeks <= 0 or embargo_weeks < 0:
        msg = "training and embargo windows must be non-negative with training > 0"
        raise ValueError(msg)
    first_valid_feature_week = max(
        _asset_ready_week(asset_registry),
        _macro_ready_week(macro_feature_registry),
    )
    candidate = first_valid_feature_week
    limit = date(2100, 1, 1)
    while candidate < limit:
        train_end = candidate - timedelta(weeks=embargo_weeks)
        if _count_fridays(first_valid_feature_week, train_end) >= min_training_weeks:
            return candidate
        candidate += timedelta(weeks=1)
    msg = "panel effective start could not be found before 2100-01-01"
    raise ValueError(msg)


def _coerce_quantiles(quantiles: NDArray[np.float64]) -> NDArray[np.float64]:
    values = np.asarray(quantiles, dtype=np.float64)
    if values.ndim == 1:
        if values.shape != (TAUS.shape[0],):
            msg = "quantiles must have shape (7,) or (n_obs, 7)"
            raise ValueError(msg)
        return values.reshape(1, -1)
    if values.ndim != MATRIX_NDIM or values.shape[1] != TAUS.shape[0]:
        msg = "quantiles must have shape (7,) or (n_obs, 7)"
        raise ValueError(msg)
    return values


def _coerce_realized(realized: NDArray[np.float64], rows: int) -> NDArray[np.float64]:
    values = np.asarray(realized, dtype=np.float64)
    if values.ndim == 0:
        return values.reshape(1)
    if values.ndim != 1 or values.shape[0] != rows:
        msg = "realized values must align to quantile rows"
        raise ValueError(msg)
    return values


def per_asset_crps(
    quantiles: NDArray[np.float64],
    realized: NDArray[np.float64],
    asset_id: str,
) -> float:
    """pure. Compute mean CRPS proxy for one asset on finite rows only."""
    q = _coerce_quantiles(quantiles)
    y = _coerce_realized(realized, q.shape[0])
    finite = np.isfinite(y) & np.isfinite(q).all(axis=1)
    if not finite.any():
        msg = f"asset {asset_id} has no finite rows for CRPS"
        raise ValueError(msg)
    residual = y[finite, None] - q[finite]
    losses = np.maximum(TAUS * residual, (TAUS - 1.0) * residual)
    return float(2.0 * np.mean(losses))


def per_asset_coverage(
    quantiles: NDArray[np.float64],
    realized: NDArray[np.float64],
    level: float,
) -> float:
    """pure. Compute one-sided empirical coverage for q10 or q90."""
    q = _coerce_quantiles(quantiles)
    y = _coerce_realized(realized, q.shape[0])
    if abs(level - 0.10) < LEVEL_TOLERANCE:
        idx = Q10_INDEX
    elif abs(level - 0.90) < LEVEL_TOLERANCE:
        idx = Q90_INDEX
    else:
        msg = "level must be 0.10 or 0.90"
        raise ValueError(msg)
    finite = np.isfinite(y) & np.isfinite(q[:, idx])
    if not finite.any():
        msg = "coverage requires at least one finite row"
        raise ValueError(msg)
    return float(np.mean(y[finite] <= q[finite, idx]))


def panel_aggregate_crps(per_asset_crps_values: Mapping[str, float] | Sequence[float]) -> float:
    """pure. Equal-weight aggregate CRPS across finite asset metrics."""
    if isinstance(per_asset_crps_values, Mapping):
        values = np.asarray(tuple(per_asset_crps_values.values()), dtype=np.float64)
    else:
        values = np.asarray(tuple(per_asset_crps_values), dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        msg = "panel aggregate CRPS requires at least one finite asset metric"
        raise ValueError(msg)
    return float(np.mean(finite))


def vol_normalized_crps(crps: float, sigma: float) -> float:
    """pure. Normalize CRPS by the corresponding expanding return volatility."""
    if not np.isfinite(crps) or not np.isfinite(sigma) or sigma <= 0.0:
        msg = "vol_normalized_crps requires finite crps and strictly positive sigma"
        raise ValueError(msg)
    return float(crps / sigma)


def effective_asset_weeks(availability: Mapping[str, NDArray[np.bool_]]) -> int:
    """pure. Count effective asset-weeks across the fixed panel."""
    return int(
        sum(
            int(np.sum(np.asarray(values, dtype=np.bool_)))
            for values in availability.values()
        ),
    )
