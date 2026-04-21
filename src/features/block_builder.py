"""Frozen production feature block from SRD v8.7 section 6."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, timedelta

import numpy as np
from numpy.typing import NDArray

from engine_types import TimeSeries

FEATURE_NAMES = ("x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10")
LAG_13W = 13
LAG_26W = 26


def _as_numpy_date(as_of: date) -> np.datetime64:
    return np.datetime64(as_of, "D")


def _index_at(series: TimeSeries, as_of: date) -> int | None:
    target = _as_numpy_date(as_of)
    week_start = _as_numpy_date(as_of - timedelta(days=6))
    timestamps = series.timestamps.astype("datetime64[D]")
    last_idx = int(np.searchsorted(timestamps, target, side="right")) - 1
    if last_idx < 0 or timestamps[last_idx] < week_start:
        return None
    return last_idx


def _value_at(series: TimeSeries, as_of: date) -> float:
    idx = _index_at(series, as_of)
    if idx is None:
        return float("nan")
    return float(series.values[idx])


def _delta_at(series: TimeSeries, as_of: date, lag_weeks: int) -> float:
    current = _value_at(series, as_of)
    lagged = _value_at(series, as_of - timedelta(weeks=lag_weeks))
    if not np.isfinite(current) or not np.isfinite(lagged):
        return float("nan")
    return current - lagged


def _spread(series: Mapping[str, TimeSeries], as_of: date) -> float:
    return _value_at(series["DGS10"], as_of) - _value_at(series["DGS2"], as_of)


def _spread_delta_13w(series: Mapping[str, TimeSeries], as_of: date) -> float:
    current = _spread(series, as_of)
    lagged = _spread(series, as_of - timedelta(weeks=LAG_13W))
    if not np.isfinite(current) or not np.isfinite(lagged):
        return float("nan")
    return current - lagged


def _safe_log(value: float) -> float:
    if not np.isfinite(value) or value <= 0.0:
        return float("nan")
    return float(np.log(value))


def build_feature_block(
    series: Mapping[str, TimeSeries],
    as_of: date,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """pure. Build SRD §6 raw 10-feature vector and missing mask."""
    vxn = _value_at(series["VXNCLS"], as_of)
    rv20 = _value_at(series["RV20_NDX"], as_of)
    vix = _value_at(series["VIXCLS"], as_of)
    vxv = _value_at(series["VXVCLS"], as_of)
    walcl = _value_at(series["WALCL"], as_of)
    walcl_delta = _delta_at(series["WALCL"], as_of, LAG_26W)

    raw = np.array(
        [
            _spread(series, as_of),
            _spread_delta_13w(series, as_of),
            _value_at(series["DGS1"], as_of),
            _delta_at(series["EFFR"], as_of, LAG_13W),
            _value_at(series["BAA10Y"], as_of),
            _delta_at(series["BAA10Y"], as_of, LAG_13W),
            _safe_log(walcl) - _safe_log(walcl - walcl_delta),
            _safe_log(vxn),
            _safe_log(vxn) - _safe_log(rv20),
            _safe_log(vix / vxv) if np.isfinite(vxv) and vxv > 0.0 else float("nan"),
        ],
        dtype=np.float64,
    )
    return raw, ~np.isfinite(raw)
