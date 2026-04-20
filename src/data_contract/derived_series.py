"""Derived data-contract series computed before features enter the pure core."""

from __future__ import annotations

from datetime import date

import numpy as np

from engine_types import TimeSeries

TRADING_DAYS_PER_YEAR = 252.0
RV20_WINDOW = 20
FRIDAY = 4


def derive_rv20_nasdaq100(
    price_series: TimeSeries,
    as_of: date,
    *,
    window: int = RV20_WINDOW,
) -> TimeSeries:
    """pure. Derive PIT RV20_NDX from trailing log returns through as_of."""
    timestamps = price_series.timestamps.astype("datetime64[D]")
    values = np.asarray(price_series.values, dtype=np.float64)
    mask = timestamps <= np.datetime64(as_of, "D")
    pit_timestamps = timestamps[mask]
    pit_values = values[mask]
    if pit_values.shape[0] <= window:
        return TimeSeries(
            series_id="RV20_NDX",
            timestamps=np.array([], dtype="datetime64[D]"),
            values=np.array([], dtype=np.float64),
            is_pseudo_pit=price_series.is_pseudo_pit,
        )
    if not np.isfinite(pit_values).all() or np.any(pit_values <= 0.0):
        msg = "NASDAQXNDX prices must be finite and positive"
        raise ValueError(msg)

    log_returns = np.diff(np.log(pit_values))
    rv_timestamps: list[np.datetime64] = []
    rv_values: list[float] = []
    for end_idx in range(window, log_returns.shape[0] + 1):
        ts = pit_timestamps[end_idx]
        weekday = date.fromisoformat(str(ts)).weekday()
        if weekday != FRIDAY:
            continue
        window_returns = log_returns[end_idx - window : end_idx]
        variance = float(np.mean(window_returns * window_returns))
        rv_timestamps.append(ts)
        rv_values.append(100.0 * float(np.sqrt(TRADING_DAYS_PER_YEAR * variance)))
    return TimeSeries(
        series_id="RV20_NDX",
        timestamps=np.asarray(rv_timestamps, dtype="datetime64[D]"),
        values=np.asarray(rv_values, dtype=np.float64),
        is_pseudo_pit=price_series.is_pseudo_pit,
    )
