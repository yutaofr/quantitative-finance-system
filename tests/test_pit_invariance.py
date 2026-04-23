from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from data_contract.yahoo_client import log_return_series
from engine_types import TimeSeries

TRADING_DAYS_PER_YEAR = 252.0


def _price_series(values: list[float]) -> TimeSeries:
    start = date(2024, 1, 2)
    return TimeSeries(
        series_id="QQQ",
        timestamps=np.asarray(
            [(start + timedelta(days=idx)).isoformat() for idx in range(len(values))],
            dtype="datetime64[D]",
        ),
        values=np.asarray(values, dtype=np.float64),
        is_pseudo_pit=False,
    )


def test_adjusted_close_log_returns_are_invariant_to_multiplicative_back_adjustment() -> None:
    prices = _price_series([100.0, 102.0, 101.0, 105.0, 106.0])
    scaled = _price_series([value * 1.2345 for value in prices.values.tolist()])

    first = log_return_series(prices)
    second = log_return_series(scaled)

    assert np.allclose(first.values, second.values)
    assert np.isclose(
        np.log(prices.values[-1] / prices.values[0]),
        np.log(scaled.values[-1] / scaled.values[0]),
    )
    assert np.isclose(
        np.sqrt(TRADING_DAYS_PER_YEAR * np.var(first.values, ddof=0)),
        np.sqrt(TRADING_DAYS_PER_YEAR * np.var(second.values, ddof=0)),
    )
