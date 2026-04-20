from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from data_contract.derived_series import derive_rv20_nasdaq100
from engine_types import TimeSeries


def _daily_prices(start: date, values: list[float]) -> TimeSeries:
    return TimeSeries(
        series_id="NASDAQXNDX",
        timestamps=np.array(
            [(start + timedelta(days=idx)).isoformat() for idx in range(len(values))],
            dtype="datetime64[D]",
        ),
        values=np.asarray(values, dtype=np.float64),
        is_pseudo_pit=False,
    )


def test_derive_rv20_uses_only_prices_at_or_before_as_of() -> None:
    start = date(2024, 11, 1)
    prices = [100.0 + idx for idx in range(35)]
    source = _daily_prices(start, prices)
    future_source = _daily_prices(start, [*prices, 1000.0, 2000.0])

    as_of = start + timedelta(days=34)
    first = derive_rv20_nasdaq100(source, as_of)
    second = derive_rv20_nasdaq100(future_source, as_of)

    assert first.timestamps.tolist() == second.timestamps.tolist()
    assert np.allclose(first.values, second.values)
    assert first.series_id == "RV20_NDX"
    assert first.values[-1] > 0.0


def test_derive_rv20_emits_only_friday_close_points() -> None:
    start = date(2024, 11, 1)
    source = _daily_prices(start, [100.0 + idx for idx in range(35)])

    rv20 = derive_rv20_nasdaq100(source, start + timedelta(days=34))

    weekdays = [
        date.fromisoformat(str(ts)).weekday()
        for ts in rv20.timestamps.astype("datetime64[D]").astype(str).tolist()
    ]
    assert set(weekdays) == {4}
