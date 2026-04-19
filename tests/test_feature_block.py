from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from engine_types import TimeSeries
from features.block_builder import FEATURE_NAMES, build_feature_block


def _weekly_series(series_id: str, values: list[float]) -> TimeSeries:
    start = date(2024, 1, 5)
    dates = [start + timedelta(weeks=idx) for idx in range(len(values))]
    return _dated_series(series_id, dates, values)


def _dated_series(series_id: str, dates: list[date], values: list[float]) -> TimeSeries:
    timestamps = np.array(
        [np.datetime64(item) for item in dates],
        dtype="datetime64[D]",
    )
    return TimeSeries(
        series_id=series_id,
        timestamps=timestamps,
        values=np.array(values, dtype=np.float64),
        is_pseudo_pit=False,
    )


def _series_map() -> dict[str, TimeSeries]:
    return {
        "DGS10": _weekly_series("DGS10", [10.0 + idx for idx in range(27)]),
        "DGS2": _weekly_series("DGS2", [2.0 + idx for idx in range(27)]),
        "DGS1": _weekly_series("DGS1", [1.0 + idx for idx in range(27)]),
        "EFFR": _weekly_series("EFFR", [3.0 + idx for idx in range(27)]),
        "BAA10Y": _weekly_series("BAA10Y", [5.0 + idx for idx in range(27)]),
        "WALCL": _weekly_series("WALCL", [100.0 + idx for idx in range(27)]),
        "VXNCLS": _weekly_series("VXNCLS", [20.0 + idx for idx in range(27)]),
        "RV20_NDX": _weekly_series("RV20_NDX", [10.0 + idx for idx in range(27)]),
        "VIXCLS": _weekly_series("VIXCLS", [15.0 + idx for idx in range(27)]),
        "VXVCLS": _weekly_series("VXVCLS", [12.0 + idx for idx in range(27)]),
    }


def test_feature_names_match_srd_section_6() -> None:
    assert FEATURE_NAMES == ("x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10")


@pytest.mark.synthetic
def test_build_feature_block_computes_the_ten_frozen_formulas() -> None:
    series = _series_map()

    x_raw, missing = build_feature_block(series, date(2024, 7, 5))

    assert not missing.any()
    assert np.allclose(
        x_raw,
        np.array(
            [
                8.0,
                0.0,
                27.0,
                13.0,
                31.0,
                13.0,
                np.log(126.0) - np.log(100.0),
                np.log(46.0),
                np.log(46.0) - np.log(36.0),
                np.log(41.0 / 38.0),
            ],
            dtype=np.float64,
        ),
    )


@pytest.mark.synthetic
def test_build_feature_block_marks_missing_without_cross_week_interpolation() -> None:
    series = _series_map()
    series["EFFR"] = _weekly_series("EFFR", [3.0 + idx for idx in range(26)])

    x_raw, missing = build_feature_block(series, date(2024, 7, 5))

    assert missing[3]
    assert np.isnan(x_raw[3])
    assert not missing[0]


@pytest.mark.synthetic
def test_build_feature_block_lags_by_calendar_week_not_prior_observation_count() -> None:
    series = _series_map()
    effr_dates = [date(2024, 1, 5) + timedelta(weeks=idx) for idx in range(27)]
    effr_values = [3.0 + idx for idx in range(27)]
    missing_lag_date = date(2024, 4, 5)
    sparse_dates = [item for item in effr_dates if item != missing_lag_date]
    sparse_values = [
        value
        for item, value in zip(effr_dates, effr_values, strict=True)
        if item != missing_lag_date
    ]
    series["EFFR"] = _dated_series("EFFR", sparse_dates, sparse_values)

    x_raw, missing = build_feature_block(series, date(2024, 7, 5))

    assert missing[3]
    assert np.isnan(x_raw[3])


@pytest.mark.synthetic
def test_build_feature_block_accepts_previous_trading_day_within_same_week() -> None:
    series = _series_map()
    dgs1_dates = [date(2024, 1, 5) + timedelta(weeks=idx) for idx in range(26)]
    dgs1_dates.append(date(2024, 7, 4))
    series["DGS1"] = _dated_series("DGS1", dgs1_dates, [1.0 + idx for idx in range(27)])

    x_raw, missing = build_feature_block(series, date(2024, 7, 5))

    assert not missing[2]
    assert x_raw[2] == 27.0
