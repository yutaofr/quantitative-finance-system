from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from engine_types import TimeSeries
from features.block_builder import build_feature_block
from features.panel_block_builder import (
    build_panel_feature_block,
    build_panel_hmm_inputs,
    fit_panel_hmm,
)
from state.ti_hmm_single import HMMModel

START_WEEK = date(2020, 1, 3)
WEEKS = 140
DAYS = 1200


def _weekly_series(series_id: str, base: float, slope: float) -> TimeSeries:
    dates = [START_WEEK + timedelta(weeks=idx) for idx in range(WEEKS)]
    values = [base + slope * idx for idx in range(WEEKS)]
    return TimeSeries(
        series_id=series_id,
        timestamps=np.asarray([item.isoformat() for item in dates], dtype="datetime64[D]"),
        values=np.asarray(values, dtype=np.float64),
        is_pseudo_pit=False,
    )


def _daily_price_series(series_id: str, level: float, drift: float) -> TimeSeries:
    start = START_WEEK - timedelta(days=4)
    dates = [start + timedelta(days=idx) for idx in range(DAYS)]
    t = np.arange(DAYS, dtype=np.float64)
    prices = level * np.exp(drift * t + 0.01 * np.sin(t / 20.0))
    return TimeSeries(
        series_id=series_id,
        timestamps=np.asarray([item.isoformat() for item in dates], dtype="datetime64[D]"),
        values=prices.astype(np.float64),
        is_pseudo_pit=False,
    )


def _macro_series(*, vix_shift: float = 0.0) -> dict[str, TimeSeries]:
    return {
        "DGS10": _weekly_series("DGS10", 2.0, 0.02),
        "DGS2": _weekly_series("DGS2", 1.0, 0.01),
        "DGS1": _weekly_series("DGS1", 0.5, 0.005),
        "EFFR": _weekly_series("EFFR", 1.5, 0.01),
        "BAA10Y": _weekly_series("BAA10Y", 2.5, 0.015),
        "WALCL": _weekly_series("WALCL", 1000.0, 2.0),
        "VIXCLS": _weekly_series("VIXCLS", 18.0 + vix_shift, 0.03),
        "VXVCLS": _weekly_series("VXVCLS", 16.0, 0.02),
        "VXNCLS": _weekly_series("VXNCLS", 22.0, 0.04),
        "RV20_NDX": _weekly_series("RV20_NDX", 14.0, 0.02),
    }


def _asset_series(
    *,
    r2k_primary_missing_until: int = 20,
    r2k_level: float = 115.0,
) -> dict[str, dict[str, TimeSeries]]:
    rvx_dates = [
        START_WEEK + timedelta(weeks=idx)
        for idx in range(r2k_primary_missing_until, WEEKS)
    ]
    rvx_values = [24.0 + 0.05 * idx for idx in range(len(rvx_dates))]
    rvx = TimeSeries(
        series_id="RVXCLS",
        timestamps=np.asarray([item.isoformat() for item in rvx_dates], dtype="datetime64[D]"),
        values=np.asarray(rvx_values, dtype=np.float64),
        is_pseudo_pit=False,
    )
    return {
        "NASDAQXNDX": {
            "target": _daily_price_series("QQQ", 100.0, 0.0008),
            "vol": _weekly_series("VXNCLS", 22.0, 0.04),
        },
        "SPX": {
            "target": _daily_price_series("SPY", 120.0, 0.0005),
            "vol": _weekly_series("VIXCLS", 18.0, 0.03),
        },
        "R2K": {
            "target": _daily_price_series("IWM", r2k_level, 0.0006),
            "vol": rvx,
            "vol_fallback": _weekly_series("VIXCLS", 18.0, 0.03),
        },
    }


def test_build_panel_feature_block_reuses_v87_macro_formulas() -> None:
    as_of = START_WEEK + timedelta(weeks=80)
    macro = _macro_series()
    assets = _asset_series()
    frame = build_panel_feature_block(macro, assets, as_of)
    reference_raw, _missing = build_feature_block(macro, as_of)

    assert np.allclose(frame.x_macro_raw[-1], reference_raw[:7])


def test_panel_micro_features_are_per_asset_scaled_without_cross_sectional_normalization() -> None:
    as_of = START_WEEK + timedelta(weeks=90)
    macro = _macro_series()
    base_frame = build_panel_feature_block(macro, _asset_series(r2k_level=115.0), as_of)
    stressed_frame = build_panel_feature_block(macro, _asset_series(r2k_level=250.0), as_of)

    assert np.allclose(
        base_frame.x_micro["NASDAQXNDX"],
        stressed_frame.x_micro["NASDAQXNDX"],
        equal_nan=True,
    )
    assert np.allclose(
        base_frame.x_micro["SPX"],
        stressed_frame.x_micro["SPX"],
        equal_nan=True,
    )
    finite = np.isfinite(base_frame.x_micro["R2K"])
    assert np.all(np.abs(base_frame.x_micro["R2K"][finite]) <= 5.0 + 1.0e-12)


def test_panel_micro_features_use_proxy_fallback_when_primary_vol_is_missing() -> None:
    as_of = START_WEEK + timedelta(weeks=60)
    frame = build_panel_feature_block(
        _macro_series(),
        _asset_series(r2k_primary_missing_until=30),
        as_of,
    )

    assert "proxy" in frame.micro_feature_mode["R2K"][:30]
    assert "primary" in frame.micro_feature_mode["R2K"][35:]


def test_hmm_label_ordering_spx_anchored(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, np.ndarray] = {}

    def fake_fit_hmm(  # noqa: PLR0913
        y_obs: np.ndarray,
        h: np.ndarray,
        rng: np.random.Generator,
        *,
        forward_52w_returns: np.ndarray | None = None,
        max_iter: int = 200,
        tolerance: float = 1.0e-6,
        restarts: int = 50,
        transition_max_iter: int = 200,
    ) -> HMMModel:
        del y_obs, h, rng, max_iter, tolerance, restarts, transition_max_iter
        captured["forward"] = np.asarray(forward_52w_returns, dtype=np.float64)
        return HMMModel(
            transition_coefs=np.zeros((3, 3), dtype=np.float64),
            emission_mean=np.zeros((3, 6), dtype=np.float64),
            emission_cov=np.tile(np.eye(6, dtype=np.float64), (3, 1, 1)),
            label_map={0: "DEFENSIVE", 1: "NEUTRAL", 2: "OFFENSIVE"},
            log_likelihood=0.0,
        )

    monkeypatch.setattr("features.panel_block_builder.fit_hmm", fake_fit_hmm)

    as_of = START_WEEK + timedelta(weeks=70)
    macro = _macro_series()
    frame = build_panel_feature_block(macro, _asset_series(), as_of)
    inputs = build_panel_hmm_inputs(frame, macro)
    fit_panel_hmm(frame, macro, np.random.default_rng(123))

    qqq_anchor = frame.target_returns["NASDAQXNDX"][
        [frame.feature_dates.index(item) for item in inputs.observation_dates]
    ]
    assert np.allclose(captured["forward"], inputs.label_anchor_returns)
    assert not np.allclose(captured["forward"], qqq_anchor)


def test_panel_hmm_uses_vix_input_not_asset_specific_vol() -> None:
    as_of = START_WEEK + timedelta(weeks=70)
    assets = _asset_series()
    base_inputs = build_panel_hmm_inputs(
        build_panel_feature_block(_macro_series(), assets, as_of),
        _macro_series(),
    )
    shifted_vix_inputs = build_panel_hmm_inputs(
        build_panel_feature_block(_macro_series(vix_shift=10.0), assets, as_of),
        _macro_series(vix_shift=10.0),
    )

    assert not np.allclose(base_inputs.y_obs[:, 4:], shifted_vix_inputs.y_obs[:, 4:])
