"""Pure panel feature construction from SRD v8.8 sections P2-P4."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
from numpy.typing import NDArray

from engine_types import TimeSeries
from errors import HMMConvergenceError
from features.pca import robust_pca_2d
from features.scaling import robust_zscore_expanding, soft_squash_clip
from state.ti_hmm_single import HMMModel, fit_hmm

FRIDAY = 4
MACRO_DIM = 7
MICRO_DIM = 3
PC_DIM = 2
OBS_DIM = 6
MIN_HMM_ROWS = 2
FORECAST_HORIZON_WEEKS = 52
LAG_13W = 13
LAG_26W = 26
LAG_4W = 4
RV20_WINDOW = 20
RV52_WINDOW = 52
TRADING_DAYS_PER_YEAR = 252.0
WEEKS_PER_YEAR = 52.0
TARGET_KEY = "target"
PRIMARY_VOL_KEY = "vol"
FALLBACK_VOL_KEY = "vol_fallback"


@dataclass(frozen=True, slots=True)
class PanelFeatureFrame:
    """pure. Shared macro plus per-asset micro feature histories."""

    as_of: date
    feature_dates: tuple[date, ...]
    x_macro_raw: NDArray[np.float64]
    x_macro: NDArray[np.float64]
    x_micro_raw: Mapping[str, NDArray[np.float64]]
    x_micro: Mapping[str, NDArray[np.float64]]
    macro_mask: NDArray[np.bool_]
    micro_mask: Mapping[str, NDArray[np.bool_]]
    asset_availability: Mapping[str, NDArray[np.bool_]]
    micro_feature_mode: Mapping[str, tuple[str, ...]]
    available_assets: tuple[str, ...]
    target_returns: Mapping[str, NDArray[np.float64]]


@dataclass(frozen=True, slots=True)
class PanelHMMInputs:
    """pure. Shared panel HMM observation history and SPX anchor returns."""

    observation_dates: tuple[date, ...]
    y_obs: NDArray[np.float64]
    h: NDArray[np.float64]
    label_anchor_returns: NDArray[np.float64]


def _as_numpy_date(value: date) -> np.datetime64:
    return np.datetime64(value, "D")


def _weekly_dates_from_series(series: Mapping[str, TimeSeries], as_of: date) -> tuple[date, ...]:
    timestamps = series["DGS10"].timestamps.astype("datetime64[D]")
    filtered = timestamps[timestamps <= _as_numpy_date(as_of)]
    return tuple(
        item
        for item in (date.fromisoformat(str(value)) for value in filtered.tolist())
        if item.weekday() == FRIDAY
    )


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


def _safe_log(value: float) -> float:
    if not np.isfinite(value) or value <= 0.0:
        return float("nan")
    return float(np.log(value))


def _scale_column(values: NDArray[np.float64]) -> NDArray[np.float64]:
    out = np.full(values.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(values)
    if finite.any():
        out[finite] = soft_squash_clip(robust_zscore_expanding(values[finite]))
    return out


def _scale_matrix(values: NDArray[np.float64]) -> NDArray[np.float64]:
    scaled = np.full(values.shape, np.nan, dtype=np.float64)
    for idx in range(values.shape[1]):
        scaled[:, idx] = _scale_column(values[:, idx])
    return scaled


def _spread(macro_series: Mapping[str, TimeSeries], as_of: date) -> float:
    return _value_at(macro_series["DGS10"], as_of) - _value_at(macro_series["DGS2"], as_of)


def _spread_delta_13w(macro_series: Mapping[str, TimeSeries], as_of: date) -> float:
    current = _spread(macro_series, as_of)
    lagged = _spread(macro_series, as_of - timedelta(weeks=LAG_13W))
    if not np.isfinite(current) or not np.isfinite(lagged):
        return float("nan")
    return current - lagged


def _macro_row(macro_series: Mapping[str, TimeSeries], as_of: date) -> NDArray[np.float64]:
    walcl = _value_at(macro_series["WALCL"], as_of)
    walcl_delta = _delta_at(macro_series["WALCL"], as_of, LAG_26W)
    return np.array(
        [
            _spread(macro_series, as_of),
            _spread_delta_13w(macro_series, as_of),
            _value_at(macro_series["DGS1"], as_of),
            _delta_at(macro_series["EFFR"], as_of, LAG_13W),
            _value_at(macro_series["BAA10Y"], as_of),
            _delta_at(macro_series["BAA10Y"], as_of, LAG_13W),
            _safe_log(walcl) - _safe_log(walcl - walcl_delta),
        ],
        dtype=np.float64,
    )


def _forward_52w_return(series: TimeSeries, as_of: date) -> float:
    current = _value_at(series, as_of)
    future = _value_at(series, as_of + timedelta(weeks=FORECAST_HORIZON_WEEKS))
    if not np.isfinite(current) or not np.isfinite(future) or current <= 0.0 or future <= 0.0:
        return float("nan")
    return float(np.log(future / current))


def _rv20_at(price_series: TimeSeries, as_of: date) -> float:
    idx = _index_at(price_series, as_of)
    if idx is None or idx < RV20_WINDOW:
        return float("nan")
    prices = np.asarray(price_series.values[: idx + 1], dtype=np.float64)
    if not np.isfinite(prices).all() or np.any(prices <= 0.0):
        return float("nan")
    returns = np.diff(np.log(prices))
    window = returns[-RV20_WINDOW:]
    return float(100.0 * np.sqrt(TRADING_DAYS_PER_YEAR * float(np.mean(window * window))))


def _weekly_returns(
    price_series: TimeSeries,
    weeks: tuple[date, ...],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    weekly_prices = np.array([_value_at(price_series, week) for week in weeks], dtype=np.float64)
    weekly_returns = np.full(weekly_prices.shape, np.nan, dtype=np.float64)
    for idx in range(1, weekly_prices.shape[0]):
        current = weekly_prices[idx]
        previous = weekly_prices[idx - 1]
        if np.isfinite(current) and np.isfinite(previous) and current > 0.0 and previous > 0.0:
            weekly_returns[idx] = float(np.log(current / previous))
    return weekly_prices, weekly_returns


def _rv52_series(price_series: TimeSeries, weeks: tuple[date, ...]) -> NDArray[np.float64]:
    _weekly_prices, weekly_returns = _weekly_returns(price_series, weeks)
    rv52 = np.full(len(weeks), np.nan, dtype=np.float64)
    for idx in range(RV52_WINDOW, len(weeks)):
        window = weekly_returns[idx - RV52_WINDOW + 1 : idx + 1]
        if np.isfinite(window).all():
            rv52[idx] = float(100.0 * np.sqrt(WEEKS_PER_YEAR * float(np.mean(window * window))))
    return rv52


def _global_term_structure(
    macro_series: Mapping[str, TimeSeries],
    weeks: tuple[date, ...],
) -> NDArray[np.float64]:
    values = []
    for week in weeks:
        vix = _value_at(macro_series["VIXCLS"], week)
        vxv = _value_at(macro_series["VXVCLS"], week)
        if not np.isfinite(vix) or not np.isfinite(vxv) or vix <= 0.0 or vxv <= 0.0:
            values.append(float("nan"))
        else:
            values.append(float(np.log(vix / vxv)))
    return np.asarray(values, dtype=np.float64)


def _micro_history_for_asset(
    asset_inputs: Mapping[str, TimeSeries],
    macro_series: Mapping[str, TimeSeries],
    weeks: tuple[date, ...],
) -> tuple[NDArray[np.float64], NDArray[np.float64], tuple[str, ...], NDArray[np.float64]]:
    if TARGET_KEY not in asset_inputs or PRIMARY_VOL_KEY not in asset_inputs:
        msg = "asset_inputs must include target and vol series"
        raise ValueError(msg)
    target_series = asset_inputs[TARGET_KEY]
    primary_vol = asset_inputs[PRIMARY_VOL_KEY]
    fallback_vol = asset_inputs.get(FALLBACK_VOL_KEY)
    term_structure = _global_term_structure(macro_series, weeks)
    rv52 = _rv52_series(target_series, weeks)

    x8: list[float] = []
    x9: list[float] = []
    modes: list[str] = []
    target_returns = np.array(
        [_forward_52w_return(target_series, week) for week in weeks],
        dtype=np.float64,
    )
    for idx, week in enumerate(weeks):
        rv20 = _rv20_at(target_series, week)
        primary = _value_at(primary_vol, week)
        fallback = float("nan") if fallback_vol is None else _value_at(fallback_vol, week)
        if np.isfinite(primary) and primary > 0.0 and np.isfinite(rv20) and rv20 > 0.0:
            x8.append(float(np.log(primary)))
            x9.append(float(np.log(primary) - np.log(rv20)))
            modes.append("primary")
        elif np.isfinite(fallback) and fallback > 0.0 and np.isfinite(rv20) and rv20 > 0.0:
            x8.append(float(np.log(fallback)))
            x9.append(float(np.log(fallback) - np.log(rv20)))
            modes.append("proxy")
        elif np.isfinite(rv20) and rv20 > 0.0 and np.isfinite(rv52[idx]) and rv52[idx] > 0.0:
            x8.append(float(np.log(rv20)))
            x9.append(float(np.log(rv20) - np.log(rv52[idx])))
            modes.append("rv_only")
        else:
            x8.append(float("nan"))
            x9.append(float("nan"))
            modes.append("rv_only")
    raw = np.column_stack(
        [
            np.asarray(x8, dtype=np.float64),
            np.asarray(x9, dtype=np.float64),
            term_structure,
        ],
    )
    return raw, _scale_matrix(raw), tuple(modes), target_returns


def build_panel_feature_block(
    macro_series: Mapping[str, TimeSeries],
    asset_series: Mapping[str, Mapping[str, TimeSeries]],
    as_of: date,
) -> PanelFeatureFrame:
    """pure. Build shared macro plus per-asset micro feature histories through as_of."""
    weeks = _weekly_dates_from_series(macro_series, as_of)
    x_macro_raw = np.vstack([_macro_row(macro_series, week) for week in weeks]).astype(np.float64)
    x_macro = _scale_matrix(x_macro_raw)
    macro_mask = ~np.isfinite(x_macro_raw)

    x_micro_raw: dict[str, NDArray[np.float64]] = {}
    x_micro: dict[str, NDArray[np.float64]] = {}
    micro_mask: dict[str, NDArray[np.bool_]] = {}
    micro_feature_mode: dict[str, tuple[str, ...]] = {}
    target_returns: dict[str, NDArray[np.float64]] = {}
    availability: dict[str, NDArray[np.bool_]] = {}
    for asset_id, inputs in asset_series.items():
        raw, scaled, modes, targets = _micro_history_for_asset(inputs, macro_series, weeks)
        x_micro_raw[asset_id] = raw
        x_micro[asset_id] = scaled
        micro_mask[asset_id] = ~np.isfinite(raw)
        micro_feature_mode[asset_id] = modes
        target_returns[asset_id] = targets
        availability[asset_id] = (
            ~macro_mask.any(axis=1)
            & ~micro_mask[asset_id].any(axis=1)
            & np.isfinite(targets)
        )
    available_assets = tuple(
        asset_id for asset_id, mask in availability.items() if mask.shape[0] > 0 and bool(mask[-1])
    )
    return PanelFeatureFrame(
        as_of=as_of,
        feature_dates=weeks,
        x_macro_raw=x_macro_raw,
        x_macro=x_macro,
        x_micro_raw=dict(x_micro_raw),
        x_micro=dict(x_micro),
        macro_mask=macro_mask,
        micro_mask=dict(micro_mask),
        asset_availability=dict(availability),
        micro_feature_mode=dict(micro_feature_mode),
        available_assets=available_assets,
        target_returns=dict(target_returns),
    )


def build_panel_hmm_inputs(
    panel_frame: PanelFeatureFrame,
    macro_series: Mapping[str, TimeSeries],
) -> PanelHMMInputs:
    """pure. Build shared panel HMM observations with VIX-based vol slots."""
    macro_finite = ~panel_frame.macro_mask.any(axis=1)
    usable_indices = np.flatnonzero(macro_finite)
    if usable_indices.shape[0] < MIN_HMM_ROWS:
        msg = "not enough finite macro rows for panel HMM"
        raise HMMConvergenceError(msg)
    x_macro = panel_frame.x_macro[usable_indices]
    pc = robust_pca_2d(x_macro)
    delta = pc[1:] - pc[:-1]
    full_vix = np.asarray(
        [_safe_log(_value_at(macro_series["VIXCLS"], week)) for week in panel_frame.feature_dates],
        dtype=np.float64,
    )
    full_vix_delta = np.full(full_vix.shape, np.nan, dtype=np.float64)
    for idx in range(LAG_4W, full_vix.shape[0]):
        if np.isfinite(full_vix[idx]) and np.isfinite(full_vix[idx - LAG_4W]):
            full_vix_delta[idx] = float(full_vix[idx] - full_vix[idx - LAG_4W])
    vix_scaled = _scale_column(full_vix)
    vix_delta_scaled = _scale_column(full_vix_delta)
    obs_indices = usable_indices[1:]
    vix_obs = vix_scaled[obs_indices]
    vix_delta_obs = vix_delta_scaled[obs_indices]
    label_anchor_returns = panel_frame.target_returns["SPX"][obs_indices]
    valid = np.isfinite(vix_obs) & np.isfinite(vix_delta_obs) & np.isfinite(label_anchor_returns)
    if np.sum(valid) < MIN_HMM_ROWS:
        msg = "not enough finite VIX/SPX rows for panel HMM"
        raise HMMConvergenceError(msg)
    y_obs = np.column_stack(
        [
            pc[1:, 0][valid],
            pc[1:, 1][valid],
            delta[:, 0][valid],
            delta[:, 1][valid],
            vix_obs[valid],
            vix_delta_obs[valid],
        ],
    )
    h = panel_frame.x_macro[obs_indices, 4][valid] + 0.5 * vix_delta_obs[valid]
    observation_dates = tuple(panel_frame.feature_dates[int(idx)] for idx in obs_indices[valid])
    return PanelHMMInputs(
        observation_dates=observation_dates,
        y_obs=y_obs,
        h=h.astype(np.float64),
        label_anchor_returns=label_anchor_returns[valid].astype(np.float64),
    )


def fit_panel_hmm(  # noqa: PLR0913
    panel_frame: PanelFeatureFrame,
    macro_series: Mapping[str, TimeSeries],
    rng: np.random.Generator,
    *,
    max_iter: int = 200,
    tolerance: float = 1.0e-6,
    restarts: int = 50,
    transition_max_iter: int = 200,
    warm_start_model: HMMModel | None = None,
) -> HMMModel:
    """pure. Fit the shared panel HMM using SPX as the label anchor."""
    inputs = build_panel_hmm_inputs(panel_frame, macro_series)
    return fit_hmm(
        inputs.y_obs,
        inputs.h,
        rng,
        max_iter=max_iter,
        tolerance=tolerance,
        restarts=restarts,
        transition_max_iter=transition_max_iter,
        forward_52w_returns=inputs.label_anchor_returns,
        warm_start_model=warm_start_model,
    )
