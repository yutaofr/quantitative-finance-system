"""Recovered original T5 residual-persistence M4 reproduction source.

io: research-only recovered implementation copied from historical pilot sources under
/tmp/qfs-sigma/repo/src/research/ for Phase 0A candidate-side reproduction.
"""

from __future__ import annotations

# ruff: noqa: E501, PLR2004, PERF401, FBT003
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, timedelta
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from app.config_loader import load_adapter_secrets
from app.panel_runner import R1_TRAIN_WINDOW, _build_asset_inputs, _slice_frame_rolling
from app.runtime_deps import build_panel_runner_deps
from data_contract.yahoo_client import YahooFinanceClient
from engine_types import TimeSeries
from features.panel_block_builder import PanelFeatureFrame, build_panel_feature_block

# Provenance:
# - /tmp/qfs-sigma/repo/src/research/run_ndx_sigma_output_transform_pilot.py
# - /tmp/qfs-sigma/repo/src/research/run_ndx_sigma_output_refinement_pilot.py
# - /tmp/qfs-sigma/repo/src/research/run_ndx_sigma_output_monotone_family_scan.py
# Formula-bearing functions _fit_t5 and _predict_t5 are copied without formula changes.

FULL_TAUS = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95], dtype=np.float64)
TARGET_ASSET = "NASDAQXNDX"
SIGMA_EPS = 1.0e-4
SIGMA_MAX_MULT = 3.0
RIDGE_MU = 1.0
RIDGE_EXOG = 1.0
CORR_CLIP = 1.0
LOGSIG_PAD = 0.75
HAR_CORE = ("NASDAQXNDX_rv1d", "NASDAQXNDX_rv5d", "NASDAQXNDX_rv22d")
IV_SLOPES = ("term_vix9d_vix", "term_vix_vix3m", "term_vix9d_vix3m")
BASE_MODEL_M4 = "M4_har_plus_iv_slopes"
WINDOWS = (
    (date(2008, 1, 4), date(2008, 12, 26)),
    (date(2017, 7, 7), date(2017, 12, 29)),
    (date(2018, 7, 6), date(2018, 12, 28)),
    (date(2020, 1, 3), date(2020, 6, 26)),
)
EXTRA_TICKERS = {
    "VIX9D": "^VIX9D",
    "VIX": "^VIX",
    "VIX3M": "^VIX3M",
    "MOVE": "^MOVE",
}
ASSETS = {
    "NASDAQXNDX": "QQQ",
    "SPX": "SPY",
    "R2K": "IWM",
}


@dataclass(frozen=True, slots=True)
class SigmaPilotContext:
    """pure. Shared immutable inputs for the recovered sigma pilot."""

    frame: PanelFeatureFrame
    weekly_returns: Mapping[str, NDArray[np.float64]]


@dataclass(frozen=True, slots=True)
class HarFeatureContext:
    """pure. Shared weekly sigma feature panel for the recovered T5 pilot."""

    base: SigmaPilotContext
    sigma_features: Mapping[str, NDArray[np.float64]]


@dataclass(frozen=True, slots=True)
class SigmaModelSpec:
    """pure. Sigma feature blocks for one recovered model."""

    name: str
    core_features: tuple[str, ...]
    exog_features: tuple[str, ...]
    penalize_exog: bool


@dataclass(frozen=True, slots=True)
class WindowArrays:
    """pure. Evaluation arrays for one recovered T5 window."""

    y: NDArray[np.float64]
    q: NDArray[np.float64]
    sigma: NDArray[np.float64]
    e: NDArray[np.float64]
    z: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class TrainBase:
    """pure. Per-week training artifact for output-layer transforms."""

    as_of: date
    train_weeks: tuple[date, ...]
    mu_beta: NDArray[np.float64]
    resid_train: NDArray[np.float64]
    sigma_fit: dict[str, Any]
    sigma_train: NDArray[np.float64]
    sigma_target: NDArray[np.float64]
    history_frame: Any
    history_returns: dict[str, NDArray[np.float64]]


def _ridge_fit(x: NDArray[np.float64], y: NDArray[np.float64], penalty: float) -> NDArray[np.float64]:
    xtx = x.T @ x
    ridge = penalty * np.eye(x.shape[1], dtype=np.float64)
    ridge[0, 0] = 0.0
    return np.linalg.solve(xtx + ridge, x.T @ y).astype(np.float64)


def _quantile_score(y_true: float, quantiles: NDArray[np.float64]) -> float:
    losses = np.maximum(FULL_TAUS * (y_true - quantiles), (FULL_TAUS - 1.0) * (y_true - quantiles))
    return float(2.0 * np.mean(losses))


def _weekly_series_returns(series: TimeSeries, feature_dates: tuple[date, ...]) -> NDArray[np.float64]:
    timestamps = series.timestamps.astype("datetime64[D]")
    values = np.asarray(series.values, dtype=np.float64)
    weekly_prices = np.full(len(feature_dates), np.nan, dtype=np.float64)
    for idx, week in enumerate(feature_dates):
        target = np.datetime64(week, "D")
        week_start = np.datetime64(week - timedelta(days=6), "D")
        last_idx = int(np.searchsorted(timestamps, target, side="right")) - 1
        if last_idx >= 0 and timestamps[last_idx] >= week_start:
            weekly_prices[idx] = values[last_idx]
    weekly_returns = np.full(len(feature_dates), np.nan, dtype=np.float64)
    for idx in range(1, len(feature_dates)):
        current = weekly_prices[idx]
        previous = weekly_prices[idx - 1]
        if np.isfinite(current) and np.isfinite(previous) and current > 0.0 and previous > 0.0:
            weekly_returns[idx] = float(np.log(current / previous))
    return weekly_returns


def _mu_design(
    frame: PanelFeatureFrame,
    asset: str,
    weekly_returns: Mapping[str, NDArray[np.float64]],
) -> NDArray[np.float64]:
    del weekly_returns
    return np.column_stack([
        np.ones(len(frame.feature_dates), dtype=np.float64),
        frame.x_macro,
        frame.x_micro[asset],
    ])


def _weekly_index(series: TimeSeries, weeks: tuple[date, ...]) -> NDArray[np.float64]:
    timestamps = series.timestamps.astype("datetime64[D]")
    values = np.asarray(series.values, dtype=np.float64)
    out = np.full(len(weeks), np.nan, dtype=np.float64)
    for idx, week in enumerate(weeks):
        target = np.datetime64(week, "D")
        week_start = np.datetime64(week - timedelta(days=6), "D")
        last_idx = int(np.searchsorted(timestamps, target, side="right")) - 1
        if last_idx >= 0 and timestamps[last_idx] >= week_start:
            out[idx] = values[last_idx]
    return out


def _daily_log_returns(series: TimeSeries) -> tuple[NDArray[np.datetime64], NDArray[np.float64]]:
    prices = np.asarray(series.values, dtype=np.float64)
    returns = np.diff(np.log(prices)).astype(np.float64)
    return series.timestamps[1:].astype("datetime64[D]"), returns


def _feature_at_week(dates: NDArray[np.datetime64], returns: NDArray[np.float64], week: date) -> Mapping[str, float]:
    target = np.datetime64(week, "D")
    last_idx = int(np.searchsorted(dates, target, side="right")) - 1
    out = {"rv1d": np.nan, "rv5d": np.nan, "rv22d": np.nan}
    if last_idx < 21:
        return out
    r1 = returns[last_idx : last_idx + 1]
    r5 = returns[last_idx - 4 : last_idx + 1]
    r22 = returns[last_idx - 21 : last_idx + 1]
    out["rv1d"] = float(np.sqrt(252.0 * np.mean(r1 * r1)))
    out["rv5d"] = float(np.sqrt(252.0 * np.mean(r5 * r5)))
    out["rv22d"] = float(np.sqrt(252.0 * np.mean(r22 * r22)))
    return out


def _safe_log(x: NDArray[np.float64] | float) -> NDArray[np.float64] | float:
    arr = np.asarray(x, dtype=np.float64)
    out = np.full(arr.shape, np.nan, dtype=np.float64)
    mask = np.isfinite(arr) & (arr > 0.0)
    out[mask] = np.log(arr[mask])
    if np.isscalar(x):
        return float(out.reshape(()))
    return out


def build_pilot_context_from_deps(end: date, deps: object, vintage_mode: str = "strict") -> SigmaPilotContext:
    macro_series = deps.fetch_macro_series(end, vintage_mode)
    asset_prices = deps.fetch_asset_prices(end + timedelta(weeks=52))
    asset_inputs = _build_asset_inputs(macro_series, asset_prices)
    frame = build_panel_feature_block(macro_series, asset_inputs, end)
    weekly_returns = {
        asset: _weekly_series_returns(asset_prices[asset], frame.feature_dates)
        for asset in ("NASDAQXNDX", "SPX", "R2K")
    }
    return SigmaPilotContext(frame=frame, weekly_returns=weekly_returns)


def build_har_context(end: date) -> HarFeatureContext:
    deps = build_panel_runner_deps(load_adapter_secrets(os.environ))
    base = build_pilot_context_from_deps(end, deps, vintage_mode="strict")
    weeks = base.frame.feature_dates
    yahoo = YahooFinanceClient(cache_root=Path("data/raw/yahoo"))
    horizon_end = end + timedelta(weeks=52)
    asset_series = {
        asset: yahoo.fetch_etf_adjusted_close(ticker, date(2000, 1, 1), horizon_end)
        for asset, ticker in ASSETS.items()
    }
    extra_series = {
        name: yahoo.fetch_etf_adjusted_close(ticker, date(2000, 1, 1), horizon_end)
        for name, ticker in EXTRA_TICKERS.items()
    }
    sigma_features: dict[str, NDArray[np.float64]] = {}
    for asset, series in asset_series.items():
        dts, rets = _daily_log_returns(series)
        rows = [_feature_at_week(dts, rets, week) for week in weeks]
        for key in ("rv1d", "rv5d", "rv22d"):
            sigma_features[f"{asset}_{key}"] = np.asarray([row[key] for row in rows], dtype=np.float64)
    for name, series in extra_series.items():
        sigma_features[name] = _weekly_index(series, weeks)
    vix9d = sigma_features["VIX9D"]
    vix = sigma_features["VIX"]
    vix3m = sigma_features["VIX3M"]
    sigma_features["term_vix9d_vix"] = np.asarray(_safe_log(vix9d / vix), dtype=np.float64)
    sigma_features["term_vix_vix3m"] = np.asarray(_safe_log(vix / vix3m), dtype=np.float64)
    sigma_features["term_vix9d_vix3m"] = np.asarray(_safe_log(vix9d / vix3m), dtype=np.float64)
    return HarFeatureContext(base=base, sigma_features=sigma_features)


def _acf1(values: NDArray[np.float64]) -> float | None:
    if values.shape[0] <= 1:
        return None
    if float(np.std(values[:-1])) <= 1.0e-12 or float(np.std(values[1:])) <= 1.0e-12:
        return None
    return float(np.corrcoef(values[:-1], values[1:])[0, 1])


def _corr(x: NDArray[np.float64], y: NDArray[np.float64]) -> float | None:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) <= 1:
        return None
    xs = x[mask]
    ys = y[mask]
    if float(np.std(xs)) <= 1.0e-12 or float(np.std(ys)) <= 1.0e-12:
        return None
    return float(np.corrcoef(xs, ys)[0, 1])


def _rank_corr(x: NDArray[np.float64], y: NDArray[np.float64]) -> float | None:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) <= 1:
        return None
    xs = x[mask]
    ys = y[mask]
    if float(np.std(xs)) <= 1.0e-12 or float(np.std(ys)) <= 1.0e-12:
        return None
    xr = np.argsort(np.argsort(xs)).astype(np.float64)
    yr = np.argsort(np.argsort(ys)).astype(np.float64)
    return float(np.corrcoef(xr, yr)[0, 1])


def _pinball_loss(y: NDArray[np.float64], q: NDArray[np.float64], tau: float, idx: int) -> float:
    diff = y - q[:, idx]
    return float(np.mean(np.maximum(tau * diff, (tau - 1.0) * diff)))


def _sigma_design(
    context: HarFeatureContext,
    model: SigmaModelSpec,
    weeks: tuple[date, ...],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    feature_dates = context.base.frame.feature_dates
    index = np.array([feature_dates.index(week) for week in weeks], dtype=np.int64)
    core_cols = [np.ones(index.shape[0], dtype=np.float64)]
    for name in model.core_features:
        core_cols.append(np.asarray(_safe_log(context.sigma_features[name][index]), dtype=np.float64))
    exog_cols = [np.asarray(context.sigma_features[name][index], dtype=np.float64) for name in model.exog_features]
    core = np.column_stack(core_cols)
    exog = np.column_stack(exog_cols) if exog_cols else np.zeros((index.shape[0], 0), dtype=np.float64)
    return core, exog


def _column_medians(values: NDArray[np.float64]) -> NDArray[np.float64]:
    out = np.zeros(values.shape[1], dtype=np.float64)
    for idx in range(values.shape[1]):
        finite = values[:, idx][np.isfinite(values[:, idx])]
        out[idx] = float(np.median(finite)) if finite.size else 0.0
    return out


def _fill_with_medians(values: NDArray[np.float64], medians: NDArray[np.float64]) -> NDArray[np.float64]:
    out = np.asarray(values, dtype=np.float64).copy()
    if out.ndim == 1:
        mask = ~np.isfinite(out)
        out[mask] = medians[mask]
        return out
    for idx in range(out.shape[1]):
        mask = ~np.isfinite(out[:, idx])
        out[mask, idx] = medians[idx]
    return out


def _fit_sigma_model(
    core_x: NDArray[np.float64],
    exog_x: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    penalize_exog: bool,
) -> NDArray[np.float64]:
    x = np.column_stack([core_x, exog_x]) if exog_x.shape[1] else core_x
    if not penalize_exog:
        return np.linalg.lstsq(x, y, rcond=None)[0].astype(np.float64)
    xtx = x.T @ x
    ridge = np.zeros_like(xtx)
    if exog_x.shape[1]:
        start = core_x.shape[1]
        ridge[start:, start:] = RIDGE_EXOG * np.eye(exog_x.shape[1], dtype=np.float64)
    return np.linalg.solve(xtx + ridge, x.T @ y).astype(np.float64)


def _predict_sigma(
    core_row: NDArray[np.float64],
    exog_row: NDArray[np.float64],
    beta: NDArray[np.float64],
    train_median: float,
) -> float:
    x = np.concatenate([core_row, exog_row]) if exog_row.size else core_row
    sigma = float(np.exp(np.clip(float(x @ beta), np.log(SIGMA_EPS), np.log(10.0))))
    return min(sigma, SIGMA_MAX_MULT * train_median)


def _train_mu(frame: Any, weekly_returns: dict[str, NDArray[np.float64]]) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    y_train = np.asarray(frame.target_returns[TARGET_ASSET], dtype=np.float64)
    mu_x_train = _mu_design(frame, TARGET_ASSET, weekly_returns)
    mu_beta = _ridge_fit(mu_x_train, y_train, RIDGE_MU)
    mu_hat_train = mu_x_train @ mu_beta
    resid_train = y_train - mu_hat_train
    return mu_beta, mu_hat_train, resid_train


def _fit_plain_sigma_model(
    context: HarFeatureContext,
    model: SigmaModelSpec,
    train_weeks: tuple[date, ...],
    sigma_target: NDArray[np.float64],
) -> tuple[dict[str, Any], dict[str, NDArray[np.float64]]]:
    core_x, exog_x = _sigma_design(context, model, train_weeks)
    finite = np.isfinite(sigma_target)
    core_medians = _column_medians(core_x[finite]) if core_x.shape[1] else np.zeros(0, dtype=np.float64)
    core_x = _fill_with_medians(core_x, core_medians) if core_x.shape[1] else core_x
    exog_medians = _column_medians(exog_x[finite]) if exog_x.shape[1] else np.zeros(0, dtype=np.float64)
    exog_x = _fill_with_medians(exog_x, exog_medians) if exog_x.shape[1] else exog_x
    beta = _fit_sigma_model(core_x[finite], exog_x[finite], sigma_target[finite], penalize_exog=model.penalize_exog)
    x_train = np.column_stack([core_x, exog_x]) if exog_x.shape[1] else core_x
    raw = np.exp(np.clip(x_train @ beta, np.log(SIGMA_EPS), np.log(10.0)))
    sigma_med = float(np.median(raw[np.isfinite(raw)]))
    sigma_train = np.minimum(raw, SIGMA_MAX_MULT * sigma_med)
    return ({
        "type": "plain",
        "model": model,
        "beta": beta,
        "core_medians": core_medians,
        "exog_medians": exog_medians,
        "sigma_train_median": sigma_med,
        "sigma_train_pred": sigma_train,
    }, {"core_x": core_x, "exog_x": exog_x})


def _predict_from_plain_sigma(context: HarFeatureContext, fit: dict[str, Any], week: date) -> float:
    core_row, exog_row = _sigma_design(context, fit["model"], (week,))
    core_f = _fill_with_medians(core_row[0], fit["core_medians"])
    exog_f = _fill_with_medians(exog_row[0], fit["exog_medians"]) if exog_row.size else exog_row[0]
    return _predict_sigma(core_f, exog_f, fit["beta"], fit["sigma_train_median"])


def _fit_sigma_by_model(
    context: HarFeatureContext,
    model_name: str,
    train_weeks: tuple[date, ...],
    sigma_target: NDArray[np.float64],
) -> tuple[dict[str, Any], dict[str, NDArray[np.float64]]]:
    if model_name != BASE_MODEL_M4:
        raise ValueError(model_name)
    spec = SigmaModelSpec(BASE_MODEL_M4, HAR_CORE, IV_SLOPES, False)
    return _fit_plain_sigma_model(context, spec, train_weeks, sigma_target)


def _predict_sigma_by_model(context: HarFeatureContext, fit: dict[str, Any], week: date) -> float:
    if fit["type"] == "plain":
        return _predict_from_plain_sigma(context, fit, week)
    raise ValueError(fit["type"])


def _same_week_mu_and_y(
    history_frame: Any,
    history_returns: dict[str, NDArray[np.float64]],
    mu_beta: NDArray[np.float64],
    as_of: date,
) -> tuple[float, float]:
    row_idx = history_frame.feature_dates.index(as_of)
    mu_x_row = _mu_design(history_frame, TARGET_ASSET, history_returns)[row_idx]
    mu_hat = float(mu_x_row @ mu_beta)
    y = float(history_frame.target_returns[TARGET_ASSET][row_idx])
    return mu_hat, y


def _lagged_return_and_resid(
    history_frame: Any,
    history_returns: dict[str, NDArray[np.float64]],
    mu_beta: NDArray[np.float64],
    as_of: date,
) -> tuple[float, float]:
    row_idx = history_frame.feature_dates.index(as_of)
    if row_idx == 0:
        return 0.0, 1.0
    prev_ret = float(history_frame.target_returns[TARGET_ASSET][row_idx - 1])
    prev_mu_x = _mu_design(history_frame, TARGET_ASSET, history_returns)[row_idx - 1]
    prev_mu = float(prev_mu_x @ mu_beta)
    prev_e = float(history_frame.target_returns[TARGET_ASSET][row_idx - 1] - prev_mu)
    return prev_ret, abs(prev_e)


def _build_train_base(context: HarFeatureContext, base_model: str, as_of: date) -> TrainBase:
    frame = context.base.frame
    train_end = as_of - timedelta(weeks=53)
    train_frame = _slice_frame_rolling(frame, train_end, R1_TRAIN_WINDOW)
    train_weeks = train_frame.feature_dates
    train_returns = {
        asset: np.asarray(context.base.weekly_returns[asset], dtype=np.float64)[
            frame.feature_dates.index(train_weeks[0]) : frame.feature_dates.index(train_weeks[-1]) + 1
        ]
        for asset in ("NASDAQXNDX", "SPX", "R2K")
    }
    mu_beta, _mu_hat_train, resid_train = _train_mu(train_frame, train_returns)
    sigma_target = np.log(np.maximum(np.abs(resid_train), SIGMA_EPS))
    sigma_fit, _aux = _fit_sigma_by_model(context, base_model, train_weeks, sigma_target)
    sigma_train = np.asarray(sigma_fit["sigma_train_pred"], dtype=np.float64)
    history_frame = _slice_frame_rolling(frame, as_of, R1_TRAIN_WINDOW + 54)
    history_weeks = history_frame.feature_dates
    history_returns = {
        asset: np.asarray(context.base.weekly_returns[asset], dtype=np.float64)[
            frame.feature_dates.index(history_weeks[0]) : frame.feature_dates.index(history_weeks[-1]) + 1
        ]
        for asset in ("NASDAQXNDX", "SPX", "R2K")
    }
    return TrainBase(
        as_of=as_of,
        train_weeks=train_weeks,
        mu_beta=mu_beta,
        resid_train=np.asarray(resid_train, dtype=np.float64),
        sigma_fit=sigma_fit,
        sigma_train=sigma_train,
        sigma_target=np.asarray(sigma_target, dtype=np.float64),
        history_frame=history_frame,
        history_returns=history_returns,
    )


def _safe_clip_sigma(
    sigma: float,
    train_sigma: NDArray[np.float64],
    train_log_sigma: NDArray[np.float64],
    extra_hi: float = LOGSIG_PAD,
) -> tuple[float, dict[str, float]]:
    log_lo = float(np.quantile(train_log_sigma, 0.10)) - extra_hi
    log_hi = float(np.quantile(train_log_sigma, 0.99))
    clipped_log = float(np.clip(math.log(max(sigma, SIGMA_EPS)), log_lo, log_hi))
    sigma_new = float(math.exp(clipped_log))
    sigma_med = float(np.median(train_sigma))
    sigma_new = min(sigma_new, SIGMA_MAX_MULT * sigma_med)
    return sigma_new, {
        "log_sigma_p10": float(np.quantile(train_log_sigma, 0.10)),
        "log_sigma_p50": float(np.quantile(train_log_sigma, 0.50)),
        "log_sigma_p90": float(np.quantile(train_log_sigma, 0.90)),
        "log_sigma_p99": float(np.quantile(train_log_sigma, 0.99)),
    }


def _fit_t5(train_base: TrainBase) -> dict[str, Any]:
    ratio = np.abs(train_base.resid_train) / np.maximum(train_base.sigma_train, SIGMA_EPS)
    x = np.log(np.maximum(np.roll(ratio, 1), SIGMA_EPS))
    x[0] = 0.0
    y = np.log(np.maximum(np.abs(train_base.resid_train), SIGMA_EPS)) - np.log(np.maximum(train_base.sigma_train, SIGMA_EPS))
    denom = float(np.dot(x, x))
    c = float(np.clip(float(np.dot(x, y) / denom) if denom > 1.0e-12 else 0.0, 0.0, 1.0))
    return {
        "c": c,
        "train_sigma": train_base.sigma_train,
        "train_log_sigma": np.log(np.maximum(train_base.sigma_train, SIGMA_EPS)),
        "monotone": False,
    }


def _predict_t5(prev_abs_e: float, prev_sigma: float, base_sigma: float, fit: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    ratio = prev_abs_e / max(prev_sigma, SIGMA_EPS)
    score = float(np.clip(fit["c"] * math.log(max(ratio, SIGMA_EPS)), -CORR_CLIP, CORR_CLIP))
    sigma = float(base_sigma * math.exp(score))
    sigma_new, aux = _safe_clip_sigma(sigma, fit["train_sigma"], fit["train_log_sigma"])
    aux["rank_order_changed_possible"] = True
    aux["c"] = fit["c"]
    return sigma_new, aux


def _base_sigma_for_week(context: HarFeatureContext, train_base: TrainBase, as_of: date) -> float:
    return _predict_sigma_by_model(context, train_base.sigma_fit, as_of)


def _eval_original_t5_window(context: HarFeatureContext, start: date, end: date) -> WindowArrays:
    frame = context.base.frame
    eval_dates = tuple(week for week in frame.feature_dates if start <= week <= end)
    ys: list[float] = []
    qs: list[NDArray[np.float64]] = []
    sigmas: list[float] = []
    residuals: list[float] = []
    z_vals: list[float] = []
    for as_of in eval_dates:
        train_base = _build_train_base(context, BASE_MODEL_M4, as_of)
        prev_ret, prev_abs_e = _lagged_return_and_resid(
            train_base.history_frame,
            train_base.history_returns,
            train_base.mu_beta,
            as_of,
        )
        del prev_ret
        prev_sigma = float(train_base.sigma_train[-1]) if len(train_base.sigma_train) else float(np.median(train_base.sigma_train))
        base_sigma = _base_sigma_for_week(context, train_base, as_of)
        fit = _fit_t5(train_base)
        sigma_hat, _aux = _predict_t5(prev_abs_e, prev_sigma, base_sigma, fit)
        std_resid = train_base.resid_train / np.maximum(train_base.sigma_train, SIGMA_EPS)
        finite_std_resid = std_resid[np.isfinite(std_resid)]
        if finite_std_resid.size == 0:
            raise ValueError("NONFINITE_STANDARDIZED_RESIDUALS")
        std_q = np.quantile(finite_std_resid, FULL_TAUS).astype(np.float64)
        std_q = np.maximum.accumulate(std_q)
        mu_hat, y = _same_week_mu_and_y(train_base.history_frame, train_base.history_returns, train_base.mu_beta, as_of)
        q = np.maximum.accumulate((mu_hat + sigma_hat * std_q).astype(np.float64))
        ys.append(y)
        qs.append(q)
        sigmas.append(sigma_hat)
        residuals.append(y - mu_hat)
        z_vals.append((y - mu_hat) / max(sigma_hat, SIGMA_EPS))
    return WindowArrays(
        y=np.asarray(ys, dtype=np.float64),
        q=np.vstack(qs).astype(np.float64),
        sigma=np.asarray(sigmas, dtype=np.float64),
        e=np.asarray(residuals, dtype=np.float64),
        z=np.asarray(z_vals, dtype=np.float64),
    )


def _window_metrics(arr: WindowArrays) -> dict[str, float | int | None]:
    y = arr.y
    q = arr.q
    sigma = arr.sigma
    e = arr.e
    z = arr.z
    sigma_blowup = int(np.sum(sigma > 2.0 * float(np.median(sigma))))
    return {
        "mean_z": float(np.mean(z)),
        "std_z": float(np.std(z)),
        "corr_next": _corr(sigma[:-1], np.abs(e)[1:]) if sigma.shape[0] > 1 else None,
        "rank_next": _rank_corr(sigma[:-1], np.abs(e)[1:]) if sigma.shape[0] > 1 else None,
        "lag1_acf_z": _acf1(z),
        "sigma_blowup": sigma_blowup,
        "pathology": 0,
        "crps": float(np.mean([_quantile_score(float(y[i]), q[i]) for i in range(len(y))])),
        "q10_coverage": float(np.mean(y <= q[:, 1])),
        "q90_coverage": float(np.mean(y <= q[:, 5])),
        "q90_exceedance": float(np.mean(y > q[:, 5])),
        "pinball_q10": _pinball_loss(y, q, 0.10, 1),
        "pinball_q50": _pinball_loss(y, q, 0.50, 3),
        "pinball_q90": _pinball_loss(y, q, 0.90, 5),
        "sigma_med": float(np.median(sigma)),
    }


def run_original_t5_reproduction() -> dict[str, Any]:
    """io: run original T5_resid_persistence_M4 across the fixed pilot windows."""

    context = build_har_context(max(end for _start, end in WINDOWS))
    windows: dict[str, Any] = {}
    for start, end in WINDOWS:
        key = f"{start.isoformat()}__{end.isoformat()}"
        try:
            arr = _eval_original_t5_window(context, start, end)
            windows[key] = {
                "window": {"start": start.isoformat(), "end": end.isoformat()},
                "metrics": _window_metrics(arr),
                "status": "PASS",
            }
        except (ValueError, FloatingPointError, np.linalg.LinAlgError, IndexError) as exc:
            windows[key] = {
                "window": {"start": start.isoformat(), "end": end.isoformat()},
                "metrics": {
                    "mean_z": f"FAILED_TO_RUN_{exc}",
                    "std_z": f"FAILED_TO_RUN_{exc}",
                    "corr_next": f"FAILED_TO_RUN_{exc}",
                    "rank_next": f"FAILED_TO_RUN_{exc}",
                    "lag1_acf_z": f"FAILED_TO_RUN_{exc}",
                    "sigma_blowup": f"FAILED_TO_RUN_{exc}",
                    "pathology": f"FAILED_TO_RUN_{exc}",
                    "crps": f"FAILED_TO_RUN_{exc}",
                },
                "status": f"FAILED_TO_RUN_{exc}",
            }
    return {
        "model": "T5_resid_persistence_M4",
        "base_model": BASE_MODEL_M4,
        "train_window": R1_TRAIN_WINDOW,
        "embargo_weeks": 53,
        "output_frequency": "weekly feature_dates / Friday-close aligned",
        "windows": windows,
    }
