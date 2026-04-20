"""Pure training artifact construction for SRD v8.7 walk-forward refits."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
from numpy.typing import NDArray

from config_types import FrozenConfig
from decision.offense_abs import OffenseThresholds
from decision.utility import UtilityZStats, excess_return, utility
from engine_types import Stance, TimeSeries
from errors import HMMConvergenceError, QuantileSolverError
from features.block_builder import build_feature_block
from features.pca import robust_pca_2d
from features.scaling import robust_zscore_expanding, soft_squash_clip
from inference.weekly import FULL_TAUS, TrainingArtifacts
from law.linear_quantiles import QRCoefs, fit_linear_quantiles, predict_interior
from law.quantile_moments import moments_from_quantiles
from law.tail_extrapolation import extrapolate_tails
from state.ti_hmm_single import HMMModel, degraded_hmm_posterior, fit_hmm, infer_hmm

FORECAST_HORIZON_WEEKS = 52
TRAINING_EMBARGO_WEEKS = 53
DEFAULT_MIN_TRAINING_WEEKS = 312
FEATURE_COUNT = 10
HMM_OBS_MIN_ROWS = 2
FRIDAY = 4
MAD_EPSILON = 1.0e-8
THRESHOLD_EPSILON = 1.0e-8
DEFAULT_LABEL_MAP: Mapping[int, Stance] = {
    0: "DEFENSIVE",
    1: "NEUTRAL",
    2: "OFFENSIVE",
}

FitHMM = Callable[
    [NDArray[np.float64], NDArray[np.float64], np.random.Generator],
    HMMModel,
]
FitQR = Callable[
    [NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
    QRCoefs,
]


@dataclass(frozen=True, slots=True)
class _TrainingMatrix:
    """pure. Aligned finite training arrays before model fitting."""

    dates: tuple[date, ...]
    x_raw: NDArray[np.float64]
    x_scaled: NDArray[np.float64]
    y_52w: NDArray[np.float64]


def _weekly_dates_from_series(series: Mapping[str, TimeSeries], as_of: date) -> tuple[date, ...]:
    timestamps = series["DGS10"].timestamps.astype("datetime64[D]")
    filtered = timestamps[timestamps <= np.datetime64(as_of, "D")]
    return tuple(
        week
        for week in (date.fromisoformat(str(value)) for value in filtered.tolist())
        if week.weekday() == FRIDAY
    )


def _index_in_week(series: TimeSeries, as_of: date) -> int | None:
    target = np.datetime64(as_of, "D")
    week_start = np.datetime64(as_of - timedelta(days=6), "D")
    timestamps = series.timestamps.astype("datetime64[D]")
    matches = np.flatnonzero((timestamps >= week_start) & (timestamps <= target))
    if matches.size == 0:
        return None
    return int(matches[-1])


def _value_in_week(series: TimeSeries, as_of: date) -> float:
    idx = _index_in_week(series, as_of)
    if idx is None:
        return float("nan")
    return float(series.values[idx])


def _forward_52w_return(series: TimeSeries, as_of: date) -> float:
    current = _value_in_week(series, as_of)
    future = _value_in_week(series, as_of + timedelta(weeks=FORECAST_HORIZON_WEEKS))
    if not np.isfinite(current) or not np.isfinite(future) or current <= 0.0 or future <= 0.0:
        return float("nan")
    return float(np.log(future / current))


def _scale_feature_history(raw_history: NDArray[np.float64]) -> NDArray[np.float64]:
    values = np.asarray(raw_history, dtype=np.float64)
    scaled = np.empty_like(values, dtype=np.float64)
    for idx in range(values.shape[1]):
        scaled[:, idx] = soft_squash_clip(robust_zscore_expanding(values[:, idx]))
    return scaled


def _training_matrix(
    as_of: date,
    series: Mapping[str, TimeSeries],
    *,
    min_training_weeks: int,
) -> _TrainingMatrix:
    if "NASDAQXNDX" not in series:
        msg = "training requires NASDAQXNDX to compute 52-week forward returns"
        raise ValueError(msg)
    train_end = as_of - timedelta(weeks=TRAINING_EMBARGO_WEEKS)
    rows: list[NDArray[np.float64]] = []
    targets: list[float] = []
    dates: list[date] = []
    for week in _weekly_dates_from_series(series, train_end):
        raw, missing = build_feature_block(series, week)
        y_52w = _forward_52w_return(series["NASDAQXNDX"], week)
        if not missing.any() and np.isfinite(y_52w):
            dates.append(week)
            rows.append(raw)
            targets.append(y_52w)
    if len(rows) < min_training_weeks:
        msg = "not enough finite training weeks after PIT cutoff"
        raise ValueError(msg)
    x_raw = np.vstack(rows).astype(np.float64)
    return _TrainingMatrix(
        dates=tuple(dates),
        x_raw=x_raw,
        x_scaled=_scale_feature_history(x_raw),
        y_52w=np.asarray(targets, dtype=np.float64),
    )


def _hmm_inputs(
    x_scaled_history: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if x_scaled_history.shape[0] < HMM_OBS_MIN_ROWS:
        msg = "not enough finite feature rows for HMM training"
        raise HMMConvergenceError(msg)
    pc = robust_pca_2d(x_scaled_history)
    delta = pc[1:] - pc[:-1]
    y_obs = np.column_stack(
        [
            pc[1:, 0],
            pc[1:, 1],
            delta[:, 0],
            delta[:, 1],
            x_scaled_history[1:, 7],
            x_scaled_history[1:, 8],
        ],
    )
    h = x_scaled_history[1:, 4] + 0.5 * x_scaled_history[1:, 8]
    return y_obs, h


def _fit_hmm_or_none(
    y_obs: NDArray[np.float64],
    h: NDArray[np.float64],
    y_52w: NDArray[np.float64],
    rng: np.random.Generator,
    *,
    fit_hmm_fn: Callable[..., HMMModel],
) -> HMMModel | None:
    try:
        return fit_hmm_fn(y_obs, h, rng, forward_52w_returns=y_52w)
    except HMMConvergenceError:
        return None


def _posterior_training_path(
    hmm_model: HMMModel | None,
    y_obs: NDArray[np.float64],
    h: NDArray[np.float64],
) -> NDArray[np.float64]:
    if hmm_model is None:
        return np.tile(degraded_hmm_posterior().post, (y_obs.shape[0], 1))
    rows: list[NDArray[np.float64]] = []
    for end_idx in range(1, y_obs.shape[0] + 1):
        try:
            rows.append(infer_hmm(hmm_model, y_obs[:end_idx], h[:end_idx]).posterior.post)
        except (HMMConvergenceError, ValueError):
            rows.append(degraded_hmm_posterior().post)
    return np.vstack(rows)


def _mad(values: NDArray[np.float64]) -> float:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return max(mad, MAD_EPSILON)


def _utility_zstats(
    er_values: NDArray[np.float64],
    es20_values: NDArray[np.float64],
    ploss_values: NDArray[np.float64],
) -> UtilityZStats:
    return UtilityZStats(
        er_med=float(np.median(er_values)),
        er_mad=_mad(er_values),
        es20_med=float(np.median(es20_values)),
        es20_mad=_mad(es20_values),
        ploss_med=float(np.median(ploss_values)),
        ploss_mad=_mad(ploss_values),
    )


def _strictly_increasing(values: NDArray[np.float64]) -> NDArray[np.float64]:
    out = np.asarray(values, dtype=np.float64).copy()
    for idx in range(1, out.shape[0]):
        out[idx] = max(out[idx], out[idx - 1] + THRESHOLD_EPSILON)
    return out


def _offense_thresholds(utilities: NDArray[np.float64]) -> OffenseThresholds:
    quantiles = np.quantile(utilities, np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]))
    extended = quantiles.copy()
    spread = 3.0 * _mad(utilities)
    extended[0] = min(extended[0], float(np.min(utilities)) - spread)
    extended[-1] = max(extended[-1], float(np.max(utilities)) + spread)
    q = _strictly_increasing(extended)
    return OffenseThresholds(
        u_q0=float(q[0]),
        u_q20=float(q[1]),
        u_q40=float(q[2]),
        u_q60=float(q[3]),
        u_q80=float(q[4]),
        u_q100=float(q[5]),
    )


def _decision_training_stats(
    x_scaled: NDArray[np.float64],
    pi: NDArray[np.float64],
    dgs1: NDArray[np.float64],
    qr_coefs: QRCoefs,
) -> tuple[UtilityZStats, OffenseThresholds]:
    er_values: list[float] = []
    es20_values: list[float] = []
    ploss_values: list[float] = []
    for row_idx in range(x_scaled.shape[0]):
        interior = predict_interior(qr_coefs, x_scaled[row_idx], pi[row_idx])
        full_quantiles, _tail_status = extrapolate_tails(interior)
        moments = moments_from_quantiles(FULL_TAUS, full_quantiles)
        er_values.append(excess_return(moments["mu_hat"], float(dgs1[row_idx])))
        es20_values.append(moments["es20"])
        ploss_values.append(moments["p_loss"])
    er = np.asarray(er_values, dtype=np.float64)
    es20 = np.asarray(es20_values, dtype=np.float64)
    ploss = np.asarray(ploss_values, dtype=np.float64)
    zstats = _utility_zstats(er, es20, ploss)
    utilities = np.array(
        [
            utility(float(er[idx]), float(es20[idx]), float(ploss[idx]), zstats)
            for idx in range(er.shape[0])
        ],
        dtype=np.float64,
    )
    return zstats, _offense_thresholds(utilities)


def build_training_artifacts(  # noqa: PLR0913
    as_of: date,
    series: Mapping[str, TimeSeries],
    cfg: FrozenConfig,
    *,
    rng: np.random.Generator,
    min_training_weeks: int = DEFAULT_MIN_TRAINING_WEEKS,
    fit_hmm_fn: Callable[..., HMMModel] = fit_hmm,
    fit_qr_fn: Callable[..., QRCoefs] = fit_linear_quantiles,
) -> TrainingArtifacts:
    """pure. Fit SRD §7-§9 artifacts from PIT-truncated history."""
    matrix = _training_matrix(as_of, series, min_training_weeks=min_training_weeks)
    y_obs, h = _hmm_inputs(matrix.x_scaled)
    y_aligned = matrix.y_52w[1:]
    x_aligned = matrix.x_scaled[1:]
    raw_aligned = matrix.x_raw[1:]
    hmm_model = _fit_hmm_or_none(
        y_obs,
        h,
        y_aligned,
        rng,
        fit_hmm_fn=fit_hmm_fn,
    )
    pi = _posterior_training_path(hmm_model, y_obs, h)
    try:
        qr_coefs = fit_qr_fn(
            x_aligned,
            pi,
            y_aligned,
            l2_alpha=cfg.l2_alpha,
            min_gap=cfg.quantile_gap,
        )
    except QuantileSolverError:
        raise
    utility_zstats, offense_thresholds = _decision_training_stats(
        x_aligned,
        pi,
        raw_aligned[:, 2],
        qr_coefs,
    )
    label_map = hmm_model.label_map if hmm_model is not None else DEFAULT_LABEL_MAP
    return TrainingArtifacts(
        utility_zstats=utility_zstats,
        offense_thresholds=offense_thresholds,
        train_distributions={
            "x1": raw_aligned[:, 0].copy(),
            "x5": raw_aligned[:, 4].copy(),
            "x9": raw_aligned[:, 8].copy(),
        },
        state_label_map=dict(label_map),
        qr_coefs=qr_coefs,
        hmm_model=hmm_model,
    )
