"""io: execute the auditable Cycle Evaluation Protocol research runner."""

from __future__ import annotations

# ruff: noqa: E501
import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from itertools import pairwise
import json
from pathlib import Path
import sys
from typing import Any, Final, Literal

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from scipy.stats import rankdata

from research.run_phase0a_benchmark_delivery import (
    FORECAST_HORIZON_WEEKS,
    TRAIN_EMBARGO_WEEKS,
    TRAIN_WINDOW_WEEKS,
    _load_weekly_forward_returns,
)
from research.run_rank_scale_hybrid_experiment import (
    ForecastSeries,
    _egarch_normal_series,
    _t5_series_by_window,
)

OUTPUT_DIR: Final[Path] = Path("artifacts/research/cycle_evaluation")
PROTOCOL_DOC: Final[Path] = Path("docs/cycle_evaluation/cycle_evaluation_protocol.md")
LABEL_BASELINE_DOC: Final[Path] = Path(
    "docs/cycle_evaluation/label_construction_and_baselines.md",
)
DECISION_DOC: Final[Path] = Path("docs/cycle_evaluation/formal_decision.md")
PRICE_PATH: Final[Path] = Path("data/raw/nasdaq/NASDAQXNDX/close.parquet")
STATE_ORDER: Final[tuple[str, str, str]] = ("CONTRACTION", "SLOWDOWN", "EXPANSION")
CURRENT_SYSTEM_OBJECTS: Final[tuple[str, str]] = (
    "T5_DERIVED_CYCLE_PROXY",
    "EGARCH_DERIVED_CYCLE_PROXY",
)
CYCLE_WINDOWS: Final[dict[str, tuple[date, date]]] = {
    "Window_2017": (date(2017, 7, 7), date(2017, 12, 29)),
    "Window_2018H2": (date(2018, 7, 6), date(2018, 12, 28)),
    "Window_2020H1": (date(2020, 1, 3), date(2020, 6, 26)),
    "Window_2008": (date(2008, 1, 4), date(2008, 12, 26)),
    "Window_2000_2002": (date(2000, 1, 7), date(2002, 12, 27)),
}
CURRENT_OBJECT_WINDOWS: Final[tuple[str, ...]] = (
    "Window_2017",
    "Window_2018H2",
    "Window_2020H1",
)
SANITY_CRISIS_WINDOWS: Final[dict[str, tuple[date, date]]] = {
    "DotCom_2000_2002": (date(2000, 3, 24), date(2002, 10, 11)),
    "GFC_2008": (date(2008, 9, 12), date(2009, 3, 13)),
    "COVID_2020H1": (date(2020, 2, 14), date(2020, 3, 27)),
}
LEAD_LOOKBACK_WEEKS: Final[int] = 26
TREND_LOOKBACK_WEEKS: Final[int] = 26
VOL_LOOKBACK_WEEKS: Final[int] = 13
FORWARD_RISK_WEEKS: Final[int] = 13
MIN_CORR_OBS: Final[int] = 3
DEGENERATE_STD_GUARD: Final[float] = 1.0e-12
MAX_CHURN_RATE: Final[float] = 0.65
MAX_FALSE_ALARM_FREQUENCY: Final[float] = 0.50
MAX_WINDOW_DISPERSION: Final[float] = 0.50

State = Literal["CONTRACTION", "SLOWDOWN", "EXPANSION"]


@dataclass(frozen=True, slots=True)
class MarketLabels:
    """pure. Forward market labels used by the cycle evaluation protocol."""

    dates: tuple[date, ...]
    forward_return: NDArray[np.float64]
    forward_risk: NDArray[np.float64]
    forward_drawdown: NDArray[np.float64]
    composite_score: NDArray[np.float64]
    states: tuple[State, ...]


@dataclass(frozen=True, slots=True)
class EvaluationSeries:
    """pure. One evaluated object's cycle score series."""

    name: str
    legal_status: str
    dates: tuple[date, ...]
    score: NDArray[np.float64]


def _to_jsonable(obj: object) -> object:  # noqa: PLR0911
    """pure. Convert payload values to strict JSON-compatible objects."""
    if isinstance(obj, EvaluationSeries):
        return {
            "name": obj.name,
            "legal_status": obj.legal_status,
            "dates": [day.isoformat() for day in obj.dates],
            "score": [_to_jsonable(value) for value in obj.score],
        }
    if isinstance(obj, np.ndarray):
        return [_to_jsonable(value) for value in obj]
    if isinstance(obj, np.floating | float):
        value = float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, np.integer | int):
        return int(obj)
    if isinstance(obj, date):
        return obj.isoformat()
    if isinstance(obj, Mapping):
        return {str(key): _to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, Sequence) and not isinstance(obj, str | bytes | bytearray):
        return [_to_jsonable(value) for value in obj]
    return obj


def _fmt(value: object) -> str:
    """pure. Format deterministic Markdown table cells."""
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6f}" if np.isfinite(value) else "nan"
    return str(value)


def _safe_zscore(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """pure. Standardize finite values with a degenerate guard."""
    out = np.full(values.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(values)
    if int(finite.sum()) < MIN_CORR_OBS:
        return out
    sample = values[finite]
    scale = float(np.std(sample, ddof=0))
    if scale <= DEGENERATE_STD_GUARD:
        out[finite] = 0.0
        return out
    out[finite] = (sample - float(np.mean(sample))) / scale
    return out


def _corr(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """pure. Pearson correlation with finite and degenerate guards."""
    finite = np.isfinite(a) & np.isfinite(b)
    if int(finite.sum()) < MIN_CORR_OBS:
        return float("nan")
    x = a[finite]
    y = b[finite]
    if float(np.std(x, ddof=0)) <= DEGENERATE_STD_GUARD:
        return float("nan")
    if float(np.std(y, ddof=0)) <= DEGENERATE_STD_GUARD:
        return float("nan")
    value = float(np.corrcoef(x, y)[0, 1])
    return value if np.isfinite(value) else float("nan")


def _rank_corr(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """pure. Spearman rank correlation with finite and degenerate guards."""
    finite = np.isfinite(a) & np.isfinite(b)
    if int(finite.sum()) < MIN_CORR_OBS:
        return float("nan")
    return _corr(
        rankdata(a[finite]).astype(np.float64),
        rankdata(b[finite]).astype(np.float64),
    )


def _tercile_bounds(values: NDArray[np.float64]) -> tuple[float, float] | None:
    """pure. Return lower and upper tercile bounds for finite values."""
    finite = values[np.isfinite(values)]
    if finite.size < MIN_CORR_OBS:
        return None
    lower, upper = np.quantile(finite, [1.0 / 3.0, 2.0 / 3.0])
    if float(upper - lower) <= DEGENERATE_STD_GUARD:
        return None
    return float(lower), float(upper)


def _cycle_states_from_scores(scores: NDArray[np.float64]) -> tuple[State, ...]:
    """pure. Map registered continuous scores into three registered cycle states."""
    bounds = _tercile_bounds(scores)
    if bounds is None:
        return tuple("SLOWDOWN" for _ in scores)
    lower, upper = bounds
    out: list[State] = []
    for value in scores:
        if not np.isfinite(value):
            out.append("SLOWDOWN")
        elif value <= lower:
            out.append("CONTRACTION")
        elif value >= upper:
            out.append("EXPANSION")
        else:
            out.append("SLOWDOWN")
    return tuple(out)


def _primary_labels_from_market_frame(
    forward_return: NDArray[np.float64],
    forward_risk: NDArray[np.float64],
    forward_drawdown: NDArray[np.float64],
) -> MarketLabels:
    """pure. Build the frozen return-risk composite label and true states."""
    composite = (
        _safe_zscore(forward_return)
        - (0.5 * _safe_zscore(forward_risk))
        - (0.5 * _safe_zscore(forward_drawdown))
    )
    return MarketLabels(
        dates=(),
        forward_return=forward_return,
        forward_risk=forward_risk,
        forward_drawdown=forward_drawdown,
        composite_score=composite,
        states=_cycle_states_from_scores(composite),
    )


def _load_market_labels() -> MarketLabels:
    """io: Load local NDX prices and construct frozen forward market labels."""
    price_frame = pd.read_parquet(PRICE_PATH)
    price_frame["date"] = pd.to_datetime(price_frame["date"])
    weekly_prices = (
        price_frame.set_index("date")
        .sort_index()["close"]
        .resample("W-FRI")
        .last()
        .dropna()
        .astype(np.float64)
    )
    weekly_returns = np.log(weekly_prices / weekly_prices.shift(1)).astype(np.float64)
    forward_return = np.log(weekly_prices.shift(-FORECAST_HORIZON_WEEKS) / weekly_prices)
    forward_risk = (
        weekly_returns.shift(-FORWARD_RISK_WEEKS)
        .rolling(FORWARD_RISK_WEEKS, min_periods=FORWARD_RISK_WEEKS)
        .std(ddof=0)
        .shift(-(FORWARD_RISK_WEEKS - 1))
    )
    forward_drawdown = _forward_max_drawdown(weekly_prices, FORWARD_RISK_WEEKS)
    frame = pd.DataFrame(
        {
            "forward_return": forward_return,
            "forward_risk": forward_risk,
            "forward_drawdown": forward_drawdown,
        },
    ).dropna()
    labels = _primary_labels_from_market_frame(
        frame["forward_return"].to_numpy(dtype=np.float64),
        frame["forward_risk"].to_numpy(dtype=np.float64),
        frame["forward_drawdown"].to_numpy(dtype=np.float64),
    )
    return MarketLabels(
        dates=tuple(ts.date() for ts in frame.index),
        forward_return=labels.forward_return,
        forward_risk=labels.forward_risk,
        forward_drawdown=labels.forward_drawdown,
        composite_score=labels.composite_score,
        states=labels.states,
    )


def _forward_max_drawdown(prices: pd.Series, horizon_weeks: int) -> pd.Series:
    """pure. Compute positive forward max drawdown over a fixed weekly horizon."""
    values = prices.to_numpy(dtype=np.float64)
    out = np.full(values.shape, np.nan, dtype=np.float64)
    for idx in range(values.size):
        end = idx + horizon_weeks + 1
        if end > values.size or not np.isfinite(values[idx]) or values[idx] <= 0.0:
            continue
        path = values[idx:end] / values[idx]
        running_peak = np.maximum.accumulate(path)
        drawdowns = (running_peak - path) / running_peak
        out[idx] = float(np.max(drawdowns))
    return pd.Series(out, index=prices.index, name="forward_max_drawdown")


def _align_labels(
    series: EvaluationSeries,
    labels: MarketLabels,
) -> tuple[EvaluationSeries, MarketLabels]:
    """pure. Align an evaluation series with market labels by date."""
    label_pos = {day: idx for idx, day in enumerate(labels.dates)}
    series_pos = {day: idx for idx, day in enumerate(series.dates)}
    common = tuple(day for day in series.dates if day in label_pos)
    series_idx = np.asarray([series_pos[day] for day in common], dtype=np.int64)
    label_idx = np.asarray([label_pos[day] for day in common], dtype=np.int64)
    return (
        EvaluationSeries(
            name=series.name,
            legal_status=series.legal_status,
            dates=common,
            score=series.score[series_idx],
        ),
        MarketLabels(
            dates=common,
            forward_return=labels.forward_return[label_idx],
            forward_risk=labels.forward_risk[label_idx],
            forward_drawdown=labels.forward_drawdown[label_idx],
            composite_score=labels.composite_score[label_idx],
            states=tuple(labels.states[idx] for idx in label_idx),
        ),
    )


def _classification_metrics(
    truth: Sequence[str],
    pred: Sequence[str],
) -> dict[str, Any]:
    """pure. Compute balanced accuracy, macro-F1, confusion, precision and recall."""
    confusion: dict[str, dict[str, int]] = {
        actual: dict.fromkeys(STATE_ORDER, 0) for actual in STATE_ORDER
    }
    for actual, guess in zip(truth, pred, strict=True):
        if actual in confusion and guess in confusion[actual]:
            confusion[actual][guess] += 1
    recalls: dict[str, float] = {}
    precisions: dict[str, float] = {}
    f1_values: list[float] = []
    for state in STATE_ORDER:
        tp = confusion[state][state]
        actual_total = sum(confusion[state].values())
        pred_total = sum(confusion[actual][state] for actual in STATE_ORDER)
        recall = float(tp / actual_total) if actual_total > 0 else float("nan")
        precision = float(tp / pred_total) if pred_total > 0 else float("nan")
        if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0.0:
            f1 = 2.0 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        recalls[state] = recall
        precisions[state] = precision
        f1_values.append(f1)
    finite_recalls = [value for value in recalls.values() if np.isfinite(value)]
    return {
        "balanced_accuracy": float(np.mean(finite_recalls)) if finite_recalls else float("nan"),
        "macro_f1": float(np.mean(f1_values)) if f1_values else float("nan"),
        "confusion_matrix": confusion,
        "per_state_precision": precisions,
        "per_state_recall": recalls,
    }


def _direction_metrics(series: EvaluationSeries, labels: MarketLabels) -> dict[str, float | int]:
    """pure. Evaluate directional cycle-score skill against return and risk labels."""
    return {
        "n_obs": len(series.dates),
        "corr_forward_return": _corr(series.score, labels.forward_return),
        "rank_corr_forward_return": _rank_corr(series.score, labels.forward_return),
        "corr_forward_risk": _corr(series.score, labels.forward_risk),
        "rank_corr_forward_risk": _rank_corr(series.score, labels.forward_risk),
        "corr_forward_drawdown": _corr(series.score, labels.forward_drawdown),
        "rank_corr_forward_drawdown": _rank_corr(series.score, labels.forward_drawdown),
        "corr_composite": _corr(series.score, labels.composite_score),
        "rank_corr_composite": _rank_corr(series.score, labels.composite_score),
    }


def _stability_metrics(series: EvaluationSeries, states: Sequence[State]) -> dict[str, float | int]:
    """pure. Compute score and state stability diagnostics."""
    if len(states) <= 1:
        return {
            "state_persistence": float("nan"),
            "transition_churn_rate": float("nan"),
            "one_step_flip_frequency": float("nan"),
            "score_autocorrelation": float("nan"),
            "score_mean_abs_change": float("nan"),
            "n_obs": len(states),
        }
    transitions = [left != right for left, right in pairwise(states)]
    direct_flips = [
        {left, right} == {"EXPANSION", "CONTRACTION"} for left, right in pairwise(states)
    ]
    diffs = np.diff(series.score)
    return {
        "state_persistence": float(1.0 - np.mean(transitions)),
        "transition_churn_rate": float(np.mean(transitions)),
        "one_step_flip_frequency": float(np.mean(direct_flips)),
        "score_autocorrelation": _corr(series.score[:-1], series.score[1:]),
        "score_mean_abs_change": float(np.nanmean(np.abs(diffs))) if diffs.size else float("nan"),
        "n_obs": len(states),
    }


def _lead_metrics(series: EvaluationSeries, states: Sequence[State]) -> dict[str, Any]:
    """pure. Compute crisis lead-time and false-alarm diagnostics."""
    state_by_day = dict(zip(series.dates, states, strict=True))
    score_by_day = dict(zip(series.dates, series.score, strict=True))
    lead_by_window: dict[str, int | None] = {}
    for name, (start, _end) in SANITY_CRISIS_WINDOWS.items():
        candidates: list[tuple[date, float]] = []
        for day in series.dates:
            weeks = int((pd.Timestamp(start) - pd.Timestamp(day)).days // 7)
            if 0 <= weeks <= LEAD_LOOKBACK_WEEKS and (
                state_by_day[day] == "CONTRACTION" or score_by_day[day] < 0.0
            ):
                candidates.append((day, score_by_day[day]))
        if candidates:
            first_day = min(day for day, _score in candidates)
            lead_by_window[name] = int((pd.Timestamp(start) - pd.Timestamp(first_day)).days // 7)
        else:
            lead_by_window[name] = None
    false_alarm_obs = 0
    eligible_obs = 0
    for day in series.dates:
        in_crisis_or_lead = False
        for start, end in SANITY_CRISIS_WINDOWS.values():
            lead_start = pd.Timestamp(start) - pd.Timedelta(weeks=LEAD_LOOKBACK_WEEKS)
            if lead_start.date() <= day <= end:
                in_crisis_or_lead = True
                break
        if not in_crisis_or_lead:
            eligible_obs += 1
            if state_by_day[day] == "CONTRACTION" or score_by_day[day] < 0.0:
                false_alarm_obs += 1
    valid_leads = [value for value in lead_by_window.values() if value is not None]
    return {
        "lead_weeks_by_crisis": lead_by_window,
        "mean_positive_lead_weeks": float(np.mean(valid_leads)) if valid_leads else float("nan"),
        "crisis_with_positive_lead_count": len(valid_leads),
        "false_alarm_count": false_alarm_obs,
        "false_alarm_eligible_obs": eligible_obs,
        "false_alarm_lead_frequency": float(false_alarm_obs / eligible_obs)
        if eligible_obs > 0
        else float("nan"),
    }


def _window_slice(
    series: EvaluationSeries,
    labels: MarketLabels,
    window: tuple[date, date],
) -> tuple[EvaluationSeries, MarketLabels]:
    """pure. Slice aligned series and labels to a preregistered window."""
    start, end = window
    idx = np.asarray(
        [pos for pos, day in enumerate(series.dates) if start <= day <= end],
        dtype=np.int64,
    )
    return (
        EvaluationSeries(
            name=series.name,
            legal_status=series.legal_status,
            dates=tuple(series.dates[pos] for pos in idx),
            score=series.score[idx],
        ),
        MarketLabels(
            dates=tuple(labels.dates[pos] for pos in idx),
            forward_return=labels.forward_return[idx],
            forward_risk=labels.forward_risk[idx],
            forward_drawdown=labels.forward_drawdown[idx],
            composite_score=labels.composite_score[idx],
            states=tuple(labels.states[pos] for pos in idx),
        ),
    )


def _evaluate_object(series: EvaluationSeries, labels: MarketLabels) -> dict[str, Any]:
    """pure. Compute the full unified protocol metric set for one object."""
    aligned_series, aligned_labels = _align_labels(series, labels)
    pred_states = _cycle_states_from_scores(aligned_series.score)
    direction = _direction_metrics(aligned_series, aligned_labels)
    classification = _classification_metrics(aligned_labels.states, pred_states)
    stability = _stability_metrics(aligned_series, pred_states)
    lead = _lead_metrics(aligned_series, pred_states)
    cross_window: dict[str, dict[str, Any]] = {}
    composite_values: list[float] = []
    for window_name, window in CYCLE_WINDOWS.items():
        sliced_series, sliced_labels = _window_slice(aligned_series, aligned_labels, window)
        sliced_states = _cycle_states_from_scores(sliced_series.score)
        window_direction = _direction_metrics(sliced_series, sliced_labels)
        cross_window[window_name] = {
            "n_obs": len(sliced_series.dates),
            "direction": window_direction,
            "classification": _classification_metrics(sliced_labels.states, sliced_states)
            if len(sliced_series.dates) > 0
            else None,
            "stability": _stability_metrics(sliced_series, sliced_states),
        }
        value = window_direction["corr_composite"]
        if isinstance(value, float) and np.isfinite(value):
            composite_values.append(value)
    dispersion = (
        float(np.std(np.asarray(composite_values, dtype=np.float64), ddof=0))
        if composite_values
        else float("nan")
    )
    worst = min(composite_values) if composite_values else float("nan")
    decision = _object_decision(
        {
            "direction": direction,
            "classification": classification,
            "stability": stability,
            "lead": lead,
            "dispersion": dispersion,
            "worst": worst,
        },
    )
    return {
        "metadata": {
            "legal_status": series.legal_status,
            "first_date": aligned_series.dates[0].isoformat() if aligned_series.dates else None,
            "last_date": aligned_series.dates[-1].isoformat() if aligned_series.dates else None,
            "n_obs": len(aligned_series.dates),
        },
        "series": aligned_series,
        "predicted_states": pred_states,
        "continuous_score_metrics": direction,
        "discrete_state_metrics": classification,
        "lead_lag_metrics": lead,
        "stability_metrics": stability,
        "cross_window_consistency_metrics": {
            "windows": cross_window,
            "corr_composite_window_dispersion": dispersion,
            "worst_window_corr_composite": worst,
        },
        "decision": decision,
    }


def _object_decision(metrics: Mapping[str, Any]) -> dict[str, str]:
    """pure. Apply preregistered layered success gates to one object."""
    direction = metrics["direction"]
    classification = metrics["classification"]
    stability = metrics["stability"]
    lead = metrics["lead"]
    dispersion = float(metrics["dispersion"])
    worst = float(metrics["worst"])
    n_obs = int(direction["n_obs"])
    churn = float(stability["transition_churn_rate"])
    return_corr = float(direction["corr_forward_return"])
    risk_corr = float(direction["corr_forward_risk"])
    rank_return = float(direction["rank_corr_forward_return"])
    rank_risk = float(direction["rank_corr_forward_risk"])
    layer_1 = "PASS"
    if n_obs < MIN_CORR_OBS or (np.isfinite(churn) and churn > MAX_CHURN_RATE):
        layer_1 = "FAIL"
    if not np.isfinite(return_corr) or not np.isfinite(risk_corr):
        layer_1 = "FAIL"
    layer_2 = "PASS"
    if return_corr <= 0.0 or risk_corr >= 0.0 or rank_return <= 0.0 or rank_risk >= 0.0:
        layer_2 = "FAIL"
    if float(classification["balanced_accuracy"]) <= (1.0 / 3.0):
        layer_2 = "FAIL"
    if int(lead["crisis_with_positive_lead_count"]) <= 0:
        layer_2 = "FAIL"
    false_alarm = float(lead["false_alarm_lead_frequency"])
    if np.isfinite(false_alarm) and false_alarm > MAX_FALSE_ALARM_FREQUENCY:
        layer_2 = "FAIL"
    layer_3 = "PASS"
    if not np.isfinite(worst) or worst <= 0.0:
        layer_3 = "FAIL"
    if np.isfinite(dispersion) and dispersion > MAX_WINDOW_DISPERSION:
        layer_3 = "FAIL"
    if layer_1 == "FAIL":
        layer_2 = "FAIL"
        layer_3 = "FAIL"
    if layer_1 == "PASS" and layer_2 == "PASS" and layer_3 == "PASS":
        overall = "PASS"
    elif layer_1 == "PASS":
        overall = "PARTIAL"
    else:
        overall = "FAIL"
    return {"layer_1": layer_1, "layer_2": layer_2, "layer_3": layer_3, "overall": overall}


def _load_weekly_prices() -> pd.Series:
    """io: Load weekly Friday-close NDX prices from local parquet."""
    price_frame = pd.read_parquet(PRICE_PATH)
    price_frame["date"] = pd.to_datetime(price_frame["date"])
    return (
        price_frame.set_index("date")
        .sort_index()["close"]
        .resample("W-FRI")
        .last()
        .dropna()
        .astype(np.float64)
    )


def _baseline_series(labels: MarketLabels) -> dict[str, EvaluationSeries]:
    """io: Build frozen baseline cycle-score series from local market history."""
    weekly_prices = _load_weekly_prices()
    weekly_returns = np.log(weekly_prices / weekly_prices.shift(1)).astype(np.float64)
    trend = np.log(weekly_prices / weekly_prices.shift(TREND_LOOKBACK_WEEKS)).astype(np.float64)
    vol = weekly_returns.rolling(VOL_LOOKBACK_WEEKS, min_periods=VOL_LOOKBACK_WEEKS).std(ddof=0)
    label_index = pd.DatetimeIndex(labels.dates)
    trend_aligned = trend.reindex(label_index).to_numpy(dtype=np.float64)
    vol_aligned = vol.reindex(label_index).to_numpy(dtype=np.float64)
    return {
        "CONSTANT_NEUTRAL_BASELINE": EvaluationSeries(
            name="CONSTANT_NEUTRAL_BASELINE",
            legal_status="baseline_constant_normal_state",
            dates=labels.dates,
            score=np.zeros(len(labels.dates), dtype=np.float64),
        ),
        "SIMPLE_26W_TREND_BASELINE": EvaluationSeries(
            name="SIMPLE_26W_TREND_BASELINE",
            legal_status="baseline_minimal_price_trend",
            dates=labels.dates,
            score=_safe_zscore(trend_aligned),
        ),
        "SIMPLE_13W_VOL_BASELINE": EvaluationSeries(
            name="SIMPLE_13W_VOL_BASELINE",
            legal_status="baseline_minimal_realized_volatility_state_machine",
            dates=labels.dates,
            score=-_safe_zscore(vol_aligned),
        ),
    }


def _series_from_forecast(name: str, status: str, forecast: ForecastSeries) -> EvaluationSeries:
    """pure. Convert an existing forecast-risk object into cycle-score semantics."""
    return EvaluationSeries(
        name=name,
        legal_status=status,
        dates=forecast.dates,
        score=-_safe_zscore(forecast.sigma),
    )


def _current_system_series(forward_returns: pd.Series) -> dict[str, EvaluationSeries]:
    """io: Build current-system T5 and EGARCH derived cycle proxies."""
    t5_by_window = _t5_series_by_window()
    series: dict[str, list[EvaluationSeries]] = {
        "T5_DERIVED_CYCLE_PROXY": [],
        "EGARCH_DERIVED_CYCLE_PROXY": [],
    }
    rank_window_names = {
        "Window_2017": "Window_2017",
        "Window_2018H2": "Window_2018",
        "Window_2020H1": "Window_2020",
    }
    for cycle_window_name in CURRENT_OBJECT_WINDOWS:
        rank_window_name = rank_window_names[cycle_window_name]
        t5_forecast = t5_by_window[rank_window_name]
        egarch_forecast = _egarch_normal_series(CYCLE_WINDOWS[cycle_window_name], forward_returns)
        series["T5_DERIVED_CYCLE_PROXY"].append(
            _series_from_forecast(
                "T5_DERIVED_CYCLE_PROXY",
                "evaluated_existing_object_t5_sigma_proxy_not_new_model",
                t5_forecast,
            ),
        )
        series["EGARCH_DERIVED_CYCLE_PROXY"].append(
            _series_from_forecast(
                "EGARCH_DERIVED_CYCLE_PROXY",
                "evaluated_existing_object_egarch_sigma_proxy_not_new_model",
                egarch_forecast,
            ),
        )
    return {
        name: _concat_series(name, values[0].legal_status, values)
        for name, values in series.items()
        if values
    }


def _concat_series(
    name: str,
    legal_status: str,
    series: Sequence[EvaluationSeries],
) -> EvaluationSeries:
    """pure. Concatenate disjoint evaluation windows into one series."""
    dates: list[date] = []
    scores: list[float] = []
    for item in series:
        dates.extend(item.dates)
        scores.extend(float(value) for value in item.score)
    order = np.argsort(np.asarray([pd.Timestamp(day).value for day in dates], dtype=np.int64))
    score_arr = np.asarray(scores, dtype=np.float64)[order]
    sorted_dates = tuple(dates[int(pos)] for pos in order)
    return EvaluationSeries(
        name=name,
        legal_status=legal_status,
        dates=sorted_dates,
        score=score_arr,
    )


def _final_decision(payload: Mapping[str, Any]) -> dict[str, Any]:
    """pure. Decide whether the current system has auditable cycle capability."""
    objects = payload["objects"]
    current_decisions = [
        objects[name]["decision"] for name in CURRENT_SYSTEM_OBJECTS if name in objects
    ]
    if any(
        item["layer_1"] == "PASS" and item["layer_2"] == "PASS" and item["layer_3"] == "PASS"
        for item in current_decisions
    ):
        return {
            "category": "CURRENT_SYSTEM_HAS_AUDITABLE_CYCLE_CAPABILITY",
            "allow_new_cycle_model": True,
            "reason": "At least one current-system object passed all protocol layers.",
        }
    if any(item["layer_1"] == "PASS" and item["layer_2"] == "PASS" for item in current_decisions):
        return {
            "category": "CURRENT_SYSTEM_HAS_LOCAL_BUT_NOT_GENERAL_CYCLE_CAPABILITY",
            "allow_new_cycle_model": False,
            "reason": (
                "At least one current-system object passed local decision gates "
                "but failed cross-window survival."
            ),
        }
    return {
        "category": "CURRENT_SYSTEM_HAS_NO_CYCLE_CAPABILITY",
        "allow_new_cycle_model": False,
        "reason": "No current-system object passed the decision and cross-window gates.",
    }


def _baseline_comparison(objects: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """pure. Build a compact baseline comparison table."""
    rows: list[dict[str, Any]] = []
    for name, result in objects.items():
        metrics = result["continuous_score_metrics"]
        decision = result["decision"]
        rows.append(
            {
                "object": name,
                "legal_status": result["metadata"]["legal_status"],
                "n_obs": result["metadata"]["n_obs"],
                "corr_forward_return": metrics["corr_forward_return"],
                "corr_forward_risk": metrics["corr_forward_risk"],
                "balanced_accuracy": result["discrete_state_metrics"]["balanced_accuracy"],
                "worst_window_corr_composite": result["cross_window_consistency_metrics"][
                    "worst_window_corr_composite"
                ],
                "layer_1": decision["layer_1"],
                "layer_2": decision["layer_2"],
                "layer_3": decision["layer_3"],
                "overall": decision["overall"],
            },
        )
    return rows


def _write_protocol_docs() -> None:
    """io: Write frozen protocol and label/baseline documentation."""
    PROTOCOL_DOC.parent.mkdir(parents=True, exist_ok=True)
    PROTOCOL_DOC.write_text(_protocol_doc_text(), encoding="utf-8")
    LABEL_BASELINE_DOC.write_text(_label_baseline_doc_text(), encoding="utf-8")


def _protocol_doc_text() -> str:
    """pure. Render the Cycle Evaluation Protocol document."""
    return f"""# Cycle Evaluation Protocol

Status: FROZEN FOR THIS AUDIT.

## 1. Task Objects

Task 1 is `cycle_score_t`, a continuous weekly score. High score means expansion. Low score means contraction.

Task 2 is `cycle_state_t`, a discrete weekly state with exactly three preregistered states: `EXPANSION`, `SLOWDOWN`, and `CONTRACTION`. Scores are mapped by terciles, with the lowest tercile labeled `CONTRACTION`, the middle tercile labeled `SLOWDOWN`, and the highest tercile labeled `EXPANSION`.

## 2. Labels

Primary endogenous forward market label: 52-week forward log return, {FORWARD_RISK_WEEKS}-week forward realized weekly volatility, and {FORWARD_RISK_WEEKS}-week forward max drawdown from local `NASDAQXNDX` prices. The frozen composite is `z(forward_return) - 0.5*z(forward_risk) - 0.5*z(forward_drawdown)`.

External sanity-check label: preregistered crisis windows `2000-03-24..2002-10-11`, `2008-09-12..2009-03-13`, and `2020-02-14..2020-03-27`. These windows are used only for lead-time and sanity overlay. They are never used for training.

## 3. Metrics

Directionality: Pearson and rank correlation of `cycle_score_t` with future return, future risk, future drawdown, and the composite label.

Classification: balanced accuracy, macro-F1, confusion matrix, and per-state precision/recall for `cycle_state_t`.

Lead time: first contraction signal inside the {LEAD_LOOKBACK_WEEKS}-week pre-crisis window, transition/crisis lead count, and false-alarm lead frequency outside crisis plus lead windows.

Stability: state persistence, transition churn rate, one-step `EXPANSION` to `CONTRACTION` flip frequency, score autocorrelation, and score mean absolute change.

Cross-window consistency: all metrics are reported for 2017, 2018H2, 2020H1, 2008, and 2000-2002 where data exists. The protocol also reports window dispersion and worst-window composite correlation.

## 4. Success Standards

Layer 1 usability gate: finite output, at least {MIN_CORR_OBS} aligned observations, no global direction undefined, and transition churn not above 0.65.

Layer 2 decision gate: positive return direction, negative risk direction, positive rank-return direction, negative rank-risk direction, balanced accuracy above constant-chance 1/3, at least one positive crisis lead, and false-alarm frequency not above 0.50.

Layer 3 cross-window survival gate: worst-window composite correlation must be positive and cross-window dispersion must not exceed 0.50.

Object result labels are `PASS`, `PARTIAL`, or `FAIL` by layer. The system-level decision is one of `CURRENT_SYSTEM_HAS_NO_CYCLE_CAPABILITY`, `CURRENT_SYSTEM_HAS_LOCAL_BUT_NOT_GENERAL_CYCLE_CAPABILITY`, or `CURRENT_SYSTEM_HAS_AUDITABLE_CYCLE_CAPABILITY`.

## 5. Evidence Boundaries

通过方向性 ≠ 通过周期判定

通过危机窗口 ≠ 可泛化到所有周期

通过单一标签体系 ≠ 真正识别经济周期
"""


def _label_baseline_doc_text() -> str:
    """pure. Render label construction and baseline documentation."""
    return """# Label Construction And Baselines

## Labels

The primary label is endogenous and market-only. It uses local `NASDAQXNDX` weekly prices to compute 52-week forward log return, 13-week forward realized weekly volatility, and 13-week forward max drawdown. The composite label rewards future return and penalizes future risk and drawdown. This is the main evaluation label.

The external sanity label is event-window only. NBER-style recession/crisis windows are represented by fixed, preregistered market crisis windows for 2000-2002, 2008, and 2020H1. These labels are never used for training or score calibration.

## Baselines

`CONSTANT_NEUTRAL_BASELINE`: always emits neutral score 0. It represents the no-cycle-skill null.

`SIMPLE_26W_TREND_BASELINE`: emits the standardized trailing 26-week NDX log return. It represents the minimum price-trend hypothesis.

`SIMPLE_13W_VOL_BASELINE`: emits negative standardized trailing 13-week realized weekly volatility. It represents the minimum realized-volatility state-machine hypothesis.

`T5_DERIVED_CYCLE_PROXY`: converts the existing T5 sigma output into cycle semantics by taking negative standardized sigma. It is an evaluated existing object, not a new model.

`EGARCH_DERIVED_CYCLE_PROXY`: converts the existing EGARCH-Normal sigma output into cycle semantics by taking negative standardized sigma. It is an evaluated existing object, not a new model.
"""


def _markdown_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    """pure. Render a Markdown table."""
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines.extend(
        "| " + " | ".join(_fmt(row.get(column, "")) for column in columns) + " |" for row in rows
    )
    return "\n".join(lines)


def _write_results(payload: Mapping[str, Any]) -> None:
    """io: Write JSON, Markdown tables, and formal decision artifacts."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "results.json").write_text(
        json.dumps(_to_jsonable(payload), allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    baseline_rows = payload["tables"]["baseline_comparison"]
    baseline_md = _markdown_table(
        baseline_rows,
        (
            "object",
            "n_obs",
            "corr_forward_return",
            "corr_forward_risk",
            "balanced_accuracy",
            "worst_window_corr_composite",
            "layer_1",
            "layer_2",
            "layer_3",
            "overall",
        ),
    )
    (OUTPUT_DIR / "baseline_comparison_table.md").write_text(baseline_md + "\n", encoding="utf-8")
    worst_rows = payload["tables"]["worst_window"]
    (OUTPUT_DIR / "worst_window_table.md").write_text(
        _markdown_table(worst_rows, ("object", "worst_window_corr_composite", "dispersion")) + "\n",
        encoding="utf-8",
    )
    lead_rows = payload["tables"]["lead_false_alarm"]
    (OUTPUT_DIR / "lead_false_alarm_summary.md").write_text(
        _markdown_table(
            lead_rows,
            (
                "object",
                "crisis_with_positive_lead_count",
                "mean_positive_lead_weeks",
                "false_alarm_lead_frequency",
            ),
        )
        + "\n",
        encoding="utf-8",
    )
    results_md = f"""# Cycle Evaluation Results

Generated by `src/research/run_cycle_evaluation_protocol.py`.

## Main Baseline Comparison

{baseline_md}

## Formal Decision

`{payload["formal_decision"]["category"]}`

Reason: {payload["formal_decision"]["reason"]}
"""
    (OUTPUT_DIR / "results.md").write_text(results_md, encoding="utf-8")
    DECISION_DOC.write_text(_decision_doc_text(payload), encoding="utf-8")


def _decision_doc_text(payload: Mapping[str, Any]) -> str:
    """pure. Render one-page formal decision."""
    rows = payload["tables"]["baseline_comparison"]
    table = _markdown_table(
        rows,
        (
            "object",
            "corr_forward_return",
            "corr_forward_risk",
            "balanced_accuracy",
            "layer_1",
            "layer_2",
            "layer_3",
            "overall",
        ),
    )
    decision = payload["formal_decision"]
    return f"""# Formal Cycle Capability Decision

## Protocol

The audit evaluates continuous `cycle_score_t` and discrete three-state `cycle_state_t` under the frozen Cycle Evaluation Protocol. The primary truth label is a forward market return-risk composite. Crisis windows are external sanity checks only.

## Labels And Baselines

Primary label: 52-week forward return minus forward risk and drawdown penalties. Baselines: constant neutral, simple 26-week trend, simple 13-week realized volatility, T5-derived cycle proxy, and EGARCH-derived cycle proxy.

## Current-System Evaluation

{table}

## Layer Outcomes

Layer 1 tests usability. Layer 2 tests decision skill. Layer 3 tests cross-window survival. Passing directionality alone is not cycle capability.

## Decision

Main category: `{decision["category"]}`.

New cycle model allowed now: `{decision["allow_new_cycle_model"]}`.

Reason: {decision["reason"]}

Future cycle-model project may be opened only after a new preregistered information set or label source is introduced, and only if it first beats the frozen baseline family under this protocol without relying on trigger, override, hard switch, MoE, fixed T5 downstream patching, or rank-scale hybrid expansion.
"""


def _build_tables(objects: Mapping[str, Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """pure. Build all required result tables."""
    baseline = _baseline_comparison(objects)
    worst: list[dict[str, Any]] = []
    lead: list[dict[str, Any]] = []
    for name, result in objects.items():
        cross = result["cross_window_consistency_metrics"]
        lead_metrics = result["lead_lag_metrics"]
        worst.append(
            {
                "object": name,
                "worst_window_corr_composite": cross["worst_window_corr_composite"],
                "dispersion": cross["corr_composite_window_dispersion"],
            },
        )
        lead.append(
            {
                "object": name,
                "crisis_with_positive_lead_count": lead_metrics["crisis_with_positive_lead_count"],
                "mean_positive_lead_weeks": lead_metrics["mean_positive_lead_weeks"],
                "false_alarm_lead_frequency": lead_metrics["false_alarm_lead_frequency"],
            },
        )
    return {
        "baseline_comparison": baseline,
        "worst_window": worst,
        "lead_false_alarm": lead,
    }


def run(argv: list[str] | None = None) -> int:
    """io: Execute the Cycle Evaluation Protocol and write all audit artifacts."""
    parser = argparse.ArgumentParser(prog="run_cycle_evaluation_protocol")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON to stdout.")
    args = parser.parse_args(argv)

    _write_protocol_docs()
    labels = _load_market_labels()
    forward_returns = _load_weekly_forward_returns()
    series_by_name = _baseline_series(labels)
    series_by_name.update(_current_system_series(forward_returns))
    objects = {name: _evaluate_object(series, labels) for name, series in series_by_name.items()}
    payload: dict[str, Any] = {
        "protocol": {
            "name": "Cycle Evaluation Protocol",
            "status": "FROZEN_FOR_THIS_AUDIT",
            "protocol_doc": PROTOCOL_DOC.as_posix(),
            "label_baseline_doc": LABEL_BASELINE_DOC.as_posix(),
            "train_window_weeks": TRAIN_WINDOW_WEEKS,
            "train_embargo_weeks": TRAIN_EMBARGO_WEEKS,
            "forecast_horizon_weeks": FORECAST_HORIZON_WEEKS,
            "cycle_windows": {
                name: [start.isoformat(), end.isoformat()]
                for name, (start, end) in CYCLE_WINDOWS.items()
            },
            "sanity_crisis_windows": {
                name: [start.isoformat(), end.isoformat()]
                for name, (start, end) in SANITY_CRISIS_WINDOWS.items()
            },
        },
        "objects": objects,
    }
    payload["tables"] = _build_tables(objects)
    payload["formal_decision"] = _final_decision(payload)
    _write_results(payload)
    text = json.dumps(
        _to_jsonable(payload),
        allow_nan=False,
        indent=2 if args.pretty else None,
        sort_keys=True,
    )
    sys.stdout.write(text + "\n")
    return 0


def main() -> None:
    """io: process exit entrypoint."""
    raise SystemExit(run())


if __name__ == "__main__":
    main()
