"""Run Phase 0A trigger lead-lag audit.

io: reads local cached market data and writes JSON results to stdout.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
NDX_CLOSE_PATH = REPO_ROOT / "data/raw/nasdaq/NASDAQXNDX/close.parquet"
YAHOO_ROOT = REPO_ROOT / "data/raw/yahoo"
FRED_ROOT = REPO_ROOT / "data/raw/fred"

LEAD_LAGS = (-5, -4, -3, -2, -1)
TARGET_SERIES = "NASDAQXNDX"
MAIN_TARGET = "next_trading_day_abs_return"
SECONDARY_TARGET = "future_5_trading_day_realized_vol_proxy"
FALSE_POSITIVE_THRESHOLD = 0.40
MIN_CORR_OBS = 5
MIN_FALSE_POSITIVE_OBS = 20
MIN_POSITIVE_CRISIS_WINDOWS = 2

CRISIS_WINDOWS = {
    "2000": (pd.Timestamp("2000-01-01"), pd.Timestamp("2002-12-31")),
    "2008": (pd.Timestamp("2008-01-01"), pd.Timestamp("2009-12-31")),
    "2020": (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-12-31")),
}


@dataclass(frozen=True, slots=True)
class TriggerSpec:
    name: str
    required_inputs: tuple[str, ...]


TRIGGERS = (
    TriggerSpec("VIX9D / VIX", ("yahoo:^VIX9D", "yahoo:^VIX")),
    TriggerSpec("VIX / VIX3M", ("yahoo:^VIX", "yahoo:^VIX3M")),
    TriggerSpec("ΔVIX9D", ("yahoo:^VIX9D",)),
    TriggerSpec("credit spread jump", ("fred:BAA10Y",)),
    TriggerSpec("FRA-OIS widening", ("missing:FRA_OIS",)),
    TriggerSpec("liquidity stress signals", ("missing:liquidity_stress",)),
    TriggerSpec("lagged realized shock", ("derived:NASDAQXNDX",)),
)


def _read_price_series(path: Path, column: str) -> pd.Series:
    df = pd.read_parquet(path)
    if "date" not in df.columns or column not in df.columns:
        raise ValueError(f"missing date/{column} columns in {path}")
    out = pd.Series(
        pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=np.float64),
        index=pd.to_datetime(df["date"]),
        name=path.parent.name,
    )
    return out.sort_index().replace([np.inf, -np.inf], np.nan).dropna()


def _latest_fred_file(series_id: str) -> Path:
    paths = sorted((FRED_ROOT / series_id).glob("as_of=*.json"))
    if not paths:
        raise FileNotFoundError(f"missing FRED cache for {series_id}")
    return paths[-1]


def _read_fred_series(series_id: str) -> pd.Series:
    path = _latest_fred_file(series_id)
    payload = json.loads(path.read_text())
    observations = payload.get("observations", [])
    if not observations:
        raise ValueError(f"empty FRED observations for {series_id}")
    dates: list[pd.Timestamp] = []
    values: list[float] = []
    for obs in observations:
        value = obs.get("value")
        if value in (None, "."):
            continue
        dates.append(pd.Timestamp(obs["date"]))
        values.append(float(value))
    return pd.Series(values, index=pd.DatetimeIndex(dates), name=series_id).sort_index()


def _corr(x: pd.Series, y: pd.Series, *, method: str = "pearson") -> float | None:
    frame = pd.concat([x, y], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < MIN_CORR_OBS:
        return None
    value = frame.iloc[:, 0].corr(frame.iloc[:, 1], method=method)
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def _fmt(value: float | None) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def _load_base_data() -> dict[str, pd.Series]:
    ndx_close = _read_price_series(NDX_CLOSE_PATH, "close")
    ndx_return = np.log(ndx_close).diff()
    future_5rv = np.sqrt(sum(ndx_return.shift(-step) ** 2 for step in range(1, 6)))
    return {
        "ndx_close": ndx_close,
        "ndx_return": ndx_return,
        "target_next_abs": ndx_return.shift(-1).abs().rename(MAIN_TARGET),
        "target_future_5rv": future_5rv.rename(SECONDARY_TARGET),
        "yahoo:^VIX9D": _read_price_series(
            YAHOO_ROOT / "^VIX9D" / "adj_close.parquet",
            "adj_close",
        ),
        "yahoo:^VIX": _read_price_series(YAHOO_ROOT / "^VIX" / "adj_close.parquet", "adj_close"),
        "yahoo:^VIX3M": _read_price_series(
            YAHOO_ROOT / "^VIX3M" / "adj_close.parquet",
            "adj_close",
        ),
        "fred:BAA10Y": _read_fred_series("BAA10Y"),
    }


def _build_trigger_series(  # noqa: PLR0911
    spec: TriggerSpec,
    data: dict[str, pd.Series],
) -> tuple[pd.Series | None, list[str]]:
    missing = [
        item
        for item in spec.required_inputs
        if item.startswith("missing:") or (not item.startswith("derived:") and item not in data)
    ]
    if missing:
        return None, missing
    if spec.name == "VIX9D / VIX":
        return (data["yahoo:^VIX9D"] / data["yahoo:^VIX"]).rename(spec.name), []
    if spec.name == "VIX / VIX3M":
        return (data["yahoo:^VIX"] / data["yahoo:^VIX3M"]).rename(spec.name), []
    if spec.name == "ΔVIX9D":
        return data["yahoo:^VIX9D"].diff().rename(spec.name), []
    if spec.name == "credit spread jump":
        return data["fred:BAA10Y"].diff().rename(spec.name), []
    if spec.name == "lagged realized shock":
        return data["ndx_return"].abs().shift(1).rename(spec.name), []
    return None, [f"unimplemented:{spec.name}"]


def _align_to_target_calendar(signal: pd.Series, target_index: pd.DatetimeIndex) -> pd.Series:
    frame = signal.sort_index().reindex(target_index).ffill()
    return frame.replace([np.inf, -np.inf], np.nan)


def _lead_lag_metrics(
    signal: pd.Series,
    target: pd.Series,
    mask: pd.Series | None = None,
) -> dict[str, dict[str, float | None]]:
    metrics: dict[str, dict[str, float | None]] = {}
    target_use = target if mask is None else target.loc[mask]
    for lag in LEAD_LAGS:
        shifted = signal.shift(abs(lag))
        signal_use = shifted if mask is None else shifted.loc[mask]
        metrics[str(lag)] = {
            "corr": _fmt(_corr(signal_use, target_use)),
            "rank_corr": _fmt(_corr(signal_use, target_use, method="spearman")),
        }
    return metrics


def _has_positive_lead(metrics: dict[str, dict[str, float | None]]) -> bool:
    for values in metrics.values():
        corr = values["corr"]
        rank = values["rank_corr"]
        if corr is not None and rank is not None and corr > 0.0 and rank > 0.0:
            return True
    return False


def _crisis_masks(index: pd.DatetimeIndex) -> dict[str, pd.Series]:
    return {
        name: pd.Series((index >= start) & (index <= end), index=index)
        for name, (start, end) in CRISIS_WINDOWS.items()
    }


def _calm_mask(index: pd.DatetimeIndex) -> pd.Series:
    mask = pd.Series(data=True, index=index)
    for start, end in CRISIS_WINDOWS.values():
        mask &= ~((index >= start) & (index <= end))
    return mask


def _false_positive_rate(signal: pd.Series, target: pd.Series) -> float | None:
    frame = pd.concat([signal, target], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if frame.empty:
        return None
    calm = _calm_mask(frame.index)
    calm_frame = frame.loc[calm]
    if len(calm_frame) < MIN_FALSE_POSITIVE_OBS:
        return None
    signal_threshold = float(calm_frame.iloc[:, 0].quantile(0.80))
    target_threshold = float(calm_frame.iloc[:, 1].quantile(0.80))
    trigger_extreme = calm_frame.iloc[:, 0] >= signal_threshold
    if int(trigger_extreme.sum()) == 0:
        return None
    false_positive = trigger_extreme & (calm_frame.iloc[:, 1] < target_threshold)
    return float(false_positive.sum() / trigger_extreme.sum())


def _coverage_summary(signal: pd.Series, target: pd.Series) -> dict[str, Any]:
    frame = pd.concat([signal, target], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    out: dict[str, Any] = {
        "available_start": None,
        "available_end": None,
        "n_aligned": len(frame),
        "crisis_counts": {},
    }
    if frame.empty:
        return out
    out["available_start"] = str(frame.index.min().date())
    out["available_end"] = str(frame.index.max().date())
    for name, mask in _crisis_masks(frame.index).items():
        out["crisis_counts"][name] = int(mask.sum())
    return out


def _evaluate_trigger(spec: TriggerSpec, data: dict[str, pd.Series]) -> dict[str, Any]:
    target = data["target_next_abs"].dropna()
    signal_raw, missing = _build_trigger_series(spec, data)
    if signal_raw is None:
        return {
            "trigger": spec.name,
            "included": "yes",
            "data_status": "FAILED_TO_RUN_DATA_MISSING",
            "missing_inputs": missing,
            "lag_metrics": {str(lag): {"corr": None, "rank_corr": None} for lag in LEAD_LAGS},
            "false_positive_rate": None,
            "crisis_consistency": dict.fromkeys(CRISIS_WINDOWS, "FAILED_TO_RUN_DATA_MISSING"),
            "positive_crisis_count": 0,
            "secondary_target_status": "FAILED_TO_RUN_DATA_MISSING",
            "final_status": "FAILED_TO_RUN_DATA_MISSING",
        }

    signal = _align_to_target_calendar(signal_raw, target.index)
    coverage = _coverage_summary(signal, target)
    lag_metrics = _lead_lag_metrics(signal, target)
    crisis_consistency: dict[str, str] = {}
    positive_crisis_count = 0
    for name, mask in _crisis_masks(target.index).items():
        window_metrics = _lead_lag_metrics(signal, target, mask=mask)
        all_missing = all(
            values["corr"] is None and values["rank_corr"] is None
            for values in window_metrics.values()
        )
        if all_missing:
            crisis_consistency[name] = "FAILED_TO_RUN_DATA_MISSING"
            continue
        is_positive = _has_positive_lead(window_metrics)
        crisis_consistency[name] = "POSITIVE" if is_positive else "NON_POSITIVE"
        positive_crisis_count += int(is_positive)

    fp_rate = _false_positive_rate(signal, target)
    lead_ok = _has_positive_lead(lag_metrics)
    false_positive_ok = fp_rate is not None and fp_rate <= FALSE_POSITIVE_THRESHOLD
    consistency_ok = positive_crisis_count >= MIN_POSITIVE_CRISIS_WINDOWS
    has_missing_crisis = any(
        value == "FAILED_TO_RUN_DATA_MISSING" for value in crisis_consistency.values()
    )
    if has_missing_crisis:
        final_status = "FAILED_TO_RUN_DATA_MISSING"
    elif lead_ok and false_positive_ok and consistency_ok:
        final_status = "PASS"
    else:
        final_status = "FAIL"

    secondary = data["target_future_5rv"].dropna()
    secondary_frame = pd.concat([signal, secondary], axis=1).dropna()
    secondary_target_status = (
        "PASS"
        if len(secondary_frame) >= MIN_FALSE_POSITIVE_OBS
        else "FAILED_TO_RUN_SECONDARY_TARGET"
    )

    return {
        "trigger": spec.name,
        "included": "yes",
        "data_status": "PASS" if not has_missing_crisis else "FAILED_TO_RUN_DATA_MISSING",
        "missing_inputs": [],
        "availability": coverage,
        "lag_metrics": lag_metrics,
        "false_positive_rate": _fmt(fp_rate),
        "lead_ok": lead_ok,
        "false_positive_ok": false_positive_ok,
        "crisis_consistency": crisis_consistency,
        "positive_crisis_count": positive_crisis_count,
        "secondary_target_status": secondary_target_status,
        "final_status": final_status,
    }


def run() -> int:
    data = _load_base_data()
    results = [_evaluate_trigger(spec, data) for spec in TRIGGERS]
    payload = {
        "audit_protocol": {
            "frequency": "daily",
            "target_series": TARGET_SERIES,
            "main_target": MAIN_TARGET,
            "secondary_target": SECONDARY_TARGET,
            "lead_window": list(LEAD_LAGS),
            "false_positive_threshold": FALSE_POSITIVE_THRESHOLD,
            "calm_regime_exclusions": {
                name: [str(start.date()), str(end.date())]
                for name, (start, end) in CRISIS_WINDOWS.items()
            },
        },
        "results": results,
        "pass_triggers": [item["trigger"] for item in results if item["final_status"] == "PASS"],
    }
    sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
