from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import cast

import numpy as np

from backtest.metrics import ceq_annualized, realized_forward_returns
from data_contract.nasdaq_client import NasdaqClient

DERIVED_PRICE_SERIES = "NASDAQXNDX"
TAUS = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95], dtype=np.float64)
STRICT_START = date(2015, 2, 20)
MIN_HISTORY = 26


@dataclass(frozen=True, slots=True)
class EvalRow:
    as_of: date
    quantiles: np.ndarray
    realized: float
    offense_final: float


def _load_rows(results_path: Path) -> list[EvalRow]:
    payloads = [json.loads(line) for line in results_path.read_text().splitlines() if line.strip()]
    series = NasdaqClient(cache_root=Path("data/raw/nasdaq")).get_series(
        DERIVED_PRICE_SERIES,
        date(2024, 12, 27),
    )
    as_of_dates = [date.fromisoformat(row["as_of_date"]) for row in payloads]
    realized = realized_forward_returns(series, as_of_dates)
    rows: list[EvalRow] = []
    for idx, row in enumerate(payloads):
        as_of = as_of_dates[idx]
        quantiles = np.array(
            [row["distribution"][key] for key in ("q05", "q10", "q25", "q50", "q75", "q90", "q95")],
            dtype=np.float64,
        )
        y_true = float(realized[idx])
        if (
            row["vintage_mode"] != "strict"
            or as_of < STRICT_START
            or not math.isfinite(y_true)
            or not np.isfinite(quantiles).all()
        ):
            continue
        rows.append(
            EvalRow(
                as_of=as_of,
                quantiles=quantiles,
                realized=y_true,
                offense_final=float(row["decision"]["offense_final"]),
            ),
        )
    return rows


def _baseline_a_quantiles(history: np.ndarray, fallback: float) -> np.ndarray:
    finite = history[np.isfinite(history)]
    if finite.shape[0] < 2:
        return np.full(TAUS.shape, fallback, dtype=np.float64)
    return np.quantile(finite, TAUS).astype(np.float64)


def _quantile_score(y_true: float, quantiles: np.ndarray) -> float:
    losses = np.maximum(TAUS * (y_true - quantiles), (TAUS - 1.0) * (y_true - quantiles))
    return float(2.0 * np.mean(losses))


def _evaluate(rows: list[EvalRow], calibrated: np.ndarray) -> dict[str, float]:
    y = np.array([row.realized for row in rows], dtype=np.float64)
    q10_hits = y <= calibrated[:, 1]
    q90_hits = y <= calibrated[:, 5]
    prod_crps = np.array(
        [_quantile_score(float(y[idx]), calibrated[idx]) for idx in range(y.shape[0])],
        dtype=np.float64,
    )
    baseline_crps = np.array(
        [
            _quantile_score(
                float(y[idx]),
                _baseline_a_quantiles(y[:idx], float(y[idx])),
            )
            for idx in range(y.shape[0])
        ],
        dtype=np.float64,
    )
    offense = np.array([row.offense_final for row in rows], dtype=np.float64)
    production_returns = offense / 100.0 * y
    baseline_b_returns = 0.5 * y
    return {
        "q10_error": abs(float(np.mean(q10_hits)) - 0.10),
        "q90_error": abs(float(np.mean(q90_hits)) - 0.90),
        "crps_improvement": float(1.0 - np.mean(prod_crps) / np.mean(baseline_crps)),
        "ceq_diff": float(ceq_annualized(production_returns) - ceq_annualized(baseline_b_returns)),
    }


def _method_a_location_scale(rows: list[EvalRow]) -> np.ndarray:
    original = np.stack([row.quantiles for row in rows])
    out = original.copy()
    for idx, row in enumerate(rows):
        if idx < MIN_HISTORY:
            continue
        history = rows[:idx]
        realized_hist = np.array([item.realized for item in history], dtype=np.float64)
        q50_hist = np.array([item.quantiles[3] for item in history], dtype=np.float64)
        spread_hist = np.array([item.quantiles[5] - item.quantiles[1] for item in history], dtype=np.float64)
        shift = float(np.median(realized_hist - q50_hist))
        realized_idr = float(np.quantile(realized_hist, 0.90) - np.quantile(realized_hist, 0.10))
        pred_idr = float(np.median(spread_hist))
        scale = 1.0 if pred_idr <= 0.0 else realized_idr / pred_idr
        scale = float(np.clip(scale, 0.25, 4.0))
        q50 = row.quantiles[3] + shift
        out[idx] = q50 + scale * (row.quantiles - row.quantiles[3])
        out[idx] = np.maximum.accumulate(out[idx])
    return out


def _isotonic_fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(x, kind="mergesort")
    xs = x[order].astype(np.float64)
    ys = y[order].astype(np.float64)
    values = ys.copy()
    weights = np.ones(xs.shape[0], dtype=np.float64)
    starts = np.arange(xs.shape[0], dtype=np.int64)
    ends = np.arange(xs.shape[0], dtype=np.int64)
    block_count = xs.shape[0]
    idx = 0
    while idx < block_count - 1:
        if values[idx] <= values[idx + 1]:
            idx += 1
            continue
        total_weight = weights[idx] + weights[idx + 1]
        pooled = (values[idx] * weights[idx] + values[idx + 1] * weights[idx + 1]) / total_weight
        values[idx] = pooled
        weights[idx] = total_weight
        ends[idx] = ends[idx + 1]
        values = np.delete(values, idx + 1)
        weights = np.delete(weights, idx + 1)
        starts = np.delete(starts, idx + 1)
        ends = np.delete(ends, idx + 1)
        block_count -= 1
        if idx > 0:
            idx -= 1
    thresholds = np.array([xs[ends[i]] for i in range(block_count)], dtype=np.float64)
    return thresholds, values


def _isotonic_predict(x: np.ndarray, thresholds: np.ndarray, values: np.ndarray) -> np.ndarray:
    block_idx = np.searchsorted(thresholds, x, side="left")
    block_idx = np.clip(block_idx, 0, values.shape[0] - 1)
    return values[cast(np.ndarray, block_idx)]


def _piecewise_map(anchor_x: np.ndarray, anchor_y: np.ndarray, values: np.ndarray) -> np.ndarray:
    x10, x50, x90 = anchor_x
    y10, y50, y90 = anchor_y
    slope_lo = (y50 - y10) / (x50 - x10) if x50 != x10 else 1.0
    slope_hi = (y90 - y50) / (x90 - x50) if x90 != x50 else 1.0
    out = np.empty_like(values)
    for idx, value in enumerate(values):
        if value <= x10:
            out[idx] = y10 + slope_lo * (value - x10)
        elif value <= x50:
            out[idx] = y10 + (y50 - y10) * (value - x10) / (x50 - x10) if x50 != x10 else y50
        elif value <= x90:
            out[idx] = y50 + (y90 - y50) * (value - x50) / (x90 - x50) if x90 != x50 else y90
        else:
            out[idx] = y90 + slope_hi * (value - x90)
    return np.maximum.accumulate(out)


def _method_b_monotone(rows: list[EvalRow]) -> np.ndarray:
    original = np.stack([row.quantiles for row in rows])
    out = original.copy()
    tau_indices = (1, 3, 5)
    for idx, row in enumerate(rows):
        if idx < MIN_HISTORY:
            continue
        history = rows[:idx]
        anchor_y = np.empty(3, dtype=np.float64)
        for anchor_idx, tau_idx in enumerate(tau_indices):
            x_hist = np.array([item.quantiles[tau_idx] for item in history], dtype=np.float64)
            y_hist = np.array([item.realized for item in history], dtype=np.float64)
            thresholds, values = _isotonic_fit(x_hist, y_hist)
            anchor_y[anchor_idx] = float(
                _isotonic_predict(
                    np.array([row.quantiles[tau_idx]], dtype=np.float64),
                    thresholds,
                    values,
                )[0],
            )
        anchor_y = np.maximum.accumulate(anchor_y)
        out[idx] = _piecewise_map(row.quantiles[[1, 3, 5]], anchor_y, row.quantiles)
    return out


def main() -> None:
    rows = _load_rows(Path("/tmp/backtest-10y-12w-v5/backtest/backtest_results.jsonl"))
    baseline = np.stack([row.quantiles for row in rows])
    method_a = _method_a_location_scale(rows)
    method_b = _method_b_monotone(rows)
    payload = {
        "baseline": _evaluate(rows, baseline),
        "method_a": _evaluate(rows, method_a),
        "method_b": _evaluate(rows, method_b),
    }
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
