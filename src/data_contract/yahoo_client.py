"""Yahoo Finance ETF adjusted-close adapter for panel targets."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from engine_types import TimeSeries
from errors import CachePoisonError

FetchHistory = Callable[[str, date, date], pd.DataFrame]
DEFAULT_CACHE_ROOT = Path("data/raw/yahoo")
ADJ_CLOSE_COLUMN = "Adj Close"
MIN_PRICE_POINTS = 2


def _default_fetch_history(ticker: str, start: date, end: date) -> pd.DataFrame:
    """io: Fetch daily ETF history from Yahoo Finance."""
    import yfinance as yf  # type: ignore[import-untyped]

    history = yf.Ticker(ticker).history(
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        auto_adjust=False,
        actions=False,
    )
    return cast(pd.DataFrame, history)


def _normalized_frame(history: pd.DataFrame) -> pd.DataFrame:
    if ADJ_CLOSE_COLUMN not in history.columns:
        msg = "Yahoo history response must include an Adj Close column"
        raise ValueError(msg)
    adj_close = history[ADJ_CLOSE_COLUMN].dropna()
    dates = pd.to_datetime(adj_close.index).tz_localize(None).date
    frame = pd.DataFrame(
        {
            "date": pd.Series(dates, dtype="object"),
            "adj_close": pd.Series(adj_close.to_numpy(dtype=np.float64), dtype=np.float64),
        },
    )
    return frame.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)


def log_return_series(price_series: TimeSeries) -> TimeSeries:
    """pure. Compute consecutive log returns from ETF adjusted close levels."""
    prices = np.asarray(price_series.values, dtype=np.float64)
    if prices.ndim != 1:
        msg = "price series must be one-dimensional"
        raise ValueError(msg)
    if prices.shape[0] < MIN_PRICE_POINTS:
        return TimeSeries(
            series_id=f"{price_series.series_id}_LOG_RETURN",
            timestamps=np.array([], dtype="datetime64[D]"),
            values=np.array([], dtype=np.float64),
            is_pseudo_pit=price_series.is_pseudo_pit,
        )
    if not np.isfinite(prices).all() or np.any(prices <= 0.0):
        msg = "adjusted close prices must be finite and positive"
        raise ValueError(msg)
    return TimeSeries(
        series_id=f"{price_series.series_id}_LOG_RETURN",
        timestamps=np.asarray(price_series.timestamps[1:], dtype="datetime64[D]"),
        values=np.diff(np.log(prices)).astype(np.float64),
        is_pseudo_pit=price_series.is_pseudo_pit,
    )


@dataclass(frozen=True, slots=True)
class YahooFinanceClient:
    """io: Fetch ETF adjusted close from Yahoo Finance with append-only parquet cache."""

    cache_root: Path = DEFAULT_CACHE_ROOT
    fetch_history: FetchHistory = _default_fetch_history

    def _cache_path(self, ticker: str) -> Path:
        return self.cache_root / ticker.upper() / "adj_close.parquet"

    def _load_cached(self, path: Path) -> pd.DataFrame | None:
        if not path.exists():
            return None
        cached = pd.read_parquet(path)
        cached["date"] = pd.to_datetime(cached["date"]).dt.date
        return cached.sort_values("date").reset_index(drop=True)

    def _write_cache(self, path: Path, frame: pd.DataFrame) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)

    def _merge_append_only(
        self,
        existing: pd.DataFrame | None,
        fetched: pd.DataFrame,
    ) -> pd.DataFrame:
        if existing is None or existing.empty:
            return fetched
        overlap = existing.merge(fetched, on="date", how="inner", suffixes=("_old", "_new"))
        if not overlap.empty and not np.array_equal(
            overlap["adj_close_old"].to_numpy(dtype=np.float64),
            overlap["adj_close_new"].to_numpy(dtype=np.float64),
        ):
            msg = "Yahoo adjusted-close cache would be overwritten with divergent data"
            raise CachePoisonError(msg)
        combined = pd.concat([existing, fetched], ignore_index=True)
        return combined.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)

    def fetch_etf_adjusted_close(
        self,
        ticker: str,
        start: date,
        end: date,
    ) -> TimeSeries:
        """io: Fetch one ETF adjusted-close series through the local parquet cache."""
        if end < start:
            msg = "end must be on or after start"
            raise ValueError(msg)
        cache_path = self._cache_path(ticker)
        cached = self._load_cached(cache_path)
        cache_end = None if cached is None or cached.empty else max(cached["date"])
        if cached is None or cache_end is None or cache_end < end:
            fetch_start = start if cache_end is None else min(end, cache_end + timedelta(days=1))
            fetched = _normalized_frame(self.fetch_history(ticker, fetch_start, end))
            cached = self._merge_append_only(cached, fetched)
            self._write_cache(cache_path, cached)
        window = cached[(cached["date"] >= start) & (cached["date"] <= end)].copy()
        return TimeSeries(
            series_id=ticker.upper(),
            timestamps=np.asarray(
                [np.datetime64(value, "D") for value in window["date"]],
                dtype="datetime64[D]",
            ),
            values=np.asarray(window["adj_close"].to_numpy(dtype=np.float64), dtype=np.float64),
            is_pseudo_pit=False,
        )


def fetch_etf_adjusted_close(
    ticker: str,
    start: date,
    end: date,
    *,
    cache_root: Path = DEFAULT_CACHE_ROOT,
    fetch_history: FetchHistory = _default_fetch_history,
) -> TimeSeries:
    """io: Convenience wrapper for ETF adjusted-close retrieval."""
    client = YahooFinanceClient(cache_root=cache_root, fetch_history=fetch_history)
    return client.fetch_etf_adjusted_close(ticker, start, end)
