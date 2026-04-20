"""Yahoo Finance adapter for NASDAQXNDX total-return daily close.

ADD §3.1: src/data_contract/nasdaq_client.py
ADD §4.5 cache path: data/raw/nasdaq/{series_id}/close.parquet
SRD §4.3: NASDAQXNDX earliest_strict_pit = 1985-10-01, daily close, no revision.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from engine_types import TimeSeries

YAHOO_XNDX_SYMBOL = "^XNDX"

# SRD §4.3 table — earliest date NASDAQXNDX data exists.
_NASDAQXNDX_EARLIEST = date(1985, 10, 1)

FetchHistory = Callable[[str, date], pd.DataFrame]


def _default_fetch_history(symbol: str, start_date: date) -> pd.DataFrame:
    """io: Fetch daily history from the free Yahoo Finance aggregate endpoint."""
    import yfinance as yf  # type: ignore[import-untyped]

    ticker = yf.Ticker(symbol)
    return cast(
        pd.DataFrame,
        ticker.history(start=start_date.isoformat(), auto_adjust=False),
    )


@dataclass(frozen=True, slots=True)
class NasdaqClient:
    """io: Fetch NASDAQXNDX daily close from Yahoo Finance; cache append-only parquet.

    NASDAQXNDX has no revisions (SRD §4.3 "daily close, no revision"), so
    is_pseudo_pit is always False.  The cache file grows monotonically as
    new dates are appended; existing rows are never overwritten.
    """

    cache_root: Path = Path("data/raw/nasdaq")
    fetch_history: FetchHistory = _default_fetch_history

    def _cache_path(self, series_id: str) -> Path:
        return self.cache_root / series_id.upper() / "close.parquet"

    def _load_cached(self, path: Path) -> pd.DataFrame | None:
        """io: Read cached parquet; returns DataFrame with columns [date, close] or None."""
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df

    def _fetch_rows(self, start_date: date) -> pd.DataFrame:
        """io: Fetch rows from Yahoo Finance starting at start_date."""
        history = self.fetch_history(YAHOO_XNDX_SYMBOL, start_date)
        if "Close" not in history.columns:
            msg = "Yahoo ^XNDX history response must include a Close column"
            raise ValueError(msg)

        close = history["Close"].dropna()
        dates = [ts.date() for ts in pd.to_datetime(close.index)]
        closes = [float(value) for value in close.to_numpy()]
        return pd.DataFrame({"date": dates, "close": closes})

    def _extend_cache(self, _series_id: str, start_date: date, cache_path: Path) -> pd.DataFrame:
        """io: Fetch new rows and append to the parquet cache."""
        new_rows = self._fetch_rows(start_date)
        existing = self._load_cached(cache_path)
        if existing is not None:
            combined = (
                pd.concat([existing, new_rows])
                .drop_duplicates(subset="date")
                .sort_values("date")
                .reset_index(drop=True)
            )
        else:
            combined = new_rows.sort_values("date").reset_index(drop=True)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(cache_path, index=False)
        return combined

    def get_series(self, series_id: str, as_of: date) -> TimeSeries:
        """io: Return a PIT TimeSeries for series_id through as_of.

        NASDAQXNDX has no vintages; is_pseudo_pit is always False.
        Fetches from Yahoo Finance only when the cache does not cover as_of.
        """
        if series_id.upper() != "NASDAQXNDX":
            msg = "NasdaqClient only supports NASDAQXNDX"
            raise ValueError(msg)
        cache_path = self._cache_path(series_id)
        cached = self._load_cached(cache_path)

        needs_fetch = cached is None or max(cached["date"]) < as_of
        if needs_fetch:
            if cached is None:
                start_date = _NASDAQXNDX_EARLIEST
            else:
                start_date = max(cached["date"]) + timedelta(days=1)
            cached = self._extend_cache(series_id, start_date, cache_path)
        if cached is None:
            msg = "NASDAQXNDX cache was not initialized after fetch"
            raise RuntimeError(msg)

        pit = cached[cached["date"] <= as_of]
        timestamps = np.asarray(
            [np.datetime64(d, "D") for d in pit["date"]],
            dtype="datetime64[D]",
        )
        values = np.asarray(pit["close"].to_numpy(), dtype=np.float64)
        return TimeSeries(
            series_id=series_id.upper(),
            timestamps=timestamps,
            values=values,
            is_pseudo_pit=False,
        )
