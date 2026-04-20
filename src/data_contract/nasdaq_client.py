"""Nasdaq Data Link adapter for NASDAQXNDX total-return daily close.

ADD §3.1: src/data_contract/nasdaq_client.py
ADD §4.5 cache path: data/raw/nasdaq/{series_id}/close.parquet
SRD §4.3: NASDAQXNDX earliest_strict_pit = 1985-10-01, daily close, no revision.

[NEEDS-HUMAN] The Nasdaq Data Link dataset code NASDAQOMX/COMP-NASDAQXNDX is inferred
from the public dataset catalogue.  Confirm this is the correct dataset code before
using in production.  If the API returns 404, replace NASDAQXNDX_DATASET below.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date, timedelta
import json
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd

from engine_types import TimeSeries

# [NEEDS-HUMAN] Confirm exact Nasdaq Data Link dataset code for NASDAQXNDX total return.
NASDAQ_DL_BASE_URL = "https://data.nasdaq.com/api/v3/datasets"
NASDAQXNDX_DATASET = "NASDAQOMX/COMP-NASDAQXNDX"

# SRD §4.3 table — earliest date NASDAQXNDX data exists on Nasdaq Data Link.
_NASDAQXNDX_EARLIEST = date(1985, 10, 1)

FetchJson = Callable[[str], Mapping[str, object]]


def _default_fetch_json(url: str) -> Mapping[str, object]:
    with urlopen(url, timeout=30) as response:  # noqa: S310
        return cast(Mapping[str, object], json.loads(response.read().decode("utf-8")))


@dataclass(frozen=True, slots=True)
class NasdaqClient:
    """io: Fetch NASDAQXNDX daily close from Nasdaq Data Link; cache append-only parquet.

    NASDAQXNDX has no revisions (SRD §4.3 "daily close, no revision"), so
    is_pseudo_pit is always False.  The cache file grows monotonically as
    new dates are appended; existing rows are never overwritten.
    """

    api_key: str
    cache_root: Path = Path("data/raw/nasdaq")
    fetch_json: FetchJson = _default_fetch_json

    def _cache_path(self, series_id: str) -> Path:
        return self.cache_root / series_id.upper() / "close.parquet"

    def _build_url(self, start_date: date) -> str:
        query = urlencode({"api_key": self.api_key, "start_date": start_date.isoformat()})
        return f"{NASDAQ_DL_BASE_URL}/{NASDAQXNDX_DATASET}.json?{query}"

    def _load_cached(self, path: Path) -> pd.DataFrame | None:
        """io: Read cached parquet; returns DataFrame with columns [date, close] or None."""
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df

    def _fetch_rows(self, start_date: date) -> pd.DataFrame:
        """io: Fetch rows from Nasdaq Data Link starting at start_date."""
        payload = self.fetch_json(self._build_url(start_date))
        dataset_data = cast(Mapping[str, Any], payload["dataset_data"])
        column_names = cast(list[str], dataset_data["column_names"])
        rows = cast(list[list[Any]], dataset_data["data"])

        # Locate the price column; NASDAQOMX datasets use "Index Value" at position 1.
        close_col_idx = 1
        for i, col in enumerate(column_names):
            if col.lower() in ("index value", "close", "value"):
                close_col_idx = i
                break

        dates: list[date] = []
        closes: list[float] = []
        for row in rows:
            dates.append(date.fromisoformat(str(row[0])))
            closes.append(float(row[close_col_idx]))
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
        Fetches from Nasdaq Data Link only when the cache does not cover as_of.
        """
        cache_path = self._cache_path(series_id)
        cached = self._load_cached(cache_path)

        needs_fetch = cached is None or max(cached["date"]) < as_of
        if needs_fetch:
            if cached is None:
                start_date = _NASDAQXNDX_EARLIEST
            else:
                start_date = max(cached["date"]) + timedelta(days=1)
            cached = self._extend_cache(series_id, start_date, cache_path)
        assert cached is not None  # always set above

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
