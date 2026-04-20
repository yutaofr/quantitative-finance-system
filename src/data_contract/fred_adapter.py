"""FRED/ALFRED point-in-time adapter with idempotent JSON caching."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np

from data_contract.vintage_registry import validate_strict_pit_available
from engine_types import TimeSeries, VintageMode

FRED_OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"

FetchJson = Callable[[str], Mapping[str, object]]


def _default_fetch_json(url: str) -> Mapping[str, object]:
    with urlopen(url, timeout=30) as response:  # noqa: S310
        return cast(Mapping[str, object], json.loads(response.read().decode("utf-8")))


@dataclass(frozen=True, slots=True)
class FredClient:
    """io: Fetch PIT series from FRED/ALFRED and cache by series/as_of."""

    api_key: str
    cache_root: Path = Path("data/raw/fred")
    fetch_json: FetchJson = _default_fetch_json

    def _cache_path(self, series_id: str, as_of: date) -> Path:
        return self.cache_root / series_id.upper() / f"as_of={as_of.isoformat()}.json"

    def _url(self, series_id: str, as_of: date) -> str:
        query = urlencode(
            {
                "series_id": series_id.upper(),
                "api_key": self.api_key,
                "file_type": "json",
                "realtime_start": as_of.isoformat(),
                "realtime_end": as_of.isoformat(),
                "observation_start": "1900-01-01",
                "observation_end": as_of.isoformat(),
            },
        )
        return f"{FRED_OBSERVATIONS_URL}?{query}"

    def _load_payload(self, series_id: str, as_of: date) -> Mapping[str, object]:
        path = self._cache_path(series_id, as_of)
        if path.exists():
            return cast(Mapping[str, object], json.loads(path.read_text(encoding="utf-8")))
        payload = self.fetch_json(self._url(series_id, as_of))
        path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        path.write_text(text, encoding="utf-8")
        return payload

    def get_series(self, series_id: str, as_of: date, vintage_mode: VintageMode) -> TimeSeries:
        """io: Return a PIT TimeSeries and reject future realtime observations."""
        if vintage_mode == "strict":
            validate_strict_pit_available(series_id, as_of)
        payload = self._load_payload(series_id, as_of)
        observations = cast(list[Mapping[str, Any]], payload.get("observations", []))
        timestamps: list[np.datetime64] = []
        values: list[float] = []
        for observation in observations:
            realtime_start = date.fromisoformat(str(observation["realtime_start"]))
            if realtime_start > as_of:
                msg = "future realtime observation returned by FRED adapter"
                raise ValueError(msg)
            obs_date = date.fromisoformat(str(observation["date"]))
            if obs_date > as_of:
                msg = "future dated observation returned by FRED adapter"
                raise ValueError(msg)
            raw_value = str(observation["value"])
            if raw_value == ".":
                continue
            timestamps.append(np.datetime64(obs_date, "D"))
            values.append(float(raw_value))
        return TimeSeries(
            series_id=series_id.upper(),
            timestamps=np.asarray(timestamps, dtype="datetime64[D]"),
            values=np.asarray(values, dtype=np.float64),
            is_pseudo_pit=vintage_mode == "pseudo",
        )
