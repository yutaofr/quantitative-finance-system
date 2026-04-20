"""FRED/ALFRED point-in-time adapter with idempotent JSON caching."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path
from typing import Any, cast
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np

from data_contract.vintage_registry import validate_strict_pit_available
from engine_types import TimeSeries, VintageMode

FRED_OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"
NO_REVISION_SERIES = frozenset(
    {
        "DGS10",
        "DGS2",
        "DGS1",
        "VIXCLS",
        "VXNCLS",
        "VXVCLS",
    },
)
CLIENT_SIDE_PIT_KEY = "__client_side_pit"
PAYLOAD_KEY = "payload"
HTTP_BAD_REQUEST = 400

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

    def _url(self, series_id: str, as_of: date, *, include_realtime: bool) -> str:
        params = {
            "series_id": series_id.upper(),
            "api_key": self.api_key,
            "file_type": "json",
            "observation_start": "1900-01-01",
            "observation_end": as_of.isoformat(),
        }
        if include_realtime:
            params["realtime_start"] = as_of.isoformat()
            params["realtime_end"] = as_of.isoformat()
        query = urlencode(params)
        return f"{FRED_OBSERVATIONS_URL}?{query}"

    def _read_cached_payload(
        self,
        path: Path,
        *,
        enforce_realtime: bool,
    ) -> tuple[Mapping[str, object], bool]:
        raw = cast(Mapping[str, object], json.loads(path.read_text(encoding="utf-8")))
        if CLIENT_SIDE_PIT_KEY in raw and PAYLOAD_KEY in raw:
            return cast(Mapping[str, object], raw[PAYLOAD_KEY]), False
        return raw, enforce_realtime

    def _write_payload(
        self,
        path: Path,
        payload: Mapping[str, object],
        *,
        enforce_realtime: bool,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        stored: Mapping[str, object] = (
            payload
            if enforce_realtime
            else {
                CLIENT_SIDE_PIT_KEY: True,
                PAYLOAD_KEY: payload,
            }
        )
        text = json.dumps(stored, sort_keys=True, separators=(",", ":"))
        path.write_text(text, encoding="utf-8")

    def _load_payload(
        self,
        series_id: str,
        as_of: date,
        vintage_mode: VintageMode,
    ) -> tuple[Mapping[str, object], bool]:
        path = self._cache_path(series_id, as_of)
        include_realtime = vintage_mode == "strict" and series_id.upper() not in NO_REVISION_SERIES
        if path.exists():
            return self._read_cached_payload(path, enforce_realtime=include_realtime)
        enforce_realtime = include_realtime
        try:
            payload = self.fetch_json(
                self._url(series_id, as_of, include_realtime=include_realtime),
            )
        except HTTPError as exc:
            if exc.code != HTTP_BAD_REQUEST or vintage_mode != "strict":
                raise
            payload = self.fetch_json(self._url(series_id, as_of, include_realtime=False))
            enforce_realtime = False
        self._write_payload(path, payload, enforce_realtime=enforce_realtime)
        return payload, enforce_realtime

    def get_series(self, series_id: str, as_of: date, vintage_mode: VintageMode) -> TimeSeries:
        """io: Return a PIT TimeSeries and reject future realtime observations."""
        if vintage_mode == "strict":
            validate_strict_pit_available(series_id, as_of)
        payload, enforce_realtime = self._load_payload(series_id, as_of, vintage_mode)
        observations = cast(list[Mapping[str, Any]], payload.get("observations", []))
        timestamps: list[np.datetime64] = []
        values: list[float] = []
        for observation in observations:
            if enforce_realtime:
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
