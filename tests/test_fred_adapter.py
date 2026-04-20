from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from data_contract.fred_adapter import FredClient


def _payload(realtime_start: str = "2024-12-27") -> dict[str, object]:
    return {
        "observations": [
            {
                "date": "2024-12-20",
                "value": "4.2",
                "realtime_start": realtime_start,
                "realtime_end": realtime_start,
            },
        ],
    }


def test_fred_client_converts_observations_to_timeseries_and_caches(tmp_path: Path) -> None:
    calls: list[str] = []

    def fetch_json(url: str) -> dict[str, object]:
        calls.append(url)
        return _payload()

    client = FredClient(api_key="secret", cache_root=tmp_path, fetch_json=fetch_json)

    first = client.get_series("DGS10", date(2024, 12, 27), "strict")
    second = client.get_series("DGS10", date(2024, 12, 27), "strict")

    assert first.series_id == "DGS10"
    assert first.values.tolist() == [4.2]
    assert second.values.tolist() == [4.2]
    assert len(calls) == 1


def test_fred_client_rejects_future_realtime_observations(tmp_path: Path) -> None:
    def fetch_json(_url: str) -> dict[str, object]:
        return _payload("2024-12-28")

    client = FredClient(
        api_key="secret",
        cache_root=tmp_path,
        fetch_json=fetch_json,
    )

    with pytest.raises(ValueError, match="future realtime"):
        client.get_series("DGS10", date(2024, 12, 27), "strict")
