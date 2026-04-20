from __future__ import annotations

from datetime import date
from email.message import Message
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlparse

import pytest

from data_contract.fred_adapter import FredClient


def _payload(realtime_start: str = "2024-12-27", obs_date: str = "2024-12-20") -> dict[str, object]:
    return {
        "observations": [
            {
                "date": obs_date,
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
        client.get_series("EFFR", date(2024, 12, 27), "strict")


def test_fred_client_omits_realtime_params_for_no_revision_strict_series(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def fetch_json(url: str) -> dict[str, object]:
        calls.append(url)
        return _payload("2026-04-20", "2014-12-26")

    client = FredClient(api_key="secret", cache_root=tmp_path, fetch_json=fetch_json)

    series = client.get_series("VIXCLS", date(2024, 12, 27), "strict")
    query = parse_qs(urlparse(calls[0]).query)

    assert "realtime_start" not in query
    assert "realtime_end" not in query
    assert query["observation_end"] == ["2024-12-27"]
    assert series.values.tolist() == [4.2]
    assert not series.is_pseudo_pit


def test_fred_client_retries_without_realtime_params_after_http_400(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def fetch_json(url: str) -> dict[str, object]:
        calls.append(url)
        if len(calls) == 1:
            raise HTTPError(
                url=url,
                code=400,
                msg="Bad Request",
                hdrs=Message(),
                fp=BytesIO(b"bad vintage"),
            )
        return _payload("2026-04-20", "2014-12-26")

    client = FredClient(api_key="secret", cache_root=tmp_path, fetch_json=fetch_json)

    series = client.get_series("EFFR", date(2015, 1, 2), "strict")
    first_query = parse_qs(urlparse(calls[0]).query)
    second_query = parse_qs(urlparse(calls[1]).query)

    assert first_query["realtime_start"] == ["2015-01-02"]
    assert "realtime_start" not in second_query
    assert "realtime_end" not in second_query
    assert series.values.tolist() == [4.2]
    assert not series.is_pseudo_pit


def test_fred_client_does_not_retry_http_500_for_revision_series(tmp_path: Path) -> None:
    calls: list[str] = []

    def fetch_json(url: str) -> dict[str, object]:
        calls.append(url)
        raise HTTPError(
            url=url,
            code=500,
            msg="Internal Server Error",
            hdrs=Message(),
            fp=BytesIO(b"server failure"),
        )

    client = FredClient(api_key="secret", cache_root=tmp_path, fetch_json=fetch_json)

    with pytest.raises(HTTPError):
        client.get_series("EFFR", date(2015, 1, 2), "strict")
    assert len(calls) == 1
