from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from data_contract.nasdaq_client import YAHOO_XNDX_SYMBOL, NasdaqClient


def test_nasdaq_client_fetches_xndx_from_free_yahoo_provider(tmp_path: Path) -> None:
    calls: list[tuple[str, date]] = []

    def fetch_history(symbol: str, start_date: date) -> pd.DataFrame:
        calls.append((symbol, start_date))
        return pd.DataFrame(
            {
                "Close": [100.0, 101.5, 102.0],
            },
            index=pd.to_datetime(["2024-01-03", "2024-01-05", "2024-01-08"]),
        )

    client = NasdaqClient(cache_root=tmp_path, fetch_history=fetch_history)

    series = client.get_series("NASDAQXNDX", date(2024, 1, 5))

    assert calls == [(YAHOO_XNDX_SYMBOL, date(1985, 10, 1))]
    assert series.series_id == "NASDAQXNDX"
    assert series.is_pseudo_pit is False
    assert series.timestamps.astype("datetime64[D]").astype(str).tolist() == [
        "2024-01-03",
        "2024-01-05",
    ]
    assert series.values.tolist() == [100.0, 101.5]
