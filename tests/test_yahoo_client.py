from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from data_contract.yahoo_client import YahooFinanceClient, log_return_series


def test_yahoo_client_caches_adjusted_close_and_computes_log_returns(tmp_path: Path) -> None:
    calls: list[tuple[str, date, date]] = []

    def fetch_history(ticker: str, start: date, end: date) -> pd.DataFrame:
        calls.append((ticker, start, end))
        return pd.DataFrame(
            { "Adj Close": [100.0, 101.0, 103.0, 104.0] },
            index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-05", "2024-01-08"]),
        )

    client = YahooFinanceClient(cache_root=tmp_path, fetch_history=fetch_history)

    first = client.fetch_etf_adjusted_close("QQQ", date(2024, 1, 2), date(2024, 1, 5))
    second = client.fetch_etf_adjusted_close("QQQ", date(2024, 1, 2), date(2024, 1, 5))
    returns = log_return_series(first)

    assert len(calls) == 1
    assert first.series_id == "QQQ"
    assert second.values.tolist() == [100.0, 101.0, 103.0]
    assert (tmp_path / "QQQ" / "adj_close.parquet").exists()
    assert np.allclose(
        returns.values,
        np.array(
            [
                np.log(101.0 / 100.0),
                np.log(103.0 / 101.0),
            ],
            dtype=np.float64,
        ),
    )
    assert returns.timestamps.astype("datetime64[D]").astype(str).tolist() == [
        "2024-01-03",
        "2024-01-05",
    ]
