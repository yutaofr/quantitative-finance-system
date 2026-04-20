"""Point-in-time data access boundary."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from data_contract.fred_adapter import FredClient
from data_contract.vintage_registry import validate_strict_pit_available
from engine_types import TimeSeries, VintageMode


def get_series_pti(
    series_id: str,
    as_of: date,
    vintage_mode: VintageMode,
    *,
    api_key: str = "",
    cache_root: Path = Path("data/raw/fred"),
) -> TimeSeries:
    """io: external PIT data access; validates SRD strict-vintage availability first."""
    if vintage_mode == "strict":
        validate_strict_pit_available(series_id, as_of)
    client = FredClient(api_key=api_key, cache_root=cache_root)
    return client.get_series(series_id, as_of, vintage_mode)
