"""Point-in-time data access boundary."""

from __future__ import annotations

from datetime import date

from data_contract.vintage_registry import validate_strict_pit_available
from engine_types import TimeSeries, VintageMode


def get_series_pti(series_id: str, as_of: date, vintage_mode: VintageMode) -> TimeSeries:
    """io: external PIT data access; validates SRD strict-vintage availability first."""
    if vintage_mode == "strict":
        validate_strict_pit_available(series_id, as_of)
    raise NotImplementedError("External PIT data adapter is not implemented yet.")
