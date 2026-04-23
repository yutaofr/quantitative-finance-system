"""Strict PIT vintage registry from SRD v8.7 section 4.3."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from typing import Literal

from errors import VintageUnavailableError

STRICT_PIT_STARTS: dict[str, date] = {
    "NASDAQXNDX": date(1985, 10, 1),
    "DGS10": date(1962, 1, 2),
    "DGS2": date(1976, 6, 1),
    "DGS1": date(1962, 1, 2),
    "EFFR": date(2000, 7, 3),
    "WALCL": date(2002, 12, 18),
    "RRPONTSYD": date(2013, 2, 1),
    "BAA10Y": date(1986, 1, 2),
    "VXNCLS": date(2001, 2, 2),
    "VIXCLS": date(1990, 1, 2),
    "VXVCLS": date(2007, 12, 4),
    "RVXCLS": date(2004, 1, 2),
}

PRODUCTION_FEATURE_SERIES = (
    "NASDAQXNDX",
    "DGS10",
    "DGS2",
    "DGS1",
    "EFFR",
    "BAA10Y",
    "WALCL",
    "VXNCLS",
    "VIXCLS",
    "VXVCLS",
)
FRIDAY_WEEKDAY = 4

FORBIDDEN_IN_PROD = frozenset(
    {
        "NFCI",
        "NFCIRISK",
        "NFCICREDIT",
        "NFCILEVERAGE",
        "NFCINONFINLEVERAGE",
        "STLFSI4",
    },
)


def is_forbidden_in_prod(series_id: str) -> bool:
    """pure. Return whether a series is barred from production features."""
    return series_id.upper() in FORBIDDEN_IN_PROD


def validate_strict_pit_available(series_id: str, as_of: date) -> None:
    """pure. Raise when strict PIT is requested before the first allowed vintage."""
    earliest = STRICT_PIT_STARTS[series_id.upper()]
    if as_of < earliest:
        msg = (
            f"{series_id.upper()} strict PIT starts at {earliest.isoformat()}, "
            f"but as_of={as_of.isoformat()} was requested"
        )
        raise VintageUnavailableError(msg)


def _next_friday(anchor: date) -> date:
    days_ahead = (FRIDAY_WEEKDAY - anchor.weekday()) % 7
    return date.fromordinal(anchor.toordinal() + days_ahead)


def compute_effective_strict_start(
    feature_registry: Sequence[str] | Mapping[str, Sequence[str] | str],
    earliest_strict_pit_registry: Mapping[str, date],
    *,
    min_training_weeks: int = 312,
    embargo_weeks: int = 53,
    weekly_calendar: Literal["Friday"] = "Friday",
) -> date:
    """pure. Compute SRD v8.7.1 strict-PIT acceptance start."""
    if weekly_calendar != "Friday" or min_training_weeks <= 0 or embargo_weeks < 0:
        msg = "effective strict start requires Friday calendar and positive windows"
        raise ValueError(msg)
    if isinstance(feature_registry, Mapping):
        series_ids: list[str] = []
        for value in feature_registry.values():
            if isinstance(value, str):
                series_ids.append(value)
            else:
                series_ids.extend(value)
    else:
        series_ids = list(feature_registry)
    if not series_ids:
        msg = "feature registry must include at least one production series"
        raise ValueError(msg)
    earliest_feature_date = max(
        earliest_strict_pit_registry[series_id.upper()] for series_id in series_ids
    )
    first_training_week = _next_friday(earliest_feature_date)
    weeks_to_first_valid_train_end = min_training_weeks - 1
    ordinal = first_training_week.toordinal() + 7 * (weeks_to_first_valid_train_end + embargo_weeks)
    return date.fromordinal(ordinal)
