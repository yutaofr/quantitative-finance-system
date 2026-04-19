"""Strict PIT vintage registry from SRD v8.7 section 4.3."""

from __future__ import annotations

from datetime import date

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
}

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
