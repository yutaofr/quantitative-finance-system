from __future__ import annotations

from datetime import date

import pytest

from data_contract.vintage_registry import (
    FORBIDDEN_IN_PROD,
    STRICT_PIT_STARTS,
    is_forbidden_in_prod,
    validate_strict_pit_available,
)
from errors import VintageUnavailableError


def test_strict_pit_start_dates_match_srd_section_4_3() -> None:
    assert STRICT_PIT_STARTS == {
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


def test_strict_pit_validation_raises_before_earliest_vintage() -> None:
    with pytest.raises(VintageUnavailableError):
        validate_strict_pit_available("RRPONTSYD", date(2013, 1, 25))


def test_strict_pit_validation_allows_start_date() -> None:
    validate_strict_pit_available("RRPONTSYD", date(2013, 2, 1))


def test_forbidden_production_series_match_srd_section_4_2() -> None:
    expected = frozenset(
        {
            "NFCI",
            "NFCIRISK",
            "NFCICREDIT",
            "NFCILEVERAGE",
            "NFCINONFINLEVERAGE",
            "STLFSI4",
        },
    )
    assert expected == FORBIDDEN_IN_PROD
    assert is_forbidden_in_prod("NFCI")
    assert not is_forbidden_in_prod("DGS10")
