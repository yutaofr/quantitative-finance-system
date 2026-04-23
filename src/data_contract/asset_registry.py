"""Frozen panel asset registry from SRD v8.8 section P2.1.1."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, timedelta
from types import MappingProxyType
from typing import Literal

MIN_TRAINING_WEEKS = 312
EMBARGO_WEEKS = 53
FIRST_TRAINABLE_OFFSET_WEEKS = MIN_TRAINING_WEEKS + EMBARGO_WEEKS - 1
PANEL_EFFECTIVE_START = date(2014, 11, 28)


@dataclass(frozen=True, slots=True)
class PanelAssetSpec:
    """pure. Frozen panel asset specification."""

    asset_id: str
    ticker: str
    description: str
    provider: str
    is_total_return: bool
    pit_classification: Literal["log_return_pit", "pseudo_pit_risk"]
    inception_date: date
    first_available_friday: date
    first_trainable_friday: date
    effective_panel_start: date
    vol_series_id: str
    vol_fallback_id: str | None
    notes: str


def _first_trainable_friday(first_available_friday: date) -> date:
    return first_available_friday + timedelta(weeks=FIRST_TRAINABLE_OFFSET_WEEKS)


PANEL_REGISTRY: Mapping[str, PanelAssetSpec] = MappingProxyType(
    {
        "NASDAQXNDX": PanelAssetSpec(
            asset_id="NASDAQXNDX",
            ticker="QQQ",
            description="Nasdaq-100 total-return proxy via ETF adjusted close",
            provider="yahoo_finance",
            is_total_return=True,
            pit_classification="log_return_pit",
            inception_date=date(1999, 3, 10),
            first_available_friday=date(1999, 3, 12),
            first_trainable_friday=_first_trainable_friday(date(1999, 3, 12)),
            effective_panel_start=PANEL_EFFECTIVE_START,
            vol_series_id="VXNCLS",
            vol_fallback_id=None,
            notes="QQQ adjusted close replaces NASDAQXNDX only on the panel challenger path.",
        ),
        "SPX": PanelAssetSpec(
            asset_id="SPX",
            ticker="SPY",
            description="S&P 500 total-return proxy via ETF adjusted close",
            provider="yahoo_finance",
            is_total_return=True,
            pit_classification="log_return_pit",
            inception_date=date(1993, 1, 29),
            first_available_friday=date(1993, 1, 29),
            first_trainable_friday=_first_trainable_friday(date(1993, 1, 29)),
            effective_panel_start=PANEL_EFFECTIVE_START,
            vol_series_id="VIXCLS",
            vol_fallback_id=None,
            notes="SPY is the invariant state-label anchor for panel HMM semantics.",
        ),
        "R2K": PanelAssetSpec(
            asset_id="R2K",
            ticker="IWM",
            description="Russell 2000 total-return proxy via ETF adjusted close",
            provider="yahoo_finance",
            is_total_return=True,
            pit_classification="log_return_pit",
            inception_date=date(2000, 5, 22),
            first_available_friday=date(2000, 5, 26),
            first_trainable_friday=_first_trainable_friday(date(2000, 5, 26)),
            effective_panel_start=PANEL_EFFECTIVE_START,
            vol_series_id="RVXCLS",
            vol_fallback_id="VIXCLS",
            notes=(
                "Use VIXCLS proxy before RVXCLS begins; "
                "fall back to RV-only if both are missing."
            ),
        ),
    },
)

PANEL_ASSET_IDS = tuple(PANEL_REGISTRY)
