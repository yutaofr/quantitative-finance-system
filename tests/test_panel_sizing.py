from __future__ import annotations

from data_contract.asset_registry import PANEL_ASSET_IDS, PANEL_REGISTRY


def test_fixed_panel_contains_exactly_the_frozen_three_assets() -> None:
    assert PANEL_ASSET_IDS == ("NASDAQXNDX", "SPX", "R2K")
    assert tuple(PANEL_REGISTRY) == PANEL_ASSET_IDS
