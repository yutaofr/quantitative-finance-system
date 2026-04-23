from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from datetime import date

import pytest

from backtest.panel_metrics import compute_panel_effective_start
from data_contract.asset_registry import PANEL_ASSET_IDS, PANEL_REGISTRY, PanelAssetSpec
from data_contract.vintage_registry import STRICT_PIT_STARTS


def test_panel_asset_registry_has_three_frozen_entries() -> None:
    assert PANEL_ASSET_IDS == ("NASDAQXNDX", "SPX", "R2K")
    assert tuple(PANEL_REGISTRY) == PANEL_ASSET_IDS
    assert all(isinstance(spec, PanelAssetSpec) for spec in PANEL_REGISTRY.values())


def test_panel_asset_registry_fields_are_all_populated() -> None:
    for spec in PANEL_REGISTRY.values():
        for field in fields(spec):
            if field.name == "vol_fallback_id":
                continue
            assert getattr(spec, field.name) is not None, (spec.asset_id, field.name)


def test_panel_asset_registry_entries_are_immutable() -> None:
    spec = PANEL_REGISTRY["NASDAQXNDX"]
    with pytest.raises(FrozenInstanceError):
        spec.ticker = "QQQM"  # type: ignore[misc]
    with pytest.raises(TypeError):
        PANEL_REGISTRY["VEU"] = spec  # type: ignore[index]


def test_panel_asset_registry_effective_start_matches_panel_scan() -> None:
    effective = compute_panel_effective_start(
        PANEL_REGISTRY,
        STRICT_PIT_STARTS,
        min_training_weeks=312,
        embargo_weeks=53,
    )
    assert effective == date(2014, 11, 28)
    assert all(spec.effective_panel_start == effective for spec in PANEL_REGISTRY.values())
