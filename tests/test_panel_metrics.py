from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from backtest.panel_metrics import (
    compute_panel_effective_start,
    effective_asset_weeks,
    panel_aggregate_crps,
    per_asset_coverage,
    per_asset_crps,
    vol_normalized_crps,
)
from data_contract.asset_registry import PANEL_REGISTRY
from data_contract.vintage_registry import STRICT_PIT_STARTS


def test_effective_start_is_deterministic() -> None:
    first = compute_panel_effective_start(
        PANEL_REGISTRY,
        STRICT_PIT_STARTS,
        min_training_weeks=312,
        embargo_weeks=53,
    )
    second = compute_panel_effective_start(
        PANEL_REGISTRY,
        STRICT_PIT_STARTS,
        min_training_weeks=312,
        embargo_weeks=53,
    )

    assert first == second == date(2014, 11, 28)


def test_panel_metric_functions_match_simple_synthetic_answers() -> None:
    quantiles = np.array(
        [
            [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3],
            [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3],
            [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3],
        ],
        dtype=np.float64,
    )
    realized = np.array([-0.2, 0.0, 0.25], dtype=np.float64)
    zero_crps = per_asset_crps(
        np.zeros(7, dtype=np.float64),
        np.array(0.0, dtype=np.float64),
        "QQQ",
    )

    assert zero_crps == 0.0
    assert per_asset_coverage(quantiles, realized, 0.10) == 1.0 / 3.0
    assert per_asset_coverage(quantiles, realized, 0.90) == 2.0 / 3.0
    assert panel_aggregate_crps({"QQQ": 0.2, "SPY": 0.4, "R2K": 0.6}) == pytest.approx(0.4)
    assert vol_normalized_crps(0.3, 0.6) == pytest.approx(0.5)
    assert effective_asset_weeks(
        {
            "QQQ": np.array([True, True, False], dtype=np.bool_),
            "SPY": np.array([True, False, False], dtype=np.bool_),
            "R2K": np.array([True, True, True], dtype=np.bool_),
        },
    ) == 6
