from __future__ import annotations

import numpy as np

from backtest.cluster_block_bootstrap import bootstrap_week_statistic_p05
from backtest.panel_metrics import panel_aggregate_crps, per_asset_coverage, per_asset_crps


def test_panel_evaluator_contract_on_synthetic_three_asset_data() -> None:
    quantiles = {
        "NASDAQXNDX": np.array(
            [
                [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3],
                [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3],
                [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3],
                [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3],
            ],
            dtype=np.float64,
        ),
        "SPX": np.array(
            [
                [-0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4],
                [-0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4],
                [-0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4],
                [-0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4],
            ],
            dtype=np.float64,
        ),
        "R2K": np.array(
            [
                [-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2],
                [-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2],
                [-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2],
                [-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2],
            ],
            dtype=np.float64,
        ),
    }
    realized = {
        "NASDAQXNDX": np.array([-0.2, 0.0, 0.2, 0.4], dtype=np.float64),
        "SPX": np.array([-0.1, 0.1, 0.3, 0.5], dtype=np.float64),
        "R2K": np.array([-0.3, -0.1, 0.1, 0.3], dtype=np.float64),
    }

    crps = {
        asset_id: per_asset_crps(quantile_matrix, realized[asset_id], asset_id)
        for asset_id, quantile_matrix in quantiles.items()
    }
    q10 = {
        asset_id: per_asset_coverage(quantile_matrix, realized[asset_id], 0.10)
        for asset_id, quantile_matrix in quantiles.items()
    }
    q90 = {
        asset_id: per_asset_coverage(quantile_matrix, realized[asset_id], 0.90)
        for asset_id, quantile_matrix in quantiles.items()
    }
    week_crps = np.array(
        [
            [0.08, 0.09, 0.10],
            [0.07, 0.08, 0.09],
            [0.10, 0.11, 0.12],
            [0.09, 0.10, 0.11],
        ],
        dtype=np.float64,
    )

    assert all(np.isfinite(value) and value > 0.0 for value in crps.values())
    assert q10 == {"NASDAQXNDX": 0.25, "SPX": 0.25, "R2K": 0.25}
    assert q90 == {"NASDAQXNDX": 0.75, "SPX": 0.75, "R2K": 0.75}
    assert panel_aggregate_crps(crps) > 0.0
    assert np.isfinite(
        bootstrap_week_statistic_p05(
            week_crps,
            statistic=lambda values: float(np.mean(values)),
            block_length=2,
            replications=64,
            rng=np.random.default_rng(7),
        ),
    )
