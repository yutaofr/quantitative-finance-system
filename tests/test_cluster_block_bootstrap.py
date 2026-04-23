from __future__ import annotations

import numpy as np

from backtest.cluster_block_bootstrap import (
    bootstrap_week_statistic_p05,
    resample_week_clusters,
    stationary_cluster_bootstrap_indices,
)


def test_stationary_cluster_bootstrap_preserves_full_week_rows() -> None:
    week_matrix = np.array(
        [
            [1.0, 10.0, 100.0],
            [2.0, 20.0, 200.0],
            [3.0, 30.0, 300.0],
            [4.0, 40.0, 400.0],
        ],
        dtype=np.float64,
    )
    resampled = resample_week_clusters(
        week_matrix,
        block_length=2,
        rng=np.random.default_rng(123),
    )

    valid_rows = {tuple(row.tolist()) for row in week_matrix}
    assert {tuple(row.tolist()) for row in resampled}.issubset(valid_rows)
    assert resampled.shape == week_matrix.shape


def test_stationary_cluster_bootstrap_is_deterministic_for_same_rng_seed() -> None:
    left = stationary_cluster_bootstrap_indices(
        12,
        block_length=3,
        rng=np.random.default_rng(8675309),
    )
    right = stationary_cluster_bootstrap_indices(
        12,
        block_length=3,
        rng=np.random.default_rng(8675309),
    )

    assert np.array_equal(left, right)


def test_bootstrap_week_statistic_returns_finite_p05() -> None:
    week_matrix = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.2, 0.3, 0.4],
            [0.0, 0.1, 0.2],
            [0.3, 0.4, 0.5],
        ],
        dtype=np.float64,
    )

    p05 = bootstrap_week_statistic_p05(
        week_matrix,
        statistic=lambda values: float(np.mean(values)),
        block_length=2,
        replications=64,
        rng=np.random.default_rng(42),
    )

    assert np.isfinite(p05)
