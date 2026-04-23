"""Clustered stationary block bootstrap for fixed-panel weekly evaluation."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

INDEX_DTYPE = np.int64
MATRIX_NDIM = 2


def stationary_cluster_bootstrap_indices(
    n_weeks: int,
    *,
    block_length: int,
    rng: np.random.Generator,
) -> NDArray[np.int64]:
    """pure. Draw stationary bootstrap week indices while preserving week clusters."""
    if n_weeks <= 0 or block_length <= 0:
        msg = "bootstrap dimensions must be strictly positive"
        raise ValueError(msg)
    restart_probability = 1.0 / float(block_length)
    indices = np.empty(n_weeks, dtype=INDEX_DTYPE)
    current = int(rng.integers(0, n_weeks))
    for idx in range(n_weeks):
        if idx > 0 and float(rng.random()) < restart_probability:
            current = int(rng.integers(0, n_weeks))
        indices[idx] = current
        current = (current + 1) % n_weeks
    return indices


def resample_week_clusters(
    week_matrix: NDArray[np.float64],
    *,
    block_length: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """pure. Resample a week-by-asset matrix by full calendar-week clusters."""
    values = np.asarray(week_matrix, dtype=np.float64)
    if values.ndim != MATRIX_NDIM:
        msg = "week_matrix must have shape (n_weeks, n_assets)"
        raise ValueError(msg)
    indices = stationary_cluster_bootstrap_indices(
        values.shape[0],
        block_length=block_length,
        rng=rng,
    )
    return values[indices]


def bootstrap_week_statistic_p05(
    week_matrix: NDArray[np.float64],
    *,
    statistic: Callable[[NDArray[np.float64]], float],
    block_length: int,
    replications: int,
    rng: np.random.Generator,
) -> float:
    """pure. Bootstrap the 5th percentile of a scalar statistic over week clusters."""
    values = np.asarray(week_matrix, dtype=np.float64)
    if values.ndim != MATRIX_NDIM:
        msg = "week_matrix must have shape (n_weeks, n_assets)"
        raise ValueError(msg)
    if replications <= 0:
        msg = "replications must be strictly positive"
        raise ValueError(msg)
    samples = np.empty(replications, dtype=np.float64)
    for idx in range(replications):
        resampled = resample_week_clusters(values, block_length=block_length, rng=rng)
        samples[idx] = statistic(resampled)
    if not np.isfinite(samples).any():
        return float("nan")
    return float(np.nanquantile(samples, 0.05))
