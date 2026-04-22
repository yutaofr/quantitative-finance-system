"""Diagnostic cycle position from SRD v8.7 section 9.7."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray


def _percentile_rank(value: float, distribution: NDArray[np.float64]) -> float:
    dist = np.asarray(distribution, dtype=np.float64)
    if dist.ndim != 1 or dist.shape[0] == 0 or not np.isfinite(dist).all():
        msg = "percentile-rank distribution must be a finite non-empty vector"
        raise ValueError(msg)
    if not np.isfinite(value):
        return 0.5
    ordered = np.sort(dist)
    if value <= ordered[0]:
        return 0.0
    if value >= ordered[-1]:
        return 1.0
    positions = np.linspace(0.0, 1.0, ordered.shape[0], dtype=np.float64)
    return float(np.interp(value, ordered, positions))


def cycle_position(
    x5_t: float,
    x9_t: float,
    x1_t: float,
    train_dist: Mapping[str, NDArray[np.float64]],
) -> float:
    """pure. Compute transparent diagnostic cycle position in [0, 100]."""
    credit_rank = _percentile_rank(x5_t, train_dist["x5"])
    vrp_rank = _percentile_rank(x9_t, train_dist["x9"])
    slope_rank = _percentile_rank(-x1_t, -np.asarray(train_dist["x1"], dtype=np.float64))
    return 100.0 * (credit_rank + vrp_rank + slope_rank) / 3.0
