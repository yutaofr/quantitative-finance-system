"""Tail extrapolation from SRD v8.7 section 8.2."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

TAIL_MULT = 0.6
INTERIOR_COUNT = 5


def extrapolate_tails(
    interior: NDArray[np.float64],
    *,
    mult: float = TAIL_MULT,
) -> tuple[NDArray[np.float64], Literal["ok", "fallback"]]:
    """pure. Add q05/q95 around q10/q25/q50/q75/q90 using SRD bounded rule."""
    values = np.asarray(interior, dtype=np.float64)
    if values.shape != (INTERIOR_COUNT,):
        msg = "interior quantiles must have shape (5,)"
        raise ValueError(msg)
    if not np.isfinite(values).all():
        msg = "interior quantiles must be finite"
        raise ValueError(msg)

    q10, q25, q50, q75, q90 = values
    q05 = q10 - mult * (q25 - q10)
    q95 = q90 + mult * (q90 - q75)
    full = np.array([q05, q10, q25, q50, q75, q90, q95], dtype=np.float64)
    if np.all(np.diff(full) >= 0.0):
        return full, "ok"
    return np.sort(full), "fallback"
