"""No-trade band hysteresis from SRD v8.7 section 9.5."""

from __future__ import annotations

import numpy as np

DEFAULT_BAND = 7.0


def apply_band(
    offense_raw_t: float,
    offense_final_prev: float,
    *,
    band: float = DEFAULT_BAND,
) -> float:
    """pure. Preserve prior offense score when raw change is inside the band."""
    if not np.isfinite(offense_raw_t) or not np.isfinite(offense_final_prev) or band < 0.0:
        msg = "hysteresis inputs must be finite and band nonnegative"
        raise ValueError(msg)
    if abs(offense_raw_t - offense_final_prev) < band:
        return offense_final_prev
    return offense_raw_t
