"""Feature scaling functions from SRD v8.7 section 5."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def robust_zscore_expanding(
    x: NDArray[np.float64],
    *,
    mad_epsilon: float = 1.0e-8,
) -> NDArray[np.float64]:
    """pure. Compute expanding-window median/MAD robust z-scores."""
    values = np.asarray(x, dtype=np.float64)
    out = np.empty_like(values, dtype=np.float64)
    for idx in range(values.shape[0]):
        window = values[: idx + 1]
        med = float(np.median(window))
        mad = float(np.median(np.abs(window - med)))
        out[idx] = (values[idx] - med) / (1.4826 * mad + mad_epsilon)
    return out


def soft_squash_clip(
    z: NDArray[np.float64],
    *,
    tanh_rescale: float = 4.0,
    clip_bound: float = 5.0,
) -> NDArray[np.float64]:
    """pure. Apply SRD soft tanh squash followed by hard clipping."""
    values = np.asarray(z, dtype=np.float64)
    squashed = tanh_rescale * np.tanh(values / tanh_rescale)
    return np.clip(squashed, -clip_bound, clip_bound)
