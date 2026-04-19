"""Deterministic PCA projection for SRD v8.7 section 6 HMM observations."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

N_COMPONENTS = 2
MATRIX_NDIM = 2


def _orient_components(components: NDArray[np.float64]) -> NDArray[np.float64]:
    oriented = components.copy()
    for idx in range(oriented.shape[0]):
        max_abs_idx = int(np.argmax(np.abs(oriented[idx])))
        if oriented[idx, max_abs_idx] < 0.0:
            oriented[idx] = -oriented[idx]
    return oriented


def _ledoit_wolf_diagonal_cov(centered: NDArray[np.float64]) -> NDArray[np.float64]:
    n_obs = centered.shape[0]
    sample_cov = (centered.T @ centered) / float(n_obs)
    target = np.diag(np.diag(sample_cov))
    diff = sample_cov - target
    shrink_den = float(np.sum(diff * diff))
    if shrink_den == 0.0:
        return target

    outer = np.einsum("ni,nj->nij", centered, centered)
    noise = outer - sample_cov
    shrink_num = float(np.sum(noise * noise)) / float(n_obs * n_obs)
    shrinkage = float(np.clip(shrink_num / shrink_den, 0.0, 1.0))
    return (1.0 - shrinkage) * sample_cov + shrinkage * target


def robust_pca_2d(x_scaled: NDArray[np.float64]) -> NDArray[np.float64]:
    """pure. Project already-scaled feature history via deterministic shrinkage PCA."""
    values = np.asarray(x_scaled, dtype=np.float64)
    if values.ndim != MATRIX_NDIM:
        msg = "x_scaled must be a 2D matrix"
        raise ValueError(msg)
    if values.shape[0] == 0 or values.shape[1] == 0:
        msg = "x_scaled must be non-empty"
        raise ValueError(msg)
    if not np.isfinite(values).all():
        msg = "x_scaled must contain only finite values"
        raise ValueError(msg)

    centered = values - np.median(values, axis=0)
    cov = _ledoit_wolf_diagonal_cov(centered)
    _, eigvecs = np.linalg.eigh(cov)
    components = _orient_components(eigvecs[:, ::-1].T[:N_COMPONENTS])
    scores = centered @ components.T
    if scores.shape[1] == N_COMPONENTS:
        return scores
    padded = np.zeros((scores.shape[0], N_COMPONENTS), dtype=np.float64)
    padded[:, : scores.shape[1]] = scores
    return padded
