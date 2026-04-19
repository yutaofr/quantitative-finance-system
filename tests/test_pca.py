from __future__ import annotations

import numpy as np
import pytest

from features.pca import robust_pca_2d


def test_robust_pca_2d_returns_two_components_per_row() -> None:
    x_scaled = np.array(
        [
            [-1.0, 0.0, 1.0],
            [0.0, 1.0, 2.0],
            [1.0, 2.0, 3.0],
        ],
        dtype=np.float64,
    )

    scores = robust_pca_2d(x_scaled)

    assert scores.shape == (3, 2)


def test_robust_pca_2d_uses_deterministic_component_signs() -> None:
    x_scaled = np.array(
        [
            [-3.0, -1.0, 0.0],
            [-1.0, -0.5, 0.0],
            [1.0, 0.5, 0.0],
            [3.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )

    scores = robust_pca_2d(x_scaled)

    assert scores[-1, 0] > scores[0, 0]


def test_robust_pca_2d_is_column_shift_invariant() -> None:
    x_scaled = np.array(
        [
            [-2.0, 0.0, 1.0],
            [-1.0, 1.0, 1.5],
            [1.0, 2.0, 2.5],
            [2.0, 3.0, 3.0],
        ],
        dtype=np.float64,
    )
    shifted = x_scaled + np.array([100.0, -50.0, 20.0], dtype=np.float64)

    assert np.allclose(robust_pca_2d(x_scaled), robust_pca_2d(shifted))


def test_robust_pca_2d_rejects_non_finite_inputs() -> None:
    x_scaled = np.array([[0.0, 1.0], [np.nan, 2.0]], dtype=np.float64)

    with pytest.raises(ValueError, match="finite"):
        robust_pca_2d(x_scaled)
