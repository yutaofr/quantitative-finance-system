from __future__ import annotations

import numpy as np
import pytest

from state.ti_hmm_single import initialize_emission_means_kmeans_pp


def test_initialize_emission_means_kmeans_pp_is_deterministic_for_seeded_rng() -> None:
    y_obs = np.array(
        [
            [-4.0, -4.0],
            [-3.0, -3.0],
            [0.0, 0.0],
            [3.0, 3.0],
            [4.0, 4.0],
        ],
        dtype=np.float64,
    )

    first = initialize_emission_means_kmeans_pp(y_obs, rng=np.random.default_rng(7), state_count=3)
    second = initialize_emission_means_kmeans_pp(y_obs, rng=np.random.default_rng(7), state_count=3)

    assert np.array_equal(first, second)


def test_initialize_emission_means_kmeans_pp_uses_only_usable_rows() -> None:
    y_obs = np.array(
        [
            [-4.0, -4.0],
            [0.0, 0.0],
            [4.0, 4.0],
            [100.0, 100.0],
        ],
        dtype=np.float64,
    )
    usable = np.array([True, True, True, False])

    means = initialize_emission_means_kmeans_pp(
        y_obs,
        rng=np.random.default_rng(11),
        state_count=3,
        usable_mask=usable,
    )

    assert not np.any(np.all(means == np.array([100.0, 100.0], dtype=np.float64), axis=1))


def test_initialize_emission_means_kmeans_pp_rejects_global_rng_substitute() -> None:
    with pytest.raises(TypeError, match="Generator"):
        initialize_emission_means_kmeans_pp(
            np.eye(3, dtype=np.float64),
            rng=None,  # type: ignore[arg-type]
            state_count=3,
        )
