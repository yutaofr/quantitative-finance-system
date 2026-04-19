from __future__ import annotations

import numpy as np

from law.tail_extrapolation import extrapolate_tails


def test_extrapolate_tails_applies_srd_section_8_2_formula() -> None:
    interior = np.array([-0.10, -0.05, 0.02, 0.10, 0.20], dtype=np.float64)

    full, status = extrapolate_tails(interior)

    assert status == "ok"
    assert np.allclose(
        full,
        np.array(
            [
                -0.13,
                -0.10,
                -0.05,
                0.02,
                0.10,
                0.20,
                0.26,
            ],
            dtype=np.float64,
        ),
    )


def test_extrapolate_tails_flags_pathological_tail_order() -> None:
    interior = np.array([0.10, 0.05, 0.02, 0.10, 0.20], dtype=np.float64)

    full, status = extrapolate_tails(interior)

    assert status == "fallback"
    assert np.all(np.diff(full) >= -1.0e-12)
