from __future__ import annotations

from hypothesis import given, strategies as st
from hypothesis.extra.numpy import arrays
import numpy as np

from features.scaling import robust_zscore_expanding, soft_squash_clip


def test_robust_zscore_expanding_uses_prior_and_current_window() -> None:
    values = np.array([1.0, 2.0, 100.0])

    out = robust_zscore_expanding(values)

    assert out[0] == 0.0
    assert np.isclose(out[1], 0.6744907594765952)
    assert np.isclose(out[2], 66.1000939828673)


@given(
    arr=arrays(
        dtype=np.float64,
        shape=st.integers(min_value=1, max_value=100),
        elements=st.floats(
            min_value=-1.0e6,
            max_value=1.0e6,
            allow_nan=False,
            allow_infinity=False,
        ),
    ),
)
def test_soft_squash_clip_is_bounded(arr: np.ndarray) -> None:
    out = soft_squash_clip(arr)
    assert np.all(np.abs(out) <= 5.0)


@given(
    arr=arrays(
        dtype=np.float64,
        shape=st.integers(min_value=2, max_value=100),
        elements=st.floats(
            min_value=-1.0e6,
            max_value=1.0e6,
            allow_nan=False,
            allow_infinity=False,
        ),
    ),
)
def test_soft_squash_clip_preserves_order(arr: np.ndarray) -> None:
    out = soft_squash_clip(np.sort(arr))
    assert np.all(np.diff(out) >= -1.0e-12)
