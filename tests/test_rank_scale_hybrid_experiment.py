from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from research.run_rank_scale_hybrid_experiment import (
    ForecastSeries,
    _hybrid_quantile_map,
    _metrics,
)

pytestmark = pytest.mark.synthetic


def test_hybrid_quantile_map_preserves_t5_ordering() -> None:
    days = (date(2020, 1, 3), date(2020, 1, 10), date(2020, 1, 17))
    t5_series = ForecastSeries(
        dates=days,
        y=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        mu=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        sigma=np.array([0.2, 0.9, 0.5], dtype=np.float64),
        e=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        crps=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        pathology=0,
    )
    egarch_series = ForecastSeries(
        dates=days,
        y=np.array([0.01, -0.02, 0.03], dtype=np.float64),
        mu=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        sigma=np.array([0.1, 0.3, 0.2], dtype=np.float64),
        e=np.array([0.01, -0.02, 0.03], dtype=np.float64),
        crps=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        pathology=0,
    )

    hybrid = _hybrid_quantile_map(t5_series, egarch_series)

    assert list(np.argsort(hybrid.sigma)) == list(np.argsort(t5_series.sigma))
    assert float(np.min(hybrid.sigma)) == pytest.approx(float(np.min(egarch_series.sigma)))
    assert float(np.max(hybrid.sigma)) == pytest.approx(float(np.max(egarch_series.sigma)))


def test_protocol_success_requires_direction_scale_and_safety() -> None:
    days = (
        date(2020, 1, 3),
        date(2020, 1, 10),
        date(2020, 1, 17),
        date(2020, 1, 24),
    )
    series = ForecastSeries(
        dates=days,
        y=np.array([0.0, 0.1, 0.0, 0.2], dtype=np.float64),
        mu=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64),
        sigma=np.array([0.05, 0.10, 0.15, 0.20], dtype=np.float64),
        e=np.array([0.0, 0.1, 0.0, 0.2], dtype=np.float64),
        crps=np.array([0.01, 0.01, 0.01, 0.01], dtype=np.float64),
        pathology=0,
    )

    metrics = _metrics(series)

    assert metrics["corr_next"] > 0.0
    assert metrics["rank_next"] > 0.0
    assert metrics["std_z"] < 1.5
    assert metrics["decision"] == "SUCCESS"
