from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from config_types import FrozenConfig
from engine_types import TimeSeries
from errors import HMMConvergenceError
from law.linear_quantiles import QRCoefs
from state.ti_hmm_single import HMMModel


def _config() -> FrozenConfig:
    return FrozenConfig(
        srd_version="8.7",
        random_seed=8675309,
        timezone="America/New_York",
        missing_rate_degraded=0.10,
        missing_rate_blocked=0.20,
        quantile_gap=1.0e-4,
        l2_alpha=2.0,
        tail_mult=0.6,
        utility_lambda=1.2,
        utility_kappa=0.8,
        band=7.0,
        score_min=0.0,
        score_max=100.0,
        block_lengths=(52, 78),
        bootstrap_replications=2000,
    )


def _series(series_id: str, values: np.ndarray) -> TimeSeries:
    start = date(2020, 1, 3)
    dates = [start + timedelta(weeks=idx) for idx in range(values.shape[0])]
    return TimeSeries(
        series_id=series_id,
        timestamps=np.array(dates, dtype="datetime64[D]"),
        values=values.astype(np.float64),
        is_pseudo_pit=False,
    )


def _series_map(n_weeks: int = 120) -> dict[str, TimeSeries]:
    idx = np.arange(n_weeks, dtype=np.float64)
    return {
        "DGS10": _series("DGS10", 4.0 + 0.02 * idx),
        "DGS2": _series("DGS2", 2.0 + 0.01 * idx),
        "DGS1": _series("DGS1", 1.0 + 0.005 * idx),
        "EFFR": _series("EFFR", 0.5 + 0.004 * idx),
        "BAA10Y": _series("BAA10Y", 1.5 + 0.003 * idx),
        "WALCL": _series("WALCL", 100.0 + idx),
        "VXNCLS": _series("VXNCLS", 20.0 + 0.1 * idx),
        "RV20_NDX": _series("RV20_NDX", 10.0 + 0.08 * idx),
        "VIXCLS": _series("VIXCLS", 15.0 + 0.05 * idx),
        "VXVCLS": _series("VXVCLS", 13.0 + 0.04 * idx),
        "NASDAQXNDX": _series("NASDAQXNDX", 1000.0 + 4.0 * idx + 0.05 * idx * idx),
    }


def test_build_training_artifacts_degrades_hmm_and_trains_qr_with_uniform_posterior() -> None:
    from inference.train import build_training_artifacts

    captured: dict[str, np.ndarray] = {}

    def fit_hmm_fn(
        _y_obs: np.ndarray,
        _h: np.ndarray,
        _rng: np.random.Generator,
        *,
        forward_52w_returns: np.ndarray | None,
    ) -> HMMModel:
        assert forward_52w_returns is not None
        raise HMMConvergenceError("forced nonconvergence")

    def fit_qr_fn(
        x_scaled: np.ndarray,
        pi: np.ndarray,
        y_52w: np.ndarray,
        *,
        l2_alpha: float,
        min_gap: float,
    ) -> QRCoefs:
        captured["x_scaled"] = x_scaled
        captured["pi"] = pi
        captured["y_52w"] = y_52w
        assert l2_alpha == 2.0
        assert min_gap == 1.0e-4
        return QRCoefs(
            a=np.array([-0.10, -0.05, 0.0, 0.05, 0.10], dtype=np.float64),
            b=np.zeros((5, x_scaled.shape[1]), dtype=np.float64),
            c=np.zeros((5, 3), dtype=np.float64),
            solver_status="ok",
        )

    artifacts = build_training_artifacts(
        date(2022, 4, 15),
        _series_map(),
        _config(),
        rng=np.random.default_rng(123),
        min_training_weeks=10,
        fit_hmm_fn=fit_hmm_fn,
        fit_qr_fn=fit_qr_fn,
    )

    assert artifacts.hmm_model is None
    assert artifacts.qr_coefs is not None
    assert artifacts.utility_zstats is not None
    assert artifacts.offense_thresholds is not None
    assert artifacts.state_label_map == {0: "DEFENSIVE", 1: "NEUTRAL", 2: "OFFENSIVE"}
    assert set(artifacts.train_distributions) == {"x1", "x5", "x9"}
    assert captured["x_scaled"].shape[0] == captured["y_52w"].shape[0]
    assert np.allclose(captured["pi"], np.array([0.25, 0.50, 0.25], dtype=np.float64))
