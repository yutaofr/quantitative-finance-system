from __future__ import annotations

import numpy as np

from law.quantile_moments import es20_from_quantiles, moments_from_quantiles, p_loss_from_quantiles

TAUS = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)


def test_moments_from_quantiles_integrates_piecewise_linear_quantile_curve() -> None:
    q_vals = np.array([-0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30], dtype=np.float64)

    moments = moments_from_quantiles(TAUS, q_vals)

    assert np.isclose(moments["mu_hat"], 0.0)
    assert moments["sigma_hat"] > 0.0


def test_p_loss_from_quantiles_interpolates_zero_crossing() -> None:
    q_vals = np.array([-0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30], dtype=np.float64)

    assert np.isclose(p_loss_from_quantiles(TAUS, q_vals), 0.50)


def test_p_loss_from_quantiles_handles_all_gain_or_all_loss() -> None:
    all_gain = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07])
    all_loss = np.array([-0.07, -0.06, -0.05, -0.04, -0.03, -0.02, -0.01])

    assert p_loss_from_quantiles(TAUS, all_gain) == 0.0
    assert p_loss_from_quantiles(TAUS, all_loss) == 1.0


def test_es20_from_quantiles_integrates_lowest_20_percent_tail() -> None:
    q_vals = np.array([-0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30], dtype=np.float64)

    es20 = es20_from_quantiles(TAUS, q_vals)

    assert es20 > 0.0
    assert np.isclose(es20, 0.19444444444444445)


def test_moments_from_quantiles_returns_downside_metrics() -> None:
    q_vals = np.array([-0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30], dtype=np.float64)

    moments = moments_from_quantiles(TAUS, q_vals)

    assert set(moments) == {"mu_hat", "sigma_hat", "p_loss", "es20"}
    assert np.isclose(moments["p_loss"], 0.50)
    assert np.isclose(moments["es20"], 0.19444444444444445)
