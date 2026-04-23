from __future__ import annotations

import numpy as np

from backtest.panel_metrics import per_asset_crps
from law.panel_quantiles import (
    fit_panel_quantiles,
    predict_panel_interior,
    predict_panel_interior_with_status,
)
from law.tail_extrapolation import extrapolate_tails

ASSET_IDS = ("NASDAQXNDX", "SPX", "R2K")


def _synthetic_panel_inputs(n_obs: int = 72) -> tuple[
    np.ndarray,
    dict[str, np.ndarray],
    np.ndarray,
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    rng = np.random.default_rng(1234)
    x_macro = rng.normal(size=(n_obs, 7)).astype(np.float64)
    pi_raw = rng.uniform(0.1, 1.0, size=(n_obs, 3))
    pi = (pi_raw / pi_raw.sum(axis=1, keepdims=True)).astype(np.float64)
    x_micro = {asset_id: rng.normal(size=(n_obs, 3)).astype(np.float64) for asset_id in ASSET_IDS}
    intercepts = {"NASDAQXNDX": 0.7, "SPX": 0.1, "R2K": -0.5}
    y = {}
    mask = {}
    for asset_id in ASSET_IDS:
        noise = 0.05 * rng.normal(size=n_obs)
        y[asset_id] = (
            intercepts[asset_id]
            + 0.08 * x_macro[:, 0]
            - 0.04 * x_macro[:, 1]
            + 0.03 * pi[:, 0]
            + 0.02 * x_micro[asset_id][:, 0]
            + noise
        ).astype(np.float64)
        mask[asset_id] = np.ones(n_obs, dtype=np.bool_)
    return x_macro, x_micro, pi, y, mask


def test_panel_quantiles_produce_non_crossing_predictions_and_finite_crps() -> None:
    x_macro, x_micro, pi, y, mask = _synthetic_panel_inputs()
    coefs = fit_panel_quantiles(x_macro, x_micro, pi, y, mask)

    for asset_id in ASSET_IDS:
        predicted = np.vstack(
            [
                predict_panel_interior(
                    coefs,
                    asset_id,
                    x_macro[idx],
                    x_micro[asset_id][idx],
                    pi[idx],
                )
                for idx in range(x_macro.shape[0])
            ],
        )
        full = np.vstack([extrapolate_tails(prediction)[0] for prediction in predicted])
        assert np.all(np.diff(predicted, axis=1) >= 1.0e-4 - 1.0e-10)
        assert per_asset_crps(full, y[asset_id], asset_id) > 0.0


def test_panel_quantiles_leave_asset_intercepts_unpenalized() -> None:
    n_obs = 60
    x_macro = np.zeros((n_obs, 7), dtype=np.float64)
    pi = np.tile(np.array([0.2, 0.5, 0.3], dtype=np.float64), (n_obs, 1))
    x_micro = {asset_id: np.zeros((n_obs, 3), dtype=np.float64) for asset_id in ASSET_IDS}
    y = {
        "NASDAQXNDX": np.full(n_obs, 0.8, dtype=np.float64),
        "SPX": np.full(n_obs, 0.1, dtype=np.float64),
        "R2K": np.full(n_obs, -0.6, dtype=np.float64),
    }
    mask = {asset_id: np.ones(n_obs, dtype=np.bool_) for asset_id in ASSET_IDS}

    coefs = fit_panel_quantiles(x_macro, x_micro, pi, y, mask)

    medians = {
        asset_id: predict_panel_interior(
            coefs,
            asset_id,
            x_macro[0],
            x_micro[asset_id][0],
            pi[0],
        )[2]
        for asset_id in ASSET_IDS
    }
    assert medians["NASDAQXNDX"] > medians["SPX"] > medians["R2K"]


def test_panel_quantiles_ignore_zero_weight_missing_rows() -> None:
    x_macro, x_micro, pi, y, mask = _synthetic_panel_inputs()
    masked_reference = {asset_id: values.copy() for asset_id, values in x_micro.items()}
    mask_with_gap = {asset_id: values.copy() for asset_id, values in mask.items()}
    mask_with_gap["SPX"][0] = False
    reference = fit_panel_quantiles(x_macro, masked_reference, pi, y, mask_with_gap)
    masked_y = {asset_id: values.copy() for asset_id, values in y.items()}
    masked_micro = {asset_id: values.copy() for asset_id, values in x_micro.items()}
    masked_y["SPX"][0] = 1.0e6
    masked_micro["SPX"][0] = np.array([np.nan, np.nan, np.nan], dtype=np.float64)

    masked = fit_panel_quantiles(x_macro, masked_micro, pi, masked_y, mask_with_gap)
    reference_prediction = predict_panel_interior(
        reference,
        "SPX",
        x_macro[1],
        x_micro["SPX"][1],
        pi[1],
    )
    masked_prediction = predict_panel_interior(
        masked,
        "SPX",
        x_macro[1],
        x_micro["SPX"][1],
        pi[1],
    )

    assert np.allclose(reference.alpha["SPX"], masked.alpha["SPX"], atol=5.0e-3)
    assert np.allclose(reference_prediction, masked_prediction, atol=5.0e-3)


def test_panel_tail_extrapolation_matches_single_asset_tail_rule() -> None:
    x_macro, x_micro, pi, y, mask = _synthetic_panel_inputs()
    coefs = fit_panel_quantiles(x_macro, x_micro, pi, y, mask)
    interior, status = predict_panel_interior_with_status(
        coefs,
        "NASDAQXNDX",
        x_macro[0],
        x_micro["NASDAQXNDX"][0],
        pi[0],
    )
    full, tail_status = extrapolate_tails(interior)

    assert status in {"ok", "rearranged", "per_asset_fallback"}
    assert tail_status in {"ok", "fallback"}
    assert np.isclose(full[0], interior[0] - 0.6 * (interior[1] - interior[0]))
    assert np.isclose(full[6], interior[4] + 0.6 * (interior[4] - interior[3]))
