from __future__ import annotations

import numpy as np
import pytest

from errors import QuantileSolverError
from law.linear_quantiles import (
    INTERIOR_TAUS,
    QRCoefs,
    fit_linear_quantiles,
    predict_interior,
    predict_interior_with_status,
)


def _training_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_scaled = np.array(
        [
            [-2.0, 0.0],
            [-1.0, 0.1],
            [0.0, 0.2],
            [1.0, 0.3],
            [2.0, 0.4],
            [3.0, 0.5],
        ],
        dtype=np.float64,
    )
    pi = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.7, 0.2, 0.1],
            [0.2, 0.6, 0.2],
            [0.1, 0.7, 0.2],
            [0.1, 0.2, 0.7],
            [0.1, 0.1, 0.8],
        ],
        dtype=np.float64,
    )
    y_52w = np.array([-0.20, -0.10, 0.0, 0.05, 0.15, 0.25], dtype=np.float64)
    return x_scaled, pi, y_52w


def test_fit_linear_quantiles_joint_solution_predicts_non_crossing_training_rows() -> None:
    x_scaled, pi, y_52w = _training_data()

    coefs = fit_linear_quantiles(x_scaled, pi, y_52w)

    assert coefs.a.shape == (5,)
    assert coefs.b.shape == (5, 2)
    assert coefs.c.shape == (5, 3)
    assert coefs.solver_status == "ok"
    for row, post in zip(x_scaled, pi, strict=True):
        preds = predict_interior(coefs, row, post)
        assert np.all(np.diff(preds) >= 1.0e-4 - 1.0e-8)


def test_predict_interior_rearranges_pathological_coefficients() -> None:
    coefs = QRCoefs(
        a=np.array([0.5, 0.4, 0.3, 0.2, 0.1], dtype=np.float64),
        b=np.zeros((5, 2), dtype=np.float64),
        c=np.zeros((5, 3), dtype=np.float64),
        solver_status="manual",
    )

    preds = predict_interior(coefs, np.zeros(2, dtype=np.float64), np.array([0.2, 0.3, 0.5]))

    assert np.all(np.diff(preds) >= 1.0e-4 - 1.0e-12)


def test_predict_interior_reports_rearranged_status_for_fallback() -> None:
    coefs = QRCoefs(
        a=np.array([0.5, 0.4, 0.3, 0.2, 0.1], dtype=np.float64),
        b=np.zeros((5, 2), dtype=np.float64),
        c=np.zeros((5, 3), dtype=np.float64),
        solver_status="ok",
    )

    _preds, status = predict_interior_with_status(
        coefs,
        np.zeros(2, dtype=np.float64),
        np.array([0.2, 0.3, 0.5], dtype=np.float64),
    )

    assert status == "rearranged"


def test_fit_linear_quantiles_rejects_solver_failure_status() -> None:
    x_scaled, pi, y_52w = _training_data()

    with pytest.raises(QuantileSolverError, match="solver"):
        fit_linear_quantiles(x_scaled, pi, y_52w, solver="NOT_A_SOLVER")


def test_interior_taus_match_srd_section_8_1() -> None:
    assert INTERIOR_TAUS == (0.10, 0.25, 0.50, 0.75, 0.90)
