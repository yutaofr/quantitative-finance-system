from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pytest

from app.challenger_artifacts import (
    challenger_fit_artifact_to_dict,
    challenger_output_to_dict,
    write_challenger_fit_artifact,
    write_challenger_output,
)
from law.student_t_location_scale import fit_student_t_location_scale, predict_student_t_quantiles


def _sample_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_scaled = np.tile(np.linspace(-1.0, 1.0, 10, dtype=np.float64), (80, 1))
    x_scaled += np.linspace(-0.5, 0.5, 80, dtype=np.float64)[:, None] * 0.1
    pi = np.tile(np.array([0.2, 0.5, 0.3], dtype=np.float64), (80, 1))
    y = np.linspace(-0.2, 0.2, 80, dtype=np.float64)
    return x_scaled, pi, y


def test_fit_student_t_location_scale_predicts_non_crossing_quantiles() -> None:
    x_scaled, pi, y = _sample_inputs()

    fit = fit_student_t_location_scale(x_scaled, pi, y, maxiter=20)
    quantiles = predict_student_t_quantiles(x_scaled[0], pi[0], fit.params)

    assert quantiles.shape == (7,)
    assert np.all(np.diff(quantiles) >= 0.0)


def test_fit_student_t_location_scale_falls_back_when_optimizer_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x_scaled, pi, y = _sample_inputs()

    class _FailedResult:
        success = False
        message = "forced-failure"

        def __init__(self, x: np.ndarray) -> None:
            self.x = x

    def _failed_minimize(
        _fun: object,
        x0: np.ndarray,
        **_kwargs: object,
    ) -> _FailedResult:
        return _FailedResult(np.asarray(x0, dtype=np.float64))

    monkeypatch.setattr("law.student_t_location_scale.minimize", _failed_minimize)

    fit = fit_student_t_location_scale(x_scaled, pi, y, maxiter=5)

    assert fit.optimization_failed is True
    assert fit.fallback_used is True
    assert fit.params.beta_mu.shape[0] == 14
    assert fit.params.beta_sigma.shape[0] == 4


def test_challenger_artifact_serialization_is_deterministic(tmp_path: Path) -> None:
    x_scaled, pi, y = _sample_inputs()
    fit = fit_student_t_location_scale(x_scaled, pi, y, maxiter=10)
    fit_payload = challenger_fit_artifact_to_dict(
        as_of=date(2024, 12, 27),
        train_end=date(2024, 6, 28),
        fit_result=fit,
    )
    output_payload = challenger_output_to_dict(
        as_of=date(2024, 12, 27),
        status="ok",
        quantiles=predict_student_t_quantiles(x_scaled[0], pi[0], fit.params),
        fit_status=fit.optimizer_status,
        optimization_failed=fit.optimization_failed,
        source_offense_final=50.0,
    )
    fit_path = tmp_path / "challenger_fit_artifact.json"
    output_path = tmp_path / "challenger_output.json"

    write_challenger_fit_artifact(fit_path, fit_payload)
    first_fit = fit_path.read_text(encoding="utf-8")
    write_challenger_fit_artifact(fit_path, fit_payload)
    assert fit_path.read_text(encoding="utf-8") == first_fit

    write_challenger_output(output_path, output_payload)
    first_output = output_path.read_text(encoding="utf-8")
    write_challenger_output(output_path, output_payload)
    assert output_path.read_text(encoding="utf-8") == first_output
