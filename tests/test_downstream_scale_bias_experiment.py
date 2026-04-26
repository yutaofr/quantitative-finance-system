from __future__ import annotations

import numpy as np
import pytest

from research.run_downstream_scale_bias_experiment import (
    ScaleBiasFit,
    _final_decision,
    _fit_scale_bias_student_t,
)

pytestmark = pytest.mark.synthetic


def test_scale_bias_mle_recovers_stable_scalar_k_on_synthetic_t_residuals() -> None:
    rng = np.random.default_rng(20260426)
    raw = rng.standard_t(df=8.0, size=240)
    variance_one = raw * np.sqrt((8.0 - 2.0) / 8.0)
    z = 2.0 * variance_one

    fit = _fit_scale_bias_student_t(z)

    assert fit.k == pytest.approx(2.0, abs=0.25)
    assert fit.nu > 2.0
    assert fit.loglik > fit.naive_loglik


def test_final_decision_requires_k_range_stability_improvement_and_coverage() -> None:
    fits = {
        "Window_2017": ScaleBiasFit(
            k=1.8,
            nu=7.0,
            loglik=-10.0,
            naive_nu=5.0,
            naive_loglik=-20.0,
            central_80_coverage=0.80,
            central_90_coverage=0.90,
        ),
        "Window_2018": ScaleBiasFit(
            k=2.1,
            nu=8.0,
            loglik=-11.0,
            naive_nu=5.0,
            naive_loglik=-21.0,
            central_80_coverage=0.80,
            central_90_coverage=0.90,
        ),
        "Window_2020": ScaleBiasFit(
            k=2.3,
            nu=6.0,
            loglik=-12.0,
            naive_nu=5.0,
            naive_loglik=-22.0,
            central_80_coverage=0.80,
            central_90_coverage=0.90,
        ),
    }

    assert _final_decision(fits)["option_1_decision"] == "SUCCESS"

    drifting = {
        **fits,
        "Window_2020": ScaleBiasFit(
            k=2.6,
            nu=6.0,
            loglik=-12.0,
            naive_nu=5.0,
            naive_loglik=-22.0,
            central_80_coverage=0.80,
            central_90_coverage=0.90,
        ),
    }

    decision = _final_decision(drifting)

    assert decision["option_1_decision"] == "FAIL"
    assert "K_RANGE_DRIFT" in decision["failed_conditions"]
