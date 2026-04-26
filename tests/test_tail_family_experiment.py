"""L1 unit tests for the tail family experiment decision logic."""

from __future__ import annotations

import numpy as np
import pytest

from research.run_tail_family_experiment import (
    K_FIXED,
    KS_P_MIN,
    NU_DRIFT_MAX,
    TailFamilyFit,
    _apply_decision,
    _calibration_errors,
    _coverage,
    _fit_window,
    _ks_test,
    _mle_nu,
    _student_t_std_cdf,
    _student_t_std_ppf,
)


def _make_fit(  # noqa: PLR0913
    *,
    nu: float = 10.0,
    loglik: float = -30.0,
    cal_errors: dict[str, float] | None = None,
    ks_stat: float = 0.05,
    ks_pvalue: float = 0.50,
    cov80: float = 0.80,
    cov90: float = 0.90,
) -> TailFamilyFit:
    if cal_errors is None:
        cal_errors = {"tau_0.10": 0.01, "tau_0.25": 0.01, "tau_0.75": 0.01, "tau_0.90": 0.01}
    return TailFamilyFit(
        nu=nu,
        loglik=loglik,
        calibration_errors=cal_errors,
        ks_statistic=ks_stat,
        ks_pvalue=ks_pvalue,
        central_80_coverage=cov80,
        central_90_coverage=cov90,
    )


def _make_fits(
    nu_2017: float = 10.0,
    nu_2018: float = 12.0,
    nu_2020: float = 8.0,
) -> dict[str, TailFamilyFit]:
    return {
        "Window_2017": _make_fit(nu=nu_2017),
        "Window_2018": _make_fit(nu=nu_2018),
        "Window_2020": _make_fit(nu=nu_2020),
    }


def test_k_fixed_is_correct_median() -> None:
    per_window_k = np.array([2.604250, 2.682572, 2.748825])
    assert pytest.approx(float(np.median(per_window_k)), abs=1e-6) == K_FIXED


def test_student_t_std_ppf_cdf_roundtrip() -> None:
    nu = 10.0
    for tau in (0.10, 0.25, 0.75, 0.90):
        q = _student_t_std_ppf(tau, nu)
        assert _student_t_std_cdf(q, nu) == pytest.approx(tau, abs=1e-6)


def test_calibration_errors_near_zero_for_correct_distribution() -> None:
    rng = np.random.default_rng(42)
    nu = 10.0
    scale = float(np.sqrt((nu - 2.0) / nu))
    u = rng.standard_t(df=nu, size=5000).astype(np.float64) * scale
    errors = _calibration_errors(u, nu)
    for err in errors.values():
        assert err < 0.03, f"calibration error {err} too large for correct distribution"


def test_ks_test_passes_for_correct_distribution() -> None:
    rng = np.random.default_rng(0)
    nu = 8.0
    scale = float(np.sqrt((nu - 2.0) / nu))
    u = rng.standard_t(df=nu, size=1000).astype(np.float64) * scale
    _ks_stat, ks_p = _ks_test(u, nu)
    assert ks_p >= KS_P_MIN, f"KS p={ks_p:.4f} should pass for correct distribution"


def test_mle_nu_recovers_known_nu() -> None:
    rng = np.random.default_rng(7)
    true_nu = 8.0
    scale = float(np.sqrt((true_nu - 2.0) / true_nu))
    u = rng.standard_t(df=true_nu, size=2000).astype(np.float64) * scale
    nu_hat, loglik = _mle_nu(u)
    assert abs(nu_hat - true_nu) < 3.0, f"MLE nu={nu_hat:.2f} too far from true {true_nu}"
    assert loglik < 0.0


def test_fit_window_runs_on_synthetic_data() -> None:
    rng = np.random.default_rng(99)
    nu = 6.0
    scale = float(np.sqrt((nu - 2.0) / nu))
    u = rng.standard_t(df=nu, size=26).astype(np.float64) * scale
    fit = _fit_window(u)
    assert np.isfinite(fit.nu)
    assert np.isfinite(fit.loglik)
    assert all(np.isfinite(v) for v in fit.calibration_errors.values())


def test_coverage_symmetric_distribution() -> None:
    rng = np.random.default_rng(3)
    nu = 100.0
    scale = float(np.sqrt((nu - 2.0) / nu))
    u = rng.standard_t(df=nu, size=10000).astype(np.float64) * scale
    cov80 = _coverage(u, nu, 0.80)
    assert abs(cov80 - 0.80) < 0.03
    cov90 = _coverage(u, nu, 0.90)
    assert abs(cov90 - 0.90) < 0.03


def test_decision_pass_all_conditions() -> None:
    fits = _make_fits(nu_2017=10.0, nu_2018=12.0, nu_2020=8.0)
    pooled = _make_fit(nu=10.0)
    dec = _apply_decision(fits, pooled)
    assert dec.pass_per_window
    assert dec.nu_drift_ok
    assert len(dec.failed_conditions) == 0


def test_decision_fail_on_calibration_breach() -> None:
    bad_cal = {"tau_0.10": 0.06, "tau_0.25": 0.01, "tau_0.75": 0.01, "tau_0.90": 0.01}
    fits = {
        "Window_2017": _make_fit(cal_errors=bad_cal),
        "Window_2018": _make_fit(),
        "Window_2020": _make_fit(),
    }
    pooled = _make_fit()
    dec = _apply_decision(fits, pooled)
    assert not dec.pass_per_window
    assert any("CALIBRATION_BREACH" in c for c in dec.failed_conditions)


def test_decision_fail_on_ks_reject() -> None:
    fits = {
        "Window_2017": _make_fit(ks_pvalue=0.03),
        "Window_2018": _make_fit(),
        "Window_2020": _make_fit(),
    }
    pooled = _make_fit()
    dec = _apply_decision(fits, pooled)
    assert not dec.pass_per_window
    assert any("KS_REJECT" in c for c in dec.failed_conditions)


def test_nu_drift_flag_when_exceeded() -> None:
    fits = _make_fits(nu_2017=5.0, nu_2018=10.0, nu_2020=40.0)
    pooled = _make_fit()
    dec = _apply_decision(fits, pooled)
    assert not dec.nu_drift_ok
    assert dec.nu_drift > NU_DRIFT_MAX
    assert not dec.pass_pooled


def test_nu_no_drift_when_within_threshold() -> None:
    fits = _make_fits(nu_2017=8.0, nu_2018=12.0, nu_2020=15.0)
    pooled = _make_fit()
    dec = _apply_decision(fits, pooled)
    assert dec.nu_drift_ok
    assert dec.nu_drift <= NU_DRIFT_MAX


def test_coverage_collapse_2020_cov90() -> None:
    fits = {
        "Window_2017": _make_fit(),
        "Window_2018": _make_fit(),
        "Window_2020": _make_fit(cov90=0.60, cov80=0.70),
    }
    pooled = _make_fit()
    dec = _apply_decision(fits, pooled)
    assert not dec.pass_per_window
    assert any("DIRECTION_DEFECT_PROPAGATED" in c for c in dec.failed_conditions)


def test_coverage_collapse_2020_cov80() -> None:
    fits = {
        "Window_2017": _make_fit(),
        "Window_2018": _make_fit(),
        "Window_2020": _make_fit(cov90=0.75, cov80=0.55),
    }
    pooled = _make_fit()
    dec = _apply_decision(fits, pooled)
    assert not dec.pass_per_window
    assert any("DIRECTION_DEFECT_PROPAGATED" in c for c in dec.failed_conditions)


def test_pass_pooled_requires_nu_drift_ok() -> None:
    # nu_drift = 35 > 30 → NU_DRIFT → pass_pooled must be False even if pooled cal/ks pass
    fits = _make_fits(nu_2017=5.0, nu_2018=10.0, nu_2020=40.0)
    pooled = _make_fit(nu=15.0)
    dec = _apply_decision(fits, pooled)
    assert not dec.pass_pooled


def test_pass_pooled_true_when_drift_ok_and_pooled_passes() -> None:
    fits = _make_fits(nu_2017=8.0, nu_2018=10.0, nu_2020=12.0)
    pooled = _make_fit(nu=10.0)
    dec = _apply_decision(fits, pooled)
    assert dec.nu_drift_ok
    assert dec.pass_pooled
