"""L1 unit tests for run_joint_location_scale_experiment.py."""

from __future__ import annotations

import numpy as np
import pytest

from research.run_joint_location_scale_experiment import (
    C_BOUND,
    CAL_TOL_ASYM,
    MEAN_TOL,
    NU_FIXED,
    STD_HI,
    STD_LO,
    JointFit,
    _apply_decision,
    _apply_location_correction,
    _calibration_error,
    _check_window,
    _corr,
    _fit_candidate,
    _neg_loglik_gaussian,
    _neg_loglik_student_t,
    _rank_corr,
    _student_t_std_logpdf,
    _student_t_std_ppf,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fit(  # noqa: PLR0913
    *,
    c: float = 0.1,
    k_new: float = 1.0,
    loglik: float = -50.0,
    mean_v: float = 0.0,
    std_v: float = 1.0,
    cal_error_025: float = 0.02,
    cal_error_075: float = 0.02,
    corr_next: float | None = 0.5,
    rank_next: float | None = 0.5,
    sigma_blowup: int = 0,
    pathology: int = 0,
) -> JointFit:
    return JointFit(
        c=c,
        k_new=k_new,
        loglik=loglik,
        mean_v=mean_v,
        std_v=std_v,
        cal_error_025=cal_error_025,
        cal_error_075=cal_error_075,
        corr_next=corr_next,
        rank_next=rank_next,
        sigma_blowup=sigma_blowup,
        pathology=pathology,
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_c_bound_preregistered() -> None:
    assert C_BOUND == 0.99


def test_nu_fixed_preregistered() -> None:
    assert NU_FIXED == 10.0


def test_cal_tol_preregistered() -> None:
    assert CAL_TOL_ASYM == 0.08


def test_mean_tol_preregistered() -> None:
    assert MEAN_TOL == 0.20


def test_std_bounds_preregistered() -> None:
    assert STD_LO == 0.80
    assert STD_HI == 1.20


# ---------------------------------------------------------------------------
# _apply_location_correction
# ---------------------------------------------------------------------------


def test_location_correction_zero_c_is_identity() -> None:
    z = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    out = _apply_location_correction(z, 0.0)
    np.testing.assert_allclose(out, z)


def test_location_correction_boundary_condition() -> None:
    """First element uses z_{t-1} = 0."""
    z = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    c = 0.5
    out = _apply_location_correction(z, c)
    # t=0: z[0] - c*0 = 1.0
    # t=1: z[1] - c*z[0] = 2.0 - 0.5 = 1.5
    # t=2: z[2] - c*z[1] = 3.0 - 1.0 = 2.0
    expected = np.array([1.0, 1.5, 2.0], dtype=np.float64)
    np.testing.assert_allclose(out, expected)


def test_location_correction_length_preserved() -> None:
    z = np.random.default_rng(0).standard_normal(50).astype(np.float64)
    out = _apply_location_correction(z, 0.3)
    assert out.shape == z.shape


# ---------------------------------------------------------------------------
# _student_t_std_logpdf and _student_t_std_ppf
# ---------------------------------------------------------------------------


def test_student_t_logpdf_finite_at_zero() -> None:
    v = np.array([0.0], dtype=np.float64)
    lp = _student_t_std_logpdf(v, 10.0)
    assert np.isfinite(lp[0])


def test_student_t_ppf_symmetry() -> None:
    lo = _student_t_std_ppf(0.25, 10.0)
    hi = _student_t_std_ppf(0.75, 10.0)
    assert pytest.approx(lo, abs=1e-10) == -hi


def test_student_t_ppf_median_zero() -> None:
    assert pytest.approx(_student_t_std_ppf(0.5, 10.0), abs=1e-10) == 0.0


# ---------------------------------------------------------------------------
# _neg_loglik_gaussian
# ---------------------------------------------------------------------------


def test_neg_loglik_gaussian_finite_at_reasonable_params() -> None:
    rng = np.random.default_rng(42)
    z = rng.standard_normal(30).astype(np.float64)
    params = np.array([0.1, 0.0], dtype=np.float64)  # c=0.1, log_k=0 → k=1
    val = _neg_loglik_gaussian(params, z)
    assert np.isfinite(val)
    assert val > 0.0


def test_neg_loglik_gaussian_penalty_at_boundary() -> None:
    z = np.ones(20, dtype=np.float64)
    params = np.array([C_BOUND, 0.0], dtype=np.float64)
    val = _neg_loglik_gaussian(params, z)
    assert val >= 1.0e14


# ---------------------------------------------------------------------------
# _neg_loglik_student_t
# ---------------------------------------------------------------------------


def test_neg_loglik_student_t_finite() -> None:
    rng = np.random.default_rng(7)
    z = rng.standard_normal(30).astype(np.float64)
    params = np.array([0.0, 0.0], dtype=np.float64)
    val = _neg_loglik_student_t(params, z, NU_FIXED)
    assert np.isfinite(val)
    assert val > 0.0


def test_neg_loglik_student_t_uses_fixed_nu() -> None:
    """Verify it does not optimize over nu."""
    rng = np.random.default_rng(99)
    z = rng.standard_normal(30).astype(np.float64)
    params = np.array([0.0, 0.0], dtype=np.float64)
    v1 = _neg_loglik_student_t(params, z, 10.0)
    v2 = _neg_loglik_student_t(params, z, 100.0)
    assert v1 != v2  # different nu → different loglik


# ---------------------------------------------------------------------------
# _corr and _rank_corr
# ---------------------------------------------------------------------------


def test_corr_perfect_positive() -> None:
    x = np.arange(10.0)
    assert pytest.approx(_corr(x, x), abs=1e-10) == 1.0


def test_corr_constant_returns_none() -> None:
    x = np.ones(10)
    y = np.arange(10.0)
    assert _corr(x, y) is None


def test_rank_corr_monotone() -> None:
    x = np.arange(10.0)
    y = x**2
    rc = _rank_corr(x, y)
    assert rc is not None
    assert pytest.approx(rc, abs=1e-10) == 1.0


def test_rank_corr_returns_none_on_single_finite() -> None:
    x = np.array([np.nan, 1.0, np.nan], dtype=np.float64)
    y = np.array([np.nan, 2.0, np.nan], dtype=np.float64)
    assert _rank_corr(x, y) is None


# ---------------------------------------------------------------------------
# _calibration_error
# ---------------------------------------------------------------------------


def test_calibration_error_gaussian_perfect_sample() -> None:
    """Large Gaussian sample → near-zero calibration error."""
    rng = np.random.default_rng(0)
    v = rng.standard_normal(5000).astype(np.float64)
    err = _calibration_error(v, 0.25, gaussian=True)
    assert err < 0.02


def test_calibration_error_student_t_perfect_sample() -> None:
    """Large Student-t(10) variance-one sample → near-zero calibration error."""
    nu = NU_FIXED
    scale = float(np.sqrt((nu - 2.0) / nu))
    rng = np.random.default_rng(1)
    v = rng.standard_t(nu, size=5000).astype(np.float64) * scale
    err = _calibration_error(v, 0.25, gaussian=False)
    assert err < 0.02


def test_calibration_error_constant_shift_detected() -> None:
    """Right-shifted distribution has large error at lower quantile."""
    rng = np.random.default_rng(2)
    v = rng.standard_normal(500).astype(np.float64) + 2.0  # shifted right
    err = _calibration_error(v, 0.25, gaussian=True)
    assert err > 0.2


# ---------------------------------------------------------------------------
# _fit_candidate
# ---------------------------------------------------------------------------


def test_fit_candidate_gaussian_recovers_near_zero_c() -> None:
    """Gaussian data with no autocorrelation → c ≈ 0."""
    rng = np.random.default_rng(5)
    z = rng.standard_normal(100).astype(np.float64)
    sigma = np.ones(100, dtype=np.float64)
    fit = _fit_candidate(z, sigma, gaussian=True)
    assert abs(fit.c) < 0.4  # loose bound, n=100 has noise


def test_fit_candidate_raises_on_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        _fit_candidate(
            np.full(10, np.nan, dtype=np.float64),
            np.ones(10, dtype=np.float64),
            gaussian=True,
        )


def test_fit_candidate_pathology_flag_structure() -> None:
    """pathology must be 0 or 1."""
    rng = np.random.default_rng(11)
    z = rng.standard_normal(40).astype(np.float64)
    sigma = np.ones(40, dtype=np.float64)
    fit = _fit_candidate(z, sigma, gaussian=True)
    assert fit.pathology in (0, 1)


def test_fit_candidate_sigma_blowup_nonneg() -> None:
    rng = np.random.default_rng(12)
    z = rng.standard_normal(40).astype(np.float64)
    sigma = np.ones(40, dtype=np.float64)
    fit = _fit_candidate(z, sigma, gaussian=True)
    assert fit.sigma_blowup >= 0


# ---------------------------------------------------------------------------
# _check_window
# ---------------------------------------------------------------------------


def test_check_window_all_pass() -> None:
    fit = _make_fit()
    fails = _check_window(fit, "Window_2017")
    assert fails == []


def test_check_window_negative_corr_next_fails() -> None:
    fit = _make_fit(corr_next=-0.1)
    fails = _check_window(fit, "Window_2017")
    assert any("corr_next_not_positive" in f for f in fails)


def test_check_window_none_corr_next_fails() -> None:
    fit = _make_fit(corr_next=None)
    fails = _check_window(fit, "Window_2017")
    assert any("corr_next_not_positive" in f for f in fails)


def test_check_window_mean_bias_fails() -> None:
    fit = _make_fit(mean_v=0.25)
    fails = _check_window(fit, "Window_2017")
    assert any("mean_v_bias" in f for f in fails)


def test_check_window_std_out_of_range_fails() -> None:
    fit_lo = _make_fit(std_v=0.75)
    fit_hi = _make_fit(std_v=1.25)
    assert any("std_v_out_of_range" in f for f in _check_window(fit_lo, "W"))
    assert any("std_v_out_of_range" in f for f in _check_window(fit_hi, "W"))


def test_check_window_cal_breach_fails() -> None:
    fit = _make_fit(cal_error_025=0.09)
    fails = _check_window(fit, "Window_2017")
    assert any("cal_025_breach" in f for f in fails)


def test_check_window_sigma_blowup_fails() -> None:
    fit = _make_fit(sigma_blowup=1)
    fails = _check_window(fit, "Window_2017")
    assert any("sigma_blowup" in f for f in fails)


def test_check_window_pathology_fails() -> None:
    fit = _make_fit(pathology=1)
    fails = _check_window(fit, "Window_2017")
    assert any("pathology" in f for f in fails)


# ---------------------------------------------------------------------------
# _apply_decision
# ---------------------------------------------------------------------------


def test_apply_decision_pass_a() -> None:
    good = _make_fit()
    fits_a = {"W_2017": good, "W_2018": good, "W_2020": good}
    bad = _make_fit(corr_next=-1.0)
    fits_b = {"W_2017": bad, "W_2018": bad, "W_2020": bad}
    decision = _apply_decision(fits_a, fits_b)
    assert decision.overall == "PASS_A"
    assert decision.pass_a is True
    assert decision.pass_b is False


def test_apply_decision_pass_b() -> None:
    bad = _make_fit(corr_next=-1.0)
    fits_a = {"W_2017": bad, "W_2018": bad, "W_2020": bad}
    good = _make_fit()
    fits_b = {"W_2017": good, "W_2018": good, "W_2020": good}
    decision = _apply_decision(fits_a, fits_b)
    assert decision.overall == "PASS_B"
    assert decision.pass_b is True


def test_apply_decision_fail_both() -> None:
    bad = _make_fit(corr_next=-1.0)
    fits_a = {"W_2017": bad, "W_2018": bad, "W_2020": bad}
    fits_b = {"W_2017": bad, "W_2018": bad, "W_2020": bad}
    decision = _apply_decision(fits_a, fits_b)
    assert decision.overall == "FAIL"
    assert decision.pass_a is False
    assert decision.pass_b is False


def test_apply_decision_one_window_fail_is_overall_fail() -> None:
    """Even a single window failing makes the candidate fail (per preregistration §7)."""
    good = _make_fit()
    bad = _make_fit(mean_v=0.9)
    fits_a = {"W_2017": good, "W_2018": bad, "W_2020": good}
    fits_b = {"W_2017": bad, "W_2018": bad, "W_2020": bad}
    decision = _apply_decision(fits_a, fits_b)
    assert decision.overall == "FAIL"
    assert decision.pass_a is False
