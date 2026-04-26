"""io: run the preregistered full tail family experiment (Student-t, global k_fixed)."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import date
import json
from pathlib import Path
import sys
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import (
    kstest,
    t as student_t,
)

# Preregistered constants — must not be modified after freezing the preregistration.
K_FIXED: Final[float] = 2.682572
NU_MIN: Final[float] = 2.05
NU_MAX: Final[float] = 100.0
CALIBRATION_TOL: Final[float] = 0.05
KS_P_MIN: Final[float] = 0.05
NU_DRIFT_MAX: Final[float] = 30.0
COVERAGE_90_MIN: Final[float] = 0.70
COVERAGE_80_MIN: Final[float] = 0.60
EVAL_TAUS: Final[tuple[float, ...]] = (0.10, 0.25, 0.75, 0.90)

PILOT_WINDOWS: Final[dict[str, tuple[date, date]]] = {
    "Window_2017": (date(2017, 7, 7), date(2017, 12, 29)),
    "Window_2018": (date(2018, 7, 6), date(2018, 12, 28)),
    "Window_2020": (date(2020, 1, 3), date(2020, 6, 26)),
}
OUTPUT_DIR: Final[Path] = Path("artifacts/research/tail_family")
PREREG_DOC: Final[Path] = Path(
    "docs/rank_scale_hybrid/07_tail_family_preregistration.md",
)


@dataclass(frozen=True, slots=True)
class TailFamilyFit:
    """pure. Fitted tail family diagnostics for one window (or pooled)."""

    nu: float
    loglik: float
    calibration_errors: dict[str, float]
    ks_statistic: float
    ks_pvalue: float
    central_80_coverage: float
    central_90_coverage: float


@dataclass(frozen=True, slots=True)
class TailFamilyDecision:
    """pure. Preregistered decision outcome for the tail family experiment."""

    pass_per_window: bool
    pass_pooled: bool
    nu_drift_ok: bool
    failed_conditions: list[str]
    nu_drift: float
    nu_min: float
    nu_max: float


def _student_t_std_logpdf(values: NDArray[np.float64], nu: float) -> NDArray[np.float64]:
    """pure. Log density of variance-one Student-t innovations."""
    scale = np.sqrt((nu - 2.0) / nu)
    standardized = values / scale
    log_norm = (
        gammaln((nu + 1.0) * 0.5)
        - gammaln(nu * 0.5)
        - 0.5 * np.log(nu * np.pi)
        - np.log(scale)
    )
    result = log_norm - ((nu + 1.0) * 0.5) * np.log1p((standardized**2) / nu)
    return np.asarray(result, dtype=np.float64)


def _student_t_std_ppf(probability: float, nu: float) -> float:
    """pure. Quantile of a variance-one Student-t innovation."""
    return float(student_t.ppf(probability, df=nu) * np.sqrt((nu - 2.0) / nu))


def _student_t_std_cdf(value: float, nu: float) -> float:
    """pure. CDF of a variance-one Student-t innovation."""
    scale = np.sqrt((nu - 2.0) / nu)
    return float(student_t.cdf(value / scale, df=nu))


def _neg_loglik_nu(params: NDArray[np.float64], u_values: NDArray[np.float64]) -> float:
    """pure. Negative log-likelihood for nu given u_t already divided by k_fixed."""
    (log_nu_shift,) = (float(v) for v in params)
    nu = float(NU_MIN + np.exp(log_nu_shift))
    if not np.isfinite(nu) or nu > NU_MAX:
        return 1.0e15
    logpdf = _student_t_std_logpdf(u_values, nu)
    if not np.isfinite(logpdf).all():
        return 1.0e15
    return float(-np.sum(logpdf))


def _mle_nu(u_values: NDArray[np.float64]) -> tuple[float, float]:
    """pure. MLE Student-t nu for u_t ~ StudentT_std(nu). Returns (nu, loglik)."""
    finite_u = np.asarray(u_values[np.isfinite(u_values)], dtype=np.float64)
    if finite_u.size == 0:
        raise ValueError("empty finite u series")
    result = minimize(
        lambda params: _neg_loglik_nu(params, finite_u),
        np.array([np.log(8.0 - NU_MIN)], dtype=np.float64),
        method="L-BFGS-B",
        bounds=((np.log(2.1 - NU_MIN), np.log(NU_MAX - NU_MIN)),),
    )
    if not result.success:
        raise ValueError(f"nu optimizer failed: {result.message}")
    nu = float(NU_MIN + np.exp(float(result.x[0])))
    loglik = float(-result.fun)
    return nu, loglik


def _calibration_errors(
    u_values: NDArray[np.float64],
    nu: float,
) -> dict[str, float]:
    """pure. Calibration error |empirical_coverage(tau) - tau| per tau."""
    finite_u = u_values[np.isfinite(u_values)]
    errors: dict[str, float] = {}
    for tau in EVAL_TAUS:
        threshold = _student_t_std_ppf(tau, nu)
        empirical = float(np.mean(finite_u <= threshold))
        errors[f"tau_{tau:.2f}"] = abs(empirical - tau)
    return errors


def _coverage(u_values: NDArray[np.float64], nu: float, central_mass: float) -> float:
    """pure. Empirical central interval coverage for u_t ~ StudentT_std(nu)."""
    tail = (1.0 - central_mass) * 0.5
    lower = _student_t_std_ppf(tail, nu)
    upper = _student_t_std_ppf(1.0 - tail, nu)
    finite_u = u_values[np.isfinite(u_values)]
    return float(np.mean((finite_u >= lower) & (finite_u <= upper)))


def _student_t_std_cdf_vec(values: NDArray[np.float64], nu: float) -> NDArray[np.float64]:
    """pure. Vectorized CDF of a variance-one Student-t innovation."""
    scale = float(np.sqrt((nu - 2.0) / nu))
    return np.asarray(student_t.cdf(values / scale, df=nu), dtype=np.float64)


def _ks_test(u_values: NDArray[np.float64], nu: float) -> tuple[float, float]:
    """pure. KS test of u_t against StudentT_std(nu). Returns (statistic, pvalue)."""
    finite_u = u_values[np.isfinite(u_values)]
    result = kstest(finite_u, lambda x: _student_t_std_cdf_vec(np.asarray(x, dtype=np.float64), nu))
    return float(result.statistic), float(result.pvalue)


def _fit_window(u_values: NDArray[np.float64]) -> TailFamilyFit:
    """pure. Fit tail family for one window's u_t = z_t / k_fixed."""
    nu, loglik = _mle_nu(u_values)
    cal_errors = _calibration_errors(u_values, nu)
    ks_stat, ks_p = _ks_test(u_values, nu)
    cov80 = _coverage(u_values, nu, 0.80)
    cov90 = _coverage(u_values, nu, 0.90)
    return TailFamilyFit(
        nu=nu,
        loglik=loglik,
        calibration_errors=cal_errors,
        ks_statistic=ks_stat,
        ks_pvalue=ks_p,
        central_80_coverage=cov80,
        central_90_coverage=cov90,
    )


def _apply_decision(
    fits: dict[str, TailFamilyFit],
    pooled_fit: TailFamilyFit,
) -> TailFamilyDecision:
    """pure. Apply the preregistered decision rules."""
    failed: list[str] = []

    # §5.1 calibration breach (per-window)
    for window_name, fit in fits.items():
        for tau_key, err in fit.calibration_errors.items():
            if err > CALIBRATION_TOL:
                failed.append(f"CALIBRATION_BREACH:{window_name}:{tau_key}:{err:.4f}")

    # §5.2 KS reject (per-window)
    for window_name, fit in fits.items():
        if fit.ks_pvalue < KS_P_MIN:
            failed.append(f"KS_REJECT:{window_name}:p={fit.ks_pvalue:.4f}")

    # §5.4 2020 direction defect propagation
    fit_2020 = fits["Window_2020"]
    if fit_2020.central_90_coverage < COVERAGE_90_MIN:
        failed.append(f"DIRECTION_DEFECT_PROPAGATED:cov90={fit_2020.central_90_coverage:.4f}")
    if fit_2020.central_80_coverage < COVERAGE_80_MIN:
        failed.append(f"DIRECTION_DEFECT_PROPAGATED:cov80={fit_2020.central_80_coverage:.4f}")

    nu_values = [fit.nu for fit in fits.values()]
    nu_drift = float(max(nu_values) - min(nu_values))
    nu_drift_ok = nu_drift <= NU_DRIFT_MAX

    pass_per_window = len(failed) == 0

    # Pooled PASS (diagnostic, §5.5)
    pooled_fails: list[str] = []
    for tau_key, err in pooled_fit.calibration_errors.items():
        if err > CALIBRATION_TOL:
            pooled_fails.append(tau_key)
    if pooled_fit.ks_pvalue < KS_P_MIN:
        pooled_fails.append("KS_REJECT")
    pass_pooled = nu_drift_ok and len(pooled_fails) == 0

    return TailFamilyDecision(
        pass_per_window=pass_per_window,
        pass_pooled=pass_pooled,
        nu_drift_ok=nu_drift_ok,
        failed_conditions=failed,
        nu_drift=nu_drift,
        nu_min=float(min(nu_values)),
        nu_max=float(max(nu_values)),
    )


def _load_u_by_window() -> dict[str, NDArray[np.float64]]:
    """io: Load T5 z_t and apply global k_fixed to produce u_t per window."""
    from research import t5_recovered_source as t5

    context = t5.build_har_context(max(end for _start, end in PILOT_WINDOWS.values()))
    u_by_window: dict[str, NDArray[np.float64]] = {}
    for window_name, (start, end) in PILOT_WINDOWS.items():
        arr = t5._eval_original_t5_window(context, start, end)
        z = np.asarray(arr.z, dtype=np.float64)
        u_by_window[window_name] = z / K_FIXED
    return u_by_window


def _markdown_report(
    fits: dict[str, TailFamilyFit],
    pooled_fit: TailFamilyFit,
    decision: TailFamilyDecision,
) -> str:
    """pure. Render the tail family result report."""
    lines = [
        "# Tail Family Results",
        "",
        "> Generated by `src/research/run_tail_family_experiment.py`.",
        f"> k_fixed = {K_FIXED}",
        "",
        "## Per-Window Diagnostics",
        "",
        "| window | nu | ks_p | c10 | c25 | c75 | c90 | cov80 | cov90 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for window_name, fit in fits.items():
        ce = fit.calibration_errors
        lines.append(
            "| "
            + " | ".join([
                window_name,
                f"{fit.nu:.4f}",
                f"{fit.ks_pvalue:.4f}",
                f"{ce.get('tau_0.10', float('nan')):.4f}",
                f"{ce.get('tau_0.25', float('nan')):.4f}",
                f"{ce.get('tau_0.75', float('nan')):.4f}",
                f"{ce.get('tau_0.90', float('nan')):.4f}",
                f"{fit.central_80_coverage:.4f}",
                f"{fit.central_90_coverage:.4f}",
            ])
            + " |",
        )
    pooled_ce = pooled_fit.calibration_errors
    lines.extend([
        "",
        "## Pooled Diagnostics",
        "",
        f"- nu_pooled: `{pooled_fit.nu:.4f}`",
        f"- ks_pvalue: `{pooled_fit.ks_pvalue:.4f}`",
        f"- cal_0.10: `{pooled_ce.get('tau_0.10', float('nan')):.4f}`",
        f"- cal_0.25: `{pooled_ce.get('tau_0.25', float('nan')):.4f}`",
        f"- cal_0.75: `{pooled_ce.get('tau_0.75', float('nan')):.4f}`",
        f"- cal_0.90: `{pooled_ce.get('tau_0.90', float('nan')):.4f}`",
        "",
        "## Protocol Decision",
        "",
        f"- PASS_PER_WINDOW: `{decision.pass_per_window}`",
        f"- PASS_POOLED: `{decision.pass_pooled}`",
        f"- NU_DRIFT_OK: `{decision.nu_drift_ok}` (drift = {decision.nu_drift:.4f})",
        f"- Failed conditions: `{', '.join(decision.failed_conditions) or 'NONE'}`",
    ])
    return "\n".join(lines) + "\n"


def _write_decision_doc(decision: TailFamilyDecision, fits: dict[str, TailFamilyFit]) -> None:
    """io: Write the one-page decision document for the tail family experiment."""
    nu_rows = "\n".join(
        f"- {name}: nu = {fit.nu:.4f}" for name, fit in fits.items()
    )
    if decision.pass_per_window:
        conclusion = (
            "PASS_PER_WINDOW: Student-t with global k_fixed provides calibrated quantile "
            "estimates across all three windows. Fixed T5 + k_fixed + per-window nu is "
            "established as a viable downstream configuration."
        )
        if decision.pass_pooled:
            conclusion += (
                "\n\nPASS_POOLED also holds: a single pooled nu may be used "
                "for a one-parameter downstream configuration."
            )
        else:
            reason = (
                "NU_DRIFT exceeded" if not decision.nu_drift_ok else "pooled calibration failed"
            )
            conclusion += f"\n\nPASS_POOLED does not hold ({reason}): per-window nu required."
    else:
        conclusion = (
            "FAIL: Student-t with global k_fixed does not provide adequate quantile "
            "calibration under the preregistered rules. "
            f"Failed conditions: {'; '.join(decision.failed_conditions)}."
        )
    doc = f"""# Tail Family Decision

## Experiment Config

- Preregistration: `{PREREG_DOC.as_posix()}`
- k_fixed: `{K_FIXED}`
- Tail family: variance-one Student-t, MLE nu per window and pooled
- Input: T5 z_t / k_fixed = u_t for windows 2017, 2018, 2020

## Nu Estimates

{nu_rows}

- nu_drift: `{decision.nu_drift:.4f}` (threshold: {NU_DRIFT_MAX})
- NU_DRIFT_OK: `{decision.nu_drift_ok}`

## Protocol Decision

- PASS_PER_WINDOW: `{decision.pass_per_window}`
- PASS_POOLED: `{decision.pass_pooled}`
- Failed conditions: `{", ".join(decision.failed_conditions) or "NONE"}`

## Conclusion

{conclusion}

## Boundary

This decision does not modify production code, SRD §18 constants, or the law layer.
It does not reopen trigger audit, crisis architecture, hard switch, override, or MoE.
"""
    (Path("docs/rank_scale_hybrid") / "08_tail_family_decision.md").write_text(
        doc,
        encoding="utf-8",
    )


def _json_default(obj: object) -> object:
    """pure. JSON encoder for numpy scalar values."""
    if isinstance(obj, np.floating | np.integer):
        return obj.item()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def run(argv: list[str] | None = None) -> int:
    """io: Execute the tail family experiment and write artifacts."""
    parser = argparse.ArgumentParser(prog="run_tail_family_experiment")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON to stdout.")
    args = parser.parse_args(argv)

    u_by_window = _load_u_by_window()

    fits: dict[str, TailFamilyFit] = {
        window_name: _fit_window(u_values)
        for window_name, u_values in u_by_window.items()
    }

    all_u = np.concatenate(list(u_by_window.values()))
    pooled_fit = _fit_window(all_u)

    decision = _apply_decision(fits, pooled_fit)

    payload: dict[str, Any] = {
        "preregistration": PREREG_DOC.as_posix(),
        "k_fixed": K_FIXED,
        "known_2020_direction_risk": "corr_next = -0.033011",
        "fits": {name: asdict(fit) for name, fit in fits.items()},
        "pooled": asdict(pooled_fit),
        "decision": asdict(decision),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "tail_family_results.json").write_text(
        json.dumps(payload, default=_json_default, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (OUTPUT_DIR / "tail_family_results.md").write_text(
        _markdown_report(fits, pooled_fit, decision),
        encoding="utf-8",
    )
    _write_decision_doc(decision, fits)

    text = json.dumps(
        payload,
        default=_json_default,
        indent=2 if args.pretty else None,
        sort_keys=True,
    )
    sys.stdout.write(text + "\n")
    return 0


def main() -> None:
    """io: process exit entrypoint."""
    raise SystemExit(run())


if __name__ == "__main__":
    main()
