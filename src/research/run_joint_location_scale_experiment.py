"""io: run the preregistered Option 2A joint location-scale falsification experiment."""

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
    norm as scipy_norm,
    t as student_t,
)

# Preregistered constants — must not be modified after freezing the preregistration.
C_BOUND: Final[float] = 0.99
K_LOWER: Final[float] = 0.1
K_UPPER: Final[float] = 10.0
_DEGENERATE_STD_GUARD: Final[float] = 1.0e-12
NU_FIXED: Final[float] = 10.0  # Candidate B only, preregistered
MEAN_TOL: Final[float] = 0.20
STD_LO: Final[float] = 0.80
STD_HI: Final[float] = 1.20
CAL_TOL_ASYM: Final[float] = 0.08  # tau_0.25 and tau_0.75 only
EVAL_TAUS: Final[tuple[float, ...]] = (0.25, 0.75)  # asymmetric quantiles per prereg

PILOT_WINDOWS: Final[dict[str, tuple[date, date]]] = {
    "Window_2017": (date(2017, 7, 7), date(2017, 12, 29)),
    "Window_2018": (date(2018, 7, 6), date(2018, 12, 28)),
    "Window_2020": (date(2020, 1, 3), date(2020, 6, 26)),
}
OUTPUT_DIR: Final[Path] = Path("artifacts/research/joint_location_scale")
PREREG_DOC: Final[Path] = Path(
    "docs/rank_scale_hybrid/11_option2a_preregistration.md",
)


@dataclass(frozen=True, slots=True)
class JointFit:
    """pure. Fitted joint (mu, sigma) diagnostics for one window and one candidate."""

    c: float
    k_new: float
    loglik: float
    mean_v: float
    std_v: float
    cal_error_025: float
    cal_error_075: float
    corr_next: float | None
    rank_next: float | None
    sigma_blowup: int
    pathology: int


@dataclass(frozen=True, slots=True)
class Option2ADecision:
    """pure. Preregistered decision outcome for the Option 2A experiment."""

    pass_a: bool
    pass_b: bool
    failed_a: list[str]
    failed_b: list[str]
    overall: str  # "PASS_A", "PASS_B", "FAIL"


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


def _apply_location_correction(
    z: NDArray[np.float64],
    c: float,
) -> NDArray[np.float64]:
    """pure. z_t_corr = z_t - c * z_{t-1}, with z_{t-1}=0 for t=0."""
    z_prev = np.concatenate([[0.0], z[:-1]])
    return (z - c * z_prev).astype(np.float64)


def _neg_loglik_gaussian(
    params: NDArray[np.float64],
    z: NDArray[np.float64],
) -> float:
    """pure. Negative log-likelihood for Candidate A (Gaussian)."""
    c, log_k = float(params[0]), float(params[1])
    if abs(c) >= C_BOUND:
        return 1.0e15
    k = float(np.exp(log_k))
    if not np.isfinite(k) or k <= 0.0:
        return 1.0e15
    z_corr = _apply_location_correction(z, c)
    v = z_corr / k
    if not np.isfinite(v).all():
        return 1.0e15
    logpdf = scipy_norm.logpdf(v) - np.log(k)
    return float(-np.sum(logpdf))


def _neg_loglik_student_t(
    params: NDArray[np.float64],
    z: NDArray[np.float64],
    nu: float,
) -> float:
    """pure. Negative log-likelihood for Candidate B (Student-t, fixed nu)."""
    c, log_k = float(params[0]), float(params[1])
    if abs(c) >= C_BOUND:
        return 1.0e15
    k = float(np.exp(log_k))
    if not np.isfinite(k) or k <= 0.0:
        return 1.0e15
    z_corr = _apply_location_correction(z, c)
    v = z_corr / k
    if not np.isfinite(v).all():
        return 1.0e15
    logpdf = _student_t_std_logpdf(v, nu) - np.log(k)
    return float(-np.sum(logpdf))


def _corr(x: NDArray[np.float64], y: NDArray[np.float64]) -> float | None:
    """pure. Pearson correlation, returns None if degenerate."""
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) <= 1:
        return None
    xs, ys = x[mask], y[mask]
    if float(np.std(xs)) <= _DEGENERATE_STD_GUARD or float(np.std(ys)) <= _DEGENERATE_STD_GUARD:
        return None
    return float(np.corrcoef(xs, ys)[0, 1])


def _rank_corr(x: NDArray[np.float64], y: NDArray[np.float64]) -> float | None:
    """pure. Spearman rank correlation, returns None if degenerate."""
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) <= 1:
        return None
    xs, ys = x[mask], y[mask]
    xr = np.argsort(np.argsort(xs)).astype(np.float64)
    yr = np.argsort(np.argsort(ys)).astype(np.float64)
    if float(np.std(xr)) <= _DEGENERATE_STD_GUARD or float(np.std(yr)) <= _DEGENERATE_STD_GUARD:
        return None
    return float(np.corrcoef(xr, yr)[0, 1])


def _calibration_error(v: NDArray[np.float64], tau: float, *, gaussian: bool) -> float:
    """pure. |empirical_coverage(tau) - tau| for Candidate A (Gaussian) or B (Student-t)."""
    threshold = float(scipy_norm.ppf(tau)) if gaussian else _student_t_std_ppf(tau, NU_FIXED)
    empirical = float(np.mean(v[np.isfinite(v)] <= threshold))
    return abs(empirical - tau)


def _fit_candidate(
    z: NDArray[np.float64],
    sigma_t5: NDArray[np.float64],
    *,
    gaussian: bool,
) -> JointFit:
    """pure. MLE joint (c, k_new) for one window, one candidate."""
    finite_mask = np.isfinite(z)
    z_clean = np.asarray(z[finite_mask], dtype=np.float64)
    sigma_clean = np.asarray(sigma_t5[finite_mask], dtype=np.float64)

    if z_clean.size == 0:
        raise ValueError("empty finite z series")

    initial = np.array([0.0, np.log(float(np.std(z_clean, ddof=0)))], dtype=np.float64)
    bounds = ((-C_BOUND, C_BOUND), (np.log(K_LOWER), np.log(K_UPPER)))

    def obj(params: NDArray[np.float64]) -> float:
        if gaussian:
            return _neg_loglik_gaussian(params, z_clean)
        return _neg_loglik_student_t(params, z_clean, NU_FIXED)

    result = minimize(obj, initial, method="L-BFGS-B", bounds=bounds)
    pathology = 0 if result.success else 1

    c = float(result.x[0])
    k_new = float(np.exp(float(result.x[1])))
    loglik = float(-result.fun)

    z_corr = _apply_location_correction(z_clean, c)
    v = z_corr / k_new

    mean_v = float(np.mean(v))
    std_v = float(np.std(v, ddof=0))
    cal_025 = _calibration_error(v, 0.25, gaussian=gaussian)
    cal_075 = _calibration_error(v, 0.75, gaussian=gaussian)

    sigma_new = sigma_clean * k_new
    sigma_blowup = int(np.sum(sigma_new > 2.0 * float(np.median(sigma_new))))

    # New residuals: e_{t+1}_new = sigma_{t+1} * z_{t+1}_corr
    e_new_abs = sigma_clean * np.abs(z_corr)
    corr_n = _corr(sigma_new[:-1], e_new_abs[1:]) if sigma_new.shape[0] > 1 else None
    rank_n = _rank_corr(sigma_new[:-1], e_new_abs[1:]) if sigma_new.shape[0] > 1 else None

    return JointFit(
        c=c,
        k_new=k_new,
        loglik=loglik,
        mean_v=mean_v,
        std_v=std_v,
        cal_error_025=cal_025,
        cal_error_075=cal_075,
        corr_next=corr_n,
        rank_next=rank_n,
        sigma_blowup=sigma_blowup,
        pathology=pathology,
    )


def _check_window(fit: JointFit, window_name: str) -> list[str]:
    """pure. Apply preregistered success criteria to one window's fit."""
    fails: list[str] = []
    if fit.corr_next is None or fit.corr_next <= 0.0:
        fails.append(f"corr_next_not_positive:{window_name}:{fit.corr_next}")
    if fit.rank_next is None or fit.rank_next <= 0.0:
        fails.append(f"rank_next_not_positive:{window_name}:{fit.rank_next}")
    if abs(fit.mean_v) >= MEAN_TOL:
        fails.append(f"mean_v_bias:{window_name}:{fit.mean_v:.4f}")
    if not (STD_LO <= fit.std_v <= STD_HI):
        fails.append(f"std_v_out_of_range:{window_name}:{fit.std_v:.4f}")
    if fit.cal_error_025 >= CAL_TOL_ASYM:
        fails.append(f"cal_025_breach:{window_name}:{fit.cal_error_025:.4f}")
    if fit.cal_error_075 >= CAL_TOL_ASYM:
        fails.append(f"cal_075_breach:{window_name}:{fit.cal_error_075:.4f}")
    if fit.sigma_blowup > 0:
        fails.append(f"sigma_blowup:{window_name}:{fit.sigma_blowup}")
    if fit.pathology > 0:
        fails.append(f"pathology:{window_name}:{fit.pathology}")
    return fails


def _apply_decision(
    fits_a: dict[str, JointFit],
    fits_b: dict[str, JointFit],
) -> Option2ADecision:
    """pure. Apply preregistered decision rules for both candidates."""
    failed_a: list[str] = []
    for window_name, fit in fits_a.items():
        failed_a.extend(_check_window(fit, window_name))

    failed_b: list[str] = []
    for window_name, fit in fits_b.items():
        failed_b.extend(_check_window(fit, window_name))

    pass_a = len(failed_a) == 0
    pass_b = len(failed_b) == 0

    if pass_a:
        overall = "PASS_A"
    elif pass_b:
        overall = "PASS_B"
    else:
        overall = "FAIL"

    return Option2ADecision(
        pass_a=pass_a,
        pass_b=pass_b,
        failed_a=failed_a,
        failed_b=failed_b,
        overall=overall,
    )


def _load_window_arrays() -> dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]]:
    """io: Load T5 z_t and sigma_t sequences for each pilot window."""
    from research import t5_recovered_source as t5

    context = t5.build_har_context(max(end for _start, end in PILOT_WINDOWS.values()))
    result: dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]] = {}
    for window_name, (start, end) in PILOT_WINDOWS.items():
        arr = t5._eval_original_t5_window(context, start, end)
        result[window_name] = (
            np.asarray(arr.z, dtype=np.float64),
            np.asarray(arr.sigma, dtype=np.float64),
        )
    return result


def _markdown_report(
    fits_a: dict[str, JointFit],
    fits_b: dict[str, JointFit],
    decision: Option2ADecision,
) -> str:
    """pure. Render the Option 2A result report."""
    lines = [
        "# Option 2A Joint Location-Scale Results",
        "",
        "> Generated by `src/research/run_joint_location_scale_experiment.py`.",
        f"> Preregistration: `{PREREG_DOC}`",
        "",
        "## Candidate A (Gaussian)",
        "",
        "| window | c | k_new | mean_v | std_v | cal@0.25 | cal@0.75 | corr_next | rank_next |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for wn, fit in fits_a.items():
        lines.append(
            "| "
            + " | ".join([
                wn,
                f"{fit.c:.4f}",
                f"{fit.k_new:.4f}",
                f"{fit.mean_v:.4f}",
                f"{fit.std_v:.4f}",
                f"{fit.cal_error_025:.4f}",
                f"{fit.cal_error_075:.4f}",
                f"{fit.corr_next:.4f}" if fit.corr_next is not None else "None",
                f"{fit.rank_next:.4f}" if fit.rank_next is not None else "None",
            ])
            + " |",
        )
    lines.extend([
        "",
        "## Candidate B (Student-t nu=10)",
        "",
        "| window | c | k_new | mean_v | std_v | cal@0.25 | cal@0.75 | corr_next | rank_next |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ])
    for wn, fit in fits_b.items():
        lines.append(
            "| "
            + " | ".join([
                wn,
                f"{fit.c:.4f}",
                f"{fit.k_new:.4f}",
                f"{fit.mean_v:.4f}",
                f"{fit.std_v:.4f}",
                f"{fit.cal_error_025:.4f}",
                f"{fit.cal_error_075:.4f}",
                f"{fit.corr_next:.4f}" if fit.corr_next is not None else "None",
                f"{fit.rank_next:.4f}" if fit.rank_next is not None else "None",
            ])
            + " |",
        )
    lines.extend([
        "",
        "## Protocol Decision",
        "",
        f"- Overall: `{decision.overall}`",
        f"- PASS_A: `{decision.pass_a}`",
        f"- PASS_B: `{decision.pass_b}`",
        f"- Failed A: `{', '.join(decision.failed_a) or 'NONE'}`",
        f"- Failed B: `{', '.join(decision.failed_b) or 'NONE'}`",
    ])
    return "\n".join(lines) + "\n"


def _write_decision_doc(
    decision: Option2ADecision,
    fits_a: dict[str, JointFit],
    fits_b: dict[str, JointFit],
) -> None:
    """io: Write the Option 2A decision document."""
    if decision.overall in ("PASS_A", "PASS_B"):
        conclusion = (
            f"{decision.overall}: joint location-scale correction absorbs location bias. "
            "Quantile calibration passes at τ ∈ {0.25, 0.75} across all three windows. "
            "Allowed to proceed to tail family preregistration with shape parameters."
        )
    else:
        conclusion = (
            "FAIL: joint location-scale correction does not adequately absorb location bias "
            "under the preregistered criteria. "
            f"Candidate A failed conditions: {'; '.join(decision.failed_a) or 'NONE'}. "
            f"Candidate B failed conditions: {'; '.join(decision.failed_b) or 'NONE'}. "
            "Research line terminated."
        )

    c_rows = "\n".join(
        f"- {wn}: c={fit.c:.4f}, k={fit.k_new:.4f}, "
        f"mean_v={fit.mean_v:.4f}, std_v={fit.std_v:.4f}, "
        f"cal@0.25={fit.cal_error_025:.4f}, cal@0.75={fit.cal_error_075:.4f}"
        for wn, fit in fits_a.items()
    )
    b_rows = "\n".join(
        f"- {wn}: c={fit.c:.4f}, k={fit.k_new:.4f}, "
        f"mean_v={fit.mean_v:.4f}, std_v={fit.std_v:.4f}, "
        f"cal@0.25={fit.cal_error_025:.4f}, cal@0.75={fit.cal_error_075:.4f}"
        for wn, fit in fits_b.items()
    )

    doc = f"""# Option 2A Decision

## Experiment Config

- Preregistration: `{PREREG_DOC.as_posix()}`
- Model: one-step residual persistence location correction
- Candidate A: Gaussian, MLE (c, k_new) per window
- Candidate B: Student-t nu={NU_FIXED:.0f}, MLE (c, k_new) per window

## Candidate A Results

{c_rows}

## Candidate B Results

{b_rows}

## Protocol Decision

- Overall: `{decision.overall}`
- PASS_A: `{decision.pass_a}`
- PASS_B: `{decision.pass_b}`

## Conclusion

{conclusion}

## Boundary

This decision does not modify production code, SRD §18 constants, or the law layer.
It does not reopen trigger audit, crisis architecture, hard switch, override, or MoE.
"""
    (Path("docs/rank_scale_hybrid") / "12_option2a_decision.md").write_text(
        doc, encoding="utf-8",
    )


def _json_default(obj: object) -> object:
    """pure. JSON encoder for numpy scalar values."""
    if isinstance(obj, np.floating | np.integer):
        return obj.item()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def run(argv: list[str] | None = None) -> int:
    """io: Execute the Option 2A joint location-scale experiment and write artifacts."""
    parser = argparse.ArgumentParser(prog="run_joint_location_scale_experiment")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON to stdout.")
    args = parser.parse_args(argv)

    arrays = _load_window_arrays()

    fits_a: dict[str, JointFit] = {
        wn: _fit_candidate(z, sigma, gaussian=True)
        for wn, (z, sigma) in arrays.items()
    }
    fits_b: dict[str, JointFit] = {
        wn: _fit_candidate(z, sigma, gaussian=False)
        for wn, (z, sigma) in arrays.items()
    }

    decision = _apply_decision(fits_a, fits_b)

    payload: dict[str, Any] = {
        "preregistration": PREREG_DOC.as_posix(),
        "nu_fixed_candidate_b": NU_FIXED,
        "known_2020_direction_risk": "corr_next_t5 = -0.033011",
        "candidate_a": {wn: asdict(fit) for wn, fit in fits_a.items()},
        "candidate_b": {wn: asdict(fit) for wn, fit in fits_b.items()},
        "decision": asdict(decision),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "results.json").write_text(
        json.dumps(payload, default=_json_default, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (OUTPUT_DIR / "results.md").write_text(
        _markdown_report(fits_a, fits_b, decision),
        encoding="utf-8",
    )
    _write_decision_doc(decision, fits_a, fits_b)

    indent = 2 if args.pretty else None
    sys.stdout.write(
        json.dumps(payload, default=_json_default, indent=indent, sort_keys=True) + "\n",
    )
    return 0


def main() -> None:
    """io: process exit entrypoint."""
    raise SystemExit(run())


if __name__ == "__main__":
    main()
