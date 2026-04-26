"""io: run the preregistered downstream scale-bias experiment."""

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
from scipy.stats import t as student_t

PILOT_WINDOWS: Final[dict[str, tuple[date, date]]] = {
    "Window_2017": (date(2017, 7, 7), date(2017, 12, 29)),
    "Window_2018": (date(2018, 7, 6), date(2018, 12, 28)),
    "Window_2020": (date(2020, 1, 3), date(2020, 6, 26)),
}
OUTPUT_DIR: Final[Path] = Path("artifacts/research/downstream_scale_bias")
PREREG_DOC: Final[Path] = Path(
    "docs/rank_scale_hybrid/05_downstream_scale_bias_preregistration.md",
)
K_MIN: Final[float] = 1.5
K_MAX: Final[float] = 2.75
K_RANGE_MAX: Final[float] = 0.60
NU_MIN: Final[float] = 2.05
NU_MAX: Final[float] = 100.0
K_LOWER_BOUND: Final[float] = 0.25
K_UPPER_BOUND: Final[float] = 10.0
COVERAGE_90_MIN: Final[float] = 0.70
COVERAGE_80_2020_MIN: Final[float] = 0.60
MIN_POSITIVE_WINDOW_IMPROVEMENTS: Final[int] = 2


@dataclass(frozen=True, slots=True)
class ScaleBiasFit:
    """pure. Fitted downstream scale-bias diagnostics for one window."""

    k: float
    nu: float
    loglik: float
    naive_nu: float
    naive_loglik: float
    central_80_coverage: float
    central_90_coverage: float


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
    return (log_norm - ((nu + 1.0) * 0.5) * np.log1p((standardized**2) / nu)).astype(
        np.float64,
    )


def _student_t_std_ppf(probability: float, nu: float) -> float:
    """pure. Quantile of a variance-one Student-t innovation."""
    return float(student_t.ppf(probability, df=nu) * np.sqrt((nu - 2.0) / nu))


def _neg_loglik(params: NDArray[np.float64], z_values: NDArray[np.float64]) -> float:
    """pure. Negative log-likelihood for free k and nu."""
    log_k, log_nu_shift = [float(value) for value in params]
    k = float(np.exp(log_k))
    nu = float(NU_MIN + np.exp(log_nu_shift))
    if not np.isfinite(k) or not np.isfinite(nu) or k <= 0.0 or nu > NU_MAX:
        return 1.0e15
    u_values = z_values / k
    logpdf = _student_t_std_logpdf(u_values, nu) - np.log(k)
    if not np.isfinite(logpdf).all():
        return 1.0e15
    return float(-np.sum(logpdf))


def _neg_loglik_naive(params: NDArray[np.float64], z_values: NDArray[np.float64]) -> float:
    """pure. Negative log-likelihood for k fixed at one and free nu."""
    (log_nu_shift,) = [float(value) for value in params]
    nu = float(NU_MIN + np.exp(log_nu_shift))
    if not np.isfinite(nu) or nu > NU_MAX:
        return 1.0e15
    logpdf = _student_t_std_logpdf(z_values, nu)
    if not np.isfinite(logpdf).all():
        return 1.0e15
    return float(-np.sum(logpdf))


def _coverage(z_values: NDArray[np.float64], *, k: float, nu: float, central_mass: float) -> float:
    """pure. Empirical central interval coverage after scale-bias correction."""
    tail = (1.0 - central_mass) * 0.5
    lower = _student_t_std_ppf(tail, nu)
    upper = _student_t_std_ppf(1.0 - tail, nu)
    u_values = z_values / k
    return float(np.mean((u_values >= lower) & (u_values <= upper)))


def _fit_scale_bias_student_t(z_values: NDArray[np.float64]) -> ScaleBiasFit:
    """pure. Jointly estimate scalar k and Student-t nu by maximum likelihood."""
    finite_z = np.asarray(z_values[np.isfinite(z_values)], dtype=np.float64)
    if finite_z.size == 0:
        raise ValueError("empty finite z series")
    initial_k = float(np.clip(np.std(finite_z, ddof=0), K_LOWER_BOUND, K_UPPER_BOUND))
    initial = np.array([np.log(initial_k), np.log(8.0 - NU_MIN)], dtype=np.float64)
    result = minimize(
        lambda params: _neg_loglik(params, finite_z),
        initial,
        method="L-BFGS-B",
        bounds=(
            (np.log(K_LOWER_BOUND), np.log(K_UPPER_BOUND)),
            (np.log(2.1 - NU_MIN), np.log(NU_MAX - NU_MIN)),
        ),
    )
    if not result.success:
        raise ValueError(f"scale-bias optimizer failed: {result.message}")

    naive_result = minimize(
        lambda params: _neg_loglik_naive(params, finite_z),
        np.array([np.log(8.0 - NU_MIN)], dtype=np.float64),
        method="L-BFGS-B",
        bounds=((np.log(2.1 - NU_MIN), np.log(NU_MAX - NU_MIN)),),
    )
    if not naive_result.success:
        raise ValueError(f"naive optimizer failed: {naive_result.message}")

    log_k, log_nu_shift = [float(value) for value in result.x]
    k = float(np.exp(log_k))
    nu = float(NU_MIN + np.exp(log_nu_shift))
    naive_nu = float(NU_MIN + np.exp(float(naive_result.x[0])))
    return ScaleBiasFit(
        k=k,
        nu=nu,
        loglik=float(-result.fun),
        naive_nu=naive_nu,
        naive_loglik=float(-naive_result.fun),
        central_80_coverage=_coverage(finite_z, k=k, nu=nu, central_mass=0.80),
        central_90_coverage=_coverage(finite_z, k=k, nu=nu, central_mass=0.90),
    )


def _final_decision(fits: dict[str, ScaleBiasFit]) -> dict[str, Any]:
    """pure. Apply the preregistered Option 1 decision rules."""
    failed_conditions: list[str] = []
    k_values = [fit.k for fit in fits.values()]
    if any(k < K_MIN or k > K_MAX for k in k_values):
        failed_conditions.append("K_OUT_OF_RANGE")
    if max(k_values) - min(k_values) > K_RANGE_MAX:
        failed_conditions.append("K_RANGE_DRIFT")
    improvements = [fit.loglik - fit.naive_loglik for fit in fits.values()]
    if sum(value > 0.0 for value in improvements) < MIN_POSITIVE_WINDOW_IMPROVEMENTS:
        failed_conditions.append("INSUFFICIENT_WINDOW_LOGLIK_IMPROVEMENT")
    if float(np.sum(improvements)) <= 0.0:
        failed_conditions.append("TOTAL_LOGLIK_NOT_IMPROVED")
    if any(fit.central_90_coverage < COVERAGE_90_MIN for fit in fits.values()):
        failed_conditions.append("COVERAGE_COLLAPSE")
    window_2020 = fits["Window_2020"]
    if (
        window_2020.central_90_coverage < COVERAGE_90_MIN
        or window_2020.central_80_coverage < COVERAGE_80_2020_MIN
    ):
        failed_conditions.append("DIRECTION_DEFECT_PROPAGATED")
    return {
        "option_1_decision": "SUCCESS" if not failed_conditions else "FAIL",
        "failed_conditions": failed_conditions,
        "k_min": float(min(k_values)),
        "k_max": float(max(k_values)),
        "k_range": float(max(k_values) - min(k_values)),
        "total_loglik_improvement": float(np.sum(improvements)),
        "positive_loglik_improvement_windows": int(sum(value > 0.0 for value in improvements)),
    }


def _json_default(obj: object) -> object:
    """pure. JSON encoder for numpy scalar values."""
    if isinstance(obj, np.floating | np.integer):
        return obj.item()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def _load_t5_z_by_window() -> dict[str, NDArray[np.float64]]:
    """io: Re-run recovered T5 and extract fixed z_t sequences."""
    from research import t5_recovered_source as t5  # noqa: PLC0415

    context = t5.build_har_context(max(end for _start, end in PILOT_WINDOWS.values()))
    z_by_window: dict[str, NDArray[np.float64]] = {}
    for window_name, (start, end) in PILOT_WINDOWS.items():
        arr = t5._eval_original_t5_window(context, start, end)
        z_by_window[window_name] = np.asarray(arr.z, dtype=np.float64)
    return z_by_window


def _markdown_report(payload: dict[str, Any]) -> str:
    """pure. Render the downstream scale-bias result report."""
    lines = [
        "# Downstream Scale-Bias Results",
        "",
        "> Generated by `src/research/run_downstream_scale_bias_experiment.py`.",
        "",
        "## Diagnostics",
        "",
        "| window | k | nu | loglik_delta | cov80 | cov90 | status |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for window_name, fit in payload["fits"].items():
        delta = fit["loglik"] - fit["naive_loglik"]
        status = "PASS" if K_MIN <= fit["k"] <= K_MAX else "K_OUT_OF_RANGE"
        lines.append(
            "| "
            + " | ".join(
                (
                    window_name,
                    f"{fit['k']:.6f}",
                    f"{fit['nu']:.6f}",
                    f"{delta:.6f}",
                    f"{fit['central_80_coverage']:.6f}",
                    f"{fit['central_90_coverage']:.6f}",
                    status,
                ),
            )
            + " |",
        )
    decision = payload["decision"]
    lines.extend(
        [
            "",
            "## Protocol Decision",
            "",
            f"- Option 1 decision: `{decision['option_1_decision']}`",
            f"- Failed conditions: `{', '.join(decision['failed_conditions']) or 'NONE'}`",
            f"- k range: `{decision['k_range']:.6f}`",
            f"- total log-likelihood improvement: `{decision['total_loglik_improvement']:.6f}`",
            "",
            "## Boundary",
            "",
            "This experiment estimates only scalar downstream scale bias `k` and Student-t `nu`.",
            "It does not modify T5, add a new sigma variant, or reopen Phase 0B.",
        ],
    )
    return "\n".join(lines) + "\n"


def _write_decision_doc(payload: dict[str, Any]) -> None:
    """io: Write the one-page decision document for Option 1."""
    decision = payload["decision"]
    if decision["option_1_decision"] == "SUCCESS":
        conclusion = (
            "Option 1 is allowed to enter full tail family modeling with fixed T5 and "
            "downstream scalar scale-bias correction."
        )
    else:
        conclusion = (
            "Option 1 is not established. T5 scale bias was not accepted as absorbable "
            "by a single stable downstream k under the preregistered rules."
        )
    doc = f"""# Downstream Scale-Bias Final Decision

## Hypothesis

T5 may remain useful for downstream tail modeling if its `std(z)≈2` scale
bias can be absorbed by one stable scalar `k`.

## Experiment Config

- Fixed input: T5 `z_t` sequences for 2017, 2018, and 2020.
- Tail family: variance-one Student-t.
- Estimated parameters per window: scalar `k` and `nu`.
- Baseline: `k = 1` with estimated `nu`.
- Preregistration: `{PREREG_DOC.as_posix()}`

## Result Summary

- Option 1 decision: `{decision["option_1_decision"]}`
- Failed conditions: `{", ".join(decision["failed_conditions"]) or "NONE"}`
- k range: `{decision["k_range"]:.6f}`
- Total log-likelihood improvement: `{decision["total_loglik_improvement"]:.6f}`

## Protocol Decision

`{decision["option_1_decision"]}`

## Allowed Next Step / Termination

{conclusion}

This decision does not reopen trigger audit, crisis architecture, hard switch, override, or MoE.
"""
    (Path("docs/rank_scale_hybrid") / "06_downstream_scale_bias_decision.md").write_text(
        doc,
        encoding="utf-8",
    )


def run(argv: list[str] | None = None) -> int:
    """io: Execute the downstream scale-bias experiment and write artifacts."""
    parser = argparse.ArgumentParser(prog="run_downstream_scale_bias_experiment")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON to stdout.")
    args = parser.parse_args(argv)

    z_by_window = _load_t5_z_by_window()
    fits = {
        window_name: _fit_scale_bias_student_t(z_values)
        for window_name, z_values in z_by_window.items()
    }
    payload: dict[str, Any] = {
        "preregistration": PREREG_DOC.as_posix(),
        "known_2020_direction_risk": "corr_next = -0.033011",
        "fits": {window_name: asdict(fit) for window_name, fit in fits.items()},
        "decision": _final_decision(fits),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "downstream_scale_bias_results.json").write_text(
        json.dumps(payload, default=_json_default, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (OUTPUT_DIR / "downstream_scale_bias_results.md").write_text(
        _markdown_report(payload),
        encoding="utf-8",
    )
    _write_decision_doc(payload)
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
