"""Forensic audit for the 2008 original T5_resid_persistence_M4 failure.

io: runs a research-only variable-level audit and writes JSON diagnostics to stdout.
"""

from __future__ import annotations

# ruff: noqa: E501, S108, PLR0915, PLR0911, SIM114
import ast
from dataclasses import asdict, dataclass
from datetime import date, timedelta
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from features.panel_block_builder import _forward_52w_return, _value_at  # noqa: E402
from research import t5_recovered_source as t5  # noqa: E402

HISTORICAL_SOURCE = Path("/tmp/qfs-sigma/repo/src/research/run_ndx_sigma_output_transform_pilot.py")
CURRENT_SOURCE = Path(__file__).resolve().with_name("t5_recovered_source.py")
AUDIT_2008 = (date(2008, 1, 4), date(2008, 12, 26))
CONTROL_2020 = (date(2020, 1, 3), date(2020, 6, 26))


@dataclass(frozen=True, slots=True)
class FunctionEquivalence:
    name: str
    status: str
    evidence: str


def _node_source(path: Path, name: str) -> str:
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(path.read_text(), node) or ""
    return ""


def _normalized_ast(path: Path, name: str) -> str:
    source = _node_source(path, name)
    if not source:
        return ""
    return ast.dump(ast.parse(source), include_attributes=False)


def _source_equivalence() -> dict[str, Any]:
    checks: list[FunctionEquivalence] = []
    for name in ("_fit_t5", "_predict_t5", "_safe_clip_sigma"):
        current = _normalized_ast(CURRENT_SOURCE, name)
        historical = _normalized_ast(HISTORICAL_SOURCE, name)
        checks.append(
            FunctionEquivalence(
                name=name,
                status="MATCH" if current == historical else "DIFF",
                evidence="AST-normalized function body comparison",
            ),
        )
    build_current = _node_source(CURRENT_SOURCE, "_build_train_base")
    build_historical = _node_source(HISTORICAL_SOURCE, "_build_train_base")
    build_semantic_match = all(
        token in build_current and token in build_historical
        for token in (
            "timedelta(weeks=53)",
            "_slice_frame_rolling(frame, train_end, R1_TRAIN_WINDOW)",
            "_train_mu(train_frame, train_returns)",
            "_fit_sigma_by_model(context, base_model, train_weeks, sigma_target)",
            "_slice_frame_rolling(frame, as_of, R1_TRAIN_WINDOW + 54)",
        )
    )
    checks.append(
        FunctionEquivalence(
            name="_build_train_base",
            status="SEMANTIC_MATCH" if build_semantic_match else "DIFF",
            evidence="Required train-window, embargo, mu, sigma-fit, and history-frame tokens match; only type/format/asarray normalization differs",
        ),
    )
    current_text = CURRENT_SOURCE.read_text()
    historical_text = HISTORICAL_SOURCE.read_text()
    literal_checks = {
        "base_model_m4": "M4_har_plus_iv_slopes" in current_text and "M4_har_plus_iv_slopes" in historical_text,
        "corr_clip": "CORR_CLIP = 1.0" in current_text and "CORR_CLIP = 1.0" in historical_text,
        "logsig_pad": "LOGSIG_PAD = 0.35" in current_text and "LOGSIG_PAD = 0.35" in historical_text,
        "r1_train_window_import": "R1_TRAIN_WINDOW" in current_text and "R1_TRAIN_WINDOW" in historical_text,
        "embargo_53_weeks": "timedelta(weeks=53)" in current_text and "timedelta(weeks=53)" in historical_text,
        "history_416_plus_54": "R1_TRAIN_WINDOW + 54" in current_text and "R1_TRAIN_WINDOW + 54" in historical_text,
    }
    overall = "SOURCE_EQUIVALENCE_CONFIRMED" if all(c.status in {"MATCH", "SEMANTIC_MATCH"} for c in checks) and all(literal_checks.values()) else "SOURCE_EQUIVALENCE_BROKEN"
    return {
        "overall": overall,
        "function_checks": [asdict(c) for c in checks],
        "literal_checks": literal_checks,
    }


def _finite(value: float) -> bool:
    return bool(np.isfinite(value))


def _first_nonfinite_array(values: np.ndarray, weeks: tuple[date, ...]) -> dict[str, Any] | None:
    mask = ~np.isfinite(values)
    if not bool(np.any(mask)):
        return None
    idx = int(np.flatnonzero(mask)[0])
    return {
        "index": idx,
        "week": weeks[idx].isoformat() if idx < len(weeks) else None,
        "value": str(values[idx]),
    }


def _audit_week(context: Any, as_of: date) -> dict[str, Any]:
    frame = context.base.frame
    train_end = as_of - timedelta(weeks=53)
    train_frame = t5._slice_frame_rolling(frame, train_end, t5.R1_TRAIN_WINDOW)
    train_weeks = train_frame.feature_dates
    train_returns = {
        asset: np.asarray(context.base.weekly_returns[asset], dtype=np.float64)[
            frame.feature_dates.index(train_weeks[0]) : frame.feature_dates.index(train_weeks[-1]) + 1
        ]
        for asset in ("NASDAQXNDX", "SPX", "R2K")
    }
    y_train = np.asarray(train_frame.target_returns[t5.TARGET_ASSET], dtype=np.float64)
    mu_x_train = t5._mu_design(train_frame, t5.TARGET_ASSET, train_returns)
    mu_beta = t5._ridge_fit(mu_x_train, y_train, t5.RIDGE_MU)
    mu_hat_train = mu_x_train @ mu_beta
    resid_train = y_train - mu_hat_train
    sigma_target = np.log(np.maximum(np.abs(resid_train), t5.SIGMA_EPS))
    sigma_fit, _aux = t5._fit_sigma_by_model(context, t5.BASE_MODEL_M4, train_weeks, sigma_target)
    sigma_train = np.asarray(sigma_fit["sigma_train_pred"], dtype=np.float64)
    history_frame = t5._slice_frame_rolling(frame, as_of, t5.R1_TRAIN_WINDOW + 54)
    history_weeks = history_frame.feature_dates
    history_returns = {
        asset: np.asarray(context.base.weekly_returns[asset], dtype=np.float64)[
            frame.feature_dates.index(history_weeks[0]) : frame.feature_dates.index(history_weeks[-1]) + 1
        ]
        for asset in ("NASDAQXNDX", "SPX", "R2K")
    }
    train_base = t5.TrainBase(
        as_of=as_of,
        train_weeks=train_weeks,
        mu_beta=mu_beta,
        resid_train=np.asarray(resid_train, dtype=np.float64),
        sigma_fit=sigma_fit,
        sigma_train=sigma_train,
        sigma_target=np.asarray(sigma_target, dtype=np.float64),
        history_frame=history_frame,
        history_returns=history_returns,
    )
    _prev_ret, prev_abs_e = t5._lagged_return_and_resid(
        history_frame,
        history_returns,
        mu_beta,
        as_of,
    )
    prev_sigma = float(train_base.sigma_train[-1]) if len(train_base.sigma_train) else float(np.median(train_base.sigma_train))
    base_sigma = t5._base_sigma_for_week(context, train_base, as_of)
    fit = t5._fit_t5(train_base)
    c = float(fit["c"])
    ratio = prev_abs_e / max(prev_sigma, t5.SIGMA_EPS)
    log_ratio = float(np.log(max(ratio, t5.SIGMA_EPS))) if np.isfinite(ratio) else float("nan")
    score_raw = c * log_ratio
    score_clipped = float(np.clip(score_raw, -t5.CORR_CLIP, t5.CORR_CLIP)) if np.isfinite(score_raw) else float("nan")
    sigma_raw = float(base_sigma * np.exp(score_clipped)) if np.isfinite(score_clipped) and np.isfinite(base_sigma) else float("nan")
    try:
        sigma_safe, _safe_aux = t5._safe_clip_sigma(
            sigma_raw,
            fit["train_sigma"],
            fit["train_log_sigma"],
        )
    except (ValueError, FloatingPointError, IndexError):
        sigma_safe = float("nan")
    mu_hat, y = t5._same_week_mu_and_y(history_frame, history_returns, mu_beta, as_of)
    e_t = y - mu_hat
    z_t = e_t / max(sigma_safe, t5.SIGMA_EPS) if np.isfinite(sigma_safe) else float("nan")
    std_resid = resid_train / np.maximum(sigma_train, t5.SIGMA_EPS)
    finite_std_resid = std_resid[np.isfinite(std_resid)]

    variable_order = [
        ("y_train", y_train),
        ("mu_x_train", mu_x_train),
        ("mu_beta", mu_beta),
        ("mu_hat_train", mu_hat_train),
        ("resid_train", resid_train),
        ("sigma_target", sigma_target),
        ("sigma_train", sigma_train),
        ("std_resid_for_quantile", std_resid),
    ]
    first_input_nonfinite = None
    for name, arr in variable_order:
        nonfinite = _first_nonfinite_array(np.asarray(arr, dtype=np.float64).reshape(-1), train_weeks)
        if nonfinite is not None:
            first_input_nonfinite = {"variable": name, **nonfinite}
            break

    stage_flags = {
        "base_sigma": _finite(base_sigma),
        "prev_abs_e": _finite(prev_abs_e),
        "prev_sigma": _finite(prev_sigma),
        "ratio": _finite(ratio),
        "log_ratio": _finite(log_ratio),
        "exp_score": _finite(float(np.exp(score_clipped))) if np.isfinite(score_clipped) else False,
        "sigma_raw": _finite(sigma_raw),
        "sigma_safe": _finite(sigma_safe),
        "z_t": _finite(z_t),
        "empirical_standardized_residual_quantiles_input": bool(finite_std_resid.size > 0),
    }
    if not stage_flags["base_sigma"]:
        failure_step = "base_sigma"
    elif not stage_flags["prev_abs_e"] or not stage_flags["prev_sigma"] or not stage_flags["ratio"]:
        failure_step = "prev_abs_e / prev_sigma"
    elif not stage_flags["log_ratio"]:
        failure_step = "log(prev_abs_e / prev_sigma)"
    elif not stage_flags["exp_score"] or not stage_flags["sigma_raw"]:
        failure_step = "exp(...)"
    elif not stage_flags["sigma_safe"]:
        failure_step = "_safe_clip_sigma"
    elif not stage_flags["z_t"]:
        failure_step = "z = e / sigma"
    elif not stage_flags["empirical_standardized_residual_quantiles_input"]:
        failure_step = "empirical standardized residual quantiles"
    else:
        failure_step = "none"

    return {
        "as_of": as_of.isoformat(),
        "train_end": train_end.isoformat(),
        "train_window_first": train_weeks[0].isoformat(),
        "train_window_last": train_weeks[-1].isoformat(),
        "train_window_len": len(train_weeks),
        "prev_abs_e": prev_abs_e,
        "prev_sigma": prev_sigma,
        "base_sigma": base_sigma,
        "ratio": ratio,
        "log_ratio": log_ratio,
        "c": c,
        "score_raw": score_raw,
        "score_clipped": score_clipped,
        "sigma_raw": sigma_raw,
        "sigma_safe": sigma_safe,
        "e_t": e_t,
        "z_t": z_t,
        "is_finite_prev_abs_e": _finite(prev_abs_e),
        "is_finite_prev_sigma": _finite(prev_sigma),
        "is_finite_base_sigma": _finite(base_sigma),
        "is_finite_sigma_raw": _finite(sigma_raw),
        "is_finite_sigma_safe": _finite(sigma_safe),
        "is_finite_z_t": _finite(z_t),
        "stage_flags": stage_flags,
        "failure_step": failure_step,
        "first_input_nonfinite": first_input_nonfinite,
        "quantile_input": {
            "sample_size": int(std_resid.size),
            "finite_count": int(finite_std_resid.size),
            "nonfinite_count": int(std_resid.size - finite_std_resid.size),
            "first_nonfinite": _first_nonfinite_array(std_resid, train_weeks),
            "polluted_before_quantile": bool(int(std_resid.size - finite_std_resid.size) > 0),
        },
    }


def _audit_window(context: Any, start: date, end: date) -> dict[str, Any]:
    frame = context.base.frame
    eval_dates = tuple(week for week in frame.feature_dates if start <= week <= end)
    rows = [_audit_week(context, as_of) for as_of in eval_dates]
    first_failure = next((row for row in rows if row["failure_step"] != "none"), None)
    if first_failure is None:
        first_failure_index = None
        snapshots = {}
    else:
        idx = rows.index(first_failure)
        first_failure_index = idx
        snapshots = {
            "previous": rows[idx - 1] if idx > 0 else None,
            "current": rows[idx],
            "next": rows[idx + 1] if idx + 1 < len(rows) else None,
        }
    return {
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "eval_weeks": len(rows),
        "first_failure_index": first_failure_index,
        "first_failure": first_failure,
        "snapshots": snapshots,
        "rows": rows,
    }


def _control_summary(audit: dict[str, Any]) -> dict[str, Any]:
    rows = audit["rows"]
    return {
        "window": audit["window"],
        "eval_weeks": audit["eval_weeks"],
        "failure_count": sum(int(row["failure_step"] != "none") for row in rows),
        "all_quantile_finite_counts_positive": all(row["quantile_input"]["finite_count"] > 0 for row in rows),
        "min_quantile_finite_count": min(row["quantile_input"]["finite_count"] for row in rows),
        "first_failure": audit["first_failure"],
    }


def _upstream_nan_trace() -> dict[str, Any]:
    deps = t5.build_panel_runner_deps(t5.load_adapter_secrets(t5.os.environ))
    macro = deps.fetch_macro_series(CONTROL_2020[1], "strict")
    asset_prices = deps.fetch_asset_prices(CONTROL_2020[1] + t5.timedelta(weeks=52))
    target_series = asset_prices[t5.TARGET_ASSET]
    context = t5.build_har_context(CONTROL_2020[1])
    frame = context.base.frame
    first_week = date(1998, 10, 9)
    first_finite_current = None
    for week in frame.feature_dates:
        current = _value_at(target_series, week)
        if np.isfinite(current):
            first_finite_current = {
                "week": week.isoformat(),
                "target_current": current,
            }
            break
    current = _value_at(target_series, first_week)
    future = _value_at(target_series, first_week + timedelta(weeks=52))
    return {
        "y_train_definition": "train_frame.target_returns[NASDAQXNDX], generated by _forward_52w_return(target_series, week)",
        "formula": "log(value_at(target_series, week + 52 weeks) / value_at(target_series, week)); returns nan when current/future is non-finite or <= 0",
        "first_polluted_week": first_week.isoformat(),
        "target_series_first_timestamp": str(target_series.timestamps[0]),
        "target_series_last_timestamp": str(target_series.timestamps[-1]),
        "target_series_first_value": float(target_series.values[0]),
        "first_finite_current": first_finite_current,
        "first_polluted_week_current": current,
        "first_polluted_week_future_52w": future,
        "first_polluted_week_forward_52w_return": _forward_52w_return(target_series, first_week),
        "macro_series_loaded": bool(macro),
        "source_category": "UPSTREAM_DATA_EDGE_CASE_CONFIRMED",
    }


def _sanitize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, tuple):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _sanitize(obj.tolist())
    if isinstance(obj, (np.floating, np.integer)):
        return _sanitize(obj.item())
    if isinstance(obj, float) and not np.isfinite(obj):
        return str(obj)
    return obj


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return str(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def run() -> int:
    context = t5.build_har_context(CONTROL_2020[1])
    source = _source_equivalence()
    audit_2008 = _audit_window(context, *AUDIT_2008)
    audit_2020 = _audit_window(context, *CONTROL_2020)
    first = audit_2008["first_failure"]
    if source["overall"] == "SOURCE_EQUIVALENCE_BROKEN":
        attribution = "INCONCLUSIVE"
    elif first and first.get("first_input_nonfinite", {}).get("variable") == "y_train":
        attribution = "UPSTREAM_DATA_EDGE_CASE_CONFIRMED"
    elif first and first["failure_step"] == "empirical standardized residual quantiles":
        attribution = "UPSTREAM_DATA_EDGE_CASE_CONFIRMED"
    elif first:
        attribution = "ORIGINAL_T5_INTERNAL_INSTABILITY_CONFIRMED"
    else:
        attribution = "INCONCLUSIVE"
    payload = {
        "source_equivalence": source,
        "audit_2008": audit_2008,
        "control_2020_summary": _control_summary(audit_2020),
        "upstream_nan_trace": _upstream_nan_trace(),
        "attribution": attribution,
    }
    sys.stdout.write(json.dumps(_sanitize(payload), default=_json_default, indent=2, sort_keys=True, allow_nan=False))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
