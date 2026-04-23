"""Imperative shell runner for the v8.8 panel challenger."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, timedelta
from functools import partial
import os
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from app.challenger_artifacts import write_challenger_report
from backtest.cluster_block_bootstrap import bootstrap_week_statistic_p05
from backtest.panel_acceptance import (
    PanelAcceptanceThresholds,
    PanelAssetMetrics,
    evaluate_panel_acceptance,
    panel_acceptance_report_to_dict,
)
from backtest.panel_metrics import (
    compute_panel_effective_start,
    effective_asset_weeks,
    per_asset_coverage,
    per_asset_crps,
    vol_normalized_crps,
)
from data_contract.asset_registry import PANEL_REGISTRY
from data_contract.vintage_registry import STRICT_PIT_STARTS
from engine_types import Stance, TimeSeries, VintageMode
from errors import HMMConvergenceError, QuantileSolverError
from features.panel_block_builder import (
    PanelFeatureFrame,
    PanelHMMInputs,
    build_panel_feature_block,
    build_panel_hmm_inputs,
    fit_panel_hmm,
)
from law.panel_quantiles import fit_panel_quantiles, predict_panel_interior_with_status
from law.quantile_moments import moments_from_quantiles
from law.tail_extrapolation import extrapolate_tails
from state.state_label_map import label_map_json_bytes
from state.ti_hmm_single import (
    HMMModel,
    degraded_hmm_posterior,
    infer_hmm,
    infer_hmm_posterior_path,
)

EXIT_OK = 0
EXIT_ACCEPTANCE_FAILED = 3
FULL_TAUS = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
WARMUP_FALLBACK_MIN_HISTORY = 2
TRAILING_COVERAGE_WEEKS = 104
DEFAULT_BACKTEST_END = date(2024, 12, 27)
PANEL_ROOT_NAME = "panel_challenger"
SMOKE_START = date(2016, 1, 1)
SMOKE_END = date(2016, 12, 30)
PIT_CLASSIFICATION_LABELS = {
    "log_return_pit": "log-return-PIT",
    "pseudo_pit_risk": "pseudo-PIT-risk",
}

FetchMacroSeries = Callable[[date, VintageMode], Mapping[str, TimeSeries]]
FetchAssetPrices = Callable[[date], Mapping[str, TimeSeries]]
WriteJson = Callable[[Path, dict[str, object]], None]


@dataclass(frozen=True, slots=True)
class PanelRunnerDeps:
    """io: Shell dependencies for the panel challenger."""

    fetch_macro_series: FetchMacroSeries
    fetch_asset_prices: FetchAssetPrices
    write_json: WriteJson = write_challenger_report


@dataclass(frozen=True, slots=True)
class PanelAssetWeekResult:
    """pure. One fitted asset block for a single panel week."""

    available: bool
    micro_feature_mode: str
    realized: float | None
    full_quantiles: NDArray[np.float64] | None
    solver_status: str | None
    tail_status: str | None
    moments: Mapping[str, float] | None


@dataclass(frozen=True, slots=True)
class PanelWeekResult:
    """pure. One fitted panel week before sequential baseline assembly."""

    as_of: date
    common_post: NDArray[np.float64]
    state_name: str
    dwell_weeks: int
    hazard_covariate: float
    panel_solver_status: str
    hmm_status: str
    label_map: Mapping[int, Stance] | None
    asset_results: Mapping[str, PanelAssetWeekResult]


def _panel_root(artifacts_root: Path) -> Path:
    return artifacts_root / PANEL_ROOT_NAME


def _panel_week_path(artifacts_root: Path, as_of: date) -> Path:
    return _panel_root(artifacts_root) / f"as_of={as_of.isoformat()}" / "panel_output.json"


def _slice_frame(frame: PanelFeatureFrame, end: date) -> PanelFeatureFrame:
    indices = [idx for idx, week in enumerate(frame.feature_dates) if week <= end]
    if not indices:
        msg = "panel frame slice requires at least one row"
        raise ValueError(msg)
    last = indices[-1] + 1
    feature_dates = frame.feature_dates[:last]
    x_micro_raw = {asset_id: values[:last].copy() for asset_id, values in frame.x_micro_raw.items()}
    x_micro = {asset_id: values[:last].copy() for asset_id, values in frame.x_micro.items()}
    micro_mask = {asset_id: values[:last].copy() for asset_id, values in frame.micro_mask.items()}
    availability = {
        asset_id: values[:last].copy() for asset_id, values in frame.asset_availability.items()
    }
    target_returns = {
        asset_id: values[:last].copy() for asset_id, values in frame.target_returns.items()
    }
    micro_modes = {
        asset_id: modes[:last] for asset_id, modes in frame.micro_feature_mode.items()
    }
    available_assets = tuple(
        asset_id
        for asset_id, mask in availability.items()
        if mask.shape[0] > 0 and bool(mask[-1])
    )
    return PanelFeatureFrame(
        as_of=feature_dates[-1],
        feature_dates=feature_dates,
        x_macro_raw=frame.x_macro_raw[:last].copy(),
        x_macro=frame.x_macro[:last].copy(),
        x_micro_raw=x_micro_raw,
        x_micro=x_micro,
        macro_mask=frame.macro_mask[:last].copy(),
        micro_mask=micro_mask,
        asset_availability=availability,
        micro_feature_mode=micro_modes,
        available_assets=available_assets,
        target_returns=target_returns,
    )


def _row_index(frame: PanelFeatureFrame, as_of: date) -> int:
    return frame.feature_dates.index(as_of)


def _build_asset_inputs(
    macro_series: Mapping[str, TimeSeries],
    asset_prices: Mapping[str, TimeSeries],
) -> dict[str, dict[str, TimeSeries]]:
    asset_inputs: dict[str, dict[str, TimeSeries]] = {}
    for asset_id, spec in PANEL_REGISTRY.items():
        payload = {
            "target": asset_prices[asset_id],
            "vol": macro_series[spec.vol_series_id],
        }
        if spec.vol_fallback_id is not None:
            payload["vol_fallback"] = macro_series[spec.vol_fallback_id]
        asset_inputs[asset_id] = payload
    return asset_inputs


def _config_int(panel_config: Mapping[str, object], key: str) -> int:
    return int(cast(Any, panel_config[key]))


def _config_float(panel_config: Mapping[str, object], key: str) -> float:
    return float(cast(Any, panel_config[key]))


def _config_str(panel_config: Mapping[str, object], key: str) -> str:
    return str(cast(Any, panel_config[key]))


def _config_block_lengths(panel_config: Mapping[str, object]) -> tuple[int, ...]:
    raw = cast(list[object], panel_config["block_lengths"])
    return tuple(int(cast(Any, value)) for value in raw)


def _baseline_a_quantiles(history: NDArray[np.float64], fallback: float) -> NDArray[np.float64]:
    finite = history[np.isfinite(history)]
    if finite.shape[0] < WARMUP_FALLBACK_MIN_HISTORY:
        return np.full(len(FULL_TAUS), fallback, dtype=np.float64)
    return np.quantile(finite, np.asarray(FULL_TAUS, dtype=np.float64)).astype(np.float64)


def _quantile_score(y_true: float, quantiles: NDArray[np.float64]) -> float:
    taus = np.asarray(FULL_TAUS, dtype=np.float64)
    losses = np.maximum(taus * (y_true - quantiles), (taus - 1.0) * (y_true - quantiles))
    return float(2.0 * np.mean(losses))


def _panel_improvement_from_week_matrix(values: NDArray[np.float64]) -> float:
    production = values[:, : len(PANEL_REGISTRY)]
    baseline = values[:, len(PANEL_REGISTRY) :]
    panel_prod = _row_mean_ignoring_nan(production)
    panel_base = _row_mean_ignoring_nan(baseline)
    finite = np.isfinite(panel_prod) & np.isfinite(panel_base) & (panel_base > 0.0)
    if not finite.any():
        return float("nan")
    return float(1.0 - np.mean(panel_prod[finite]) / np.mean(panel_base[finite]))


def _row_mean_ignoring_nan(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    out = np.full(matrix.shape[0], np.nan, dtype=np.float64)
    for idx in range(matrix.shape[0]):
        finite = matrix[idx][np.isfinite(matrix[idx])]
        if finite.size > 0:
            out[idx] = float(np.mean(finite))
    return out


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _weekly_panel_improvement(
    model_row: NDArray[np.float64],
    baseline_row: NDArray[np.float64],
) -> float:
    finite = np.isfinite(model_row) & np.isfinite(baseline_row) & (baseline_row > 0.0)
    if not finite.any():
        return 0.0
    return float(1.0 - np.mean(model_row[finite]) / np.mean(baseline_row[finite]))


def _trailing_coverage(
    quantiles_history: list[NDArray[np.float64]],
    realized_history: list[float],
) -> tuple[float, float]:
    if not quantiles_history or not realized_history:
        return 0.0, 0.0
    quantiles = np.vstack(quantiles_history[-TRAILING_COVERAGE_WEEKS:]).astype(np.float64)
    realized = np.asarray(realized_history[-TRAILING_COVERAGE_WEEKS:], dtype=np.float64)
    return (
        per_asset_coverage(quantiles, realized, 0.10),
        per_asset_coverage(quantiles, realized, 0.90),
    )


def _payload_is_finite(value: object) -> bool:
    if value is None or isinstance(value, str | bool | int):
        return True
    if isinstance(value, float):
        return bool(np.isfinite(value))
    if isinstance(value, list):
        return all(_payload_is_finite(item) for item in value)
    if isinstance(value, Mapping):
        return all(_payload_is_finite(item) for item in value.values())
    return True


def _mode_fraction(count: int, total: int) -> float | None:
    if total <= 0:
        return None
    return float(count / total)


def _common_state_payload(
    posterior: NDArray[np.float64],
    *,
    state_name: str,
    dwell_weeks: int,
    hazard_covariate: float,
) -> dict[str, object]:
    return {
        "post": [float(value) for value in posterior.tolist()],
        "state_name": state_name,
        "dwell_weeks": dwell_weeks,
        "hazard_covariate": hazard_covariate,
        "label_anchor": "SPX",
    }


def _fit_panel_training_state(
    train_frame: PanelFeatureFrame,
    macro_series: Mapping[str, TimeSeries],
    rng: np.random.Generator,
) -> tuple[HMMModel | None, PanelHMMInputs | None, NDArray[np.float64], str]:
    try:
        hmm_model = fit_panel_hmm(train_frame, macro_series, rng)
        hmm_inputs = build_panel_hmm_inputs(train_frame, macro_series)
        posterior_path = infer_hmm_posterior_path(hmm_model, hmm_inputs.y_obs, hmm_inputs.h)
        return hmm_model, hmm_inputs, posterior_path, "ok"
    except HMMConvergenceError:
        try:
            hmm_inputs = build_panel_hmm_inputs(train_frame, macro_series)
        except HMMConvergenceError:
            return None, None, np.empty((0, 3), dtype=np.float64), "degenerate"
        degraded = np.tile(degraded_hmm_posterior().post, (hmm_inputs.y_obs.shape[0], 1))
        return None, hmm_inputs, degraded, "degenerate"


def _current_hmm_state(
    history_frame: PanelFeatureFrame,
    macro_series: Mapping[str, TimeSeries],
    hmm_model: HMMModel | None,
) -> tuple[NDArray[np.float64], str, int, float]:
    if hmm_model is None:
        degraded = degraded_hmm_posterior()
        return degraded.post, degraded.state_name, 0, 0.0
    try:
        inputs = build_panel_hmm_inputs(history_frame, macro_series)
        result = infer_hmm(hmm_model, inputs.y_obs, inputs.h)
        return (
            result.posterior.post,
            result.posterior.state_name,
            result.dwell_weeks,
            result.hazard_covariate,
        )
    except HMMConvergenceError:
        degraded = degraded_hmm_posterior()
        return degraded.post, degraded.state_name, 0, 0.0


def _write_panel_label_map(
    artifacts_root: Path,
    label_map: Mapping[int, Stance],
) -> None:
    path = _panel_root(artifacts_root) / "panel_state_label_map.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(label_map_json_bytes(label_map))


def _panel_worker_count(total_weeks: int) -> int:
    if total_weeks <= 1:
        return 1
    configured = os.environ.get("PANEL_MAX_WORKERS", os.environ.get("BACKTEST_MAX_WORKERS", "0"))
    if configured:
        try:
            parsed = int(configured)
        except ValueError:
            parsed = 0
        if parsed > 0:
            return min(total_weeks, parsed)
    return 1


def _evaluate_panel_week(
    as_of: date,
    *,
    panel_frame: PanelFeatureFrame,
    macro_series: Mapping[str, TimeSeries],
    panel_config: Mapping[str, object],
) -> PanelWeekResult:
    train_frame = _slice_frame(
        panel_frame,
        as_of - timedelta(weeks=_config_int(panel_config, "embargo_weeks")),
    )
    history_frame = _slice_frame(panel_frame, as_of)
    rng = np.random.default_rng(8675309 + as_of.toordinal())
    hmm_model, hmm_inputs, posterior_path, hmm_status = _fit_panel_training_state(
        train_frame,
        macro_series,
        rng,
    )
    label_map = hmm_model.label_map if hmm_model is not None else None
    if hmm_inputs is None or posterior_path.shape[0] == 0:
        common_post, state_name, dwell_weeks, hazard_covariate = _current_hmm_state(
            history_frame,
            macro_series,
            None,
        )
        qr_coefs = None
        panel_solver_status = "per_asset_fallback"
    else:
        train_obs_rows = [
            _row_index(train_frame, obs_date) for obs_date in hmm_inputs.observation_dates
        ]
        train_pi = posterior_path
        try:
            qr_coefs = fit_panel_quantiles(
                train_frame.x_macro[np.asarray(train_obs_rows, dtype=np.int64)],
                {
                    asset_id: train_frame.x_micro[asset_id][
                        np.asarray(train_obs_rows, dtype=np.int64)
                    ]
                    for asset_id in PANEL_REGISTRY
                },
                train_pi,
                {
                    asset_id: train_frame.target_returns[asset_id][
                        np.asarray(train_obs_rows, dtype=np.int64)
                    ]
                    for asset_id in PANEL_REGISTRY
                },
                {
                    asset_id: train_frame.asset_availability[asset_id][
                        np.asarray(train_obs_rows, dtype=np.int64)
                    ]
                    for asset_id in PANEL_REGISTRY
                },
                l2_alpha_macro=_config_float(panel_config, "l2_alpha_macro"),
                l2_alpha_micro=_config_float(panel_config, "l2_alpha_micro"),
                min_gap=_config_float(panel_config, "min_gap"),
            )
            panel_solver_status = qr_coefs.solver_status
        except QuantileSolverError:
            qr_coefs = None
            panel_solver_status = "per_asset_fallback"
        common_post, state_name, dwell_weeks, hazard_covariate = _current_hmm_state(
            history_frame,
            macro_series,
            hmm_model,
        )

    row_idx = _row_index(history_frame, as_of)
    asset_results: dict[str, PanelAssetWeekResult] = {}
    for asset_id in PANEL_REGISTRY:
        mode = history_frame.micro_feature_mode[asset_id][row_idx]
        if qr_coefs is None or not bool(history_frame.asset_availability[asset_id][row_idx]):
            asset_results[asset_id] = PanelAssetWeekResult(
                available=False,
                micro_feature_mode=mode,
                realized=None,
                full_quantiles=None,
                solver_status=None,
                tail_status=None,
                moments=None,
            )
            continue
        interior, solver_status = predict_panel_interior_with_status(
            qr_coefs,
            asset_id,
            history_frame.x_macro[row_idx],
            history_frame.x_micro[asset_id][row_idx],
            common_post,
            min_gap=_config_float(panel_config, "min_gap"),
        )
        full_quantiles, tail_status = extrapolate_tails(
            interior,
            mult=_config_float(panel_config, "tail_mult"),
        )
        realized = float(history_frame.target_returns[asset_id][row_idx])
        moments = {
            key: float(value)
            for key, value in moments_from_quantiles(FULL_TAUS, full_quantiles).items()
        }
        asset_results[asset_id] = PanelAssetWeekResult(
            available=True,
            micro_feature_mode=mode,
            realized=realized,
            full_quantiles=full_quantiles,
            solver_status=solver_status,
            tail_status=tail_status,
            moments=moments,
        )

    return PanelWeekResult(
        as_of=as_of,
        common_post=common_post,
        state_name=state_name,
        dwell_weeks=dwell_weeks,
        hazard_covariate=hazard_covariate,
        panel_solver_status=panel_solver_status,
        hmm_status=hmm_status,
        label_map=label_map,
        asset_results=asset_results,
    )


def run_panel_backtest_job(  # noqa: PLR0912, PLR0915
    *,
    start: date | None,
    end: date,
    panel_config: Mapping[str, object],
    artifacts_root: Path,
    deps: PanelRunnerDeps,
) -> int:
    """io: Run the v8.8 fixed-panel challenger backtest or smoke stage."""
    vintage_mode: VintageMode = "strict"
    macro_series = deps.fetch_macro_series(end, vintage_mode)
    asset_prices = deps.fetch_asset_prices(end + timedelta(weeks=52))
    asset_inputs = _build_asset_inputs(macro_series, asset_prices)
    panel_frame = build_panel_feature_block(macro_series, asset_inputs, end)
    effective_start = compute_panel_effective_start(
        PANEL_REGISTRY,
        STRICT_PIT_STARTS,
        min_training_weeks=_config_int(panel_config, "min_train"),
        embargo_weeks=_config_int(panel_config, "embargo_weeks"),
    )
    run_start = effective_start if start is None else max(start, effective_start)
    eval_dates = tuple(week for week in panel_frame.feature_dates if run_start <= week <= end)
    panel_root = _panel_root(artifacts_root)
    panel_root.mkdir(parents=True, exist_ok=True)

    model_week_crps = np.full((len(eval_dates), len(PANEL_REGISTRY)), np.nan, dtype=np.float64)
    baseline_week_crps = np.full((len(eval_dates), len(PANEL_REGISTRY)), np.nan, dtype=np.float64)
    quantiles_by_asset: dict[str, list[NDArray[np.float64]]] = {
        asset_id: [] for asset_id in PANEL_REGISTRY
    }
    baseline_by_asset: dict[str, list[NDArray[np.float64]]] = {
        asset_id: [] for asset_id in PANEL_REGISTRY
    }
    realized_by_asset: dict[str, list[float]] = {asset_id: [] for asset_id in PANEL_REGISTRY}
    micro_mode_counts: dict[str, dict[str, int]] = {
        asset_id: {"primary": 0, "proxy": 0, "rv_only": 0} for asset_id in PANEL_REGISTRY
    }
    blocked_weeks = 0
    solver_failure_weeks = 0
    all_output_finite = True
    last_label_map: Mapping[int, Stance] | None = None

    worker = partial(
        _evaluate_panel_week,
        panel_frame=panel_frame,
        macro_series=macro_series,
        panel_config=panel_config,
    )
    worker_count = _panel_worker_count(len(eval_dates))
    if worker_count == 1:
        week_results = [worker(as_of) for as_of in eval_dates]
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            week_results = list(executor.map(worker, eval_dates))

    for week_index, week_result in enumerate(week_results):
        if week_result.label_map is not None:
            last_label_map = week_result.label_map
        asset_payloads: dict[str, object] = {}
        missing_assets: list[str] = []
        all_asset_quantiles: dict[str, NDArray[np.float64]] = {}
        week_mode = (
            "DEGRADED"
            if week_result.panel_solver_status == "per_asset_fallback"
            or week_result.hmm_status != "ok"
            else "NORMAL"
        )
        if week_result.panel_solver_status != "ok" or week_result.hmm_status != "ok":
            solver_failure_weeks += 1
        for asset_position, asset_id in enumerate(PANEL_REGISTRY):
            asset_result = week_result.asset_results[asset_id]
            if not asset_result.available:
                asset_payloads[asset_id] = {
                    "available": False,
                    "micro_feature_mode": asset_result.micro_feature_mode,
                    "distribution": None,
                    "diagnostics": None,
                }
                missing_assets.append(asset_id)
                continue
            if (
                asset_result.realized is None
                or asset_result.full_quantiles is None
                or asset_result.solver_status is None
                or asset_result.tail_status is None
                or asset_result.moments is None
            ):
                msg = f"available asset result is incomplete for {asset_id}"
                raise ValueError(msg)
            prior = np.asarray(realized_by_asset[asset_id], dtype=np.float64)
            baseline_q = _baseline_a_quantiles(prior, asset_result.realized)
            quantiles_by_asset[asset_id].append(asset_result.full_quantiles)
            baseline_by_asset[asset_id].append(baseline_q)
            realized_by_asset[asset_id].append(asset_result.realized)
            micro_mode_counts[asset_id][asset_result.micro_feature_mode] += 1
            trailing_q10, trailing_q90 = _trailing_coverage(
                quantiles_by_asset[asset_id],
                realized_by_asset[asset_id],
            )
            model_week_crps[week_index, asset_position] = _quantile_score(
                asset_result.realized,
                asset_result.full_quantiles,
            )
            baseline_week_crps[week_index, asset_position] = _quantile_score(
                asset_result.realized,
                baseline_q,
            )
            all_asset_quantiles[asset_id] = asset_result.full_quantiles
            asset_payloads[asset_id] = {
                "available": True,
                "micro_feature_mode": asset_result.micro_feature_mode,
                "distribution": {
                    "q05": float(asset_result.full_quantiles[0]),
                    "q10": float(asset_result.full_quantiles[1]),
                    "q25": float(asset_result.full_quantiles[2]),
                    "q50": float(asset_result.full_quantiles[3]),
                    "q75": float(asset_result.full_quantiles[4]),
                    "q90": float(asset_result.full_quantiles[5]),
                    "q95": float(asset_result.full_quantiles[6]),
                    "mu_hat": asset_result.moments["mu_hat"],
                    "sigma_hat": asset_result.moments["sigma_hat"],
                    "p_loss": asset_result.moments["p_loss"],
                    "es20": asset_result.moments["es20"],
                },
                "diagnostics": {
                    "solver_status": asset_result.solver_status,
                    "tail_status": asset_result.tail_status,
                    "coverage_q10_trailing_104w": trailing_q10,
                    "coverage_q90_trailing_104w": trailing_q90,
                },
            }
        if not all_asset_quantiles:
            week_mode = "BLOCKED"
            blocked_weeks += 1
        weekly_panel_improvement = _weekly_panel_improvement(
            model_week_crps[week_index],
            baseline_week_crps[week_index],
        )
        payload = {
            "as_of_date": week_result.as_of.isoformat(),
            "srd_version": "8.8.0",
            "panel_size": _config_int(panel_config, "panel_size"),
            "available_assets": list(all_asset_quantiles),
            "pit_classification": PIT_CLASSIFICATION_LABELS.get(
                _config_str(panel_config, "pit_classification"),
                _config_str(panel_config, "pit_classification"),
            ),
            "common": {
                "mode": week_mode,
                "vintage_mode": vintage_mode,
                "state": _common_state_payload(
                    week_result.common_post,
                    state_name=week_result.state_name,
                    dwell_weeks=week_result.dwell_weeks,
                    hazard_covariate=week_result.hazard_covariate,
                ),
            },
            "assets": asset_payloads,
            "panel_diagnostics": {
                "panel_solver_status": week_result.panel_solver_status,
                "panel_crps_vs_baseline_a": weekly_panel_improvement,
                "effective_asset_weeks": len(all_asset_quantiles),
                "missing_assets_this_week": missing_assets,
            },
        }
        all_output_finite = all_output_finite and _payload_is_finite(payload)
        deps.write_json(_panel_week_path(artifacts_root, week_result.as_of), payload)

    if last_label_map is not None:
        _write_panel_label_map(artifacts_root, last_label_map)
    combined = np.hstack([model_week_crps, baseline_week_crps])
    bootstrap_p05 = {
        int(block_length): bootstrap_week_statistic_p05(
            combined,
            statistic=_panel_improvement_from_week_matrix,
            block_length=int(block_length),
            replications=_config_int(panel_config, "B"),
            rng=np.random.default_rng(4242 + int(block_length)),
        )
        for block_length in _config_block_lengths(panel_config)
    }
    per_asset_metrics: dict[str, PanelAssetMetrics] = {}
    for asset_id in PANEL_REGISTRY:
        if not quantiles_by_asset[asset_id]:
            continue
        model_q = np.vstack(quantiles_by_asset[asset_id]).astype(np.float64)
        baseline_q = np.vstack(baseline_by_asset[asset_id]).astype(np.float64)
        realized_array = np.asarray(realized_by_asset[asset_id], dtype=np.float64)
        crps = per_asset_crps(model_q, realized_array, asset_id)
        baseline_crps = per_asset_crps(baseline_q, realized_array, asset_id)
        sigma = (
            float(np.nanstd(realized_array, ddof=1))
            if realized_array.shape[0] > 1
            else float("nan")
        )
        per_asset_metrics[asset_id] = PanelAssetMetrics(
            q10_coverage=per_asset_coverage(model_q, realized_array, 0.10),
            q90_coverage=per_asset_coverage(model_q, realized_array, 0.90),
            crps=crps,
            baseline_a_crps=baseline_crps,
            effective_weeks=int(model_q.shape[0]),
            vol_normalized_crps=(
                vol_normalized_crps(crps, sigma)
                if np.isfinite(sigma) and sigma > 0.0
                else None
            ),
        )
    thresholds = PanelAcceptanceThresholds(
        coverage_tolerance=_config_float(panel_config, "coverage_tol"),
        coverage_collapse_limit=_config_float(panel_config, "coverage_collapse"),
        crps_improvement_min=_config_float(panel_config, "crps_min_improve"),
        blocked_cap=_config_float(panel_config, "blocked_cap"),
        block_lengths=_config_block_lengths(panel_config),
    )
    blocked_proportion = float(blocked_weeks / len(eval_dates)) if eval_dates else 1.0
    acceptance = evaluate_panel_acceptance(
        per_asset_metrics,
        bootstrap_p05_by_block=bootstrap_p05,
        blocked_proportion=blocked_proportion,
        all_finite=all_output_finite,
        thresholds=thresholds,
    )
    aggregate_crps = (
        float(np.mean([metrics.crps for metrics in per_asset_metrics.values()]))
        if per_asset_metrics
        else float("nan")
    )
    aggregate_baseline_crps = (
        float(np.mean([metrics.baseline_a_crps for metrics in per_asset_metrics.values()]))
        if per_asset_metrics
        else float("nan")
    )
    aggregate_vol_normalized = (
        float(
            np.mean(
                [
                    metrics.vol_normalized_crps
                    for metrics in per_asset_metrics.values()
                    if metrics.vol_normalized_crps is not None
                ],
            ),
        )
        if any(metrics.vol_normalized_crps is not None for metrics in per_asset_metrics.values())
        else None
    )
    overall_mode_totals = {"primary": 0, "proxy": 0, "rv_only": 0}
    for counts in micro_mode_counts.values():
        for mode, count in counts.items():
            overall_mode_totals[mode] += count
    overall_mode_total = sum(overall_mode_totals.values())
    report = {
        "srd_version": "8.8.0",
        "backtest_start": run_start.isoformat(),
        "backtest_end": end.isoformat(),
        "panel_effective_start": effective_start.isoformat(),
        "acceptance": panel_acceptance_report_to_dict(acceptance),
        "per_asset_metrics": {
            asset_id: {
                "q10_coverage": metrics.q10_coverage,
                "q90_coverage": metrics.q90_coverage,
                "crps": metrics.crps,
                "baseline_a_crps": metrics.baseline_a_crps,
                "crps_improvement": (
                    float(1.0 - metrics.crps / metrics.baseline_a_crps)
                    if metrics.baseline_a_crps > 0.0
                    else None
                ),
                "effective_weeks": metrics.effective_weeks,
                "vol_normalized_crps": metrics.vol_normalized_crps,
                "micro_feature_mode_breakdown": {
                    mode: {
                        "count": micro_mode_counts[asset_id][mode],
                        "fraction": _mode_fraction(
                            micro_mode_counts[asset_id][mode],
                            metrics.effective_weeks,
                        ),
                    }
                    for mode in ("primary", "proxy", "rv_only")
                },
            }
            for asset_id, metrics in per_asset_metrics.items()
        },
        "panel_aggregate_metrics": {
            "raw_crps": _finite_or_none(aggregate_crps),
            "baseline_a_crps": _finite_or_none(aggregate_baseline_crps),
            "crps_improvement": _finite_or_none(_panel_improvement_from_week_matrix(combined)),
            "crps_bootstrap_p05": {
                str(key): _finite_or_none(value) for key, value in bootstrap_p05.items()
            },
            "vol_normalized_crps": aggregate_vol_normalized,
            "ceq_diff_incomparable": "incomparable: Decision layer not adapted for panel",
            "blocked_proportion": blocked_proportion,
            "total_asset_weeks": effective_asset_weeks(
                {
                    asset_id: np.isfinite(np.asarray(values, dtype=np.float64))
                    for asset_id, values in realized_by_asset.items()
                },
            ),
        },
        "micro_feature_mode_breakdown": {
            "overall": {
                mode: {
                    "count": overall_mode_totals[mode],
                    "fraction": _mode_fraction(overall_mode_totals[mode], overall_mode_total),
                }
                for mode in ("primary", "proxy", "rv_only")
            },
            "solver_failure_weeks": solver_failure_weeks,
        },
    }
    deps.write_json(panel_root / "panel_comparison_report.json", cast(dict[str, object], report))
    if start == SMOKE_START and end == SMOKE_END:
        smoke_passed = solver_failure_weeks == 0 and blocked_weeks == 0 and all_output_finite
        return EXIT_OK if smoke_passed else EXIT_ACCEPTANCE_FAILED
    if start is None:
        return EXIT_OK if acceptance.passed else EXIT_ACCEPTANCE_FAILED
    return EXIT_OK if all_output_finite else EXIT_ACCEPTANCE_FAILED
