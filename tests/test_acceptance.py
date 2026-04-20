from __future__ import annotations

import ast
from collections.abc import Mapping
from datetime import date, timedelta
import hashlib
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
import numpy as np
import pytest

from app.output_serializer import serialize_weekly_output, to_serializable_dict
from backtest.acceptance import (
    AcceptanceMetrics,
    AcceptancePrerequisites,
    AcceptanceThresholds,
    evaluate_acceptance,
)
from backtest.walkforward import run_walkforward
from config_types import FrozenConfig
from decision.offense_abs import OffenseThresholds
from decision.utility import UtilityZStats
from engine_types import TimeSeries, WeeklyOutput
from inference.weekly import TrainingArtifacts, run_weekly
from law.linear_quantiles import QRCoefs
from state.ti_hmm_single import HMMModel

SCHEMA_ROOT_KEYS = (
    "as_of_date",
    "srd_version",
    "mode",
    "vintage_mode",
    "state",
    "distribution",
    "decision",
    "diagnostics",
)
STATE_KEYS = ("post", "state_name", "dwell_weeks", "hazard_covariate")
DISTRIBUTION_KEYS = (
    "q05",
    "q10",
    "q25",
    "q50",
    "q75",
    "q90",
    "q95",
    "q05_ci_low",
    "q05_ci_high",
    "q95_ci_low",
    "q95_ci_high",
    "mu_hat",
    "sigma_hat",
    "p_loss",
    "es20",
)
DECISION_KEYS = (
    "excess_return",
    "utility",
    "offense_raw",
    "offense_final",
    "stance",
    "cycle_position",
)
DIAGNOSTICS_KEYS = (
    "missing_rate",
    "quantile_solver_status",
    "tail_extrapolation_status",
    "hmm_status",
    "coverage_q10_trailing_104w",
    "coverage_q90_trailing_104w",
)
ENGINE_DISTRIBUTION_FIELDS = (
    "q05",
    "q10",
    "q25",
    "q50",
    "q75",
    "q90",
    "q95",
    "q05_ci_low",
    "q05_ci_high",
    "q95_ci_low",
    "q95_ci_high",
    "mu_hat",
    "sigma_hat",
    "p_loss",
    "es20",
)
SRD_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": list(SCHEMA_ROOT_KEYS),
    "properties": {
        "as_of_date": {"type": "string", "format": "date"},
        "srd_version": {"const": "8.7"},
        "mode": {"enum": ["NORMAL", "DEGRADED", "BLOCKED"]},
        "vintage_mode": {"enum": ["strict", "pseudo"]},
        "state": {
            "type": "object",
            "additionalProperties": False,
            "required": list(STATE_KEYS),
            "properties": {
                "post": {
                    "type": "array",
                    "minItems": 3,
                    "maxItems": 3,
                    "items": {"type": "number"},
                },
                "state_name": {"enum": ["DEFENSIVE", "NEUTRAL", "OFFENSIVE"]},
                "dwell_weeks": {"type": "integer", "minimum": 0},
                "hazard_covariate": {"type": "number"},
            },
        },
        "distribution": {
            "type": "object",
            "additionalProperties": False,
            "required": list(DISTRIBUTION_KEYS),
            "properties": {key: {"type": "number"} for key in DISTRIBUTION_KEYS},
        },
        "decision": {
            "type": "object",
            "additionalProperties": False,
            "required": list(DECISION_KEYS),
            "properties": {
                "excess_return": {"type": "number"},
                "utility": {"type": "number"},
                "offense_raw": {"type": "number"},
                "offense_final": {"type": "number"},
                "stance": {"enum": ["DEFENSIVE", "NEUTRAL", "OFFENSIVE"]},
                "cycle_position": {"type": "number"},
            },
        },
        "diagnostics": {
            "type": "object",
            "additionalProperties": False,
            "required": list(DIAGNOSTICS_KEYS),
            "properties": {
                "missing_rate": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "quantile_solver_status": {
                    "enum": ["ok", "rearranged", "failed"],
                },
                "tail_extrapolation_status": {"enum": ["ok", "fallback"]},
                "hmm_status": {"enum": ["ok", "degenerate", "em_nonconverge"]},
                "coverage_q10_trailing_104w": {"type": "number"},
                "coverage_q90_trailing_104w": {"type": "number"},
            },
        },
    },
}
EXPECTED_ENGINE_FIELDS = {
    "WeeklyOutput": (
        ("as_of_date", "date"),
        ("srd_version", "Literal['8.7']"),
        ("mode", "Mode"),
        ("vintage_mode", "VintageMode"),
        ("state", "WeeklyState"),
        ("distribution", "DistributionOutput"),
        ("decision", "DecisionOutput"),
        ("diagnostics", "DiagnosticsOutput"),
    ),
    "WeeklyState": (
        ("post", "NDArray[np.float64]"),
        ("state_name", "Stance"),
        ("dwell_weeks", "int"),
        ("hazard_covariate", "float"),
    ),
    "DistributionOutput": tuple((key, "float") for key in ENGINE_DISTRIBUTION_FIELDS),
    "DecisionOutput": (
        ("excess_return", "float"),
        ("utility", "float"),
        ("offense_raw", "float"),
        ("offense_final", "float"),
        ("stance", "Stance"),
        ("cycle_position", "float"),
    ),
    "DiagnosticsOutput": (
        ("missing_rate", "float"),
        ("quantile_solver_status", "str"),
        ("tail_extrapolation_status", "str"),
        ("hmm_status", "str"),
        ("coverage_q10_trailing_104w", "float"),
        ("coverage_q90_trailing_104w", "float"),
    ),
}
EXPECTED_WALKFORWARD_SHA256 = "446fb45cdee7bcfe62dc31eb5924ad61dced1811197f770bc8c8da5a6d01d603"


def _config() -> FrozenConfig:
    return FrozenConfig(
        srd_version="8.7",
        random_seed=8675309,
        timezone="America/New_York",
        missing_rate_degraded=0.10,
        missing_rate_blocked=0.20,
        quantile_gap=1.0e-4,
        l2_alpha=2.0,
        tail_mult=0.6,
        utility_lambda=1.2,
        utility_kappa=0.8,
        band=7.0,
        score_min=0.0,
        score_max=100.0,
        block_lengths=(52, 78),
        bootstrap_replications=2000,
    )


def _acceptance_thresholds() -> AcceptanceThresholds:
    return AcceptanceThresholds(
        coverage_tolerance=0.03,
        crps_improvement_min=0.05,
        ceq_floor=-0.005,
        max_drawdown_tolerance=0.03,
        turnover_cap=1.5,
        blocked_cap=0.15,
        block_lengths=(52, 78),
    )


def _engine_dataclass_fields() -> dict[str, tuple[tuple[str, str], ...]]:
    tree = ast.parse(Path("src/engine_types.py").read_text(encoding="utf-8"))
    fields: dict[str, tuple[tuple[str, str], ...]] = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        if not any(
            isinstance(decorator, ast.Call) and getattr(decorator.func, "id", "") == "dataclass"
            for decorator in node.decorator_list
        ):
            continue
        fields[node.name] = tuple(
            (statement.target.id, ast.unparse(statement.annotation))
            for statement in node.body
            if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name)
        )
    return fields


def _validator() -> Draft202012Validator:
    Draft202012Validator.check_schema(SRD_OUTPUT_SCHEMA)
    return Draft202012Validator(SRD_OUTPUT_SCHEMA)


def _assert_no_nonfinite_numbers(value: object) -> None:
    if isinstance(value, Mapping):
        for nested in value.values():
            _assert_no_nonfinite_numbers(nested)
        return
    if isinstance(value, list):
        for nested in value:
            _assert_no_nonfinite_numbers(nested)
        return
    if isinstance(value, float | int):
        assert np.isfinite(float(value))


def _series(series_id: str, start: date, values: np.ndarray) -> TimeSeries:
    timestamps = np.array(
        [(start + timedelta(weeks=idx)).isoformat() for idx in range(values.shape[0])],
        dtype="datetime64[D]",
    )
    return TimeSeries(
        series_id=series_id,
        timestamps=timestamps,
        values=values.astype(np.float64, copy=True),
        is_pseudo_pit=False,
    )


def _mock_series(start: date, weeks: int = 100) -> dict[str, TimeSeries]:
    idx = np.arange(weeks, dtype=np.float64)
    return {
        "DGS10": _series("DGS10", start, 3.0 + 0.010 * idx),
        "DGS2": _series("DGS2", start, 2.0 + 0.008 * idx),
        "DGS1": _series("DGS1", start, 0.020 + 0.0001 * idx),
        "EFFR": _series("EFFR", start, 0.015 + 0.0001 * idx),
        "BAA10Y": _series("BAA10Y", start, 1.5 + 0.002 * idx),
        "WALCL": _series("WALCL", start, 100.0 + idx),
        "VXNCLS": _series("VXNCLS", start, 20.0 + 0.05 * idx),
        "RV20_NDX": _series("RV20_NDX", start, 10.0 + 0.03 * idx),
        "VIXCLS": _series("VIXCLS", start, 15.0 + 0.04 * idx),
        "VXVCLS": _series("VXVCLS", start, 12.0 + 0.03 * idx),
    }


def _inject_missing(
    series: Mapping[str, TimeSeries],
    *,
    as_of: date,
    series_ids: tuple[str, ...],
) -> dict[str, TimeSeries]:
    mutated: dict[str, TimeSeries] = {}
    target = np.datetime64(as_of, "D")
    for series_id, time_series in series.items():
        values = time_series.values.copy()
        if series_id in series_ids:
            indices = np.flatnonzero(time_series.timestamps.astype("datetime64[D]") == target)
            values[indices] = np.nan
        mutated[series_id] = TimeSeries(
            series_id=time_series.series_id,
            timestamps=time_series.timestamps.copy(),
            values=values,
            is_pseudo_pit=time_series.is_pseudo_pit,
        )
    return mutated


def _hmm_model() -> HMMModel:
    return HMMModel(
        transition_coefs=np.zeros((3, 3), dtype=np.float64),
        emission_mean=np.zeros((3, 6), dtype=np.float64),
        emission_cov=np.stack([np.eye(6, dtype=np.float64)] * 3),
        label_map={0: "DEFENSIVE", 1: "NEUTRAL", 2: "OFFENSIVE"},
        log_likelihood=-1.0,
    )


def _training_artifacts() -> TrainingArtifacts:
    return TrainingArtifacts(
        utility_zstats=UtilityZStats(
            er_med=0.0,
            er_mad=1.0,
            es20_med=0.0,
            es20_mad=1.0,
            ploss_med=0.5,
            ploss_mad=0.1,
        ),
        offense_thresholds=OffenseThresholds(-2.0, -1.0, 0.0, 1.0, 2.0, 3.0),
        train_distributions={
            "x1": np.array([0.0, 1.0, 2.0], dtype=np.float64),
            "x5": np.array([0.0, 1.0, 2.0], dtype=np.float64),
            "x9": np.array([0.0, 1.0, 2.0], dtype=np.float64),
        },
        state_label_map={0: "DEFENSIVE", 1: "NEUTRAL", 2: "OFFENSIVE"},
        qr_coefs=QRCoefs(
            a=np.array([-0.20, -0.10, 0.0, 0.10, 0.20], dtype=np.float64),
            b=np.zeros((5, 10), dtype=np.float64),
            c=np.zeros((5, 3), dtype=np.float64),
            solver_status="ok",
        ),
        hmm_model=_hmm_model(),
    )


def _fit_training_artifacts(
    _as_of: date,
    _history: Mapping[str, TimeSeries],
    _cfg: FrozenConfig,
) -> TrainingArtifacts:
    return _training_artifacts()


def _infer_weekly(
    as_of: date,
    _cfg: FrozenConfig,
    history: Mapping[str, TimeSeries],
    training_artifacts: TrainingArtifacts,
) -> WeeklyOutput:
    return run_weekly(as_of, "strict", history, training_artifacts)


def _walkforward_bytes(series: Mapping[str, TimeSeries], start: date, end: date) -> bytes:
    result = run_walkforward(
        start=start,
        end=end,
        series=series,
        cfg=_config(),
        fit_training_artifacts=_fit_training_artifacts,
        infer_weekly=_infer_weekly,
    )
    return b"".join(serialize_weekly_output(output) for output in result.outputs)


def _acceptance_outputs() -> tuple[WeeklyOutput, ...]:
    start = date(2024, 1, 5)
    series = _mock_series(date(2023, 2, 3), weeks=100)
    result = run_walkforward(
        start=start,
        end=start + timedelta(weeks=5),
        series=series,
        cfg=_config(),
        fit_training_artifacts=_fit_training_artifacts,
        infer_weekly=_infer_weekly,
    )
    return result.outputs


@pytest.mark.acceptance
def test_srd_output_schema_matches_engine_types_and_serializer() -> None:
    fields = _engine_dataclass_fields()
    sample = run_weekly(
        date(2024, 12, 27),
        "strict",
        _mock_series(date(2023, 2, 3), weeks=100),
        _training_artifacts(),
    )
    payload = to_serializable_dict(sample)

    assert {name: fields[name] for name in EXPECTED_ENGINE_FIELDS} == EXPECTED_ENGINE_FIELDS
    assert tuple(payload) == SCHEMA_ROOT_KEYS
    assert tuple(payload["state"]) == STATE_KEYS
    assert tuple(payload["distribution"]) == DISTRIBUTION_KEYS
    assert tuple(payload["decision"]) == DECISION_KEYS
    assert tuple(payload["diagnostics"]) == DIAGNOSTICS_KEYS
    assert not list(_validator().iter_errors(payload))
    _assert_no_nonfinite_numbers(payload)


@pytest.mark.acceptance
def test_determinism_smoke_mock_walkforward_emits_byte_stable_json() -> None:
    start = date(2024, 1, 5)
    end = start + timedelta(weeks=5)
    payload = _walkforward_bytes(_mock_series(date(2023, 2, 3), weeks=100), start, end)

    assert hashlib.sha256(payload).hexdigest() == EXPECTED_WALKFORWARD_SHA256
    assert b"NaN" not in payload
    assert b"Infinity" not in payload


@pytest.mark.acceptance
def test_acceptance_report_maps_srd_section_16_items() -> None:
    report = evaluate_acceptance(
        _acceptance_outputs(),
        prerequisites=AcceptancePrerequisites(
            bit_identical_determinism_ok=True,
            vintage_strict_pit_ok=True,
            research_firewall_ok=True,
            state_label_map_stable=True,
        ),
        metrics=AcceptanceMetrics(
            q10_empirical_coverage=0.10,
            q90_empirical_coverage=0.90,
            crps_improvement=0.06,
            crps_bootstrap_p05_by_block={52: 0.01, 78: 0.02},
            ceq_diff=0.0,
            ceq_bootstrap_p05_by_block={52: -0.004, 78: -0.003},
            max_drawdown_diff=0.02,
        ),
        thresholds=_acceptance_thresholds(),
    )

    assert [item.name for item in report.items] == [
        "bit_identical_determinism",
        "vintage_strict_pit",
        "research_firewall",
        "state_label_map_stability",
        "quantile_non_crossing",
        "tail_extrapolation_safety",
        "interior_coverage",
        "crps_vs_baseline_a",
        "ceq_vs_baseline_b",
        "max_drawdown_vs_baseline_b",
        "turnover_and_blocked_proportion",
    ]
    assert report.passed
    assert report.by_name["turnover_and_blocked_proportion"].observed["blocked_proportion"] == 0.0


@pytest.mark.acceptance
@pytest.mark.parametrize(
    ("missing_series_ids", "expected_mode", "expected_offense"),
    [
        (("EFFR",), "DEGRADED", 50.0),
        (("DGS10", "VXNCLS"), "BLOCKED", 50.0),
    ],
)
def test_weekly_blackbox_missing_data_falls_back_to_visible_modes(
    missing_series_ids: tuple[str, ...],
    expected_mode: str,
    expected_offense: float,
) -> None:
    as_of = date(2024, 12, 27)
    series = _inject_missing(
        _mock_series(date(2023, 2, 3), weeks=100),
        as_of=as_of,
        series_ids=missing_series_ids,
    )

    output = run_weekly(as_of, "strict", series, _training_artifacts())
    payload = to_serializable_dict(output)

    assert output.mode == expected_mode
    if expected_mode == "BLOCKED":
        assert output.decision.offense_final == expected_offense
    else:
        assert 30.0 <= output.decision.offense_final <= 70.0
    assert output.decision.stance == "NEUTRAL"
    assert not list(_validator().iter_errors(payload))
    _assert_no_nonfinite_numbers(payload)
