from __future__ import annotations

from datetime import date, timedelta
import json
from typing import cast

import numpy as np
import pytest

from backtest.acceptance import (
    AcceptancePrerequisites,
    AcceptanceThresholds,
    acceptance_report_to_dict,
    evaluate_backtest_acceptance,
)
from backtest.metrics import realized_forward_returns
from backtest.walkforward import BacktestResult
from engine_types import (
    DecisionOutput,
    DiagnosticsOutput,
    DistributionOutput,
    TimeSeries,
    WeeklyOutput,
    WeeklyState,
)


def _thresholds() -> AcceptanceThresholds:
    return AcceptanceThresholds(
        coverage_tolerance=0.03,
        crps_improvement_min=0.05,
        ceq_floor=-0.005,
        max_drawdown_tolerance=0.03,
        turnover_cap=1.5,
        blocked_cap=0.15,
        block_lengths=(52, 78),
    )


def _prerequisites() -> AcceptancePrerequisites:
    return AcceptancePrerequisites(
        bit_identical_determinism_ok=True,
        vintage_strict_pit_ok=True,
        research_firewall_ok=True,
        state_label_map_stable=True,
    )


def _output(as_of: date, y: float, *, vintage_mode: str = "strict") -> WeeklyOutput:
    return WeeklyOutput(
        as_of_date=as_of,
        srd_version="8.7.1",
        mode="NORMAL",
        vintage_mode=vintage_mode,  # type: ignore[arg-type]
        state=WeeklyState(
            post=np.array([0.25, 0.50, 0.25], dtype=np.float64),
            state_name="NEUTRAL",
            dwell_weeks=1,
            hazard_covariate=0.0,
        ),
        distribution=DistributionOutput(
            q05=y - 0.05,
            q10=y - 0.03,
            q25=y - 0.01,
            q50=y,
            q75=y + 0.01,
            q90=y + 0.03,
            q95=y + 0.05,
            q05_ci_low=y - 0.08,
            q05_ci_high=y - 0.02,
            q95_ci_low=y + 0.02,
            q95_ci_high=y + 0.08,
            mu_hat=y,
            sigma_hat=0.05,
            p_loss=0.4,
            es20=0.02,
        ),
        decision=DecisionOutput(
            excess_return=y,
            utility=0.5,
            offense_raw=50.0,
            offense_final=50.0,
            stance="NEUTRAL",
            cycle_position=50.0,
        ),
        diagnostics=DiagnosticsOutput(
            missing_rate=0.0,
            quantile_solver_status="ok",
            tail_extrapolation_status="ok",
            hmm_status="ok",
            coverage_q10_trailing_104w=0.1,
            coverage_q90_trailing_104w=0.9,
        ),
    )


def test_realized_forward_returns_aligns_52w_target_without_future_leakage() -> None:
    start = date(2020, 1, 3)
    prices = np.exp(np.arange(60, dtype=np.float64) * 0.01)
    series = TimeSeries(
        series_id="NASDAQXNDX",
        timestamps=np.array(
            [(start + timedelta(weeks=idx)).isoformat() for idx in range(60)],
            dtype="datetime64[D]",
        ),
        values=prices,
        is_pseudo_pit=False,
    )

    realized = realized_forward_returns(
        series,
        (start, start + timedelta(weeks=7), start + timedelta(weeks=8)),
    )

    assert np.allclose(realized[:2], np.array([0.52, 0.52], dtype=np.float64))
    assert np.isnan(realized[2])


def test_evaluate_backtest_acceptance_uses_strict_finite_segment_and_serializes_report() -> None:
    start = date(2024, 1, 5)
    strict_outputs = tuple(
        _output(start + timedelta(weeks=idx), float(idx) / 1000.0) for idx in range(12)
    )
    result = BacktestResult(
        outputs=(_output(start - timedelta(weeks=1), 0.0, vintage_mode="pseudo"), *strict_outputs),
        realized_52w_returns=(0.0, *(float(idx) / 1000.0 for idx in range(12))),
    )

    report = evaluate_backtest_acceptance(
        result,
        prerequisites=_prerequisites(),
        thresholds=_thresholds(),
        bootstrap_replications=64,
        rng=np.random.default_rng(123),
        effective_strict_start=date(2014, 11, 28),
    )
    payload = acceptance_report_to_dict(report)

    assert [item.name for item in report.items][-5:] == [
        "interior_coverage",
        "crps_vs_baseline_a",
        "ceq_vs_baseline_b",
        "max_drawdown_vs_baseline_b",
        "turnover_and_blocked_proportion",
    ]
    assert payload["passed"] == report.passed
    items = cast(list[dict[str, object]], payload["items"])
    assert items[0]["name"] == "bit_identical_determinism"
    assert json.loads(json.dumps(payload, sort_keys=True)) == payload


def test_evaluate_backtest_acceptance_excludes_strict_warmup_before_effective_start() -> None:
    effective_start = date(2014, 11, 28)
    result = BacktestResult(
        outputs=(
            _output(date(2014, 11, 21), 10.0),
            _output(effective_start, 0.0),
        ),
        realized_52w_returns=(10.0, 0.0),
    )

    report = evaluate_backtest_acceptance(
        result,
        prerequisites=_prerequisites(),
        thresholds=_thresholds(),
        bootstrap_replications=8,
        rng=np.random.default_rng(123),
        effective_strict_start=effective_start,
    )

    coverage = report.by_name["interior_coverage"].observed
    assert coverage["q10_error"] == 0.10
    assert coverage["q90_error"] == pytest.approx(0.10)
