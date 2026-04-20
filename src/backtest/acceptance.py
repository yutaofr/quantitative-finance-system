"""SRD §16 acceptance report evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np

from backtest.metrics import (
    bootstrap_ceq_diff_p05,
    bootstrap_crps_improvement_p05,
    ceq_annualized,
    crps_improvement_ratio,
    max_drawdown,
    strict_metric_series,
)
from backtest.walkforward import BacktestResult
from config_types import FrozenConfig
from engine_types import WeeklyOutput

MIN_TURNOVER_OBSERVATIONS = 2


@dataclass(frozen=True, slots=True)
class AcceptancePrerequisites:
    """pure. External structural gates already verified by focused tests or CI jobs."""

    bit_identical_determinism_ok: bool
    vintage_strict_pit_ok: bool
    research_firewall_ok: bool
    state_label_map_stable: bool


@dataclass(frozen=True, slots=True)
class AcceptanceMetrics:
    """pure. Strict-PIT statistical metrics from the backtest metrics layer."""

    q10_empirical_coverage: float
    q90_empirical_coverage: float
    crps_improvement: float
    crps_bootstrap_p05_by_block: Mapping[int, float]
    ceq_diff: float
    ceq_bootstrap_p05_by_block: Mapping[int, float]
    max_drawdown_diff: float


@dataclass(frozen=True, slots=True)
class AcceptanceThresholds:
    """pure. Frozen SRD §16 thresholds injected from config or test fixtures."""

    coverage_tolerance: float
    crps_improvement_min: float
    ceq_floor: float
    max_drawdown_tolerance: float
    turnover_cap: float
    blocked_cap: float
    block_lengths: tuple[int, ...]


def acceptance_thresholds_from_config(cfg: FrozenConfig) -> AcceptanceThresholds:
    """pure. Load frozen SRD §16 acceptance thresholds from runtime config."""
    return AcceptanceThresholds(
        coverage_tolerance=cfg.coverage_tol,
        crps_improvement_min=cfg.crps_min_improve,
        ceq_floor=cfg.ceq_floor,
        max_drawdown_tolerance=cfg.maxdd_tol,
        turnover_cap=cfg.turnover_cap,
        blocked_cap=cfg.blocked_cap,
        block_lengths=cfg.block_lengths,
    )


@dataclass(frozen=True, slots=True)
class AcceptanceItem:
    """pure. One named SRD §16 gate result."""

    name: str
    passed: bool
    observed: Mapping[str, float]
    threshold: str


@dataclass(frozen=True, slots=True)
class AcceptanceReport:
    """pure. Ordered SRD §16 acceptance gate report."""

    items: tuple[AcceptanceItem, ...]

    @property
    def passed(self) -> bool:
        """pure. Return whether all SRD §16 gates passed."""
        return all(item.passed for item in self.items)

    @property
    def by_name(self) -> Mapping[str, AcceptanceItem]:
        """pure. Return report items keyed by stable item name."""
        return {item.name: item for item in self.items}


def acceptance_report_to_dict(report: AcceptanceReport) -> dict[str, object]:
    """pure. Render an acceptance report as deterministic JSON-compatible data."""

    def finite_or_null(value: float) -> float | None:
        if not np.isfinite(value):
            return None
        return value

    return {
        "passed": report.passed,
        "items": [
            {
                "name": item.name,
                "passed": item.passed,
                "observed": {
                    key: finite_or_null(value) for key, value in sorted(item.observed.items())
                },
                "threshold": item.threshold,
            }
            for item in report.items
        ],
    }


def _bool_item(name: str, *, passed: bool, threshold: str) -> AcceptanceItem:
    return AcceptanceItem(
        name=name,
        passed=passed,
        observed={"passed": 1.0 if passed else 0.0},
        threshold=threshold,
    )


def _all_finite(outputs: Sequence[WeeklyOutput]) -> bool:
    for output in outputs:
        values = np.array(
            [
                output.state.hazard_covariate,
                output.distribution.q05,
                output.distribution.q10,
                output.distribution.q25,
                output.distribution.q50,
                output.distribution.q75,
                output.distribution.q90,
                output.distribution.q95,
                output.distribution.q05_ci_low,
                output.distribution.q05_ci_high,
                output.distribution.q95_ci_low,
                output.distribution.q95_ci_high,
                output.distribution.mu_hat,
                output.distribution.sigma_hat,
                output.distribution.p_loss,
                output.distribution.es20,
                output.decision.excess_return,
                output.decision.utility,
                output.decision.offense_raw,
                output.decision.offense_final,
                output.decision.cycle_position,
                output.diagnostics.missing_rate,
                output.diagnostics.coverage_q10_trailing_104w,
                output.diagnostics.coverage_q90_trailing_104w,
            ],
            dtype=np.float64,
        )
        if not np.isfinite(values).all() or not np.isfinite(output.state.post).all():
            return False
    return True


def _quantiles_non_crossing(output: WeeklyOutput) -> bool:
    quantiles = np.array(
        [
            output.distribution.q05,
            output.distribution.q10,
            output.distribution.q25,
            output.distribution.q50,
            output.distribution.q75,
            output.distribution.q90,
            output.distribution.q95,
        ],
        dtype=np.float64,
    )
    return bool(np.isfinite(quantiles).all() and np.all(np.diff(quantiles) >= 0.0))


def _turnover(outputs: Sequence[WeeklyOutput]) -> float:
    if len(outputs) < MIN_TURNOVER_OBSERVATIONS:
        return 0.0
    offense = np.array([output.decision.offense_final for output in outputs], dtype=np.float64)
    if not np.isfinite(offense).all():
        return float("nan")
    weekly_turnover = np.abs(np.diff(offense)) / 100.0
    return float(52.0 * np.mean(weekly_turnover))


def _blocked_proportion(outputs: Sequence[WeeklyOutput]) -> float:
    if not outputs:
        return 1.0
    blocked = sum(1 for output in outputs if output.mode == "BLOCKED")
    return float(blocked / len(outputs))


def _required_blocks_positive(values: Mapping[int, float], block_lengths: tuple[int, ...]) -> bool:
    return all(values.get(block_length, float("-1.0")) > 0.0 for block_length in block_lengths)


def _required_blocks_above_floor(
    values: Mapping[int, float],
    *,
    floor: float,
    block_lengths: tuple[int, ...],
) -> bool:
    return all(values.get(block_length, float("-1.0")) > floor for block_length in block_lengths)


def evaluate_acceptance(
    outputs: Sequence[WeeklyOutput],
    *,
    prerequisites: AcceptancePrerequisites,
    metrics: AcceptanceMetrics,
    thresholds: AcceptanceThresholds,
) -> AcceptanceReport:
    """pure. Evaluate the ordered SRD §16 acceptance gates."""
    q10_error = abs(metrics.q10_empirical_coverage - 0.10)
    q90_error = abs(metrics.q90_empirical_coverage - 0.90)
    turnover = _turnover(outputs)
    blocked_proportion = _blocked_proportion(outputs)
    items = (
        _bool_item(
            "bit_identical_determinism",
            passed=prerequisites.bit_identical_determinism_ok,
            threshold="same inputs emit byte-identical JSON",
        ),
        _bool_item(
            "vintage_strict_pit",
            passed=prerequisites.vintage_strict_pit_ok,
            threshold="strict PIT requests before earliest vintage raise",
        ),
        _bool_item(
            "research_firewall",
            passed=prerequisites.research_firewall_ok,
            threshold="production modules do not import research",
        ),
        _bool_item(
            "state_label_map_stability",
            passed=prerequisites.state_label_map_stable,
            threshold="same seed yields byte-identical label map",
        ),
        _bool_item(
            "quantile_non_crossing",
            passed=all(_quantiles_non_crossing(output) for output in outputs),
            threshold="q05 <= q10 <= q25 <= q50 <= q75 <= q90 <= q95",
        ),
        _bool_item(
            "tail_extrapolation_safety",
            passed=all(
                output.diagnostics.tail_extrapolation_status in {"ok", "fallback"}
                for output in outputs
            )
            and _all_finite(outputs),
            threshold="tail status is SRD-valid and output has no NaN/Inf",
        ),
        AcceptanceItem(
            name="interior_coverage",
            passed=q10_error <= thresholds.coverage_tolerance
            and q90_error <= thresholds.coverage_tolerance,
            observed={"q10_error": q10_error, "q90_error": q90_error},
            threshold="both absolute errors <= 0.03",
        ),
        AcceptanceItem(
            name="crps_vs_baseline_a",
            passed=metrics.crps_improvement >= thresholds.crps_improvement_min
            and _required_blocks_positive(
                metrics.crps_bootstrap_p05_by_block,
                thresholds.block_lengths,
            ),
            observed={"improvement": metrics.crps_improvement},
            threshold="mean improvement >= 5%; block p05 > 0 for 52 and 78",
        ),
        AcceptanceItem(
            name="ceq_vs_baseline_b",
            passed=metrics.ceq_diff > thresholds.ceq_floor
            and _required_blocks_above_floor(
                metrics.ceq_bootstrap_p05_by_block,
                floor=thresholds.ceq_floor,
                block_lengths=thresholds.block_lengths,
            ),
            observed={"diff": metrics.ceq_diff},
            threshold="block p05 and point diff > -50 bp/yr",
        ),
        AcceptanceItem(
            name="max_drawdown_vs_baseline_b",
            passed=metrics.max_drawdown_diff <= thresholds.max_drawdown_tolerance,
            observed={"diff": metrics.max_drawdown_diff},
            threshold="production max drawdown excess <= 300 bp",
        ),
        AcceptanceItem(
            name="turnover_and_blocked_proportion",
            passed=turnover <= thresholds.turnover_cap
            and blocked_proportion <= thresholds.blocked_cap,
            observed={"turnover": turnover, "blocked_proportion": blocked_proportion},
            threshold="turnover <= 1.5/yr and blocked proportion <= 15%",
        ),
    )
    return AcceptanceReport(items=items)


def _metric_value_or_fail(value: float) -> float:
    if not np.isfinite(value):
        return float("nan")
    return value


def evaluate_backtest_acceptance(  # noqa: PLR0913
    result: BacktestResult,
    *,
    prerequisites: AcceptancePrerequisites,
    thresholds: AcceptanceThresholds,
    bootstrap_replications: int,
    rng: np.random.Generator,
    effective_strict_start: date,
) -> AcceptanceReport:
    """pure. Compute strict-PIT metrics from a walk-forward result and evaluate gates."""
    if not result.realized_52w_returns:
        msg = "backtest acceptance requires realized_52w_returns"
        raise ValueError(msg)
    series = strict_metric_series(
        result.outputs,
        result.realized_52w_returns,
        effective_strict_start=effective_strict_start,
    )
    q10_coverage = float(np.mean(series.q10_hits)) if series.q10_hits.size else float("nan")
    q90_coverage = float(np.mean(series.q90_hits)) if series.q90_hits.size else float("nan")
    crps_improvement = crps_improvement_ratio(
        series.crps_production,
        series.crps_baseline_a,
    )
    crps_bootstrap = {
        block_length: bootstrap_crps_improvement_p05(
            series,
            block_length=block_length,
            replications=bootstrap_replications,
            rng=rng,
        )
        for block_length in thresholds.block_lengths
    }
    production_ceq = ceq_annualized(series.production_returns)
    baseline_ceq = ceq_annualized(series.baseline_b_returns)
    ceq_diff = production_ceq - baseline_ceq
    ceq_bootstrap = {
        block_length: bootstrap_ceq_diff_p05(
            series,
            block_length=block_length,
            replications=bootstrap_replications,
            rng=rng,
        )
        for block_length in thresholds.block_lengths
    }
    max_drawdown_diff = max_drawdown(series.production_returns) - max_drawdown(
        series.baseline_b_returns,
    )
    metrics = AcceptanceMetrics(
        q10_empirical_coverage=_metric_value_or_fail(q10_coverage),
        q90_empirical_coverage=_metric_value_or_fail(q90_coverage),
        crps_improvement=_metric_value_or_fail(crps_improvement),
        crps_bootstrap_p05_by_block=crps_bootstrap,
        ceq_diff=_metric_value_or_fail(ceq_diff),
        ceq_bootstrap_p05_by_block=ceq_bootstrap,
        max_drawdown_diff=_metric_value_or_fail(max_drawdown_diff),
    )
    return evaluate_acceptance(
        result.outputs,
        prerequisites=prerequisites,
        metrics=metrics,
        thresholds=thresholds,
    )
