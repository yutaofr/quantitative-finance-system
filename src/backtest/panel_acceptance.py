"""Pure panel challenger acceptance gates from SRD v8.8 section P7."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class PanelAssetMetrics:
    """pure. Per-asset panel evaluation metrics on a common acceptance segment."""

    q10_coverage: float
    q90_coverage: float
    crps: float
    baseline_a_crps: float
    effective_weeks: int
    vol_normalized_crps: float | None = None


@dataclass(frozen=True, slots=True)
class PanelAcceptanceThresholds:
    """pure. Frozen panel acceptance thresholds from SRD v8.8 §P10."""

    coverage_tolerance: float
    coverage_collapse_limit: float
    crps_improvement_min: float
    blocked_cap: float
    block_lengths: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PanelAcceptanceItem:
    """pure. One named panel acceptance gate result."""

    name: str
    passed: bool
    observed: Mapping[str, float]
    threshold: str


@dataclass(frozen=True, slots=True)
class PanelAcceptanceReport:
    """pure. Ordered panel acceptance report."""

    items: tuple[PanelAcceptanceItem, ...]

    @property
    def passed(self) -> bool:
        """pure. Return whether all gates passed."""
        return all(item.passed for item in self.items)

    @property
    def by_name(self) -> Mapping[str, PanelAcceptanceItem]:
        """pure. Return items keyed by stable gate name."""
        return {item.name: item for item in self.items}


def panel_acceptance_report_to_dict(report: PanelAcceptanceReport) -> dict[str, object]:
    """pure. Convert a panel acceptance report into deterministic JSON-compatible data."""

    def finite_or_null(value: float) -> float | None:
        return value if np.isfinite(value) else None

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


def panel_crps_improvement(per_asset_metrics: Mapping[str, PanelAssetMetrics]) -> float:
    """pure. Equal-weight panel CRPS improvement versus Baseline_A."""
    crps = np.asarray([metric.crps for metric in per_asset_metrics.values()], dtype=np.float64)
    baseline = np.asarray(
        [metric.baseline_a_crps for metric in per_asset_metrics.values()],
        dtype=np.float64,
    )
    finite = np.isfinite(crps) & np.isfinite(baseline) & (baseline > 0.0)
    if not finite.any():
        return float("nan")
    return float(1.0 - np.mean(crps[finite]) / np.mean(baseline[finite]))


def _coverage_item(
    name: str,
    *,
    target: float,
    observed: Mapping[str, float],
    tolerance: float,
) -> PanelAcceptanceItem:
    passed = all(abs(value - target) <= tolerance for value in observed.values())
    return PanelAcceptanceItem(
        name=name,
        passed=passed,
        observed=dict(observed),
        threshold=f"|coverage - {target:.2f}| <= {tolerance:.2f} for every asset",
    )


def evaluate_panel_acceptance(
    per_asset_metrics: Mapping[str, PanelAssetMetrics],
    *,
    bootstrap_p05_by_block: Mapping[int, float],
    blocked_proportion: float,
    all_finite: bool,
    thresholds: PanelAcceptanceThresholds,
) -> PanelAcceptanceReport:
    """pure. Evaluate the fixed-panel structural/statistical acceptance gates."""
    q10_coverage = {
        asset_id: metrics.q10_coverage for asset_id, metrics in per_asset_metrics.items()
    }
    q90_coverage = {
        asset_id: metrics.q90_coverage for asset_id, metrics in per_asset_metrics.items()
    }
    improvement = panel_crps_improvement(per_asset_metrics)
    collapse_errors = {
        f"{asset_id}.q10_error": abs(metrics.q10_coverage - 0.10)
        for asset_id, metrics in per_asset_metrics.items()
    }
    collapse_errors.update(
        {
            f"{asset_id}.q90_error": abs(metrics.q90_coverage - 0.90)
            for asset_id, metrics in per_asset_metrics.items()
        },
    )
    block_pass = all(
        bootstrap_p05_by_block.get(block_length, float("-inf")) > 0.0
        for block_length in thresholds.block_lengths
    )
    bootstrap_observed = {
        f"bootstrap_p05_{block}": value for block, value in bootstrap_p05_by_block.items()
    }
    items = (
        _coverage_item(
            "per_asset_q10_coverage",
            target=0.10,
            observed=q10_coverage,
            tolerance=thresholds.coverage_tolerance,
        ),
        _coverage_item(
            "per_asset_q90_coverage",
            target=0.90,
            observed=q90_coverage,
            tolerance=thresholds.coverage_tolerance,
        ),
        PanelAcceptanceItem(
            name="panel_crps_vs_baseline_a",
            passed=bool(
                np.isfinite(improvement)
                and improvement >= thresholds.crps_improvement_min
                and block_pass
            ),
            observed={
                "improvement": improvement,
                **bootstrap_observed,
            },
            threshold=(
                f"improvement >= {thresholds.crps_improvement_min:.2f} and "
                f"bootstrap p05 > 0 for block_lengths={thresholds.block_lengths}"
            ),
        ),
        PanelAcceptanceItem(
            name="no_asset_coverage_collapse",
            passed=all(
                error <= thresholds.coverage_collapse_limit for error in collapse_errors.values()
            ),
            observed=collapse_errors,
            threshold=f"every q10/q90 error <= {thresholds.coverage_collapse_limit:.2f}",
        ),
        PanelAcceptanceItem(
            name="blocked_proportion",
            passed=bool(
                np.isfinite(blocked_proportion) and blocked_proportion <= thresholds.blocked_cap
            ),
            observed={"blocked_proportion": blocked_proportion},
            threshold=f"blocked_proportion <= {thresholds.blocked_cap:.2f}",
        ),
        PanelAcceptanceItem(
            name="finite_output_safety",
            passed=all_finite,
            observed={"all_finite": 1.0 if all_finite else 0.0},
            threshold="all panel outputs must be finite",
        ),
    )
    return PanelAcceptanceReport(items=items)
