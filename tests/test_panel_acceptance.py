from __future__ import annotations

from backtest.panel_acceptance import (
    PanelAcceptanceThresholds,
    PanelAssetMetrics,
    evaluate_panel_acceptance,
    panel_acceptance_report_to_dict,
)


def _thresholds() -> PanelAcceptanceThresholds:
    return PanelAcceptanceThresholds(
        coverage_tolerance=0.05,
        coverage_collapse_limit=0.08,
        crps_improvement_min=0.05,
        blocked_cap=0.15,
        block_lengths=(52, 78),
    )


def test_panel_acceptance_passes_for_controlled_metrics() -> None:
    report = evaluate_panel_acceptance(
        {
            "NASDAQXNDX": PanelAssetMetrics(0.10, 0.90, 0.20, 0.30, 100),
            "SPX": PanelAssetMetrics(0.11, 0.89, 0.21, 0.31, 100),
            "R2K": PanelAssetMetrics(0.09, 0.91, 0.22, 0.33, 100),
        },
        bootstrap_p05_by_block={52: 0.01, 78: 0.02},
        blocked_proportion=0.10,
        all_finite=True,
        thresholds=_thresholds(),
    )
    payload = panel_acceptance_report_to_dict(report)

    assert report.passed
    assert payload["passed"] is True
    assert report.by_name["panel_crps_vs_baseline_a"].passed


def test_panel_acceptance_flags_coverage_collapse_and_blocked_overflow() -> None:
    report = evaluate_panel_acceptance(
        {
            "NASDAQXNDX": PanelAssetMetrics(0.00, 0.90, 0.28, 0.30, 100),
            "SPX": PanelAssetMetrics(0.11, 0.89, 0.29, 0.31, 100),
            "R2K": PanelAssetMetrics(0.09, 1.00, 0.32, 0.33, 100),
        },
        bootstrap_p05_by_block={52: -0.01, 78: 0.02},
        blocked_proportion=0.20,
        all_finite=False,
        thresholds=_thresholds(),
    )

    assert not report.passed
    assert not report.by_name["per_asset_q10_coverage"].passed
    assert not report.by_name["no_asset_coverage_collapse"].passed
    assert not report.by_name["blocked_proportion"].passed
    assert not report.by_name["finite_output_safety"].passed
