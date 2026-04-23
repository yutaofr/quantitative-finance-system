from __future__ import annotations

from datetime import date
import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from app.panel_runner import PanelRunnerDeps, run_panel_backtest_job
from engine_types import TimeSeries
from features.panel_block_builder import PanelFeatureFrame
from law.panel_quantiles import PanelQRCoefs

ASSET_IDS = ("NASDAQXNDX", "SPX", "R2K")


def _dummy_series(series_id: str) -> TimeSeries:
    return TimeSeries(
        series_id=series_id,
        timestamps=np.asarray(["2024-01-05"], dtype="datetime64[D]"),
        values=np.asarray([1.0], dtype=np.float64),
        is_pseudo_pit=False,
    )


def _panel_frame() -> PanelFeatureFrame:
    weeks = (
        date(2024, 1, 5),
        date(2024, 1, 12),
        date(2024, 1, 19),
        date(2024, 1, 26),
    )
    x_macro = np.tile(np.linspace(-0.3, 0.3, 7, dtype=np.float64), (4, 1))
    micro_base = np.array(
        [
            [0.10, 0.20, 0.30],
            [0.20, 0.30, 0.40],
            [0.30, 0.40, 0.50],
            [0.40, 0.50, 0.60],
        ],
        dtype=np.float64,
    )
    return PanelFeatureFrame(
        as_of=weeks[-1],
        feature_dates=weeks,
        x_macro_raw=x_macro.copy(),
        x_macro=x_macro.copy(),
        x_micro_raw={
            "NASDAQXNDX": micro_base.copy(),
            "SPX": micro_base.copy() + 0.1,
            "R2K": micro_base.copy() + 0.2,
        },
        x_micro={
            "NASDAQXNDX": micro_base.copy(),
            "SPX": micro_base.copy() + 0.1,
            "R2K": micro_base.copy() + 0.2,
        },
        macro_mask=np.zeros((4, 7), dtype=np.bool_),
        micro_mask={asset_id: np.zeros((4, 3), dtype=np.bool_) for asset_id in ASSET_IDS},
        asset_availability={asset_id: np.ones(4, dtype=np.bool_) for asset_id in ASSET_IDS},
        micro_feature_mode={
            "NASDAQXNDX": ("primary", "primary", "primary", "primary"),
            "SPX": ("primary", "primary", "primary", "primary"),
            "R2K": ("proxy", "primary", "rv_only", "primary"),
        },
        available_assets=ASSET_IDS,
        target_returns={
            "NASDAQXNDX": np.array([0.00, 0.01, 0.03, 0.02], dtype=np.float64),
            "SPX": np.array([-0.01, 0.00, 0.02, 0.01], dtype=np.float64),
            "R2K": np.array([-0.02, 0.01, 0.04, 0.03], dtype=np.float64),
        },
    )


def _fixed_coefs() -> PanelQRCoefs:
    return PanelQRCoefs(
        asset_ids=ASSET_IDS,
        alpha={
            "NASDAQXNDX": np.array([-0.03, -0.01, 0.01, 0.03, 0.05], dtype=np.float64),
            "SPX": np.array([-0.04, -0.02, 0.00, 0.02, 0.04], dtype=np.float64),
            "R2K": np.array([-0.05, -0.03, 0.01, 0.05, 0.08], dtype=np.float64),
        },
        b=np.zeros((5, 7), dtype=np.float64),
        c=np.zeros((5, 3), dtype=np.float64),
        delta=np.zeros((5, 3), dtype=np.float64),
        solver_status="ok",
        model_status="NORMAL",
    )


def _fake_effective_start() -> date:
    return date(2024, 1, 12)


def _fake_fit_state(
    train_frame: PanelFeatureFrame,
) -> tuple[SimpleNamespace, SimpleNamespace, np.ndarray, str]:
    return (
        SimpleNamespace(label_map={0: "DEFENSIVE", 1: "NEUTRAL", 2: "OFFENSIVE"}),
        SimpleNamespace(observation_dates=(train_frame.feature_dates[-1],)),
        np.asarray([[0.2, 0.5, 0.3]], dtype=np.float64),
        "ok",
    )


def _fake_current_state() -> tuple[np.ndarray, str, int, float]:
    return (
        np.asarray([0.2, 0.5, 0.3], dtype=np.float64),
        "NEUTRAL",
        4,
        0.1,
    )


def test_panel_runner_emits_serializable_weekly_and_report_payloads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    frame = _panel_frame()
    coefs = _fixed_coefs()
    writes: dict[str, dict[str, object]] = {}

    def fake_write_json(path: Path, payload: dict[str, object]) -> None:
        json.dumps(payload, allow_nan=False, sort_keys=True)
        writes[path.name] = payload

    monkeypatch.setattr(
        "app.panel_runner.compute_panel_effective_start",
        lambda *_args, **_kwargs: _fake_effective_start(),
    )
    monkeypatch.setattr(
        "app.panel_runner.build_panel_feature_block",
        lambda *_args, **_kwargs: frame,
    )
    monkeypatch.setattr(
        "app.panel_runner._fit_panel_training_state",
        lambda train_frame, *_args, **_kwargs: _fake_fit_state(train_frame),
    )
    monkeypatch.setattr(
        "app.panel_runner._current_hmm_state",
        lambda *_args, **_kwargs: _fake_current_state(),
    )
    monkeypatch.setattr(
        "app.panel_runner.fit_panel_quantiles",
        lambda *_args, **_kwargs: coefs,
    )
    monkeypatch.setattr("app.panel_runner._write_panel_label_map", lambda *_args, **_kwargs: None)

    deps = PanelRunnerDeps(
        fetch_macro_series=lambda *_args, **_kwargs: {
            "VXNCLS": _dummy_series("VXNCLS"),
            "VIXCLS": _dummy_series("VIXCLS"),
            "RVXCLS": _dummy_series("RVXCLS"),
        },
        fetch_asset_prices=lambda *_args, **_kwargs: {
            asset_id: _dummy_series(asset_id) for asset_id in ASSET_IDS
        },
        write_json=fake_write_json,
    )
    panel_config = {
        "panel_size": 3,
        "min_train": 1,
        "embargo_weeks": 0,
        "l2_alpha_macro": 2.0,
        "l2_alpha_micro": 2.0,
        "min_gap": 1.0e-4,
        "tail_mult": 0.6,
        "coverage_tol": 0.05,
        "coverage_collapse": 0.08,
        "crps_min_improve": 0.05,
        "blocked_cap": 0.15,
        "block_lengths": [2],
        "B": 8,
        "pit_classification": "log_return_pit",
    }

    code = run_panel_backtest_job(
        start=date(2024, 1, 12),
        end=date(2024, 1, 26),
        panel_config=panel_config,
        artifacts_root=tmp_path,
        deps=deps,
    )

    assert code == 0
    weekly = writes["panel_output.json"]
    report = writes["panel_comparison_report.json"]
    weekly_diag = cast(dict[str, object], weekly["panel_diagnostics"])
    weekly_assets = cast(dict[str, object], weekly["assets"])
    r2k = cast(dict[str, object], weekly_assets["R2K"])
    r2k_diag = cast(dict[str, object], r2k["diagnostics"])
    aggregate = cast(dict[str, object], report["panel_aggregate_metrics"])
    mode_breakdown = cast(dict[str, object], report["micro_feature_mode_breakdown"])
    overall_modes = cast(dict[str, object], mode_breakdown["overall"])
    primary_overall = cast(dict[str, object], overall_modes["primary"])

    assert weekly["pit_classification"] == "log-return-PIT"
    assert weekly_diag["panel_solver_status"] == "ok"
    assert cast(float, weekly_diag["panel_crps_vs_baseline_a"]) < 0.0
    assert cast(float, r2k_diag["coverage_q10_trailing_104w"]) >= 0.0
    assert aggregate["raw_crps"] is not None
    assert (
        aggregate["ceq_diff_incomparable"]
        == "incomparable: Decision layer not adapted for panel"
    )
    assert "micro_feature_mode_breakdown" in report
    assert cast(int, primary_overall["count"]) >= 1
