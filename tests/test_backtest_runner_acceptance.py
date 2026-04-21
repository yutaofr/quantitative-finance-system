from __future__ import annotations

from collections.abc import Mapping
from datetime import date, timedelta
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pytest

from app.backtest_runner import BacktestRunnerDeps, run_backtest_job, write_backtest_jsonl
from config_types import FrozenConfig
from engine_types import (
    DecisionOutput,
    DiagnosticsOutput,
    DistributionOutput,
    TimeSeries,
    WeeklyOutput,
    WeeklyState,
)
from inference.weekly import TrainingArtifacts


def _config() -> FrozenConfig:
    return FrozenConfig(
        srd_version="8.7.1",
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
        bootstrap_replications=8,
    )


def _output(as_of: date) -> WeeklyOutput:
    return WeeklyOutput(
        as_of_date=as_of,
        srd_version="8.7.1",
        mode="NORMAL",
        vintage_mode="strict",
        state=WeeklyState(
            post=np.array([0.25, 0.50, 0.25], dtype=np.float64),
            state_name="NEUTRAL",
            dwell_weeks=1,
            hazard_covariate=0.0,
        ),
        distribution=DistributionOutput(
            q05=-0.05,
            q10=-0.03,
            q25=-0.01,
            q50=0.0,
            q75=0.01,
            q90=0.03,
            q95=0.05,
            q05_ci_low=-0.08,
            q05_ci_high=-0.02,
            q95_ci_low=0.02,
            q95_ci_high=0.08,
            mu_hat=0.0,
            sigma_hat=0.05,
            p_loss=0.4,
            es20=0.02,
        ),
        decision=DecisionOutput(
            excess_return=0.0,
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


def test_run_backtest_job_writes_acceptance_report_next_to_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fetch_series(_as_of: date, _vintage_mode: str) -> Mapping[str, TimeSeries]:
        start = date(2024, 1, 5)
        timestamps = np.array(
            [(start + timedelta(weeks=idx)).isoformat() for idx in range(60)],
            dtype="datetime64[D]",
        )
        return {
            "NASDAQXNDX": TimeSeries(
                series_id="NASDAQXNDX",
                timestamps=timestamps,
                values=np.ones(60, dtype=np.float64),
                is_pseudo_pit=False,
            ),
            "DGS10": TimeSeries(
                series_id="DGS10",
                timestamps=timestamps,
                values=np.ones(60, dtype=np.float64),
                is_pseudo_pit=False,
            ),
        }

    def fit_training_artifacts(
        _as_of: date,
        _series: Mapping[str, TimeSeries],
        _cfg: FrozenConfig,
        _feature_cache: Any = None,
    ) -> TrainingArtifacts:
        return TrainingArtifacts(
            utility_zstats=None,
            offense_thresholds=None,
            train_distributions={},
            state_label_map={0: "DEFENSIVE", 1: "NEUTRAL", 2: "OFFENSIVE"},
        )

    def infer_weekly(
        as_of: date,
        _cfg: FrozenConfig,
        _series: Mapping[str, TimeSeries],
        _training_artifacts: TrainingArtifacts,
        _feature_cache: Any = None,
    ) -> WeeklyOutput:
        return _output(as_of)

    monkeypatch.setattr(
        "inference.train.compute_effective_strict_acceptance_start_from_series",
        lambda *_args, **_kwargs: date(2024, 1, 5),
    )
    monkeypatch.setattr(
        "app.backtest_runner.compute_effective_strict_acceptance_start_from_series",
        lambda *_args, **_kwargs: date(2024, 1, 5),
    )
    monkeypatch.setattr(
        "inference.train.build_feature_block",
        lambda _series, _week: (
            np.zeros(10, dtype=np.float64),
            np.zeros(10, dtype=np.bool_),
        ),
    )
    monkeypatch.setattr(
        "app.backtest_runner.build_feature_block",
        lambda _series, _week: (
            np.zeros(10, dtype=np.float64),
            np.zeros(10, dtype=np.bool_),
        ),
    )

    output_path = tmp_path / "backtest" / "backtest_results.jsonl"
    code = run_backtest_job(
        start=date(2024, 1, 5),
        end=date(2024, 1, 12),
        cfg=_config(),
        output_path=output_path,
        deps=BacktestRunnerDeps(
            fetch_series=fetch_series,
            fit_training_artifacts=fit_training_artifacts,
            infer_weekly=infer_weekly,
            write_result=write_backtest_jsonl,
        ),
    )

    report = json.loads((tmp_path / "backtest" / "acceptance_report.json").read_text())
    metadata = json.loads((tmp_path / "backtest" / "backtest_metadata.json").read_text())
    assert code in {0, 3}
    assert output_path.exists()
    assert report["items"][0]["name"] == "bit_identical_determinism"
    assert metadata["actual_start"] == "2024-01-05"
    assert metadata["effective_strict_start"] == "2024-01-05"


def test_run_backtest_job_clips_start_to_effective_strict_week(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_as_of: list[date] = []

    def fetch_series(_as_of: date, _vintage_mode: str) -> Mapping[str, TimeSeries]:
        start = date(2024, 1, 5)
        timestamps = np.array(
            [(start + timedelta(weeks=idx)).isoformat() for idx in range(10)],
            dtype="datetime64[D]",
        )
        return {
            "NASDAQXNDX": TimeSeries(
                series_id="NASDAQXNDX",
                timestamps=timestamps,
                values=np.ones(10, dtype=np.float64),
                is_pseudo_pit=False,
            ),
            "DGS10": TimeSeries(
                series_id="DGS10",
                timestamps=timestamps,
                values=np.ones(10, dtype=np.float64),
                is_pseudo_pit=False,
            ),
        }

    def fit_training_artifacts(
        _as_of: date,
        _series: Mapping[str, TimeSeries],
        _cfg: FrozenConfig,
        _feature_cache: Any = None,
    ) -> TrainingArtifacts:
        return TrainingArtifacts(
            utility_zstats=None,
            offense_thresholds=None,
            train_distributions={},
            state_label_map={0: "DEFENSIVE", 1: "NEUTRAL", 2: "OFFENSIVE"},
        )

    def infer_weekly(
        as_of: date,
        _cfg: FrozenConfig,
        _series: Mapping[str, TimeSeries],
        _training_artifacts: TrainingArtifacts,
        _feature_cache: Any = None,
    ) -> WeeklyOutput:
        seen_as_of.append(as_of)
        return _output(as_of)

    monkeypatch.setattr(
        "app.backtest_runner.compute_effective_strict_acceptance_start_from_series",
        lambda *_args, **_kwargs: date(2024, 1, 19),
    )
    monkeypatch.setattr(
        "app.backtest_runner.build_feature_block",
        lambda _series, _week: (
            np.zeros(10, dtype=np.float64),
            np.zeros(10, dtype=np.bool_),
        ),
    )

    code = run_backtest_job(
        start=date(2024, 1, 5),
        end=date(2024, 1, 26),
        cfg=_config(),
        output_path=tmp_path / "backtest" / "backtest_results.jsonl",
        deps=BacktestRunnerDeps(
            fetch_series=fetch_series,
            fit_training_artifacts=fit_training_artifacts,
            infer_weekly=infer_weekly,
            write_result=write_backtest_jsonl,
        ),
    )

    metadata = json.loads((tmp_path / "backtest" / "backtest_metadata.json").read_text())
    assert code in {0, 3}
    assert seen_as_of == [date(2024, 1, 19), date(2024, 1, 26)]
    assert metadata["actual_start"] == "2024-01-19"
    assert metadata["effective_strict_start"] == "2024-01-19"


def test_run_backtest_job_preserves_output_order_with_process_workers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fetch_series(_as_of: date, _vintage_mode: str) -> Mapping[str, TimeSeries]:
        start = date(2024, 1, 5)
        timestamps = np.array(
            [(start + timedelta(weeks=idx)).isoformat() for idx in range(8)],
            dtype="datetime64[D]",
        )
        return {
            "NASDAQXNDX": TimeSeries(
                series_id="NASDAQXNDX",
                timestamps=timestamps,
                values=np.ones(8, dtype=np.float64),
                is_pseudo_pit=False,
            ),
            "DGS10": TimeSeries(
                series_id="DGS10",
                timestamps=timestamps,
                values=np.ones(8, dtype=np.float64),
                is_pseudo_pit=False,
            ),
        }

    def fit_training_artifacts(
        _as_of: date,
        _series: Mapping[str, TimeSeries],
        _cfg: FrozenConfig,
        _feature_cache: Any = None,
    ) -> TrainingArtifacts:
        return TrainingArtifacts(
            utility_zstats=None,
            offense_thresholds=None,
            train_distributions={},
            state_label_map={0: "DEFENSIVE", 1: "NEUTRAL", 2: "OFFENSIVE"},
        )

    def infer_weekly(
        as_of: date,
        _cfg: FrozenConfig,
        _series: Mapping[str, TimeSeries],
        _training_artifacts: TrainingArtifacts,
        _feature_cache: Any = None,
    ) -> WeeklyOutput:
        if as_of == date(2024, 1, 12):
            time.sleep(0.02)
        return _output(as_of)

    monkeypatch.setattr(
        "app.backtest_runner.compute_effective_strict_acceptance_start_from_series",
        lambda *_args, **_kwargs: date(2024, 1, 5),
    )
    monkeypatch.setattr(
        "app.backtest_runner.build_feature_block",
        lambda _series, _week: (
            np.zeros(10, dtype=np.float64),
            np.zeros(10, dtype=np.bool_),
        ),
    )

    output_path = tmp_path / "backtest" / "backtest_results.jsonl"
    code = run_backtest_job(
        start=date(2024, 1, 5),
        end=date(2024, 1, 19),
        cfg=_config(),
        output_path=output_path,
        deps=BacktestRunnerDeps(
            fetch_series=fetch_series,
            fit_training_artifacts=fit_training_artifacts,
            infer_weekly=infer_weekly,
            write_result=write_backtest_jsonl,
            max_workers=2,
        ),
    )

    lines = output_path.read_text(encoding="utf-8").splitlines()
    payloads = [json.loads(line) for line in lines]
    assert code in {0, 3}
    assert [payload["as_of_date"] for payload in payloads] == [
        "2024-01-05",
        "2024-01-12",
        "2024-01-19",
    ]
