from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from pathlib import Path

import numpy as np

from app.cli import ExitCode
from app.weekly_runner import WeeklyRunnerDeps, run_weekly_job
from config_types import FrozenConfig
from engine_types import (
    DecisionOutput,
    DiagnosticsOutput,
    DistributionOutput,
    TimeSeries,
    WeeklyOutput,
    WeeklyState,
)
from errors import HMMConvergenceError, QuantileSolverError
from inference.weekly import TrainingArtifacts


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


def _artifacts() -> TrainingArtifacts:
    return TrainingArtifacts(
        utility_zstats=None,
        offense_thresholds=None,
        train_distributions={},
        state_label_map={0: "DEFENSIVE", 1: "NEUTRAL", 2: "OFFENSIVE"},
    )


def _series_map() -> dict[str, TimeSeries]:
    timestamps = np.array(["2024-12-20", "2024-12-27"], dtype="datetime64[D]")
    values = np.array([1.0, 2.0], dtype=np.float64)
    return {
        "DGS10": TimeSeries(
            series_id="DGS10",
            timestamps=timestamps,
            values=values,
            is_pseudo_pit=False,
        ),
    }


def _normal_output(as_of: date) -> WeeklyOutput:
    return WeeklyOutput(
        as_of_date=as_of,
        srd_version="8.7",
        mode="NORMAL",
        vintage_mode="strict",
        state=WeeklyState(
            post=np.array([0.25, 0.50, 0.25], dtype=np.float64),
            state_name="NEUTRAL",
            dwell_weeks=3,
            hazard_covariate=0.2,
        ),
        distribution=DistributionOutput(
            q05=-0.10,
            q10=-0.05,
            q25=-0.01,
            q50=0.03,
            q75=0.08,
            q90=0.12,
            q95=0.16,
            q05_ci_low=-0.15,
            q05_ci_high=-0.09,
            q95_ci_low=0.10,
            q95_ci_high=0.20,
            mu_hat=0.04,
            sigma_hat=0.10,
            p_loss=0.40,
            es20=0.08,
        ),
        decision=DecisionOutput(
            excess_return=0.02,
            utility=0.70,
            offense_raw=55.0,
            offense_final=55.0,
            stance="NEUTRAL",
            cycle_position=47.0,
        ),
        diagnostics=DiagnosticsOutput(
            missing_rate=0.0,
            quantile_solver_status="ok",
            tail_extrapolation_status="ok",
            hmm_status="ok",
            coverage_q10_trailing_104w=0.87,
            coverage_q90_trailing_104w=0.89,
        ),
    )


def _fetch_series(_as_of: date, _vintage_mode: str) -> dict[str, TimeSeries]:
    return _series_map()


def _write_capture(captured: list[WeeklyOutput], output: WeeklyOutput, _path: Path) -> None:
    captured.append(output)


def _infer_success(
    as_of: date,
    _cfg: FrozenConfig,
    _series: Mapping[str, TimeSeries],
    _training_artifacts: TrainingArtifacts,
) -> WeeklyOutput:
    return _normal_output(as_of)


def _raise_hmm(
    _as_of: date,
    _cfg: FrozenConfig,
    _series: Mapping[str, TimeSeries],
    _training_artifacts: TrainingArtifacts,
) -> WeeklyOutput:
    raise HMMConvergenceError("em failed")


def _raise_quantile(
    _as_of: date,
    _cfg: FrozenConfig,
    _series: Mapping[str, TimeSeries],
    _training_artifacts: TrainingArtifacts,
) -> WeeklyOutput:
    raise QuantileSolverError("solver failed")


def test_run_weekly_job_writes_success_output(tmp_path: Path) -> None:
    captured: list[WeeklyOutput] = []

    deps = WeeklyRunnerDeps(
        fetch_series=_fetch_series,
        load_training_artifacts=lambda: _artifacts(),
        infer_weekly=_infer_success,
        write_output=lambda output, path: _write_capture(captured, output, path),
    )

    code = run_weekly_job(
        as_of=date(2024, 12, 27),
        vintage_mode="strict",
        cfg=_config(),
        output_path=tmp_path / "production_output.json",
        deps=deps,
    )

    assert code == int(ExitCode.OK)
    assert captured[-1].mode == "NORMAL"


def test_run_weekly_job_degrades_hmm_convergence_failures(tmp_path: Path) -> None:
    captured: list[WeeklyOutput] = []

    deps = WeeklyRunnerDeps(
        fetch_series=_fetch_series,
        load_training_artifacts=lambda: _artifacts(),
        infer_weekly=_raise_hmm,
        write_output=lambda output, path: _write_capture(captured, output, path),
    )

    code = run_weekly_job(
        as_of=date(2024, 12, 27),
        vintage_mode="strict",
        cfg=_config(),
        output_path=tmp_path / "production_output.json",
        deps=deps,
    )

    assert code == int(ExitCode.OK)
    assert captured[-1].mode == "DEGRADED"
    assert captured[-1].diagnostics.hmm_status == "em_nonconverge"
    assert captured[-1].decision.offense_final == 50.0


def test_run_weekly_job_blocks_quantile_solver_failures(tmp_path: Path) -> None:
    captured: list[WeeklyOutput] = []

    deps = WeeklyRunnerDeps(
        fetch_series=_fetch_series,
        load_training_artifacts=lambda: _artifacts(),
        infer_weekly=_raise_quantile,
        write_output=lambda output, path: _write_capture(captured, output, path),
    )

    code = run_weekly_job(
        as_of=date(2024, 12, 27),
        vintage_mode="strict",
        cfg=_config(),
        output_path=tmp_path / "production_output.json",
        deps=deps,
    )

    assert code == int(ExitCode.BLOCKED)
    assert captured[-1].mode == "BLOCKED"
    assert captured[-1].decision.stance == "NEUTRAL"
    assert captured[-1].diagnostics.quantile_solver_status == "failed"
