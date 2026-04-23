from __future__ import annotations

from collections.abc import Mapping
from datetime import date

import numpy as np
import pytest

from backtest.walkforward import run_walkforward
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
        bootstrap_replications=2000,
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
            q05=-0.1,
            q10=-0.05,
            q25=-0.01,
            q50=0.03,
            q75=0.08,
            q90=0.12,
            q95=0.15,
            q05_ci_low=-0.2,
            q05_ci_high=-0.1,
            q95_ci_low=0.1,
            q95_ci_high=0.2,
            mu_hat=0.04,
            sigma_hat=0.1,
            p_loss=0.4,
            es20=0.08,
        ),
        decision=DecisionOutput(
            excess_return=0.02,
            utility=0.7,
            offense_raw=55.0,
            offense_final=55.0,
            stance="NEUTRAL",
            cycle_position=50.0,
        ),
        diagnostics=DiagnosticsOutput(
            missing_rate=0.0,
            quantile_solver_status="ok",
            tail_extrapolation_status="ok",
            hmm_status="ok",
            coverage_q10_trailing_104w=0.8,
            coverage_q90_trailing_104w=0.8,
        ),
    )


def _fit_training_artifacts(
    _as_of: date,
    history: Mapping[str, TimeSeries],
    _cfg: FrozenConfig,
    *,
    seen_fit_max: list[np.datetime64],
) -> TrainingArtifacts:
    seen_fit_max.append(history["DGS10"].timestamps.max())
    return TrainingArtifacts(
        utility_zstats=None,
        offense_thresholds=None,
        train_distributions={},
        state_label_map={0: "DEFENSIVE", 1: "NEUTRAL", 2: "OFFENSIVE"},
    )


def _infer_weekly(
    as_of: date,
    _cfg: FrozenConfig,
    history: Mapping[str, TimeSeries],
    _training_artifacts: TrainingArtifacts,
    *,
    seen_infer_max: list[np.datetime64],
) -> WeeklyOutput:
    seen_infer_max.append(history["DGS10"].timestamps.max())
    return _output(as_of)


def test_run_walkforward_only_exposes_history_up_to_each_week() -> None:
    timestamps = np.array(
        ["2024-01-05", "2024-01-12", "2024-01-19"],
        dtype="datetime64[D]",
    )
    series = {
        "DGS10": TimeSeries(
            series_id="DGS10",
            timestamps=timestamps,
            values=np.array([1.0, 2.0, 3.0], dtype=np.float64),
            is_pseudo_pit=False,
        ),
    }
    seen_fit_max: list[np.datetime64] = []
    seen_infer_max: list[np.datetime64] = []

    def fit_with_cache(
        as_of: date,
        history: Mapping[str, TimeSeries],
        cfg: FrozenConfig,
        _feature_cache: object = None,
    ) -> TrainingArtifacts:
        return _fit_training_artifacts(
            as_of,
            history,
            cfg,
            seen_fit_max=seen_fit_max,
        )

    def infer_with_cache(
        as_of: date,
        cfg: FrozenConfig,
        history: Mapping[str, TimeSeries],
        training_artifacts: TrainingArtifacts,
        _feature_cache: object = None,
    ) -> WeeklyOutput:
        return _infer_weekly(
            as_of,
            cfg,
            history,
            training_artifacts,
            seen_infer_max=seen_infer_max,
        )

    result = run_walkforward(
        start=date(2024, 1, 5),
        end=date(2024, 1, 12),
        series=series,
        cfg=_config(),
        fit_training_artifacts=fit_with_cache,
        infer_weekly=infer_with_cache,
    )

    assert [output.as_of_date for output in result.outputs] == [
        date(2024, 1, 5),
        date(2024, 1, 12),
    ]
    assert seen_fit_max == [np.datetime64("2024-01-05"), np.datetime64("2024-01-12")]
    assert seen_infer_max == [np.datetime64("2024-01-05"), np.datetime64("2024-01-12")]


def test_run_walkforward_raises_when_strict_start_precedes_effective_start() -> None:
    timestamps = np.array(["2024-01-05", "2024-01-12"], dtype="datetime64[D]")
    series = {
        "DGS10": TimeSeries(
            series_id="DGS10",
            timestamps=timestamps,
            values=np.array([1.0, 2.0], dtype=np.float64),
            is_pseudo_pit=False,
        ),
    }

    with pytest.raises(ValueError, match="effective strict start"):
        run_walkforward(
            start=date(2024, 1, 5),
            end=date(2024, 1, 12),
            series=series,
            cfg=_config(),
            fit_training_artifacts=lambda *_args, **_kwargs: TrainingArtifacts(
                utility_zstats=None,
                offense_thresholds=None,
                train_distributions={},
                state_label_map={0: "DEFENSIVE", 1: "NEUTRAL", 2: "OFFENSIVE"},
            ),
            infer_weekly=lambda *_args, **_kwargs: _output(date(2024, 1, 5)),
            effective_strict_start=date(2024, 1, 12),
            strict_start_policy="raise",
        )
