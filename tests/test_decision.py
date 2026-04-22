from __future__ import annotations

import numpy as np

from decision.cycle_position import cycle_position
from decision.hysteresis import apply_band
from decision.offense_abs import OffenseThresholds, offense_raw, stance_from_offense
from decision.utility import UtilityZStats, excess_return, utility


def test_utility_uses_training_frozen_zstats() -> None:
    zstats = UtilityZStats(
        er_med=0.02,
        er_mad=0.01,
        es20_med=0.10,
        es20_mad=0.02,
        ploss_med=0.40,
        ploss_mad=0.10,
    )

    value = utility(er=0.04, es20=0.08, p_loss=0.30, zstats=zstats)

    assert value > 0.0


def test_excess_return_is_simple_dgs1_difference() -> None:
    assert excess_return(mu_hat=0.08, dgs1=0.03) == 0.05


def test_offense_raw_maps_absolute_threshold_segments_and_clips() -> None:
    th = OffenseThresholds(u_q0=-2.0, u_q20=-1.0, u_q40=0.0, u_q60=1.0, u_q80=2.0, u_q100=3.0)

    assert offense_raw(-1.5, th) == 10.0
    assert offense_raw(0.5, th) == 50.0
    assert offense_raw(2.5, th) == 90.0
    assert offense_raw(-10.0, th) == 0.0
    assert offense_raw(10.0, th) == 100.0


def test_apply_band_preserves_previous_score_inside_band() -> None:
    assert apply_band(offense_raw_t=54.0, offense_final_prev=50.0, band=7.0) == 50.0
    assert apply_band(offense_raw_t=60.0, offense_final_prev=50.0, band=7.0) == 60.0


def test_stance_from_offense_matches_srd_cutoffs() -> None:
    assert stance_from_offense(35.0) == "DEFENSIVE"
    assert stance_from_offense(50.0) == "NEUTRAL"
    assert stance_from_offense(65.0) == "OFFENSIVE"


def test_cycle_position_averages_credit_vrp_and_inverted_slope_ranks() -> None:
    train_dist = {
        "x5": np.array([1.0, 2.0, 3.0], dtype=np.float64),
        "x9": np.array([0.1, 0.2, 0.3], dtype=np.float64),
        "x1": np.array([-1.0, 0.0, 1.0], dtype=np.float64),
    }

    assert cycle_position(x5_t=2.0, x9_t=0.2, x1_t=0.0, train_dist=train_dist) == 50.0


def test_cycle_position_uses_neutral_rank_for_missing_input() -> None:
    train_dist = {
        "x5": np.array([1.0, 2.0, 3.0], dtype=np.float64),
        "x9": np.array([0.1, 0.2, 0.3], dtype=np.float64),
        "x1": np.array([-1.0, 0.0, 1.0], dtype=np.float64),
    }

    assert cycle_position(x5_t=float("nan"), x9_t=0.2, x1_t=0.0, train_dist=train_dist) == 50.0
