from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from decision.offense_abs import OffenseThresholds
from decision.utility import UtilityZStats
from engine_types import TimeSeries
from inference.weekly import TrainingArtifacts, run_weekly
from law.linear_quantiles import QRCoefs


def _series(series_id: str, start: date, values: list[float]) -> TimeSeries:
    timestamps = np.array(
        [(start + timedelta(weeks=idx)).isoformat() for idx in range(len(values))],
        dtype="datetime64[D]",
    )
    return TimeSeries(
        series_id=series_id,
        timestamps=timestamps,
        values=np.array(values, dtype=np.float64),
        is_pseudo_pit=False,
    )


def _series_map(as_of: date) -> dict[str, TimeSeries]:
    start = as_of - timedelta(weeks=30)
    base = [10.0 + idx for idx in range(31)]
    return {
        "DGS10": _series("DGS10", start, base),
        "DGS2": _series("DGS2", start, [value - 1.0 for value in base]),
        "DGS1": _series("DGS1", start, [0.02] * 31),
        "EFFR": _series("EFFR", start, [0.01 + idx * 0.001 for idx in range(31)]),
        "BAA10Y": _series("BAA10Y", start, [0.03 + idx * 0.001 for idx in range(31)]),
        "WALCL": _series("WALCL", start, [100.0 + idx for idx in range(31)]),
        "VXNCLS": _series("VXNCLS", start, [20.0 + idx * 0.1 for idx in range(31)]),
        "RV20_NDX": _series("RV20_NDX", start, [10.0 + idx * 0.1 for idx in range(31)]),
        "VIXCLS": _series("VIXCLS", start, [15.0 + idx * 0.1 for idx in range(31)]),
        "VXVCLS": _series("VXVCLS", start, [12.0 + idx * 0.1 for idx in range(31)]),
    }


def test_run_weekly_assembles_output_from_training_artifacts() -> None:
    as_of = date(2024, 12, 27)
    artifacts = TrainingArtifacts(
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
    )

    output = run_weekly(as_of, "strict", _series_map(as_of), artifacts)

    assert output.as_of_date == as_of
    assert output.mode == "DEGRADED"
    assert output.diagnostics.hmm_status == "degenerate"
    assert output.distribution.q05 <= output.distribution.q10
    assert output.decision.stance in {"DEFENSIVE", "NEUTRAL", "OFFENSIVE"}
