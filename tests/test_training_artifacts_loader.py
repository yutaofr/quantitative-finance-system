from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from app.training_artifacts import load_training_artifacts, write_training_artifacts
from decision.offense_abs import OffenseThresholds
from decision.utility import UtilityZStats
from inference.weekly import TrainingArtifacts
from law.linear_quantiles import QRCoefs


def test_load_training_artifacts_reads_json_arrays(tmp_path: Path) -> None:
    root = tmp_path / "training"
    root.mkdir()
    (root / "utility_zstats.json").write_text(
        json.dumps(
            {
                "er_med": 0.0,
                "er_mad": 1.0,
                "es20_med": 0.0,
                "es20_mad": 1.0,
                "ploss_med": 0.5,
                "ploss_mad": 0.1,
            },
        ),
        encoding="utf-8",
    )
    (root / "offense_thresholds.json").write_text(
        json.dumps({"u_q0": -2, "u_q20": -1, "u_q40": 0, "u_q60": 1, "u_q80": 2, "u_q100": 3}),
        encoding="utf-8",
    )
    (root / "state_label_map.json").write_text(
        json.dumps({"0": "DEFENSIVE", "1": "NEUTRAL", "2": "OFFENSIVE"}),
        encoding="utf-8",
    )
    (root / "train_distributions.json").write_text(
        json.dumps({"x1": [1.0, 2.0], "x5": [0.1, 0.2], "x9": [0.3, 0.4]}),
        encoding="utf-8",
    )
    (root / "qr_coefs.json").write_text(
        json.dumps(
            {
                "a": [-0.1, -0.05, 0.0, 0.05, 0.1],
                "b": [[0.0] * 10] * 5,
                "c": [[0.0, 0.0, 0.0]] * 5,
                "solver_status": "ok",
            },
        ),
        encoding="utf-8",
    )
    eye6 = [[1.0 if row == col else 0.0 for col in range(6)] for row in range(6)]
    (root / "hmm_model.json").write_text(
        json.dumps(
            {
                "transition_coefs": [[0.0, 0.0, 0.0]] * 3,
                "emission_mean": [[0.0] * 6, [1.0] * 6, [2.0] * 6],
                "emission_cov": [eye6] * 3,
                "label_map": {"0": "DEFENSIVE", "1": "NEUTRAL", "2": "OFFENSIVE"},
                "log_likelihood": -12.5,
            },
        ),
        encoding="utf-8",
    )

    artifacts = load_training_artifacts(root)

    assert artifacts.utility_zstats is not None
    assert artifacts.offense_thresholds is not None
    assert artifacts.qr_coefs is not None
    assert artifacts.hmm_model is not None
    assert artifacts.state_label_map[0] == "DEFENSIVE"
    assert artifacts.train_distributions["x1"].tolist() == [1.0, 2.0]


def test_write_training_artifacts_skips_stale_hmm_when_model_degrades(tmp_path: Path) -> None:
    root = tmp_path / "training"
    root.mkdir()
    (root / "hmm_model.json").write_text("{}", encoding="utf-8")
    artifacts = TrainingArtifacts(
        utility_zstats=UtilityZStats(
            er_med=0.0,
            er_mad=1.0,
            es20_med=0.0,
            es20_mad=1.0,
            ploss_med=0.5,
            ploss_mad=0.1,
        ),
        offense_thresholds=OffenseThresholds(
            u_q0=-2.0,
            u_q20=-1.0,
            u_q40=0.0,
            u_q60=1.0,
            u_q80=2.0,
            u_q100=3.0,
        ),
        train_distributions={
            "x1": np.array([1.0, 2.0], dtype=np.float64),
            "x5": np.array([0.1, 0.2], dtype=np.float64),
            "x9": np.array([0.3, 0.4], dtype=np.float64),
        },
        state_label_map={0: "DEFENSIVE", 1: "NEUTRAL", 2: "OFFENSIVE"},
        qr_coefs=QRCoefs(
            a=np.array([-0.1, -0.05, 0.0, 0.05, 0.1], dtype=np.float64),
            b=np.zeros((5, 10), dtype=np.float64),
            c=np.zeros((5, 3), dtype=np.float64),
            solver_status="ok",
        ),
        hmm_model=None,
    )

    write_training_artifacts(artifacts, root)
    reloaded = load_training_artifacts(root)

    assert not (root / "hmm_model.json").exists()
    assert reloaded.hmm_model is None
    assert (root / "state_label_map.json").read_text(encoding="utf-8") == (
        '{"0":"DEFENSIVE","1":"NEUTRAL","2":"OFFENSIVE"}\n'
    )
