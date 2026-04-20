from __future__ import annotations

import json
from pathlib import Path

from app.training_artifacts import load_training_artifacts


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

    artifacts = load_training_artifacts(root)

    assert artifacts.utility_zstats is not None
    assert artifacts.offense_thresholds is not None
    assert artifacts.qr_coefs is not None
    assert artifacts.state_label_map[0] == "DEFENSIVE"
    assert artifacts.train_distributions["x1"].tolist() == [1.0, 2.0]
