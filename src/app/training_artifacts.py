"""Shell loader for persisted training artifacts."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, cast

import numpy as np

from decision.offense_abs import OffenseThresholds
from decision.utility import UtilityZStats
from engine_types import Stance
from inference.weekly import TrainingArtifacts
from law.linear_quantiles import QRCoefs
from state.ti_hmm_single import HMMModel


def _read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    path.write_text(text + "\n", encoding="utf-8")


def load_training_artifacts(root: Path = Path("artifacts/training")) -> TrainingArtifacts:
    """io: Load persisted training artifacts into frozen in-memory contracts."""
    zstats_raw = _read_json(root / "utility_zstats.json")
    thresholds_raw = _read_json(root / "offense_thresholds.json")
    label_raw = _read_json(root / "state_label_map.json")
    distributions_raw = _read_json(root / "train_distributions.json")
    qr_raw = _read_json(root / "qr_coefs.json")
    hmm_raw = _read_json(root / "hmm_model.json") if (root / "hmm_model.json").exists() else None
    return TrainingArtifacts(
        utility_zstats=UtilityZStats(**zstats_raw),
        offense_thresholds=OffenseThresholds(**thresholds_raw),
        train_distributions={
            key: np.asarray(value, dtype=np.float64) for key, value in distributions_raw.items()
        },
        state_label_map={int(key): cast(Stance, value) for key, value in label_raw.items()},
        qr_coefs=QRCoefs(
            a=np.asarray(qr_raw["a"], dtype=np.float64),
            b=np.asarray(qr_raw["b"], dtype=np.float64),
            c=np.asarray(qr_raw["c"], dtype=np.float64),
            solver_status=str(qr_raw["solver_status"]),
        ),
        hmm_model=(
            None
            if hmm_raw is None
            else HMMModel(
                transition_coefs=np.asarray(hmm_raw["transition_coefs"], dtype=np.float64),
                emission_mean=np.asarray(hmm_raw["emission_mean"], dtype=np.float64),
                emission_cov=np.asarray(hmm_raw["emission_cov"], dtype=np.float64),
                label_map={
                    int(key): cast(Stance, value) for key, value in hmm_raw["label_map"].items()
                },
                log_likelihood=float(hmm_raw["log_likelihood"]),
            )
        ),
    )


def write_training_artifacts(
    artifacts: TrainingArtifacts,
    root: Path = Path("artifacts/training"),
) -> None:
    """io: Persist frozen training artifacts as deterministic JSON files."""
    if (
        artifacts.utility_zstats is None
        or artifacts.offense_thresholds is None
        or artifacts.qr_coefs is None
    ):
        msg = "cannot persist incomplete training artifacts"
        raise ValueError(msg)
    _write_json(root / "utility_zstats.json", asdict(artifacts.utility_zstats))
    _write_json(root / "offense_thresholds.json", asdict(artifacts.offense_thresholds))
    _write_json(
        root / "state_label_map.json",
        {str(key): artifacts.state_label_map[key] for key in sorted(artifacts.state_label_map)},
    )
    _write_json(
        root / "train_distributions.json",
        {
            key: [float(value) for value in artifacts.train_distributions[key].tolist()]
            for key in sorted(artifacts.train_distributions)
        },
    )
    _write_json(
        root / "qr_coefs.json",
        {
            "a": [float(value) for value in artifacts.qr_coefs.a.tolist()],
            "b": [[float(value) for value in row] for row in artifacts.qr_coefs.b.tolist()],
            "c": [[float(value) for value in row] for row in artifacts.qr_coefs.c.tolist()],
            "solver_status": artifacts.qr_coefs.solver_status,
        },
    )
    hmm_path = root / "hmm_model.json"
    if artifacts.hmm_model is None:
        if hmm_path.exists():
            hmm_path.unlink()
        return
    _write_json(
        hmm_path,
        {
            "emission_cov": [
                [[float(value) for value in row] for row in state]
                for state in artifacts.hmm_model.emission_cov.tolist()
            ],
            "emission_mean": [
                [float(value) for value in row]
                for row in artifacts.hmm_model.emission_mean.tolist()
            ],
            "label_map": {
                str(key): artifacts.hmm_model.label_map[key]
                for key in sorted(artifacts.hmm_model.label_map)
            },
            "log_likelihood": artifacts.hmm_model.log_likelihood,
            "transition_coefs": [
                [float(value) for value in row]
                for row in artifacts.hmm_model.transition_coefs.tolist()
            ],
        },
    )
