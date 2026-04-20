"""Shell loader for persisted training artifacts."""

from __future__ import annotations

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
