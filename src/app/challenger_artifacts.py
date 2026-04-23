"""Deterministic shell serialization for challenger shadow artifacts."""

from __future__ import annotations

from datetime import date
import json
from pathlib import Path
from typing import Any

import numpy as np

from law.student_t_location_scale import StudentTFitResult


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    path.write_text(text + "\n", encoding="utf-8")


def challenger_fit_artifact_to_dict(
    *,
    as_of: date,
    train_end: date,
    fit_result: StudentTFitResult,
) -> dict[str, Any]:
    """io: Convert one challenger fit result into deterministic JSON."""
    return {
        "as_of_date": as_of.isoformat(),
        "train_end": train_end.isoformat(),
        "objective_value": fit_result.objective_value,
        "optimization_failed": fit_result.optimization_failed,
        "fallback_used": fit_result.fallback_used,
        "optimizer_status": fit_result.optimizer_status,
        "nu": fit_result.params.nu,
        "beta_mu": [float(value) for value in fit_result.params.beta_mu.tolist()],
        "beta_sigma": [float(value) for value in fit_result.params.beta_sigma.tolist()],
        "train_rows": fit_result.train_rows,
    }


def challenger_output_to_dict(  # noqa: PLR0913
    *,
    as_of: date,
    status: str,
    quantiles: np.ndarray | None,
    fit_status: str,
    optimization_failed: bool,
    source_offense_final: float,
) -> dict[str, Any]:
    """io: Convert one challenger weekly shadow result into deterministic JSON."""
    distribution = None
    if quantiles is not None:
        distribution = {
            "q05": float(quantiles[0]),
            "q10": float(quantiles[1]),
            "q25": float(quantiles[2]),
            "q50": float(quantiles[3]),
            "q75": float(quantiles[4]),
            "q90": float(quantiles[5]),
            "q95": float(quantiles[6]),
        }
    return {
        "as_of_date": as_of.isoformat(),
        "distribution": distribution,
        "fit_status": fit_status,
        "optimization_failed": optimization_failed,
        "source_offense_final": source_offense_final,
        "status": status,
    }


def write_challenger_fit_artifact(path: Path, payload: dict[str, Any]) -> None:
    """io: Persist deterministic challenger fit artifact JSON."""
    _write_json(path, payload)


def write_challenger_output(path: Path, payload: dict[str, Any]) -> None:
    """io: Persist deterministic challenger output JSON."""
    _write_json(path, payload)


def write_challenger_report(path: Path, payload: dict[str, Any]) -> None:
    """io: Persist deterministic challenger comparison report JSON."""
    _write_json(path, payload)
