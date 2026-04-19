"""Shared frozen data contracts for the production engine."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import Literal

import numpy as np
from numpy.typing import NDArray

VintageMode = Literal["strict", "pseudo"]
Mode = Literal["NORMAL", "DEGRADED", "BLOCKED"]
Stance = Literal["DEFENSIVE", "NEUTRAL", "OFFENSIVE"]


@dataclass(frozen=True, slots=True)
class SeriesPITRequest:
    """pure. Request metadata for point-in-time series access."""

    series_id: str
    as_of: date
    vintage_mode: VintageMode


@dataclass(frozen=True, slots=True)
class TimeSeries:
    """pure. Immutable numeric time series returned by data adapters."""

    series_id: str
    timestamps: NDArray[np.datetime64]
    values: NDArray[np.float64]
    is_pseudo_pit: bool


@dataclass(frozen=True, slots=True)
class FeatureFrame:
    """pure. Scaled production feature matrix and missing-data mask."""

    as_of: date
    feature_names: tuple[str, ...]
    x_raw: NDArray[np.float64]
    x_scaled: NDArray[np.float64]
    missing_mask: NDArray[np.bool_]
    metadata: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class WeeklyState:
    """pure. Weekly state payload in the SRD v8.7 output contract."""

    post: NDArray[np.float64]
    state_name: Stance
    dwell_weeks: int
    hazard_covariate: float


@dataclass(frozen=True, slots=True)
class DistributionOutput:
    """pure. Seven-point law plus tail intervals and implied moments."""

    q05: float
    q10: float
    q25: float
    q50: float
    q75: float
    q90: float
    q95: float
    q05_ci_low: float
    q05_ci_high: float
    q95_ci_low: float
    q95_ci_high: float
    mu_hat: float
    sigma_hat: float
    p_loss: float
    es20: float


@dataclass(frozen=True, slots=True)
class DecisionOutput:
    """pure. Final decision payload from SRD sections 9.2-9.7."""

    excess_return: float
    utility: float
    offense_raw: float
    offense_final: float
    stance: Stance
    cycle_position: float


@dataclass(frozen=True, slots=True)
class DiagnosticsOutput:
    """pure. Output diagnostics required by SRD v8.7 section 11."""

    missing_rate: float
    quantile_solver_status: str
    tail_extrapolation_status: str
    hmm_status: str
    coverage_q10_trailing_104w: float
    coverage_q90_trailing_104w: float


@dataclass(frozen=True, slots=True)
class WeeklyOutput:
    """pure. Frozen weekly production output before shell serialization."""

    as_of_date: date
    srd_version: Literal["8.7"]
    mode: Mode
    vintage_mode: VintageMode
    state: WeeklyState
    distribution: DistributionOutput
    decision: DecisionOutput
    diagnostics: DiagnosticsOutput
