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

