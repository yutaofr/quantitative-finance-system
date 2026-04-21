"""Frozen shell-visible configuration contracts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

SRD_VERSION: Literal["8.7.1"] = "8.7.1"


@dataclass(frozen=True, slots=True)
class FrozenConfig:
    """pure. Frozen runtime config assembled by the app shell."""

    srd_version: Literal["8.7.1"]
    random_seed: int
    timezone: str
    missing_rate_degraded: float
    missing_rate_blocked: float
    quantile_gap: float
    l2_alpha: float
    tail_mult: float
    utility_lambda: float
    utility_kappa: float
    band: float
    score_min: float
    score_max: float
    block_lengths: tuple[int, ...]
    bootstrap_replications: int
    strict_pit_start: date = date(2012, 1, 6)
    coverage_tol: float = 0.03
    crps_min_improve: float = 0.05
    ceq_floor: float = -0.005
    maxdd_tol: float = 0.03
    turnover_cap: float = 1.5
    blocked_cap: float = 0.15
