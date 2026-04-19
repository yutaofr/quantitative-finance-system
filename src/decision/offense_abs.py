"""Absolute-threshold offense mapping from SRD v8.7 section 9.4 and 9.6."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from engine_types import Stance

SCORE_MIN = 0.0
SCORE_MAX = 100.0
SEGMENT_WIDTH = 20.0
DEFENSIVE_CUTOFF = 35.0
OFFENSIVE_CUTOFF = 65.0


@dataclass(frozen=True, slots=True)
class OffenseThresholds:
    """pure. Persisted absolute utility thresholds."""

    u_q0: float
    u_q20: float
    u_q40: float
    u_q60: float
    u_q80: float
    u_q100: float


def _segment(value: float, left: float, right: float, base: float) -> float:
    if right <= left:
        msg = "offense thresholds must be strictly increasing"
        raise ValueError(msg)
    return base + SEGMENT_WIDTH * (value - left) / (right - left)


def offense_raw(u_t: float, th: OffenseThresholds) -> float:
    """pure. Map utility to SRD §9.4 raw offense score clipped to [0, 100]."""
    if not np.isfinite(u_t):
        msg = "utility must be finite"
        raise ValueError(msg)
    if u_t < th.u_q20:
        value = _segment(u_t, th.u_q0, th.u_q20, 0.0)
    elif u_t < th.u_q40:
        value = _segment(u_t, th.u_q20, th.u_q40, 20.0)
    elif u_t < th.u_q60:
        value = _segment(u_t, th.u_q40, th.u_q60, 40.0)
    elif u_t < th.u_q80:
        value = _segment(u_t, th.u_q60, th.u_q80, 60.0)
    else:
        value = _segment(u_t, th.u_q80, th.u_q100, 80.0)
    return float(np.clip(value, SCORE_MIN, SCORE_MAX))


def stance_from_offense(offense_final: float) -> Stance:
    """pure. Map final offense score to SRD §9.6 stance."""
    if offense_final <= DEFENSIVE_CUTOFF:
        return "DEFENSIVE"
    if offense_final >= OFFENSIVE_CUTOFF:
        return "OFFENSIVE"
    return "NEUTRAL"
