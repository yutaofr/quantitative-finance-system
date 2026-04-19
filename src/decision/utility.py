"""Utility score components from SRD v8.7 section 9.2-9.3."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MAD_SCALE = 1.4826
MAD_EPSILON = 1.0e-8
LAMBDA_ES20 = 1.2
KAPPA_PLOSS = 0.8


@dataclass(frozen=True, slots=True)
class UtilityZStats:
    """pure. Frozen training-window robust z-score statistics."""

    er_med: float
    er_mad: float
    es20_med: float
    es20_mad: float
    ploss_med: float
    ploss_mad: float


def excess_return(mu_hat: float, dgs1: float) -> float:
    """pure. Compute SRD §9.2 52-week excess return against DGS1."""
    return mu_hat - dgs1


def _robust_z(value: float, median: float, mad: float) -> float:
    if not np.isfinite(value) or not np.isfinite(median) or not np.isfinite(mad):
        msg = "utility z-score inputs must be finite"
        raise ValueError(msg)
    return (value - median) / (MAD_SCALE * mad + MAD_EPSILON)


def utility(  # noqa: PLR0913
    er: float,
    es20: float,
    p_loss: float,
    zstats: UtilityZStats,
    *,
    lam: float = LAMBDA_ES20,
    kappa: float = KAPPA_PLOSS,
) -> float:
    """pure. Compute SRD §9.3 frozen-z utility."""
    return (
        _robust_z(er, zstats.er_med, zstats.er_mad)
        - lam * _robust_z(es20, zstats.es20_med, zstats.es20_mad)
        - kappa * _robust_z(p_loss, zstats.ploss_med, zstats.ploss_mad)
    )
