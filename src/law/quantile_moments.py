"""Moments and downside metrics from SRD v8.7 section 9.1 quantile curves."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

MIN_QUANTILE_POINTS = 2
FIRST_POWER = 1
SECOND_POWER = 2
MEAN_DENOMINATOR = 2.0
SECOND_MOMENT_DENOMINATOR = 3.0


def _validate_curve(
    taus: Sequence[float],
    q_vals: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    tau = np.asarray(taus, dtype=np.float64)
    q = np.asarray(q_vals, dtype=np.float64)
    if tau.ndim != 1 or q.ndim != 1 or tau.shape != q.shape:
        msg = "taus and q_vals must be aligned 1D arrays"
        raise ValueError(msg)
    if tau.shape[0] < MIN_QUANTILE_POINTS:
        msg = "at least two quantile points are required"
        raise ValueError(msg)
    if not np.isfinite(tau).all() or not np.isfinite(q).all():
        msg = "quantile curve inputs must be finite"
        raise ValueError(msg)
    if np.any(np.diff(tau) <= 0.0) or tau[0] < 0.0 or tau[-1] > 1.0:
        msg = "taus must be strictly increasing within [0, 1]"
        raise ValueError(msg)
    if np.any(np.diff(q) < 0.0):
        msg = "q_vals must be non-decreasing"
        raise ValueError(msg)
    return tau, q


def _linear_quantile_at(
    tau: NDArray[np.float64],
    q: NDArray[np.float64],
    prob: float,
) -> float:
    return float(np.interp(prob, tau, q))


def _integrate_quantile_power(
    tau: NDArray[np.float64],
    q: NDArray[np.float64],
    *,
    power: int,
    upper: float | None = None,
) -> float:
    limit = tau[-1] if upper is None else upper
    total = 0.0
    for idx in range(1, tau.shape[0]):
        left_tau = float(tau[idx - 1])
        right_tau = float(tau[idx])
        if left_tau >= limit:
            break
        segment_right = min(right_tau, limit)
        width = segment_right - left_tau
        if width <= 0.0:
            continue
        left_q = float(q[idx - 1])
        right_q = _linear_quantile_at(tau, q, segment_right)
        if power == FIRST_POWER:
            total += width * (left_q + right_q) / MEAN_DENOMINATOR
        elif power == SECOND_POWER:
            total += (
                width
                * (left_q * left_q + left_q * right_q + right_q * right_q)
                / SECOND_MOMENT_DENOMINATOR
            )
        else:
            msg = "only first and second powers are supported"
            raise ValueError(msg)
    return total


def p_loss_from_quantiles(taus: Sequence[float], q_vals: NDArray[np.float64]) -> float:
    """pure. Estimate F(0) by linear interpolation on the quantile curve."""
    tau, q = _validate_curve(taus, q_vals)
    if q[0] >= 0.0:
        return 0.0
    if q[-1] < 0.0:
        return 1.0
    return float(np.interp(0.0, q, tau))


def es20_from_quantiles(taus: Sequence[float], q_vals: NDArray[np.float64]) -> float:
    """pure. Return positive 20% expected shortfall from piecewise-linear quantiles."""
    tau, q = _validate_curve(taus, q_vals)
    alpha = 0.20
    if alpha <= float(tau[0]):
        msg = "quantile curve must include probability mass below ES alpha"
        raise ValueError(msg)
    q_alpha = _linear_quantile_at(tau, q, alpha)
    extended_tau = np.concatenate([tau[tau < alpha], np.array([alpha], dtype=np.float64)])
    extended_q = np.concatenate([q[tau < alpha], np.array([q_alpha], dtype=np.float64)])
    observed_tail_mass = alpha - float(tau[0])
    tail_area = _integrate_quantile_power(extended_tau, extended_q, power=FIRST_POWER)
    tail_mean = tail_area / observed_tail_mass
    return -tail_mean


def moments_from_quantiles(
    taus: Sequence[float],
    q_vals: NDArray[np.float64],
) -> Mapping[str, float]:
    """pure. Compute SRD §9.1 moments from a piecewise-linear quantile curve."""
    tau, q = _validate_curve(taus, q_vals)
    prob_mass = float(tau[-1] - tau[0])
    mean = _integrate_quantile_power(tau, q, power=FIRST_POWER) / prob_mass
    second_moment = _integrate_quantile_power(tau, q, power=SECOND_POWER) / prob_mass
    variance = max(second_moment - mean * mean, 0.0)
    return {
        "mu_hat": mean,
        "sigma_hat": float(np.sqrt(variance)),
        "p_loss": p_loss_from_quantiles(taus, q_vals),
        "es20": es20_from_quantiles(taus, q_vals),
    }
