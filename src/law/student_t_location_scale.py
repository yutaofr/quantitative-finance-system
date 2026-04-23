from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize  # type: ignore[import-untyped]
from scipy.stats import t as student_t  # type: ignore[import-untyped]

TAUS_FULL = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95], dtype=np.float64)
TAUS_CRPS = np.linspace(0.05, 0.95, 19, dtype=np.float64)
NU = 5.0
L2 = 1.0e-3
COVERAGE_WEIGHT = 10.0
LOG_SIGMA_MIN = -4.0
LOG_SIGMA_MAX = 2.0


@dataclass(frozen=True, slots=True)
class StudentTLawParams:
    """pure. Frozen Student-t location-scale law parameters."""

    beta_mu: NDArray[np.float64]
    beta_sigma: NDArray[np.float64]
    nu: float = NU


@dataclass(frozen=True, slots=True)
class StudentTFitResult:
    """pure. Deterministic fit result with optimizer fallback metadata."""

    params: StudentTLawParams
    objective_value: float
    optimizer_status: str
    optimization_failed: bool
    fallback_used: bool
    train_rows: int


def _quantile_score(
    y_true: NDArray[np.float64],
    quantiles: NDArray[np.float64],
    taus: NDArray[np.float64],
) -> float:
    delta = y_true[:, None] - quantiles
    losses = np.maximum(taus[None, :] * delta, (taus[None, :] - 1.0) * delta)
    return float(2.0 * np.mean(losses))


def _design_matrix_mu(
    x_scaled: NDArray[np.float64],
    pi: NDArray[np.float64],
) -> NDArray[np.float64]:
    return np.column_stack(
        [
            np.ones(x_scaled.shape[0], dtype=np.float64),
            np.asarray(x_scaled, dtype=np.float64),
            np.asarray(pi, dtype=np.float64),
        ],
    )


def _design_matrix_sigma(pi: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.column_stack(
        [np.ones(pi.shape[0], dtype=np.float64), np.asarray(pi, dtype=np.float64)],
    )


def _predict_matrix(
    x_mu: NDArray[np.float64],
    x_sigma: NDArray[np.float64],
    params: StudentTLawParams,
    taus: NDArray[np.float64],
) -> NDArray[np.float64]:
    mu = x_mu @ params.beta_mu
    log_sigma = np.clip(x_sigma @ params.beta_sigma, LOG_SIGMA_MIN, LOG_SIGMA_MAX)
    sigma = np.exp(log_sigma)
    z = student_t.ppf(taus, df=params.nu).astype(np.float64)
    quantiles = mu[:, None] + sigma[:, None] * z[None, :]
    return np.maximum.accumulate(quantiles, axis=1)


def predict_student_t_quantiles(
    x_scaled: NDArray[np.float64],
    pi: NDArray[np.float64],
    params: StudentTLawParams,
) -> NDArray[np.float64]:
    """pure. Predict the seven SRD quantiles from one row or a matrix of rows."""
    x_scaled_arr = np.asarray(x_scaled, dtype=np.float64)
    pi_arr = np.asarray(pi, dtype=np.float64)
    scalar_input = x_scaled_arr.ndim == 1
    if scalar_input:
        x_scaled_arr = x_scaled_arr[None, :]
        pi_arr = pi_arr[None, :]
    quantiles = _predict_matrix(
        _design_matrix_mu(x_scaled_arr, pi_arr),
        _design_matrix_sigma(pi_arr),
        params,
        TAUS_FULL,
    )
    return quantiles[0] if scalar_input else quantiles


def _objective(
    theta: NDArray[np.float64],
    x_mu: NDArray[np.float64],
    x_sigma: NDArray[np.float64],
    y_true: NDArray[np.float64],
) -> float:
    mu_dim = x_mu.shape[1]
    params = StudentTLawParams(
        beta_mu=theta[:mu_dim],
        beta_sigma=theta[mu_dim:],
    )
    crps_proxy = _quantile_score(
        y_true,
        _predict_matrix(x_mu, x_sigma, params, TAUS_CRPS),
        TAUS_CRPS,
    )
    eval_quantiles = _predict_matrix(x_mu, x_sigma, params, TAUS_FULL)
    q10_cov = float(np.mean(y_true <= eval_quantiles[:, 1]))
    q90_cov = float(np.mean(y_true <= eval_quantiles[:, 5]))
    coverage_penalty = (q10_cov - 0.10) ** 2 + (q90_cov - 0.90) ** 2
    regularization = L2 * float(np.sum(theta * theta))
    return crps_proxy + COVERAGE_WEIGHT * coverage_penalty + regularization


def _initial_theta(
    x_mu: NDArray[np.float64],
    x_sigma: NDArray[np.float64],
    y_true: NDArray[np.float64],
) -> NDArray[np.float64]:
    ridge_mu = 1.0e-3 * np.eye(x_mu.shape[1], dtype=np.float64)
    beta_mu0 = np.linalg.solve(x_mu.T @ x_mu + ridge_mu, x_mu.T @ y_true)
    residual = y_true - x_mu @ beta_mu0
    sigma_target = np.log(np.maximum(np.abs(residual), 1.0e-3))
    ridge_sigma = 1.0e-3 * np.eye(x_sigma.shape[1], dtype=np.float64)
    beta_sigma0 = np.linalg.solve(x_sigma.T @ x_sigma + ridge_sigma, x_sigma.T @ sigma_target)
    return np.concatenate([beta_mu0, beta_sigma0]).astype(np.float64)


def fit_student_t_location_scale(
    x_scaled: NDArray[np.float64],
    pi: NDArray[np.float64],
    y_52w: NDArray[np.float64],
    *,
    theta0: NDArray[np.float64] | None = None,
    maxiter: int = 80,
) -> StudentTFitResult:
    """pure. Fit the shadow Student-t law with deterministic optimizer fallback."""
    x_mu = _design_matrix_mu(
        np.asarray(x_scaled, dtype=np.float64),
        np.asarray(pi, dtype=np.float64),
    )
    x_sigma = _design_matrix_sigma(np.asarray(pi, dtype=np.float64))
    y_true = np.asarray(y_52w, dtype=np.float64)
    initial = (
        _initial_theta(x_mu, x_sigma, y_true)
        if theta0 is None
        else np.asarray(theta0, dtype=np.float64)
    )
    trial_iters = (maxiter, maxiter * 2, maxiter * 4)
    last_theta = initial
    last_status = "not_started"
    fallback_used = False
    for budget in trial_iters:
        try:
            result = minimize(
                _objective,
                last_theta,
                args=(x_mu, x_sigma, y_true),
                method="Powell",
                options={"maxiter": budget, "xtol": 1.0e-4, "ftol": 1.0e-4},
            )
        except (ValueError, FloatingPointError) as exc:
            last_status = f"exception:{type(exc).__name__}"
            fallback_used = True
            break
        last_theta = np.asarray(result.x, dtype=np.float64)
        last_status = str(result.message)
        if result.success:
            params = StudentTLawParams(
                beta_mu=last_theta[: x_mu.shape[1]],
                beta_sigma=last_theta[x_mu.shape[1] :],
            )
            return StudentTFitResult(
                params=params,
                objective_value=float(result.fun),
                optimizer_status=last_status,
                optimization_failed=False,
                fallback_used=fallback_used,
                train_rows=int(y_true.shape[0]),
            )
        fallback_used = True
    params = StudentTLawParams(
        beta_mu=last_theta[: x_mu.shape[1]],
        beta_sigma=last_theta[x_mu.shape[1] :],
    )
    return StudentTFitResult(
        params=params,
        objective_value=_objective(last_theta, x_mu, x_sigma, y_true),
        optimizer_status=last_status,
        optimization_failed=True,
        fallback_used=True,
        train_rows=int(y_true.shape[0]),
    )
