# Hypothesis A Preregistration: Direct Conditional Density

## Model Object

Directly model the 52-week forward log return `y_t` with a low-DOF conditional Student-t density:

`y_t | x_t ~ t(mu_t, sigma_t, nu)`

Features are embargo-safe functions of past target history only: lagged forward return, rolling mean, and rolling standard deviation, all shifted by 53 weeks.

## Essential Difference From Old Line

This model does not consume fixed T5 sigma, z, rank, trigger, or downstream tail outputs. It jointly estimates location, scale, and restricted shape from `y_t` rather than fitting a scale layer and then repairing it.

## Free-Degree Cap

- Linear `mu_t` ridge head: intercept plus three features.
- Linear log-scale ridge head: intercept plus three features.
- Shape is selected only from the frozen grid `{5, 8, 12, 30}`.
- No window-specific tuning after seeing results.

## Comparison Baselines

T5, EGARCH-Normal, and the old failed Option 2A Student-t candidate.

## Success Standard

Must satisfy the unified comparison protocol in `comparison_protocol.md`, including direction not worse than T5 in all three windows, scale not worse than EGARCH-Normal within the frozen tolerance, no blowup/pathology, and at least 3% material CRPS improvement against the old failed instance.

## Forbidden Actions

No T5 output input. No trigger. No override. No hard switch. No MoE. No bootstrap batch. No model-family expansion after seeing results.

## Failure Interpretation Boundary

Failure only rejects this low-DOF direct Student-t density specification. It does not reject all possible conditional density models.
