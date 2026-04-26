# Hypothesis D Preregistration: Low-Complexity Latent State

## Model Object

Model `y_t` with a two-state latent proxy based on embargoed rolling target volatility. Each state emits a direct empirical normal proxy with state-specific mean and standard deviation.

## Essential Difference From Old Line

The state is not a trigger, crisis override, hard switch, or MoE gate. It is a descriptive two-bucket latent proxy derived only from historical target dispersion and used to produce a direct density forecast.

## Free-Degree Cap

- Two states.
- One frozen median threshold on the training volatility feature.
- State-specific empirical mean and standard deviation.
- Fallback to pooled training distribution only if a state has fewer than 20 observations.

## Comparison Baselines

T5, EGARCH-Normal, and the old failed Option 2A Student-t candidate.

## Success Standard

Must satisfy `comparison_protocol.md`.

## Forbidden Actions

No trigger audit. No crisis classifier. No override/hard switch/MoE. No T5 input. No state count expansion after seeing results.

## Failure Interpretation Boundary

Failure rejects this specific two-state latent proxy. It does not reject all state-space models.
