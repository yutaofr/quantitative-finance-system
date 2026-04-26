# Hypothesis C Preregistration: Decoupled Direction and Scale Heads

## Model Object

Model direction/ranking and cardinal scale as separate heads:

- ordinal head: ridge regression to an in-sample normal-score rank target;
- scale head: ridge regression to log absolute deviation from the training median;
- combiner: `mu_t = median_train + tanh(score_t) * sigma_t`, with normal scoring diagnostics.

## Essential Difference From Old Line

The ordinal head is learned from `y_t` ranks, not from T5 rank output. The scale head is learned from target-history dispersion, not EGARCH output. The combiner is a new density proxy rather than downstream patching of T5.

## Free-Degree Cap

- One linear ordinal head with intercept plus three features.
- One linear log-scale head with intercept plus three features.
- Fixed `tanh` combiner.
- Normal density proxy for CRPS.

## Comparison Baselines

T5, EGARCH-Normal, and the old failed Option 2A Student-t candidate.

## Success Standard

Must satisfy `comparison_protocol.md`.

## Forbidden Actions

No T5 rank or sigma input. No EGARCH scale input. No trigger. No override. No MoE. No hidden search over combiner functions.

## Failure Interpretation Boundary

Failure rejects this low-DOF decoupled-head implementation. It does not reject all rank/scale decoupling if future specifications introduce different features under a new preregistration.
