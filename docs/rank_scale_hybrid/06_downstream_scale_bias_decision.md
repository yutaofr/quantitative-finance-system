# Downstream Scale-Bias Final Decision

## Hypothesis

T5 may remain useful for downstream tail modeling if its `std(z)≈2` scale
bias can be absorbed by one stable scalar `k`.

## Experiment Config

- Fixed input: T5 `z_t` sequences for 2017, 2018, and 2020.
- Tail family: variance-one Student-t.
- Estimated parameters per window: scalar `k` and `nu`.
- Baseline: `k = 1` with estimated `nu`.
- Preregistration: `docs/rank_scale_hybrid/05_downstream_scale_bias_preregistration.md`

## Result Summary

- Option 1 decision: `SUCCESS`
- Failed conditions: `NONE`
- k range: `0.144575`
- Total log-likelihood improvement: `108.725342`

## Protocol Decision

`SUCCESS`

## Allowed Next Step / Termination

Option 1 is allowed to enter full tail family modeling with fixed T5 and downstream scalar scale-bias correction.

This decision does not reopen trigger audit, crisis architecture, hard switch, override, or MoE.
