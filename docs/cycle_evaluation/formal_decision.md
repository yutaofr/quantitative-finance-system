# Formal Cycle Capability Decision

## Protocol

The audit evaluates continuous `cycle_score_t` and discrete three-state `cycle_state_t` under the frozen Cycle Evaluation Protocol. The primary truth label is a forward market return-risk composite. Crisis windows are external sanity checks only.

## Labels And Baselines

Primary label: 52-week forward return minus forward risk and drawdown penalties. Baselines: constant neutral, simple 26-week trend, simple 13-week realized volatility, T5-derived cycle proxy, and EGARCH-derived cycle proxy.

## Current-System Evaluation

| object | corr_forward_return | corr_forward_risk | balanced_accuracy | layer_1 | layer_2 | layer_3 | overall |
|---|---|---|---|---|---|---|---|
| CONSTANT_NEUTRAL_BASELINE | nan | nan | 0.333333 | FAIL | FAIL | FAIL | FAIL |
| SIMPLE_26W_TREND_BASELINE | 0.088123 | -0.117031 | 0.327386 | PASS | FAIL | FAIL | PARTIAL |
| SIMPLE_13W_VOL_BASELINE | 0.261514 | -0.472605 | 0.386486 | PASS | FAIL | FAIL | PARTIAL |
| T5_DERIVED_CYCLE_PROXY | 0.009592 | -0.133948 | 0.364495 | PASS | FAIL | FAIL | PARTIAL |
| EGARCH_DERIVED_CYCLE_PROXY | -0.193822 | 0.435191 | 0.248578 | PASS | FAIL | FAIL | PARTIAL |

## Layer Outcomes

Layer 1 tests usability. Layer 2 tests decision skill. Layer 3 tests cross-window survival. Passing directionality alone is not cycle capability.

## Decision

Main category: `CURRENT_SYSTEM_HAS_NO_CYCLE_CAPABILITY`.

New cycle model allowed now: `False`.

Reason: No current-system object passed the decision and cross-window gates.

Future cycle-model project may be opened only after a new preregistered information set or label source is introduced, and only if it first beats the frozen baseline family under this protocol without relying on trigger, override, hard switch, MoE, fixed T5 downstream patching, or rank-scale hybrid expansion.
