# New Hypotheses Unified Comparison Protocol

## Scope

This protocol governs the minimum falsification run for independent new hypotheses A-D. It is frozen before running `src/research/run_new_hypotheses_parallel_experiment.py`.

## Windows

- `Window_2017`: 2017-07-07 to 2017-12-29
- `Window_2018`: 2018-07-06 to 2018-12-28
- `Window_2020`: 2020-01-03 to 2020-06-26

## Baselines

1. T5: frozen reproduced values from `docs/phase0b/02_benchmark_results_filled.md`.
2. EGARCH-Normal: frozen benchmark values from `docs/phase0b/02_benchmark_results_filled.md`.
3. Old failed instance: Option 2A Student-t candidate from `artifacts/research/joint_location_scale/results.json`.

## Core Metrics

Every hypothesis must emit:

- `mean_z`
- `std_z`
- `corr_next`
- `rank_next`
- `lag1_acf_z`
- `sigma_blowup`
- `pathology`
- `crps`

For direct quantiles, `z_t` is the preregistered median/IQR equivalent residual in `hypothesis_B_prereg.md`.

## Unified Continuation Gate

A hypothesis is `WORTH_CONTINUING` only if all conditions hold:

1. Three-window `corr_next` is never below T5 for the corresponding window.
2. Three-window `std_z` is never worse than EGARCH-Normal by more than `0.35`.
3. `sigma_blowup = 0` in all windows.
4. `pathology = 0` in all windows.
5. Mean CRPS improves at least 3% versus the old failed Option 2A Student-t candidate.

The 3% threshold is the frozen materiality threshold. Smaller improvements are treated as noise-level.

## Single-Line Categories

- `WORTH_CONTINUING`
- `PROMISING_BUT_INSUFFICIENT`
- `FAILED`
- `INVALID_IMPLEMENTATION`

## Final Categories

- `SELECT_ONE_CONTINUE`
- `NO_MODEL_WORTH_CONTINUING`
- `INCONCLUSIVE_DUE_TO_IMPLEMENTATION_LIMITS`

If multiple candidates pass, the single winner is the one with the highest material CRPS improvement score. No tie-retention is allowed.
