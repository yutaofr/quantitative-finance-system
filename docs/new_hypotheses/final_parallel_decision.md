# Final Parallel Decision: Independent New Hypotheses

## Verdict

`NO_MODEL_WORTH_CONTINUING`

No candidate met the unified continuation gate. The research program should terminate at this layer unless a future SRD/ADD-level change authorizes a materially different information set or model class.

## Artifacts

- Runner: `src/research/run_new_hypotheses_parallel_experiment.py`
- JSON: `artifacts/research/new_hypotheses/parallel_results.json`
- Markdown summary: `artifacts/research/new_hypotheses/parallel_results.md`
- Protocol: `docs/new_hypotheses/comparison_protocol.md`

## Candidate Decisions

| hypothesis | single-line decision | protocol pass | mean CRPS | material improvement vs old failed |
|---|---|---:|---:|---:|
| A_DIRECT_DENSITY | FAILED | false | 0.112426 | -0.618128 |
| B_DIRECT_QUANTILES | FAILED | false | 0.108365 | -0.559667 |
| C_DECOUPLED_HEADS | FAILED | false | 0.112287 | -0.616116 |
| D_LATENT_STATE | FAILED | false | 0.094073 | -0.353968 |

## Three-Window Summary

### A_DIRECT_DENSITY

| window | mean_z | std_z | corr_next | rank_next | lag1_acf_z | sigma_blowup | pathology | CRPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2017 | 0.575405 | 1.846378 | 0.095972 | 0.153077 | 0.918859 | 0 | 0 | 0.060277 |
| 2018 | -0.223410 | 1.912996 | -0.494812 | -0.312308 | 0.909260 | 0 | 0 | 0.056174 |
| 2020 | 5.567826 | 1.429744 | -0.215228 | -0.149231 | 0.731422 | 0 | 0 | 0.220829 |

### B_DIRECT_QUANTILES

| window | mean_z | std_z | corr_next | rank_next | lag1_acf_z | sigma_blowup | pathology | CRPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2017 | 282.670875 | 859.674429 | -1.000000 | -1.000000 | 0.921179 | 0 | 0 | 0.064695 |
| 2018 | -146.903262 | 869.953964 | -1.000000 | -1.000000 | 0.904579 | 0 | 0 | 0.061556 |
| 2020 | 2485.539350 | 589.465987 | -1.000000 | -1.000000 | 0.676230 | 0 | 0 | 0.198843 |

For B, `-1.000000` marks preregistered comparable direction failure after equivalent sigma degeneracy made correlation undefined.

### C_DECOUPLED_HEADS

| window | mean_z | std_z | corr_next | rank_next | lag1_acf_z | sigma_blowup | pathology | CRPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2017 | -0.099166 | 2.071659 | -0.284682 | -0.216154 | 0.929821 | 0 | 0 | 0.060405 |
| 2018 | -0.727066 | 1.509023 | -0.429840 | -0.394615 | 0.914566 | 0 | 0 | 0.064407 |
| 2020 | 4.810865 | 1.066806 | 0.464638 | 0.450000 | 0.603034 | 0 | 0 | 0.212048 |

### D_LATENT_STATE

| window | mean_z | std_z | corr_next | rank_next | lag1_acf_z | sigma_blowup | pathology | CRPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2017 | -0.288824 | 0.739704 | -0.630827 | -0.621538 | 0.870253 | 0 | 0 | 0.046869 |
| 2018 | -0.259826 | 1.037856 | 0.157690 | 0.255346 | 0.938162 | 0 | 0 | 0.059354 |
| 2020 | 3.076532 | 1.051237 | -0.168426 | -0.194615 | 0.795502 | 0 | 0 | 0.175995 |

## Unified Baseline Comparison

| object | direction gate vs T5 | scale gate vs EGARCH | blowup/pathology | material CRPS improvement | outcome |
|---|---:|---:|---:|---:|---|
| A_DIRECT_DENSITY | -0.881750 | -0.269173 | pass | -0.618128 | FAILED |
| B_DIRECT_QUANTILES | -1.629802 | -867.426695 | pass | -0.559667 | FAILED |
| C_DECOUPLED_HEADS | -0.914484 | -0.019353 | pass | -0.616116 | FAILED |
| D_LATENT_STATE | -1.260629 | 0.109334 | pass | -0.353968 | FAILED |

All four candidates fail the direction gate. A, B, and C also fail the scale gate. All four have worse mean CRPS than the old failed Option 2A baseline under the frozen materiality definition.

## Final Selection

No winner is selected. The final category is `NO_MODEL_WORTH_CONTINUING`.

## Termination Recommendation

Terminate this parallel new-hypothesis plan. The negative result is not an implementation-limit result: all runners produced complete three-window diagnostics and no candidate was marked `INVALID_IMPLEMENTATION`.

The termination scope is limited to the four preregistered low-complexity candidates and the current target-history-only information set. It does not authorize returning to fixed T5 downstream patching, trigger search, override, hard switch, MoE, or rank-scale hybrid parameter expansion.
