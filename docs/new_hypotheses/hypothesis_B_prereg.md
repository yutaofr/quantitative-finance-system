# Hypothesis B Preregistration: Direct Conditional Quantile Family

## Model Object

Directly model five conditional quantiles of `y_t`:

`q_tau(t) for tau in {0.10, 0.25, 0.50, 0.75, 0.90}`

The fitted quantile surface uses the same embargo-safe target-history features as Hypothesis A.

## Essential Difference From Old Line

This model never constructs a single intermediate T5-style sigma layer. It treats the conditional quantile family as the primary object and derives comparable diagnostics only after the quantile surface is fitted.

## Free-Degree Cap

- Five independent linear quantile heads.
- Intercept plus three features per quantile.
- Monotonicity is enforced by fixed rearrangement at prediction time.
- No additional tail family or scale repair.

## Comparison Baselines

T5, EGARCH-Normal, and the old failed Option 2A Student-t candidate.

## Success Standard

Must satisfy the unified comparison protocol in `comparison_protocol.md`. Standardized residuals are defined from the median and IQR-implied equivalent sigma:

`z_t = (y_t - q_0.50(t)) / ((q_0.75(t) - q_0.25(t)) / (Phi^-1(0.75) - Phi^-1(0.25)))`

CRPS is the integrated pinball score on the five preregistered quantiles.

## Forbidden Actions

No T5 input. No sigma-first skeleton. No trigger. No bootstrap. No post-result expansion to extra quantiles.

## Failure Interpretation Boundary

Failure rejects this low-DOF linear direct quantile family. It does not reject nonlinear quantile surfaces or panel quantile models.
