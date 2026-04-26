# Label Construction And Baselines

## Labels

The primary label is endogenous and market-only. It uses local `NASDAQXNDX` weekly prices to compute 52-week forward log return, 13-week forward realized weekly volatility, and 13-week forward max drawdown. The composite label rewards future return and penalizes future risk and drawdown. This is the main evaluation label.

The external sanity label is event-window only. NBER-style recession/crisis windows are represented by fixed, preregistered market crisis windows for 2000-2002, 2008, and 2020H1. These labels are never used for training or score calibration.

## Baselines

`CONSTANT_NEUTRAL_BASELINE`: always emits neutral score 0. It represents the no-cycle-skill null.

`SIMPLE_26W_TREND_BASELINE`: emits the standardized trailing 26-week NDX log return. It represents the minimum price-trend hypothesis.

`SIMPLE_13W_VOL_BASELINE`: emits negative standardized trailing 13-week realized weekly volatility. It represents the minimum realized-volatility state-machine hypothesis.

`T5_DERIVED_CYCLE_PROXY`: converts the existing T5 sigma output into cycle semantics by taking negative standardized sigma. It is an evaluated existing object, not a new model.

`EGARCH_DERIVED_CYCLE_PROXY`: converts the existing EGARCH-Normal sigma output into cycle semantics by taking negative standardized sigma. It is an evaluated existing object, not a new model.
