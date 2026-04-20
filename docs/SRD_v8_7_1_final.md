# SRD v8.7.1 (exec-final)
## QQQ Cycle-Conditional Law Engine — Executable Specification

**Status**: Frozen for implementation. Supersedes SRD v8.7.
**Audience**: AI coding agents / quantitative research engineers.
**Language**: Python 3.11+.
**Execution frequency**: Weekly (Friday close, NY time).
**Prediction horizon**: 52 weeks.
**Asset**: Nasdaq-100 total-return (`NASDAQXNDX`).
**Document precedence**: v8.7.1 > v8.7. Where the two conflict, v8.7.1 governs.

---

## 0. How to read this document

This SRD is an implementation contract. It is written so a coding agent can produce code that compiles, backtests, and emits output in a deterministic way without inventing missing math or tuning parameters.

Normative keywords:

- **MUST**: hard requirement, enforced by unit test
- **MUST NOT**: forbidden, enforced by unit test
- **SHOULD**: default unless a written deviation record is attached to the config
- **MAY**: optional, not required for v8.7

Every numeric constant in this document is a frozen governance parameter, not a hyperparameter. A coding agent MUST NOT tune them without a written spec revision.

---

## 1. Scope redefinition (v8.7 change vs v8.6)

v8.6 attempted to deliver (a) a conditional 52-week return distribution, (b) an offense score, and (c) a diagnostic layer on one production stack that simultaneously used two HMMs, a diffusion-map geometry, an EVT tail engine, conformal calibration, and a challenger/MIDAS pool. The sample size available for QQQ/NDX, the number of independent crisis events, the true ALFRED vintage start dates for several core features, and the 52-week overlapping-target autocorrelation structure are all incompatible with that parameter budget.

v8.7 splits the work into two tracks:

- **Production track** — the only track that gates acceptance. Narrow, linear, PIT-clean, block-bootstrap-verified.
- **Research track** — every advanced idea from v8.6 is preserved in code, but runs as a diagnostic sidecar that cannot influence the production output.

The stated goal of "helping the user judge the current cycle position to inform the QQQ expected-return distribution" is preserved. The cycle position is exposed as an explicit output field (`cycle_position`) computed from transparent macro signals, not as a latent HMM state.

---

## 2. Frozen design decisions (v8.7)

1. Single **3-state time-inhomogeneous HMM** in the production track. No shadow HMM. No Hungarian matching. No arithmetic posterior fusion.
2. State semantics are fixed at training time by sorting states on in-sample expected 52-week return; the mapping is persisted and MUST NOT drift between runs.
3. **Linear non-crossing quantile regression** for the interior law. No spline surfaces. No nonlinear interactions in production.
4. **No EVT in production.** Q05/Q95 are produced by bounded geometric extrapolation from calibrated interior quantiles, with block-bootstrap confidence intervals. EVT remains in the research track.
5. **10 frozen production features.** The feature list in §6 is closed. No additions without a spec revision.
6. **Two-layer PIT protocol with computed strict-acceptance start.**
   (a) NFCI family, STLFSI4, MIDAS challenger series are excluded from the production feature set.
   (b) The production backtest window is split into pseudo-PIT and strict-PIT.
   (c) The strict-PIT acceptance segment does **not** start from a fixed calendar date. It starts from `effective_strict_start`, defined as the earliest weekly inference date for which all mandatory production features can produce finite values under `vintage_mode="strict"` and for which the minimum training window and embargo requirements of §15.1 are simultaneously satisfiable.
   (d) Under the frozen v8.7.1 production feature set and constants, `effective_strict_start = 2014-11-28` (Friday close, NY time).
   (e) Only the strict-PIT segment from `effective_strict_start` onward enters §16 acceptance statistics.
7. **Soft squash + hard clip** replaces v8.6 winsorization. Features are passed through a scaled `tanh` then hard-clipped at `[-5, 5]`.
8. **Offense score uses absolute thresholds calibrated on training data**, not a rolling percentile rank. Percentile rank remains as a diagnostic field.
9. **Block bootstrap is mandatory** for every inferential statement. Fixed block lengths `{52, 78}`, `B = 2000` replications, both must be reported.
10. **Risk-free benchmark remains `DGS1`**, matched to the 52-week horizon.
11. **No-trade band** is retained as a hard rule, but the threshold is a named config parameter (`band`) with default `7`.
12. Geometry (diffusion map, Nyström, topology), EVT (Bayesian GPD, conformal), MIDAS, challenger promotion, and double-HMM fusion are moved to the research track (§12). They MUST NOT enter the production JSON output.

---

## 3. Architecture overview

```
                 ┌──────────────────────────┐
                 │  Data contract (PIT)     │
                 └────────────┬─────────────┘
                              │
         ┌────────────────────┴────────────────────┐
         │                                         │
   ┌─────▼──────┐                         ┌────────▼────────┐
   │ Production │                         │    Research     │
   │   track    │                         │    sidecar      │
   └─────┬──────┘                         └────────┬────────┘
         │                                         │
  Features (10, frozen)               geometry / double-HMM /
         │                            EVT / MIDAS / challenger
  Scaling (tanh + clip)                        │
         │                              research_report.json
  Single 3-state TI-HMM                        │
         │                               (NOT gated by acceptance)
  Linear quantile regression
         │
  Tail extrapolation + bootstrap CI
         │
  Absolute-threshold offense
         │
  Hysteresis band
         │
  production_output.json  ◄── gated by §16 acceptance
```

---

## 4. Data contract

### 4.1 Production-track series (allowed)

Core target:
- `NASDAQXNDX`

Rates:
- `DGS10`, `DGS2`, `DGS1`, `EFFR`

Liquidity:
- `WALCL`, `RRPONTSYD`

Credit:
- `BAA10Y`

Volatility:
- `VXNCLS`, `VIXCLS`, `VXVCLS`

Self-constructed (from `NASDAQXNDX` price only):
- `RV20_NDX` = annualized realized volatility over trailing 20 trading days, resampled to Friday close
- `VRP_NDX` = `log(VXNCLS) − log(RV20_NDX)`
- `TS` = `log(VIXCLS / VXVCLS)`

### 4.2 Production-track series (excluded)

The following series are **excluded** from the production feature set because their ALFRED real-time vintages do not cover 2001–2010, which would contaminate any pre-2011 backtest with post-hoc recalculation:

- `NFCI`, `NFCIRISK`, `NFCICREDIT`, `NFCILEVERAGE`, `NFCINONFINLEVERAGE`
- `STLFSI4`
- All BEA quarterly / monthly challenger candidates (`A679RC1Q027SBEA`, `Y005RC1A027NBEA`, `IPG3344S`, `CAPUTLG3344SQ`, etc.)

These series remain available to the research track (§12). A coding agent MUST NOT read them from inside `features/block_builder.py`.

### 4.3 Point-in-time rule

All production-track series MUST be queried via:

```python
get_series_pti(series_id: str, as_of: date, vintage_mode: Literal["strict","pseudo"]) -> pd.Series
```

Semantics:

* `vintage_mode="strict"`: raise if the requested `as_of` date is earlier than the first ALFRED real-time vintage for `series_id`. Use ALFRED `get_archival_series(..., realtime_start=as_of, realtime_end=as_of)`.
* `vintage_mode="pseudo"`: fall back to the latest revised FRED series, flag the returned frame with `is_pseudo_pit=True`.

The production backtest window is two-segmented:

* **2001-02-02 → 2011-12-30**: `vintage_mode="pseudo"`. Output is computed, but this segment MUST be excluded from §16 acceptance statistics.
* **2012-01-06 → 2014-11-21**: `vintage_mode="strict"` MAY be computed for diagnostics, but this segment MUST be treated as **strict-pre-acceptance warmup** and MUST be excluded from §16 acceptance statistics because the frozen production feature set plus §15.1 training requirements do not permit a full valid acceptance window earlier than `2014-11-28`.
* **2014-11-28 → today**: `vintage_mode="strict"`. This is the only segment that enters §16 acceptance statistics under v8.7.1.

Series-level earliest strict-PIT dates (as of writing) that the agent MUST hard-code in `data_contract/vintage_registry.py`:

| Series       | earliest_strict_pit | notes                           |
| ------------ | ------------------- | ------------------------------- |
| `NASDAQXNDX` | 1985-10-01          | daily close, no revision        |
| `DGS10`      | 1962-01-02          | daily                           |
| `DGS2`       | 1976-06-01          | daily                           |
| `DGS1`       | 1962-01-02          | daily                           |
| `EFFR`       | 2000-07-03          | daily                           |
| `WALCL`      | 2002-12-18          | weekly H.4.1                    |
| `RRPONTSYD`  | 2013-02-01          | before this date, research-only |
| `BAA10Y`     | 1986-01-02          | computed spread                 |
| `VXNCLS`     | 2001-02-02          | before this date, not usable    |
| `VIXCLS`     | 1990-01-02          | daily                           |
| `VXVCLS`     | 2007-12-04          | before this date, research-only |

When a production feature's earliest_strict_pit postdates the backtest `as_of`, the feature MUST be marked missing and handled under §4.4.

The coding agent MUST implement:

```python
compute_effective_strict_start(
    feature_registry,
    earliest_strict_pit_registry,
    min_training_weeks=312,
    embargo_weeks=53,
    weekly_calendar="Friday"
) -> date
```

For v8.7.1, under the frozen feature list in §6 and frozen constants in §18, this function MUST return `2014-11-28`.

### 4.4 Missing-data policy

At each week `t`:
- If any **required** production input is missing, that feature is flagged and the `mode` field degrades per §10.
- If the missing-rate across the 10 frozen features exceeds 10% at `t`, set `mode = DEGRADED`.
- If the missing-rate exceeds 20%, or if two of the three blocks (target/rates, volatility, credit) are both missing, set `mode = BLOCKED`.

Imputation rules:
- **Never** carry forward revisions from later vintages.
- Within a single week, if a daily series is missing for Friday but available for Thursday, use Thursday's value (weekly timestamps are not required to be exactly Friday).
- No imputation across series.

---

## 5. Scaling

### 5.1 Expanding-window robust z-score

For each feature `x` and each week `t`:

```
med_t = median(x[:t])
mad_t = median(|x[:t] − med_t|)
z_t   = (x_t − med_t) / (1.4826 * mad_t + 1e-8)
```

The scaler uses an **expanding** training window that grows with `t`, not a fixed-length rolling window. This preserves stationarity assumptions across regimes without re-anchoring.

### 5.2 Soft squash + hard clip

After z-scoring, apply:

```
z_soft_t = 4 * tanh(z_t / 4)
z_clipped_t = clip(z_soft_t, -5, 5)
```

Rationale: `tanh` preserves ordering in the tail, unlike winsorization, but bounds leverage. The residual hard clip at `[-5, 5]` guards the optimizer.

### 5.3 Anchor scaling (removed from production)

The v8.6 anchor-scaling mechanism tied to a 260-week geometry window is removed from the production path. It is retained only inside the research sidecar.

---

## 6. Feature block (frozen, 10 features)

The production feature vector `X_t ∈ ℝ^10` is:

| idx | name          | formula                                |
|----:|---------------|----------------------------------------|
| 1   | `x1`          | `DGS10 − DGS2` (level)                 |
| 2   | `x2`          | `Δ13w (DGS10 − DGS2)`                  |
| 3   | `x3`          | `DGS1` (level)                         |
| 4   | `x4`          | `Δ13w EFFR`                            |
| 5   | `x5`          | `BAA10Y` (level)                       |
| 6   | `x6`          | `Δ13w BAA10Y`                          |
| 7   | `x7`          | `Δ26w log(WALCL)`                      |
| 8   | `x8`          | `log(VXNCLS)` (level)                  |
| 9   | `x9`          | `VRP_NDX = log(VXNCLS) − log(RV20_NDX)`|
| 10  | `x10`         | `TS = log(VIXCLS / VXVCLS)`            |

All 10 features are scaled per §5. The scaled vector is `X_t^z ∈ ℝ^10`.

Two auxiliary aggregates are computed:

- `h_t = x5_t + 0.5 * x9_t` (credit + volatility stress composite, used only in HMM hazard)
- `pc1_t, pc2_t` = first two principal components of `X_t^z` computed via an expanding-window robust PCA (MCD or LEDOIT-WOLF shrinkage); used only as HMM observations, not in the quantile regression.

The quantile regression itself consumes `X_t^z` directly, not the PCA components. This preserves interpretability.

---

## 7. State engine (production — single TI-HMM)

### 7.1 Model

One time-inhomogeneous hidden Markov model.

- States `K = 3`, labels `{DEFENSIVE, NEUTRAL, OFFENSIVE}`.
- Observation vector `y_t = [pc1_t, pc2_t, Δ1w pc1_t, Δ1w pc2_t, x8_t, x9_t] ∈ ℝ^6`.
- Emission: multivariate Gaussian per state, full covariance with shrinkage (Ledoit-Wolf target = diagonal of pooled covariance, λ learned on training set and frozen).
- Transition hazard for each state `j`:

```
P(S_t ≠ j | S_{t−1} = j, d_t, h_t) = σ(β_{0j} + β_{1j} d_t + β_{2j} h_t)
```

where `d_t` is the current dwell duration (weeks since last transition) and `h_t` is as defined in §6.

### 7.2 Training

- Right-censor-aware log-likelihood is mandatory (last unfinished episode contributes via a survival term).
- EM with 50 random restarts. Retain the model with the best penalized log-likelihood on the training window.
- Label identification rule (MUST): after EM, compute the in-sample average forward 52-week return in each decoded state. Sort ascending. Assign:
  - lowest avg → `DEFENSIVE`
  - middle → `NEUTRAL`
  - highest → `OFFENSIVE`
- The mapping is persisted to `artifacts/state_label_map.json`. Every subsequent inference MUST reuse this map.

### 7.3 Failure fallback

The state engine outputs `DEGRADED` with a uniform posterior over the three states (`[0.25, 0.50, 0.25]`) when any of the following hold:

- `EM` did not converge within 200 iterations on the training window
- posterior contains `NaN` or `Inf` at any inference step
- any state's smoothed occupancy probability is below `0.01` averaged over 26 consecutive weeks (degenerate state)
- the sign of the in-sample `avg forward return` ordering flips when refit on a rolling window

When the fallback fires:

```
state_post = [0.25, 0.50, 0.25]
state_name = "NEUTRAL"
model_status = "DEGRADED"
```

---

## 8. Conditional law engine (production — linear quantile regression)

### 8.1 Interior quantiles

For `τ ∈ {0.10, 0.25, 0.50, 0.75, 0.90}`:

```
Q_τ(R_{t,52} | F_t) = a_τ + b_τ^⊤ X_t^z + c_τ^⊤ π_t
```

where `π_t ∈ ℝ^3` is the smoothed HMM posterior.

Implementation MUST:

- jointly solve all 5 quantile regressions in one optimization problem
- enforce `Q_{τ_{k+1}} − Q_{τ_k} ≥ 1e-4` (1 basis point) for all adjacent `τ_k`
- apply an L2 penalty `α = 2.0` on `b_τ` and `c_τ` (not on intercepts)
- use the asymmetric check-loss (`pinball`) objective

A convex solver (e.g., `cvxpy` with ECOS) is sufficient. If the solver fails at any `t`, set `model_status = "DEGRADED"` and fall back to the unconstrained single-τ solutions followed by rearrangement (Chernozhukov et al. rearrangement).

### 8.2 Tail extrapolation

No EVT in production. Compute:

```
q05 = q10 − 0.6 * (q25 − q10)
q95 = q90 + 0.6 * (q90 − q75)
```

The multiplier `0.6` is frozen. It is a governance choice: it assumes the tail behaves roughly like a scaled-Student-t with 5 degrees of freedom in the region just outside the estimated interior quantiles. This is **not** a distribution assumption imposed on the full law; it is a bounded extrapolation rule.

### 8.3 Bootstrap confidence intervals

At each strict-PIT-era `t`, also emit:

- `q05_ci_low`, `q05_ci_high`
- `q95_ci_low`, `q95_ci_high`

computed by stationary block bootstrap of residuals (`block_length = 52`, `B = 2000`).

### 8.4 Tail failure fallback

If `q95 < q90` or `q05 > q10` after extrapolation (numerical pathology), replace the offending tail with the empirical unconditional quantile on the training window and set `model_status = "DEGRADED"`.

---

## 9. Decision engine

### 9.1 Distribution moments

Compute by numerical quadrature on the quantile curve:

- `mu_hat_t` = implied mean, by integrating `r dF` where `F` is a piecewise-linear CDF between the 7 quantile points
- `sigma_hat_t` = implied standard deviation, same integration
- `p_loss_t` = F(0), estimated by linear interpolation between the two quantile points bracketing 0
- `es20_t` = negative of conditional mean on the lowest 20% of the distribution (expected shortfall at 20%)

### 9.2 Excess return

```
ER_t = mu_hat_t − DGS1_t
```

Simple difference; because the horizon is 52 weeks and `DGS1` is a 1-year yield, no horizon rescaling is applied. The `exp(μ + 0.5σ²) − 1` approximation from v8.6 §10.2 is removed to avoid distributional assumptions that conflict with the piecewise-linear quantile form.

### 9.3 Utility

```
U_t = z_52w(ER_t) − 1.2 * z_52w(es20_t) − 0.8 * z_52w(p_loss_t)
```

where `z_52w(.)` is a robust z-score computed on the **training window only**, using the median and MAD of the target quantity. The mean and scale are persisted at training time to `artifacts/utility_zstats.json` and used unchanged at every inference step.

This differs from v8.6, where the z-score was rolling. A rolling z-score anchors "good" to the recent past and systematically misclassifies persistent bull or bear regimes. The training-frozen version preserves absolute comparability across regimes.

The weights `λ = 1.2`, `κ = 0.8` are frozen. The entropy penalty from v8.6 is removed: entropy of a 3-state posterior is a degenerate feature when posterior mass is concentrated.

### 9.4 Offense mapping (absolute thresholds)

Compute absolute thresholds on the training window (strict-PIT era only):

```
u_q20 = 20th percentile of U_t over training window
u_q40 = 40th percentile
u_q60 = 60th percentile
u_q80 = 80th percentile
```

These four numbers are persisted. At inference:

```
if   U_t < u_q20: offense_raw = 20 * (U_t − u_q0) / (u_q20 − u_q0)
elif U_t < u_q40: offense_raw = 20 + 20 * (U_t − u_q20) / (u_q40 − u_q20)
elif U_t < u_q60: offense_raw = 40 + 20 * (U_t − u_q40) / (u_q60 − u_q40)
elif U_t < u_q80: offense_raw = 60 + 20 * (U_t − u_q60) / (u_q80 − u_q60)
else:             offense_raw = 80 + 20 * (U_t − u_q80) / (u_q100 − u_q80)
```

clipped to `[0, 100]`, where `u_q0, u_q100` are the min and max of `U` on the training window, extended by ±3 MAD to cover out-of-sample excursions.

### 9.5 No-trade band (hysteresis)

```python
band = 7   # config
if abs(offense_raw_t - offense_final_{t-1}) < band:
    offense_final_t = offense_final_{t-1}
else:
    offense_final_t = offense_raw_t
```

### 9.6 Stance mapping

```
offense_final ≤ 35 → DEFENSIVE
35 < offense_final < 65 → NEUTRAL
offense_final ≥ 65 → OFFENSIVE
```

### 9.7 Diagnostic cycle position

An explicit, transparent "cycle position" field `cycle_position ∈ [0, 100]` is emitted:

```
cycle_position_t = 100 * average(
    percentile_rank(x5_t on training window),   # credit spread level
    percentile_rank(x9_t on training window),   # VRP level
    percentile_rank(-x1_t on training window),  # inverted slope
)
```

Low values imply "early cycle / low stress"; high values imply "late cycle / stress building". This is a diagnostic output, not an input to the decision engine. It exists to give the end-user the "where in the cycle are we" view that motivates the project, without letting the view contaminate the model.

---

## 10. Modes

Three modes only:

### NORMAL
- all 10 features present this week
- HMM posterior valid
- quantile solver succeeded with non-crossing constraint satisfied
- offense_final defined

### DEGRADED
- 10–20% of features missing OR
- HMM fallback triggered (§7.3) OR
- quantile solver triggered fallback rearrangement (§8.1) OR
- tail extrapolation fallback triggered (§8.4)

In DEGRADED:
- decision output is still emitted
- `offense_final` is clamped to `[30, 70]` (no extreme stances)

### BLOCKED
- >20% of features missing OR
- any of {target, rates, volatility} entire block missing for two consecutive weeks OR
- utility or offense computation produced NaN/Inf after fallbacks

In BLOCKED:
- `offense_final = 50`
- `stance = "NEUTRAL"`
- `mode = "BLOCKED"`

Rebase from v8.6 §7.5 is deleted; nothing in the production track uses an anchored window.

---

## 11. Output contract

The production pipeline MUST emit exactly this JSON object per week:

```json
{
  "as_of_date": "YYYY-MM-DD",
  "srd_version": "8.7",
  "mode": "NORMAL|DEGRADED|BLOCKED",
  "vintage_mode": "strict|pseudo",
  "state": {
    "post": [0.0, 0.0, 0.0],
    "state_name": "DEFENSIVE|NEUTRAL|OFFENSIVE",
    "dwell_weeks": 0,
    "hazard_covariate": 0.0
  },
  "distribution": {
    "q05": 0.0,
    "q10": 0.0,
    "q25": 0.0,
    "q50": 0.0,
    "q75": 0.0,
    "q90": 0.0,
    "q95": 0.0,
    "q05_ci_low": 0.0,
    "q05_ci_high": 0.0,
    "q95_ci_low": 0.0,
    "q95_ci_high": 0.0,
    "mu_hat": 0.0,
    "sigma_hat": 0.0,
    "p_loss": 0.0,
    "es20": 0.0
  },
  "decision": {
    "excess_return": 0.0,
    "utility": 0.0,
    "offense_raw": 0.0,
    "offense_final": 0.0,
    "stance": "DEFENSIVE|NEUTRAL|OFFENSIVE",
    "cycle_position": 0.0
  },
  "diagnostics": {
    "missing_rate": 0.0,
    "quantile_solver_status": "ok|rearranged|failed",
    "tail_extrapolation_status": "ok|fallback",
    "hmm_status": "ok|degenerate|em_nonconverge",
    "coverage_q10_trailing_104w": 0.0,
    "coverage_q90_trailing_104w": 0.0
  }
}
```

---

## 12. Research track (sidecar, NOT gated by acceptance)

All v8.6 advanced features live here. They MUST NOT write into the production output path.

Preserved modules (retained as code, demoted in the pipeline):

- `research/geometry/` — robust PCA, diffusion map, Nyström, topology, projection gate, rebase, degradation score `G_t`
- `research/state/shadow_hmm.py` — ambient-space HMM
- `research/state/state_matching.py` — Hungarian alignment on co-occurrence
- `research/state/posterior_fusion.py` — arithmetic mixture; geometric pooling allowed here as an experimental variant
- `research/law/spline_quantile.py` — v8.6 §9.2 spline surface
- `research/law/conformal.py` — dual-tail adaptive conformal
- `research/law/evt_bayes.py` — hierarchical Bayesian GPD
- `research/features/midas.py` — MIDAS smoothing of monthly/quarterly series
- `research/features/challenger_pool.py` — BEA / IP series, challenger promotion tests

Research output is written to `research_report.json` per week and is not consumed by the production engine. A coding agent MUST fail CI if `production_output.json` imports anything under `research/`.

Promotion from research to production requires:

1. A spec revision raising the document version from v8.7 to v8.8+
2. The candidate module passes the acceptance tests of §16 *by itself*, i.e. the production pipeline with the candidate module enabled must beat the production pipeline without it on every acceptance metric, with a block-bootstrap p-value below 0.05 on both `block_length = 52` and `78`.
3. The candidate does not increase `BLOCKED`-week proportion.
4. Data-contract PIT status does not degrade.

---

## 13. Repository layout

```
repo/
  pyproject.toml
  README.md
  configs/
    base.yaml
    data.yaml
    features.yaml
    state.yaml
    law.yaml
    decision.yaml
    backtest.yaml
    research.yaml
  data_contract/
    fred_client.py
    alfred_client.py
    vintage_registry.py          # §4.3 hard-coded start dates
    point_in_time.py             # get_series_pti(..., vintage_mode=)
    calendars.py
  features/
    scaling.py                   # tanh squash + clip §5
    block_builder.py             # produces 10 frozen features §6
    pca.py                       # robust PCA for HMM observation
  state/
    ti_hmm_single.py             # §7, production single HMM
    censored_likelihood.py
    state_label_map.py           # E[R_52] ordering §7.2
  law/
    linear_quantiles.py          # §8.1, constrained linear quantile regression
    quantile_moments.py          # §9.1, moments from piecewise-linear CDF
    tail_extrapolation.py        # §8.2, bounded geometric extrapolation
  decision/
    utility.py                   # §9.3, frozen-z utility
    offense_abs.py               # §9.4, absolute-threshold mapping
    hysteresis.py                # §9.5, no-trade band
    cycle_position.py            # §9.7, diagnostic cycle coordinate
  backtest/
    walkforward.py
    block_bootstrap.py           # §15, stationary block bootstrap
    metrics.py                   # CRPS, coverage, PIT, CEQ, max-dd, turnover
    acceptance.py                # §16, emits pass/fail against thresholds
  research/
    geometry/
      robust_pca.py
      diffusion_map.py
      nystrom.py
      topology.py
      gate.py
      rebase.py
    state/
      shadow_hmm.py
      state_matching.py
      posterior_fusion.py
    law/
      spline_quantile.py
      conformal.py
      evt_bayes.py
      tail_stitch.py
    features/
      midas.py
      challenger_pool.py
    sidecar.py                   # runs all research modules, writes research_report.json
  tests/
    test_vintage_registry.py
    test_point_in_time.py
    test_scaling.py
    test_feature_block.py
    test_ti_hmm_single.py
    test_state_label_map.py
    test_linear_quantiles.py
    test_tail_extrapolation.py
    test_utility.py
    test_offense_abs.py
    test_hysteresis.py
    test_block_bootstrap.py
    test_acceptance.py
    test_research_isolation.py   # enforces research → production firewall
```

---

## 14. Implementation order

Strict order:

1. `data_contract/vintage_registry.py` + `point_in_time.py`
2. `features/scaling.py` + `features/block_builder.py` (reproduces all 10 features from saved FRED snapshots)
3. `features/pca.py`
4. `state/ti_hmm_single.py` + `state_label_map.py`
5. `law/linear_quantiles.py` + `law/tail_extrapolation.py` + `law/quantile_moments.py`
6. `decision/utility.py` + `decision/offense_abs.py` + `decision/hysteresis.py` + `decision/cycle_position.py`
7. `backtest/block_bootstrap.py` + `backtest/metrics.py`
8. `backtest/walkforward.py`
9. `backtest/acceptance.py`
10. `research/sidecar.py` (isolated, last)

Unit tests MUST accompany each module. The walkforward harness MUST NOT run before modules 1–7 pass their tests.

---

## 15. Backtest and inference protocol

### 15.1 Walk-forward

- Frequency: weekly, Friday close.
- Minimum training window: `312` weeks (6 years).
- Refit frequency: weekly (rolling refit of quantile regression and HMM).
- Forecast horizon: 52 weeks ahead.
- Training segment ends `52 + 1 = 53` weeks before the current inference week (to keep the realized target out of training).
- The walkforward harness MUST compute `effective_strict_start` before any production acceptance run.
- If the user-supplied `START` is earlier than `effective_strict_start`, the harness MUST allow only:
  - pseudo-PIT backtest output, and/or
  - strict-PIT diagnostic warmup output with `acceptance_eligible = false`.
- The harness MUST NOT compute §16 acceptance statistics on any week earlier than `effective_strict_start`.
- Under the frozen v8.7.1 feature set and constants, `effective_strict_start = 2014-11-28`.

### 15.2 Block bootstrap

Stationary block bootstrap (Politis and Romano 1994) with geometrically distributed block lengths having **expected** length equal to the stated `block_length`. Both `block_length ∈ {52, 78}` are run.

```
B = 2000
for block_length in [52, 78]:
    bootstrap_stats[block_length] = []
    for b in 1..B:
        resample weeks with stationary block
        compute coverage, CRPS, CEQ, max_dd, turnover
        bootstrap_stats[block_length].append(stats)
report 2.5, 50, 97.5 percentiles for each statistic
```

### 15.3 Baselines

Two baselines run alongside the production model and have the same output contract:

- `Baseline_A`: expanding-window empirical quantiles of 52-week forward returns. No features. No state.
- `Baseline_B`: `mu` = expanding-window mean of 52-week log returns; `sigma` = expanding-window std of 52-week log returns; implied quantiles under a normal distribution.

### 15.4 Turnover

```
turnover_t = |offense_final_t − offense_final_{t-1}| / 100
annualized_turnover = 52 * mean(turnover)
```

---

## 16. Acceptance tests (gated on strict-PIT acceptance segment `effective_strict_start` → only)

A coding agent's implementation passes v8.7.1 iff every item below passes, with all acceptance statistics computed only on the strict-PIT acceptance segment starting at `effective_strict_start`.

For v8.7.1, given the frozen production feature set in §6 and frozen constants in §18, `effective_strict_start = 2014-11-28`.

### 16.1 Deterministic and structural

1. `data_contract/vintage_registry.py` is hard-coded per §4.3; any series requested before its earliest_strict_pit raises.
2. No production module imports from `research/`. `tests/test_research_isolation.py` enforces this.
3. Every inference week emits a JSON conforming to §11. No missing fields.
4. `state_label_map.json` is byte-identical across two independent training runs on the same seed.
5. Quantile non-crossing: for every emitted week, `q05 ≤ q10 ≤ q25 ≤ q50 ≤ q75 ≤ q90 ≤ q95`.

### 16.2 Statistical (strict-PIT acceptance segment `effective_strict_start` →)

Numerical thresholds, frozen:

6. **Interior coverage**:
   - `|empirical_coverage(q10) − 0.10| ≤ 0.03` over the full strict-PIT acceptance segment
   - `|empirical_coverage(q90) − 0.90| ≤ 0.03` over the full strict-PIT acceptance segment
   - computed with HAC-robust (Hodrick 1992) standard errors; bootstrap 95% CI reported

7. **CRPS vs Baseline_A**:
   - mean CRPS improvement ≥ 5% (lower is better)
   - block-bootstrap 5th percentile of improvement must exceed 0%, for both block_length ∈ {52, 78}

8. **CEQ vs Baseline_B**:
   - block-bootstrap 5th percentile of `CEQ_production − CEQ_Baseline_B` > `−50 bp/yr`, for both block lengths

9. **Max drawdown**:
   - `max_dd_production − max_dd_Baseline_B ≤ 300 bp`

10. **Turnover**:
    - annualized turnover of `offense_final` ≤ 1.5 (i.e., equivalent to at most 1.5 full stance changes per year)

11. **Safety**:
    - zero `NaN` or `Inf` in production output over the full strict-PIT acceptance segment
    - `BLOCKED` proportion of weeks ≤ 15%

### 16.3 Diagnostic (report only, not pass/fail)

12. PIT histogram entropy vs uniform: reported with block-bootstrap 95% CI against Baseline_A; not gated.
13. Calibration tail coverage (q05, q95): reported with bootstrap CI; not gated (tail extrapolation is not a strong empirical claim).
14. Cycle-position series plotted over the full segment with overlay of NBER recession bars: qualitative report only.

Items 12–14 are NOT acceptance gates; they are research-quality diagnostics.

---

## 17. Explicit prohibitions (v8.7)

A coding agent MUST NOT:

1. Use future returns for state alignment, state labeling, or quantile fitting.
2. Train a second (shadow) HMM inside the production pipeline.
3. Include `NFCI`, `NFCIRISK`, `NFCICREDIT`, `NFCILEVERAGE`, `STLFSI4`, or any MIDAS/challenger series in the production feature set.
4. Replace the linear quantile regression with a spline, kernel, or neural quantile model in the production pipeline.
5. Use EVT (GPD, POT, hierarchical Bayesian tail) inside the production pipeline.
6. Use a rolling-window z-score for the utility scale; training-window z-stats are frozen at training time.
7. Use a rolling percentile rank as the final offense mapping; absolute thresholds §9.4 are mandatory.
8. Use `DGS3MO` as risk-free benchmark; `DGS1` is mandatory.
9. Compute acceptance statistics on the pseudo-PIT segment (2001–2011).
10. Allow any module under `research/` to write into the production output JSON.
11. Modify any of the frozen numeric constants in this document without a spec revision to v8.8.
12. Treat any week earlier than `effective_strict_start` as acceptance-eligible under the production pipeline.

---

## 18. Frozen numeric constants (audit table)

For grep-ability. A revision MUST update this table and increment the spec version.

| symbol                        | value       | section |
|------------------------------|------------:|---------|
| embargo weeks                | 53          | §15.1   |
| effective_strict_start       | 2014-11-28  | §4.3/§15.1/§16 |
| hard clip bound         | ±5        | §5.2    |
| tanh rescale factor     | 4         | §5.2    |
| HMM state count K       | 3         | §7.1    |
| HMM EM restarts         | 50        | §7.2    |
| HMM EM max iters        | 200       | §7.3    |
| degenerate-state window | 26w       | §7.3    |
| quantile gap            | 1e-4      | §8.1    |
| L2 penalty α            | 2.0       | §8.1    |
| tail extrapolation mult | 0.6       | §8.2    |
| utility λ (es20 weight) | 1.2       | §9.3    |
| utility κ (ploss weight)| 0.8       | §9.3    |
| offense threshold grid  | 20/40/60/80 | §9.4  |
| stance cutoffs          | 35, 65    | §9.6    |
| no-trade band           | 7         | §9.5    |
| degraded missing lower  | 10%       | §10     |
| blocked missing upper   | 20%       | §10     |
| min training weeks      | 312       | §15.1   |
| bootstrap replications  | 2000      | §15.2   |
| bootstrap block lengths | 52, 78    | §15.2   |
| coverage tolerance      | 0.03      | §16.2-6 |
| CRPS improvement min    | 5%        | §16.2-7 |
| CEQ floor vs Baseline_B | −50 bp/yr | §16.2-8 |
| max-dd tolerance        | 300 bp    | §16.2-9 |
| turnover cap            | 1.5/yr    | §16.2-10|
| blocked cap             | 15%       | §16.2-11|

---

## 19. Open items deferred to v8.8+

Items deferred beyond this spec. Each must enter via a named spec revision and pass the §16 acceptance suite independently; none may be bundled.

- §19.1–§19.6: cross-asset panel extension (primary v8.8 priority, fully specified below)
- Reactivation of Bayesian EVT conditional on research-track acceptance (separate revision; MUST NOT be bundled with §19.1)
- Reactivation of geometry engine as a predictive feature (currently diagnostic-only in `research/`)
- Options-implied NDX term structure replacement for `VXN / VXV`
- Execution-cost model beyond the fixed `band` hysteresis
- Monotonic neural quantile surface as candidate for spline replacement

### 19.1 v8.8 first priority — cross-asset tail-stress extension

The first priority for v8.8 is a **cross-asset panel extension** covering `SPX`, `R2K`, `VEU`, and `EEM` in addition to the existing `NASDAQXNDX` target.

Rationale and precise claim:

- Under v8.7 strict-PIT 2012→ only, the production model observes a small number of true large-drawdown episodes on a single asset path. The tail evidence for QQQ alone is insufficient to validate tail behavior under a fresh, unseen crisis.
- Without changing the weekly cadence, 52-week forecast horizon, public-data-only constraint, or strict-PIT acceptance protocol, a cross-asset panel is the only primary path that can materially increase tail-relevant information.
- The valid claim is: the panel extension increases the number of crisis-period **asset-path observations** and the diversity of tail realizations across assets. It MUST NOT be described as creating 4–5× independent crisis events. Cross-asset returns remain conditionally correlated inside a shared global shock; independence is not obtained by asset count.

### 19.2 Scope of the v8.8 panel extension

The v8.8 panel extension MUST preserve the v8.7 production/research separation.

Production-track additions:

1. The target set expands from one asset to five assets:
   - `NASDAQXNDX`
   - `SPX`
   - `R2K`
   - `VEU`
   - `EEM`

2. The macro production feature set in §6 remains the common state input. No research-only features are promoted automatically.

3. The production law becomes a **panel quantile system**:

   \[
   Q_{\tau,a}(R_{t,52}) = \alpha_{\tau,a} + b_{\tau}^\top X_t^z + c_{\tau}^\top \pi_t + \delta_{\tau,a}
   \]

   where:
   - `a` indexes assets,
   - `α_{τ,a}` is an asset-specific intercept,
   - `b_τ` and `c_τ` are **shared across assets** (pooled macro and state sensitivities),
   - `δ_{τ,a}` is an optional penalized asset-specific deviation term that defaults to zero in v8.8 unless explicitly enabled by spec revision.

   Joint estimation MUST keep the L2 penalty `α = 2.0` on `b_τ` and `c_τ` (same as §8.1), add an L2 penalty `α_δ = 20.0` on `δ_{τ,a}` when enabled (strongly shrunk toward zero by default), and retain non-crossing constraints per asset.

4. The production state engine remains a single 3-state TI-HMM. Its state semantics remain ordered by in-sample forward 52-week return, but the ordering statistic MUST be computed on the **panel-average** forward return across the five production assets, not on `NASDAQXNDX` alone. The label map is persisted as `artifacts/panel_state_label_map.json`.

5. Tail extrapolation remains the bounded non-EVT rule from §8.2 of v8.7. Reactivating EVT is a separate v8.8+ item and MUST NOT be bundled with the panel extension.

Research-track additions:

6. Any asset-specific implied-volatility replacement, geometry reactivation, Bayesian EVT, or challenger/MIDAS block remains under `research/` and is still blocked from the production JSON path unless promoted by explicit spec revision and independent acceptance.

### 19.3 Data contract for the panel

1. Each production asset MUST have exactly one public, PIT-clean, total-return target series defined in `data_contract/asset_registry.py`. Candidate canonical sources (subject to the same vintage-registry discipline as §4.3):

   | asset        | target candidate                    | notes                                 |
   |--------------|-------------------------------------|---------------------------------------|
   | `NASDAQXNDX` | Nasdaq-100 TR                       | existing v8.7 target                  |
   | `SPX`        | S&P 500 TR (e.g., `SP500TR`)        | TR variant only; price-only forbidden |
   | `R2K`        | Russell 2000 TR                     | licensing caveat noted                |
   | `VEU`        | FTSE All-World ex-US TR proxy       | ETF TR acceptable if reconstructed    |
   | `EEM`        | MSCI Emerging Markets TR proxy      | ETF TR acceptable if reconstructed    |

   The exact series identifiers MUST be pinned in `asset_registry.py` before v8.8 implementation begins. Selection between native index TR and ETF-derived TR is a governance decision; whichever is chosen MUST be frozen for the life of the spec.

2. If a given asset lacks a strict-PIT-clean target history for a week `t`, that asset-week MUST be excluded from acceptance statistics for that asset only; it MUST NOT contaminate the other assets' evaluation.

3. The panel acceptance window is the intersection of:
   - the strict-PIT segment already defined in §4.3, and
   - the availability window of each asset's own target series.

   The effective start date of the panel segment is typically later than 2012 because EM and ex-US TR histories become clean PIT later than US TR histories; this is expected.

### 19.4 Statistical protocol for tail-stress validation

The v8.8 panel extension MUST change the bootstrap resampling unit from "single-asset week" to "calendar-week cluster".

1. Resampling unit:
   - sample weeks by stationary block bootstrap exactly as in §15.2,
   - when a week is selected, include the full cross-section of all available assets for that week,
   - MUST NOT resample asset-weeks independently.

2. Report both:
   - **asset-level** metrics (`CRPS`, `coverage(q10)`, `coverage(q90)`, `CEQ`, `max_dd`, `turnover`) for each asset separately,
   - **panel-level** metrics, computed as the equal-weight average across available assets at each week.

3. Acceptance for panel promotion requires all of:
   - mean panel `CRPS` improvement vs panel `Baseline_A` ≥ 5%,
   - block-bootstrap 5th percentile of panel `CRPS` improvement > 0 for both block lengths `{52, 78}`,
   - **no individual asset** has `q10` or `q90` coverage error exceeding `±0.05`,
   - panel `CEQ` 5th percentile vs panel `Baseline_B` > `−50 bp/yr`,
   - `BLOCKED` proportion does not rise relative to the v8.7 single-asset production model.

4. A coding agent MUST report the effective number of asset-week observations used in each acceptance metric, asset by asset and in panel aggregate. This is an audit requirement, not a pass/fail threshold: inflated apparent sample size via asset-week independence is the primary statistical risk and must be visible.

### 19.5 Output contract for v8.8

The production output becomes a per-week panel object:

```json
{
  "as_of_date": "YYYY-MM-DD",
  "srd_version": "8.8",
  "panel_assets": ["NASDAQXNDX", "SPX", "R2K", "VEU", "EEM"],
  "panel_diagnostics": {
    "available_assets": 5,
    "panel_crps_vs_baseline_a": 0.0,
    "panel_ceq_vs_baseline_b": 0.0,
    "effective_asset_weeks": 0
  },
  "common": {
    "mode": "NORMAL|DEGRADED|BLOCKED",
    "vintage_mode": "strict|pseudo",
    "state": {
      "post": [0.0, 0.0, 0.0],
      "state_name": "DEFENSIVE|NEUTRAL|OFFENSIVE",
      "dwell_weeks": 0,
      "hazard_covariate": 0.0
    },
    "cycle_position": 0.0
  },
  "assets": {
    "NASDAQXNDX": { "distribution": {}, "decision": {}, "diagnostics": {} },
    "SPX":        { "distribution": {}, "decision": {}, "diagnostics": {} },
    "R2K":        { "distribution": {}, "decision": {}, "diagnostics": {} },
    "VEU":        { "distribution": {}, "decision": {}, "diagnostics": {} },
    "EEM":        { "distribution": {}, "decision": {}, "diagnostics": {} }
  }
}
```

Each per-asset block retains the `distribution`, `decision`, and `diagnostics` schemas of v8.7 §11. The `mode`, `vintage_mode`, `state`, and `cycle_position` fields are lifted to the `common` block because they are driven by shared macro state and data contract, not by asset-specific latent labels.

### 19.6 Explicit non-claims

The v8.8 panel extension MUST NOT claim:

- that five assets create five independent crisis histories,
- that panel expansion by itself validates EVT,
- that tail robustness demonstrated on the panel automatically transfers to QQQ in the next unseen crisis.

The valid claims are narrower and MUST be stated in any research or marketing material:

- panel expansion increases tail-relevant information,
- it improves cross-sectional stress comparability,
- it provides a materially stronger basis for evaluating whether the production law survives beyond a single-asset historical path,
- it does not eliminate the fundamental scarcity of 52-week-horizon independent tail events.

### 19.7 v8.8 repository increment

Additions (no deletions from v8.7):

```
data_contract/
  asset_registry.py               # §19.3, pinned per-asset TR target series
law/
  panel_quantiles.py              # §19.2, joint panel quantile system
backtest/
  cluster_block_bootstrap.py      # §19.4, week-cluster resampling
  panel_metrics.py                # §19.4, panel CRPS / coverage / CEQ
  panel_acceptance.py             # §19.4, panel acceptance gating
tests/
  test_asset_registry.py
  test_panel_quantiles.py
  test_cluster_block_bootstrap.py
  test_panel_metrics.py
```

The v8.7 single-asset production stack MUST remain runnable in parallel for at least one revision cycle after v8.8 is declared. Dual-run is required so that regression on the single-asset path is visible and cannot be silently traded away for panel-average improvement.

---

## 20. Final implementation note

If any module fails its acceptance criteria, the coding agent MUST prefer, in this order:

1. Tighten the failing module while respecting §18 frozen constants;
2. Fall back to the approved baseline path (rearranged quantiles, uniform-prior HMM, training-window empirical tails);
3. Declare the week `BLOCKED` per §10;
4. Halt the walkforward and surface a structured error.

A coding agent MUST NOT invent a new unsanctioned mechanism to pass acceptance.

This document is the implementation contract. v8.6 is superseded.
