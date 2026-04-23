# SRD v8.8.0 (revision-3)
## Cross-Asset Panel Law Engine — Executable Specification

**Status**: Frozen for implementation after 5 expert reviews (revision-4). Supersedes SRD v8.7.1 §19.
**Audience**: AI coding agents / quantitative research engineers.
**Language**: Python 3.11+.
**Execution frequency**: Weekly (Friday close, NY time).
**Prediction horizon**: 52 weeks.
**Assets**: Fixed 3-asset panel (QQQ, SPY, IWM). See §P2.
**Document precedence**: v8.8.0 §P* > v8.7.1 §19. All v8.7.1 §1–§18 remain in force for the single-asset production path.

---

## 0. How to read this document

This SRD is an **additive extension** to SRD v8.7.1. It does NOT replace v8.7.1 §1–§18.

**Inheritance rule**: All v8.7.1 sections §1–§18 remain frozen and binding for the single-asset production path. This document adds sections §P1–§P14 that define the panel challenger track. The panel track runs in physical isolation from the v8.7.1 production path.

**Normative keywords**: Same as v8.7.1 §0.

**Relationship to v8.7.1 §19**: This document fully specifies what v8.7.1 §19 sketched. Where this document conflicts with §19, this document governs.

**Scope boundary**: This document specifies **v8.8.0 only** — a fixed 3-asset panel (QQQ, SPY, IWM). All references to VEU, EEM, 5-asset panel, and dynamic panel sizing are confined to the **Appendix A (v8.8.1 Deferred)** section and carry zero normative weight for v8.8.0 implementation.

---

## P1. Panel scope and design decisions

### P1.1 Rationale

Under v8.7.1 strict-PIT (2014-11-28 →), the production model observes approximately 2 true large-drawdown episodes on a single asset (NASDAQXNDX). The tail evidence for QQQ alone is mathematically insufficient to validate tail behavior under an unseen crisis. The cross-asset panel is the only primary path that increases tail-relevant information without relaxing PIT or changing the forecast horizon.

### P1.2 Valid claims

The panel extension increases tail-relevant information and cross-sectional stress comparability. It provides a materially stronger basis for evaluating whether the production law survives beyond a single-asset historical path.

### P1.3 Explicit non-claims

The panel extension MUST NOT claim:
- that five assets create five independent crisis histories
- that panel expansion by itself validates EVT
- that tail robustness demonstrated on the panel automatically transfers to QQQ in the next unseen crisis

### P1.4 Frozen panel design decisions

1. **Phased panel rollout**: v8.8.0 implements a **fixed 3-asset panel** (QQQ+SPY+IWM). v8.8.1 MAY expand to 5 assets (+VEU+EEM) only after the 3-asset evaluator, artifact schema, and strict-PIT contract are fully validated. All via Yahoo Finance ETF Adjusted Close as Total Return proxy. No paid data providers.
2. **Shared 3-state TI-HMM**: Same single HMM as v8.7.1, trained on shared macro features. State label ordering uses **SPX** forward 52-week return as invariant anchor (§P4.2).
3. **Joint panel quantile regression**: Shared macro coefficients `b_τ, c_τ` across all assets. Asset-specific intercepts `α_{τ,a}`. Shared micro coefficients `δ_τ` for asset-specific volatility features.
4. **Dynamic panel sizing deferred to v8.8.1**: v8.8.0 uses a fixed 3-asset set. The 3→5 dynamic expansion is architecturally supported but gated behind v8.8.1 validation.
5. **Masked objective**: When an asset is missing at week `t`, its contribution to the joint loss is zeroed. Other assets continue contributing to shared parameters.
6. **Decision layer freeze**: The v8.7.1 decision layer (utility, offense, hysteresis) is NOT applied to panel assets. Panel validation is Law-only (CRPS, coverage). CEQ is reported as "incomparable" in this phase.
7. **Production isolation**: Panel output writes to `artifacts/panel_challenger/`. It MUST NOT touch `production_output.json` or any v8.7.1 artifact.
8. **Tail extrapolation**: Same bounded non-EVT rule as v8.7.1 §8.2 (`mult = 0.6`), applied per asset.
9. **Micro-feature degradation**: When implied vol is unavailable, micro features MUST degrade to pure realized volatility (RV20/RV52 ratio z-score). Same functional interface, no special code path.

---

## P2. Panel data contract

### P2.1 Panel target series (all Yahoo Finance ETF Adjusted Close)

**v8.8.0 scope (fixed 3-asset panel)**:

| asset_id      | ticker | description                      | inception       | pit_classification  | earliest_strict_pit | notes |
|---------------|--------|----------------------------------|-----------------|---------------------|---------------------|-------|
| `NASDAQXNDX`  | `QQQ`  | Nasdaq-100 TR proxy              | 1999-03-10      | **log-return-PIT**  | 1999-03-10          | Panel uses QQQ ETF adj close |
| `SPX`         | `SPY`  | S&P 500 TR proxy                 | 1993-01-29      | **log-return-PIT**  | 1993-01-29          | SPDR S&P 500 ETF |
| `R2K`         | `IWM`  | Russell 2000 TR proxy            | 2000-05-22      | **log-return-PIT**  | 2000-05-22          | iShares Russell 2000 ETF |

### P2.1.0 PIT classification: log-return-PIT

Yahoo Finance Adjusted Close retrospectively recalculates all historical prices by a multiplicative factor whenever a dividend or split is processed. This means the **price level** visible today for a past date differs from what was observable at that past date.

However, **log returns between consecutive adjusted closes are invariant to multiplicative back-adjustments**. When Yahoo applies a new adjustment factor `f` to all historical prices, every `AdjClose_t` is multiplied by the same `f`, and the ratio `AdjClose_t / AdjClose_{t-1}` cancels `f`. This invariance holds for all derived log-return quantities:

- Target `R_{a,t,52}` = 52-week log return = `log(AdjClose_{t+52} / AdjClose_t)` — invariant
- `RV20_a` = std of daily log returns (annualized) — invariant
- `RV52_a` = std of weekly log returns (annualized) — invariant

**Important**: this is NOT the claim that `AdjClose` returns equal `RawClose` returns (they differ on ex-dividend dates because adjusted close incorporates dividends as reinvested capital). The claim is narrower: **a future back-adjustment does not alter the already-computed log returns**. This makes log-return-derived quantities immune to the retrospective recalculation that invalidates price-level PIT.

We define the classification **`log-return-PIT`** to mean:
- Log returns derived from Adjusted Close are stable against future provider back-adjustments
- Price levels are NOT PIT-safe (never used in v8.8.0)
- This is a **provider-methodology-conditional** PIT approximation, weaker than ALFRED vintage-PIT
- If Yahoo Finance changes its adjustment **methodology** (not just factor) mid-history, this classification is revoked

**Validation requirement**: `tests/test_pit_invariance.py` MUST verify that log returns computed from today's adjusted close are invariant to subsequent back-adjustments by comparing returns computed on two different download dates for the same historical window. If they diverge, the `log-return-PIT` classification is revoked.

**NASDAQXNDX special rule**: For the panel, QQQ ETF adjusted close is used as the target (consistent with other ETF-based assets). The v8.7.1 production path continues to use `NASDAQXNDX` index directly. A coding agent MUST maintain both data paths.

### P2.1.1 Asset data contract registry (hard requirement)

Before any panel code is written, `data_contract/asset_registry.py` MUST contain, for each asset, the following fields verified against actual data pulls (not assumptions):

```python
@dataclass(frozen=True, slots=True)
class PanelAssetSpec:
    asset_id: str               # e.g. "SPX"
    ticker: str                 # e.g. "SPY"
    description: str
    provider: str               # "yahoo_finance"
    is_total_return: bool       # True (ETF adj close includes dividends)
    pit_classification: Literal["log_return_pit", "pseudo_pit_risk"]
    inception_date: date
    first_available_friday: date # verified by actual data fetch
    first_trainable_friday: date # inception + 312w + 53w embargo
    effective_panel_start: date  # computed by scan, NOT hand-calculated
    vol_series_id: str          # e.g. "VIXCLS"
    vol_fallback_id: str | None
    notes: str
```

The `effective_panel_start` MUST be computed by `compute_panel_effective_start()` which scans week-by-week to find the earliest Friday where all required features produce finite values AND the training window + embargo requirements are satisfied. This is the same discipline applied to v8.7.1's `effective_strict_start = 2014-11-28`.

### P2.2 Panel asset-specific volatility series (micro features)

Each v8.8.0 panel asset requires an implied volatility series for constructing asset-specific micro features (`x8_a`, `x9_a`, `x10_a`).

| asset_id     | primary_vol_series | source      | earliest_strict_pit | fallback_vol_series | fallback_start |
|--------------|--------------------|-------------|---------------------|---------------------|----------------|
| `NASDAQXNDX` | `VXNCLS`           | FRED        | 2001-02-02          | none                | —              |
| `SPX`        | `VIXCLS`           | FRED        | 1990-01-02          | none                | —              |
| `R2K`        | `RVXCLS`           | FRED        | 2004-01-02          | `VIXCLS`            | before 2004-01-02 |

**Fallback protocol**: When `primary_vol_series` is unavailable at a given `as_of`, the `fallback_vol_series` MUST be used. The `micro_feature_mode` diagnostic field MUST be set to `"proxy"` for that asset-week. The fallback uses the **same functional interface** — `get_series_pti(fallback_series_id, as_of, vintage_mode)` — no special code path.

### P2.3 Shared macro features (inherited from v8.7.1)

The v8.7.1 §6 feature block (x1–x7) is reused exactly as the shared macro feature set `X^{macro}_t ∈ ℝ^7`. No additions, no modifications.

| idx | name | formula                     | shared across all assets |
|----:|------|-----------------------------|--------------------------|
| 1   | x1   | `DGS10 − DGS2` (level)     | ✔                        |
| 2   | x2   | `Δ13w (DGS10 − DGS2)`      | ✔                        |
| 3   | x3   | `DGS1` (level)              | ✔                        |
| 4   | x4   | `Δ13w EFFR`                 | ✔                        |
| 5   | x5   | `BAA10Y` (level)            | ✔                        |
| 6   | x6   | `Δ13w BAA10Y`               | ✔                        |
| 7   | x7   | `Δ26w log(WALCL)`           | ✔                        |

### P2.4 Asset-specific micro features (per asset, 3 features each)

For each asset `a` with its designated volatility series `VOL_a`:

| idx   | name      | formula (primary mode)                                | formula (RV-only fallback)                     |
|------:|-----------|-------------------------------------------------------|------------------------------------------------|
| 8     | `x8_a`    | `log(VOL_a)` (implied vol level)                      | `log(RV20_a)` (realized vol level)             |
| 9     | `x9_a`    | `log(VOL_a) − log(RV20_a)` (VRP)                     | `log(RV20_a) − log(RV52_a)` (RV ratio z-score)|
| 10    | `x10_a`   | `log(VIXCLS / VXVCLS)` (global vol term structure)    | same (always available)                        |

Where:
- `RV20_a` = annualized realized volatility over trailing 20 trading days of asset `a`'s ETF adjusted close, resampled to Friday close
- `RV52_a` = annualized realized volatility over trailing 52 trading weeks (for fallback mode)
- `x10_a` is identical across all assets (global vol term structure) — this avoids requiring per-asset vol term structures

**Micro-feature degradation protocol**: If `VOL_a` is unavailable at `as_of` (neither primary nor fallback vol series available), the micro features degrade to the RV-only fallback formulas above. The `micro_feature_mode` diagnostic MUST be set to `"rv_only"`. The fallback uses the same functional interface — no special code path.

**No cross-sectional standardization**: Micro features are scaled per-asset only (expanding robust z-score + soft squash + hard clip at ±5, per v8.7.1 §5). Cross-sectional standardization across the N=3 panel is explicitly **prohibited** — with only 3 assets, cross-sectional mean/std would create artificial collinearity and distort signals (an idiosyncratic shock in R2K would mechanically force QQQ and SPX micro features negative). The per-asset expanding z-score already normalizes each asset's features to comparable scales.

### P2.5 Panel PIT windows (v8.8.0 — fixed 3-asset)

v8.8.0 uses a fixed 3-asset panel. Dynamic sizing is deferred to v8.8.1.

```
v8.8.0 fixed panel: QQQ (1999-03-10), SPY (1993-01-29), IWM (2000-05-22)
Binding constraint: IWM inception = 2000-05-22
```

**Panel `effective_panel_start` computation (MUST be scanned, not hand-calculated)**:

The `effective_panel_start` for the 3-asset panel MUST be computed by the same scanning approach used for v8.7.1:

```python
def compute_panel_effective_start(
    asset_registry: Mapping[str, PanelAssetSpec],
    macro_feature_registry: Mapping[str, VintageSpec],
    min_training_weeks: int = 312,
    embargo_weeks: int = 53,
    weekly_calendar: str = "Friday",
) -> date:
    """pure. Scan week-by-week to find the earliest valid panel acceptance date.
    
    For each candidate Friday, verify:
    1. All panel assets have target data available
    2. All macro features produce finite values under strict vintage
    3. At least min_training_weeks of history exist before embargo
    4. The embargo window is fully clearable
    """
```

The **exact date MUST come from the scan function**, not from manual calculation or approximation. Do NOT hard-code or reference any approximate date in implementation code. This is the same discipline that produced `2014-11-28` for v8.7.1.

### P2.6 v8.8.1 expansion prerequisites (informational)

Before VEU and EEM can be added in v8.8.1:

1. A PIT audit MUST confirm whether Yahoo Finance Adjusted Close for these ETFs provides true as-of-date values or retrospectively recalculated values
2. If pseudo-PIT only, VEU/EEM acceptance statistics MUST be computed separately and labeled as `pseudo_pit`
3. The dynamic panel sizing logic (3→4→5 assets) MUST be validated with the 3-asset evaluator already working correctly
4. Loss function weights, sample sizes, and aggregate metrics MUST be explicitly re-defined for the expanded panel

---

## P3. Panel feature engineering

### P3.1 Feature block output

`features/panel_block_builder.py` MUST produce:

```python
def build_panel_feature_block(
    macro_series: Mapping[str, TimeSeries],
    asset_series: Mapping[str, Mapping[str, TimeSeries]],
    as_of: date,
) -> PanelFeatureFrame:
    """pure. Build shared macro + per-asset micro feature matrices."""
```

Returns a `PanelFeatureFrame`:

```python
@dataclass(frozen=True, slots=True)
class PanelFeatureFrame:
    as_of: date
    x_macro: NDArray[np.float64]          # shape (T, 7), shared
    x_micro: Mapping[str, NDArray[np.float64]]  # {asset_id: shape (T, 3)}
    macro_mask: NDArray[np.float64]       # shape (T, 7), bool
    micro_mask: Mapping[str, NDArray[np.float64]]  # {asset_id: shape (T, 3)}
    asset_availability: Mapping[str, NDArray[np.bool_]]  # {asset_id: shape (T,)}
    micro_feature_mode: Mapping[str, str]  # {asset_id: "primary"|"proxy"}
```

### P3.2 Scaling

All features (macro and micro) are scaled per v8.7.1 §5:
1. Expanding-window robust z-score (median/MAD)
2. Soft squash: `4 * tanh(z / 4)`
3. Hard clip: `[-5, 5]`

Macro scalers are shared. Micro scalers are per-asset (each asset's `x8_a` has its own expanding median/MAD).

---

## P4. Panel state engine

### P4.1 Model (macro-anchored shared state)

The panel reuses the v8.7.1 single 3-state TI-HMM architecture exactly. Same K=3, same EM protocol with 50 restarts.

Two changes from v8.7.1:

1. **State label ordering**: uses **SPX forward 52-week return** as the invariant anchor (not panel-average, not QQQ-specific). SPX is chosen because it is the most liquid, longest-history, and most representative single-asset measure of US macro conditions. This ensures label semantics remain stationary regardless of future panel expansion.
2. **HMM observation vector**: volatility slots use **VIXCLS** (market-wide implied vol) instead of NASDAQXNDX-specific `VXNCLS`. This makes the state engine truly shared-macro.

### P4.2 SPX-anchored state ordering

After EM convergence, states are sorted by:

```
state_label(k) = rank_ascending(mean(R_{SPX,t,52} | z_t = k))
```

where `R_{SPX,t,52}` is the 52-week forward log return of SPY. This produces:
- State 0 → DEFENSIVE (lowest mean SPX forward return)
- State 1 → NEUTRAL
- State 2 → OFFENSIVE (highest mean SPX forward return)

**Rationale**: Using a single invariant anchor (SPX) avoids the structural break that would occur if the panel-average return baseline shifts when assets are added/removed in v8.8.1.

### P4.3 Label map persistence

The panel label map is persisted to `artifacts/panel_challenger/panel_state_label_map.json`. It MUST NOT overwrite the v8.7.1 `artifacts/training/state_label_map.json`.

### P4.4 HMM observation vector (shared macro only)

The HMM observation vector `y_t ∈ ℝ^6` is constructed from **purely shared macro inputs**:

```
y_t = [pc1_t, pc2_t, Δ1w pc1_t, Δ1w pc2_t, log(VIXCLS_t), Δ4w log(VIXCLS_t)]
```

The last two slots use **VIXCLS** — the CBOE S&P 500 Volatility Index — which is a market-wide measure, not specific to any single panel asset. This replaces the v8.7.1 use of NASDAQXNDX-specific `VXNCLS`/`VRP` in the observation vector.

**Why not QQQ-specific vol**: Including a single asset's vol feature in the "shared" state engine would asymmetrically anchor the macro state to that asset, contradicting the panel's symmetric design. VIX captures the same macro vol information without the asymmetry.

---

## P5. Panel law engine (joint quantile regression)

### P5.1 Panel quantile equation

For each asset `a` and quantile `τ ∈ {0.10, 0.25, 0.50, 0.75, 0.90}`:

```
Q_{τ,a}(R_{a,t,52} | F_t) = α_{τ,a} + b_τ^⊤ X_t^{macro,z} + c_τ^⊤ π_t + δ_τ^⊤ X_{t,a}^{micro,z}
```

Where:
- `α_{τ,a} ∈ ℝ` — asset-specific intercept (absorbs ETF expense ratio drift and asset-level location)
- `b_τ ∈ ℝ^7` — **shared** macro sensitivity (7 macro features)
- `c_τ ∈ ℝ^3` — **shared** HMM state sensitivity (3-state posterior)
- `δ_τ ∈ ℝ^3` — **shared** micro sensitivity (3 micro features, same coefficients applied to each asset's own micro features)
- `π_t ∈ ℝ^3` — smoothed HMM posterior (same for all assets)

### P5.2 Joint optimization

Implementation MUST:

1. Jointly solve all 5 quantiles × N_t assets in **one** optimization problem per training window
2. The objective is the sum of pinball losses across all quantiles and all available assets:
   ```
   min Σ_τ Σ_{a ∈ available(t)} Σ_t ρ_τ(R_{a,t,52} − Q_{τ,a,t})
       + α_macro * (‖b‖² + ‖c‖²)
       + α_micro * ‖δ‖²
       ```
3. Enforce non-crossing per asset: `Q_{τ_{k+1},a} − Q_{τ_k,a} ≥ 1e-4` for all adjacent τ, for each asset `a`
4. L2 penalties (frozen):
   - `α_macro = 2.0` on `b_τ` and `c_τ` (same as v8.7.1 §8.1)
   - `α_micro = 2.0` on `δ_τ`
5. **Intercept freedom**: `α_{τ,a}` are **unpenalized** — they freely absorb each asset's unconditional drift rate and risk premium differences. QQQ and R2K have fundamentally different unconditional return distributions; penalizing their intercept spread would force `b_τ` and `c_τ` to erroneously absorb structural drift differences. This is a deliberate design choice: intercepts handle location, shared parameters handle conditional variation.

### P5.3 Masked objective (missing asset handling)

When asset `a` has no valid target return at week `t`:

1. Exclude `a`'s pinball loss contribution for that week
2. Exclude `a`'s non-crossing constraint for that week
3. **Keep** the shared parameters `b_τ`, `c_τ`, `δ_τ` receiving gradients from the remaining healthy assets
4. Set `asset_availability[a][t] = False` in the output

This is implemented as a weight mask `w_{a,t} ∈ {0, 1}` on each term of the loss function.

### P5.4 Solver

Use `cvxpy` with ECOS backend (same as v8.7.1). If the joint solver fails:

1. Fall back to per-asset independent quantile regressions (each using its own macro + micro, **no cross-asset coefficient sharing is preserved**)
2. Apply Chernozhukov rearrangement per asset
3. Set `panel_solver_status = "per_asset_fallback"`
4. Set `panel_model_status = "DEGRADED"`
5. Fallback weeks are valid outputs but count as `DEGRADED`. Acceptance statistics **include** these weeks — they are NOT silently excluded.

### P5.5 Tail extrapolation (per asset)

Same as v8.7.1 §8.2, applied independently to each asset:

```
q05_a = q10_a − 0.6 * (q25_a − q10_a)
q95_a = q90_a + 0.6 * (q90_a − q75_a)
```

The multiplier `0.6` is frozen.

---

## P6. Panel backtest protocol (v8.8.0 — fixed 3-asset)

### P6.1 Walk-forward (panel extension)

Same weekly Friday-close cadence as v8.7.1 §15.1. Additional rules:

- **Minimum training window**: 312 weeks (unchanged)
- **Embargo**: 53 weeks (unchanged)
- **Fixed asset set**: v8.8.0 uses a fixed 3-asset panel (QQQ, SPY, IWM) for all weeks. No dynamic panel sizing. The asset set does NOT change during the walk-forward.
- **Refit frequency**: Weekly (same as v8.7.1)

### P6.2 Cluster block bootstrap

The bootstrap resampling unit changes from "single-asset week" to **"calendar-week cluster"**:

1. Sample weeks by stationary block bootstrap exactly as v8.7.1 §15.2
2. When a week is selected, include the **full cross-section** of all 3 assets for that week
3. MUST NOT resample asset-weeks independently (this would inflate apparent sample size)
4. Both `block_length ∈ {52, 78}`, `B = 2000` (unchanged)

### P6.3 Baselines (per asset)

For each panel asset, two baselines run alongside:

- `Baseline_A_a`: expanding-window empirical quantiles of 52-week forward returns for asset `a`
- `Baseline_B_a`: normal-distribution implied quantiles from expanding-window mean and std of 52-week log returns for asset `a`

Panel-aggregate baselines are the equal-weight average across the 3 assets.

### P6.4 Staged validation protocol

Panel validation MUST follow this staged sequence. Each stage gates the next:

**Stage 1 — Smoke Run**:
- Window: 2016-01-01 to 2016-12-31 (1 year)
- Assets: 3 (QQQ, SPY, IWM)
- Pass criteria: Pipeline runs without error; solver converges; artifacts written to correct isolation path; no OOM
- Estimated time: ~15 minutes

**Stage 2 — One-Year Backtest**:
- Window: 2020-01-01 to 2020-12-31 (includes COVID crisis)
- Assets: 3 (QQQ, SPY, IWM)
- Pass criteria: All 3 assets produce non-crossing quantiles; CRPS computable; coverage computable; no NaN/Inf
- Estimated time: ~30 minutes

**Stage 3 — Full Panel Backtest (parallel)**:
- Window: scanned `panel_effective_start` to 2024-12-27
- Assets: 3 (QQQ, SPY, IWM) — fixed throughout
- Pass criteria: §P7 acceptance gates
- Estimated time: ~4 hours (parallelizable across workers)
- Workers: up to 12, partitioned by time segments

---

## P7. Panel acceptance tests

### P7.1 Structural gates (must all pass)

1. **Production isolation**: No panel module imports from `src/law/linear_quantiles.py` or writes to `production_output.json`. `tests/test_panel_isolation.py` enforces this.
2. **v8.7.1 regression**: The v8.7.1 single-asset production path MUST produce byte-identical output before and after panel code is added. `tests/test_v87_regression.py` runs the v8.7.1 golden snapshot and compares.
3. **Quantile non-crossing (per asset)**: For every emitted asset-week, `q05_a ≤ q10_a ≤ q25_a ≤ q50_a ≤ q75_a ≤ q90_a ≤ q95_a`.
4. **Panel state label map stability**: Two independent training runs on the same seed produce byte-identical `panel_state_label_map.json`.
5. **Fixed panel consistency**: At every week, exactly 3 assets (QQQ, SPY, IWM) are present. `tests/test_panel_sizing.py` verifies no asset appears/disappears.

### P7.2 Statistical gates (log-return-PIT acceptance segment)

All statistics computed on the fixed 3-asset panel from scanned `panel_effective_start` onward.

6. **Per-asset coverage (relaxed tolerance)**:
   - For each asset `a`: `|empirical_coverage(q10_a) − 0.10| ≤ 0.05`
   - For each asset `a`: `|empirical_coverage(q90_a) − 0.90| ≤ 0.05`
   - Note: tolerance is `0.05` (not `0.03` as in v8.7.1) because per-asset sample sizes are smaller

7. **Panel-aggregate CRPS**:
   - Panel CRPS = equal-weight average of per-asset CRPS
   - Mean panel CRPS improvement vs panel `Baseline_A` ≥ 5%
   - Cluster block-bootstrap 5th percentile of improvement > 0% for both `block_length ∈ {52, 78}`

8. **No individual asset coverage collapse**:
   - No single asset may have `q10` or `q90` coverage error exceeding `±0.08`
   - This prevents the panel aggregate from masking a single catastrophic asset

9. **BLOCKED proportion**: Panel `BLOCKED` proportion of weeks ≤ 15% (same as v8.7.1)

10. **Safety**: Zero NaN or Inf in panel output over the full strict-PIT acceptance segment

### P7.3 Diagnostic metrics (report only, not pass/fail)

11. **CEQ comparison**: Report per-asset and panel-aggregate CEQ vs Baseline_B. Explicitly label as **"incomparable — Decision layer not adapted for panel"**.
12. **Per-asset CRPS**: Report individually for each asset, with cluster bootstrap CI.
13. **Vol-normalized CRPS**: In addition to raw CRPS, report `CRPS_a / σ_a` where `σ_a` is the expanding-window std of 52-week returns for asset `a`. This prevents high-vol assets (R2K) from dominating the panel aggregate. Report both raw and vol-normalized panel aggregates.
14. **Effective asset-weeks**: Report the total number of asset-week observations used in each metric, asset by asset and in panel aggregate. This is an audit requirement.
15. **Micro feature mode breakdown**: Report fraction of asset-weeks using `"primary"` vs `"proxy"` vs `"rv_only"` volatility series.

### P7.4 Panel evaluator contract (hard requirement)

Before any backtest is run, the panel evaluator MUST be fully specified and tested:

1. **Per-asset metrics**: CRPS, coverage(q10), coverage(q90) — computed independently per asset using only that asset's target returns and quantile predictions
2. **Panel aggregate metrics**: Equal-weight average of per-asset metrics across available assets at each week. The denominator is the number of available assets at that week, not the total panel size.
3. **Fixed vs dynamic panel**: In v8.8.0 (fixed 3-asset), all metrics use the same asset set throughout. No dynamic weighting complications.
4. **Comparable baseline**: All acceptance metrics compare the panel model against per-asset baselines computed on the same asset and time window. Never compare a panel metric against a single-asset baseline from a different time window.

### P7.5 Acceptance flow

```
Stage 1 (smoke) → pass → Stage 2 (1-year) → pass → Stage 3 (full) → §P7.1 + §P7.2 → ACCEPT/REJECT
```

If any §P7.2 gate fails, the panel challenger is rejected. The v8.7.1 production path remains unchanged. No silent promotion.

---

## P8. Panel output contract

### P8.1 Per-week panel output

The panel challenger MUST emit exactly this JSON object per week, written to `artifacts/panel_challenger/as_of=YYYY-MM-DD/panel_output.json`:

```json
{
  "as_of_date": "YYYY-MM-DD",
  "srd_version": "8.8.0",
  "panel_size": 3,
  "available_assets": ["NASDAQXNDX", "SPX", "R2K"],
  "pit_classification": "log-return-PIT",
  "common": {
    "mode": "NORMAL|DEGRADED|BLOCKED",
    "vintage_mode": "strict|pseudo",
    "state": {
      "post": [0.0, 0.0, 0.0],
      "state_name": "DEFENSIVE|NEUTRAL|OFFENSIVE",
      "dwell_weeks": 0,
      "hazard_covariate": 0.0,
      "label_anchor": "SPX"
    }
  },
  "assets": {
    "NASDAQXNDX": {
      "available": true,
      "micro_feature_mode": "primary|proxy|rv_only",
      "distribution": {
        "q05": 0.0, "q10": 0.0, "q25": 0.0, "q50": 0.0,
        "q75": 0.0, "q90": 0.0, "q95": 0.0,
        "mu_hat": 0.0, "sigma_hat": 0.0,
        "p_loss": 0.0, "es20": 0.0
      },
      "diagnostics": {
        "solver_status": "ok|rearranged|per_asset_fallback",
        "tail_status": "ok|fallback",
        "coverage_q10_trailing_104w": 0.0,
        "coverage_q90_trailing_104w": 0.0
      }
    }
  },
  "panel_diagnostics": {
    "panel_solver_status": "ok|per_asset_fallback",
    "panel_crps_vs_baseline_a": 0.0,
    "effective_asset_weeks": 0,
    "missing_assets_this_week": []
  }
}
```

Each asset block under `"assets"` follows the same schema. Assets not available at a given week have `"available": false` and null distribution/diagnostics.

### P8.2 Panel comparison report

After a full backtest, emit `artifacts/panel_challenger/panel_comparison_report.json`:

```json
{
  "srd_version": "8.8",
  "backtest_start": "YYYY-MM-DD",
  "backtest_end": "YYYY-MM-DD",
  "panel_effective_start": "YYYY-MM-DD",
  "acceptance": {
    "passed": true,
    "items": []
  },
  "per_asset_metrics": {
    "NASDAQXNDX": {
      "q10_coverage": 0.0, "q90_coverage": 0.0,
      "crps_improvement": 0.0, "effective_weeks": 0
    }
  },
  "panel_aggregate_metrics": {
    "crps_improvement": 0.0,
    "crps_bootstrap_p05": {"52": 0.0, "78": 0.0},
    "ceq_diff_incomparable": 0.0,
    "blocked_proportion": 0.0,
    "total_asset_weeks": 0
  }
}
```

---

## P9. Panel repository layout

Additions only. No deletions from v8.7.1.

```
src/
  data_contract/
    asset_registry.py              # §P2.1, pinned per-asset ETF target + vol series
    yahoo_client.py                # §P2.1, Yahoo Finance ETF adjusted close fetcher
  features/
    panel_block_builder.py         # §P3, shared macro + per-asset micro
  law/
    panel_quantiles.py             # §P5, joint panel quantile system
  backtest/
    cluster_block_bootstrap.py     # §P6.2, week-cluster resampling
    panel_metrics.py               # §P7, per-asset + panel aggregate metrics
    panel_acceptance.py            # §P7, panel acceptance gates
  engine_types.py                  # extended with PanelWeeklyOutput, PanelFeatureFrame
configs/
    panel.yaml                     # panel-specific config (asset list, penalties, etc.)
tests/
    test_asset_registry.py
    test_panel_block_builder.py
    test_panel_quantiles.py
    test_cluster_block_bootstrap.py
    test_panel_metrics.py
    test_panel_acceptance.py
    test_panel_isolation.py        # panel ↛ production firewall
    test_v87_regression.py         # v8.7.1 golden snapshot unchanged
artifacts/
    panel_challenger/              # physically isolated output directory
        panel_state_label_map.json
        panel_utility_zstats.json  # (if Decision layer is ever activated)
```

The v8.7.1 single-asset production stack MUST remain runnable via `make weekly` / `make backtest`. Panel is invoked via `make panel-smoke` / `make panel-backtest`.

---

## P10. Panel frozen constants (audit table)

All v8.7.1 §18 constants remain in force. These are panel-specific additions:

| symbol                             | value       | section  |
|------------------------------------|-------------|----------|
| panel assets (v8.8.0)              | 3           | §P1.4   |
| minimum viable panel size          | 3           | §P2.5   |
| macro feature count                | 7           | §P2.3   |
| micro feature count (per asset)    | 3           | §P2.4   |
| α_macro (shared L2)               | 2.0         | §P5.2   |
| α_micro (micro L2)                | 2.0         | §P5.2   |
| α_{τ,a} intercept penalty         | 0 (free)    | §P5.2   |
| quantile gap (per asset)          | 1e-4        | §P5.2   |
| tail mult (per asset)             | 0.6         | §P5.5   |
| per-asset coverage tolerance      | 0.05        | §P7.2-6 |
| per-asset coverage collapse limit | 0.08        | §P7.2-8 |
| panel CRPS improvement min        | 5%          | §P7.2-7 |
| blocked cap                       | 15%         | §P7.2-9 |
| bootstrap replications            | 2000        | §P6.2   |
| bootstrap block lengths           | 52, 78      | §P6.2   |
| min training weeks                | 312         | §P6.1   |
| embargo weeks                     | 53          | §P6.1   |
| HMM state label anchor            | SPX         | §P4.2   |
| HMM vol input                     | VIXCLS      | §P4.4   |
| PIT classification                | log-return-PIT | §P2.1.0 |

**Governance**: Changing any value requires a spec revision to v8.9.

---

## P11. Panel implementation order (v8.8.0 — fixed 3-asset)

Strict order. Each step gates the next. Unit tests MUST accompany each module.

**Phase 0 — Data contract hardening (MUST complete before any model code)**:

0a. `data_contract/asset_registry.py` — define 3 assets (QQQ, SPY, IWM) with `PanelAssetSpec` dataclass. All fields verified against actual data fetch.
0b. `data_contract/yahoo_client.py` — Yahoo Finance ETF adjusted close fetcher (IO adapter)
0c. Verify: actual data fetch for all 3 assets, confirm `first_available_friday`, confirm finite values
0d. `compute_panel_effective_start()` — scan and compute exact date
0e. `configs/panel.yaml` — panel-specific config with scanned dates

**Phase 1 — Panel evaluator contract (MUST complete before backtest)**:

1a. `backtest/panel_metrics.py` — per-asset CRPS, coverage, vol-normalized CRPS. Fully specified and unit-tested.
1b. `backtest/panel_acceptance.py` — §P7 acceptance gates with evaluator contract (§P7.4)
1c. `backtest/cluster_block_bootstrap.py` — week-cluster resampling
1d. `tests/test_panel_evaluator_contract.py` — verify evaluator against synthetic data

**Phase 2 — Model implementation**:

2a. `features/panel_block_builder.py` — shared macro (reuse v8.7.1) + per-asset micro features (per-asset z-score only, no cross-sectional standardization)
2b. `state/` — no new modules; verify **SPX-anchored** label map via existing `ti_hmm_single.py` with VIXCLS observation slots
2c. `law/panel_quantiles.py` — joint 3-asset quantile regression with masked objective, unpenalized intercepts

**Phase 3 — Integration and validation**:

3a. Integration wiring in `app/cli.py` — `panel-smoke`, `panel-backtest` commands
3b. `tests/test_panel_isolation.py` + `tests/test_v87_regression.py` (structural)
3c. Stage 1 smoke run (2016, 1 year)
3d. Stage 2 one-year backtest (2020, COVID)
3e. Stage 3 full backtest (scanned start → 2024-12-27)

---

## P12. Panel explicit prohibitions

A coding agent MUST NOT:

1. Modify any v8.7.1 production module (`linear_quantiles.py`, `decision/*`, `walkforward.py`) for panel purposes. Panel code goes in new files only.
2. Write panel output to `production_output.json` or any v8.7.1 artifact path.
3. Import `src/law/panel_quantiles.py` from any v8.7.1 production module.
4. Use `panel_state_label_map.json` in the v8.7.1 production inference path.
5. Treat per-asset CEQ as an acceptance gate (Decision layer is not adapted).
6. Resample asset-weeks independently in bootstrap (inflates apparent sample size).
7. Hard-code a fixed `effective_panel_start` date. It MUST be computed dynamically from the asset registry and training window requirements.
8. Silently drop weeks when a single asset is missing. Use masked objective (§P5.3).
9. Mix price-index and total-return targets within the panel. All must be ETF Adjusted Close.
10. Use any paid data provider for panel target or volatility series.
11. Promote the panel challenger to production without explicit §P7 acceptance passage AND a spec revision to v8.9.
12. Modify any v8.7.1 §18 frozen constant for panel purposes.

---

## P13. Migration path from panel challenger to production

This section is informational. It describes the conditions under which a future v8.9 spec revision may promote the panel to production.

### P13.1 Prerequisites for promotion

1. `panel_comparison_report.json` shows all §P7.1 structural gates passed
2. `panel_comparison_report.json` shows all §P7.2 statistical gates passed
3. No individual asset shows coverage collapse (§P7.2-8)
4. Panel CRPS improvement is statistically significant at p < 0.05 on both block lengths
5. v8.7.1 single-asset path shows no regression

### P13.2 What a v8.9 spec would need to define

- Adapted Decision layer (utility, offense, hysteresis) for multi-asset context
- Portfolio-level CEQ that accounts for cross-asset diversification
- Updated `production_output.json` schema with per-asset blocks
- Retirement timeline for v8.7.1 single-asset path

These are NOT specified in v8.8. They are deferred to v8.9.

---

## P14. Final implementation note

The panel challenger is a research-grade extension that runs in full physical isolation from the v8.7.1 production path. Its sole purpose is to validate whether cross-asset information improves the conditional law. If validation succeeds, a separate spec revision (v8.9) will integrate the panel into production.

A coding agent encountering ambiguity in this document MUST:
1. Check v8.7.1 for inherited rules
2. If still ambiguous, mark `[NEEDS-HUMAN]` in the PR description
3. Never invent unsanctioned mechanisms to pass acceptance

This document is the panel implementation contract. v8.7.1 §19 is superseded.

---

**Version**: SRD v8.8 (exec-final)
**Effective**: Upon user approval
**Companion documents**: SRD v8.7.1 (inherited), ADD v1.0 (to be updated to ADD v1.1 for panel)
**Revision trigger**: Panel acceptance results, v8.9 planning
