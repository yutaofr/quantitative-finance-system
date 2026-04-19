# SRD v8.7 Production Foundation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the SRD v8.7 production engine in the strict order mandated by SRD §14, starting with testable local contracts before external data I/O.

**Architecture:** Follow ADD §3.1 Functional Core / Imperative Shell. `src/data_contract/` owns PIT and registry I/O boundaries; `src/features/`, `src/law/`, `src/decision/`, `src/backtest/`, and `src/inference/` stay pure and receive data explicitly.

**Tech Stack:** Python 3.11, frozen dataclasses, NumPy, pandas, cvxpy/ECOS, pytest, hypothesis, ruff, mypy strict.

---

### Phase 1: Data Contract Foundation

**SRD:** §4.3, §4.4, §16.1-1  
**ADD:** §4.2, §4.3, Appendix B

**Files:**
- Create: `src/engine_types.py`
- Create: `src/errors.py`
- Create: `src/data_contract/vintage_registry.py`
- Create: `src/data_contract/point_in_time.py`
- Test: `tests/test_vintage_registry.py`

**Deviation from ADD:** ADD Appendix B names `src/types.py`, but this repository sets
`pythonpath = ["src"]`; a top-level `types.py` shadows Python's standard-library
`types` module and makes mypy fail. Use `src/engine_types.py` for the same frozen
contracts.

**Steps:**
1. Write failing tests for all SRD §4.3 earliest strict-PIT dates and forbidden production series.
2. Run `uv run pytest tests/test_vintage_registry.py -q` and confirm import/test failure.
3. Implement frozen shared types and registry constants exactly from SRD §4.3.
4. Implement strict vintage validation that raises `VintageUnavailableError` before earliest strict-PIT.
5. Run `uv run pytest tests/test_vintage_registry.py -q`.

### Phase 2: Scaling Core

**SRD:** §5.1, §5.2, §18  
**ADD:** §5.1, Appendix B, Appendix C

**Files:**
- Create: `src/features/scaling.py`
- Test: `tests/test_scaling.py`

**Steps:**
1. Write failing unit/property tests for expanding robust z-score and soft squash clip bounds/monotonicity.
2. Run `uv run pytest tests/test_scaling.py -q` and confirm failure.
3. Implement `robust_zscore_expanding` and `soft_squash_clip` as pure NumPy functions.
4. Run `uv run pytest tests/test_scaling.py -q`.

### Phase 3: Feature Block

**SRD:** §6  
**ADD:** §5.1, Appendix B

**Files:**
- Create: `src/features/block_builder.py`
- Test: `tests/test_feature_block.py`

**Steps:**
1. Test the 10 frozen feature formulas on explicitly supplied weekly `TimeSeries` inputs.
2. Test missing masks without cross-week interpolation.
3. Implement only SRD §6 formulas; do not add research series.
4. Run `uv run pytest tests/test_feature_block.py -q`.

### Phase 4: Law And Decision Pure Cores

**SRD:** §8, §9  
**ADD:** §5.5, §5.6, Appendix B

**Files:**
- Create: `src/law/tail_extrapolation.py`
- Create: `src/law/quantile_moments.py`
- Create: `src/decision/utility.py`
- Create: `src/decision/offense_abs.py`
- Create: `src/decision/hysteresis.py`
- Create: `src/decision/cycle_position.py`
- Test: matching `tests/test_*.py`

**Steps:**
1. Start with tail extrapolation and hysteresis because they are deterministic SRD formulas.
2. Add quantile moment tests for non-crossing curves and interpolation edge cases.
3. Add offense threshold tests for each grid segment and clamp behavior.
4. Run targeted tests after each module.

### Phase 5: Stateful Model And Walkforward

**SRD:** §7, §15, §16  
**ADD:** §5.4, §7

**Files:**
- Create: `src/state/state_label_map.py`
- Create: `src/state/ti_hmm_single.py`
- Create: `src/backtest/block_bootstrap.py`
- Create: `src/backtest/metrics.py`
- Create: `src/backtest/walkforward.py`
- Create: `src/inference/weekly.py`

**Steps:**
1. Implement `state_label_map` before HMM training so label persistence is deterministic.
2. Implement stationary block bootstrap with injected `rng`.
3. Add HMM only after feature/PCA/law/decision unit tests are stable.
4. Build walkforward last, using only production modules and no research imports.

### Verification Gates

Run after every phase:

```bash
uv run ruff check src tests
uv run mypy src tests
uv run pytest -q
```

Run before PR:

```bash
uv run pytest tests/test_fp_purity.py tests/test_layer_boundaries.py tests/test_research_isolation.py -q
```
