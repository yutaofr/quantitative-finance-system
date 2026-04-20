# QQQ Cycle-Conditional Law Engine

A production-ready quantitative finance inference engine implementing SRD v8.7 and ADD v1.0.

This repository strictly adheres to a **spec-first, zero-lookahead, bit-identical** design philosophy. The core mathematical models (features, scaling, HMM, non-crossing quantile regression, decision utility) are implemented as 100% pure functions (Functional Core). All side-effects (API fetching, file I/O, caching) are isolated in the Imperative Shell.

## Key Features

- **Strict Point-In-Time (PIT) Emulation**: Hard truncation of historical data boundaries to eliminate look-ahead bias during training and walk-forward backtesting.
- **Log-space HMM Engine**: Custom robust implementation of a Time-Inhomogeneous Hidden Markov Model with log-sum-exp stabilization, right-censored tail handling, and K-Means++ defensive seeding.
- **Joint Convex Quantile Regression**: Non-crossing conditional quantile fitting via `cvxpy`.
- **Absolute Determinism**: Guaranteed byte-identical JSON outputs for any given configuration and historical dataset.
- **Anti-Corruption Layers**: Graceful `DEGRADED` and `BLOCKED` fallback modes with comprehensive starvation and semantic label-flip guards.

## Local Bootstrap

1. Initialize your environment configuration:
   ```bash
   make init-env
   ```
   *Edit `.env` to add your `FRED_API_KEY` (and `CBOE_TOKEN` if applicable). Yahoo Finance (`^XNDX`) is used as a free fallback for the Nasdaq 100 price series.*

2. Install dependencies using `uv`:
   ```bash
   make sync
   ```

## Running the Engine (Docker)

All production jobs should be executed via Docker to ensure environment parity.

**1. Weekly Inference**
Run the weekly production inference for a specific date (or `auto` for today):
```bash
make weekly AS_OF=2024-12-27
```

**2. Training**
Train the HMM and Quantile Regression models using a defined historical window:
```bash
make train WINDOW=312w
```

**3. Walk-forward Backtest**
Run a full walk-forward backtest over a specified date range:
```bash
make backtest START=2015-01-02 END=2024-12-27
```

**4. Artifact Verification**
Re-run a specific date to verify that the generated `production_output.json` perfectly matches its expected SHA256 hash:
```bash
make verify AS_OF=2024-12-27
```

## Development and Testing

The project relies on an exhaustive matrix of tests and structural guards:

- **Standard Tests:**
  ```bash
  make test
  ```
- **Architectural Guards** (Verifies functional purity, import boundaries, and research isolation):
  ```bash
  make purity
  make research-firewall
  ```
- **Acceptance Gates** (Runs the SRD §16 Acceptance Report matrix):
  ```bash
  make acceptance
  ```
- **Formatting and Linting**:
  ```bash
  make lint
  make type
  make format
  ```
