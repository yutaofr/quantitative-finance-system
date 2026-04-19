# QQQ Cycle-Conditional Law Engine

Implementation scaffold for SRD v8.7 and ADD v1.0.

This repository is intentionally spec-first. Production math must be implemented only where SRD
defines it, and the research sidecar must never influence `production_output.json`.

## Local Bootstrap

```bash
make init-env
make sync
make lint
make type
make test
```

Docker runtime:

```bash
make build
make weekly AS_OF=2024-12-27
```

The initial scaffold exposes a health-checkable CLI. Production commands that are not yet
implemented exit explicitly instead of emitting placeholder market data.
