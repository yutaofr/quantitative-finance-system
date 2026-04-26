"""Run Phase 0A original T5_resid_persistence_M4 reproduction.

io: executes the recovered research-only T5 path and writes JSON diagnostics to stdout.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def run() -> int:
    module = __import__("research.t5_recovered_source", fromlist=["run_original_t5_reproduction"])
    payload = module.run_original_t5_reproduction()
    sys.stdout.write(json.dumps(payload, default=_json_default, indent=2, sort_keys=True))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
