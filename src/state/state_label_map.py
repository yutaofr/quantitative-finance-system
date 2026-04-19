"""Deterministic state semantic mapping from SRD v8.7 section 7.2."""

from __future__ import annotations

from collections.abc import Mapping
import json
import math

from engine_types import Stance

STATE_COUNT = 3
ORDERED_STANCES: tuple[Stance, ...] = ("DEFENSIVE", "NEUTRAL", "OFFENSIVE")


def build_label_map(forward_52w_returns_by_state: Mapping[int, float]) -> dict[int, Stance]:
    """pure. Map HMM state indices to fixed semantics by ascending forward return."""
    if len(forward_52w_returns_by_state) != STATE_COUNT:
        msg = "state label map requires exactly 3 states"
        raise ValueError(msg)

    pairs: list[tuple[int, float]] = []
    for state_idx, avg_return in forward_52w_returns_by_state.items():
        if not math.isfinite(avg_return):
            msg = "state average forward returns must be finite"
            raise ValueError(msg)
        pairs.append((state_idx, avg_return))

    ordered = sorted(pairs, key=lambda item: (item[1], item[0]))
    return {state_idx: ORDERED_STANCES[rank] for rank, (state_idx, _) in enumerate(ordered)}


def label_map_json_bytes(label_map: Mapping[int, Stance]) -> bytes:
    """pure. Return canonical bytes suitable for app-layer artifact persistence."""
    payload = {str(key): label_map[key] for key in sorted(label_map)}
    return json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
