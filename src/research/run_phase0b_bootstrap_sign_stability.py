"""Run Phase 0B bootstrap sign-stability prereg skeleton.

io: writes prereg execution plan payload; no architecture experiment is run here.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

BLOCK_LENGTH = 20
N_BOOT = 400
SEED = 20260426
THRESHOLD = 0.60
METRICS = ("corr_next", "rank_next")


@dataclass(frozen=True, slots=True)
class BootstrapConfig:
    bootstrap_type: str = "block_bootstrap"
    block_length: int = BLOCK_LENGTH
    n_boot: int = N_BOOT
    seed: int = SEED
    metrics: tuple[str, str] = METRICS
    threshold: float = THRESHOLD


@dataclass(frozen=True, slots=True)
class WindowResult:
    window: str
    corr_next_sign_stability: float | None
    rank_next_sign_stability: float | None
    status: str
    reason: str


def _sample_block_starts(n_obs: int, *, rng: np.random.Generator, n_blocks: int) -> np.ndarray:
    max_start = n_obs - BLOCK_LENGTH
    if max_start < 0:
        raise ValueError("n_obs is smaller than block length")
    return rng.integers(0, max_start + 1, size=n_blocks)


def _block_bootstrap_mean_sign_frequency(values: np.ndarray, *, rng: np.random.Generator) -> float:
    n_obs = int(values.shape[0])
    n_blocks = int(np.ceil(n_obs / BLOCK_LENGTH))
    positive_count = 0
    for _ in range(N_BOOT):
        starts = _sample_block_starts(n_obs, rng=rng, n_blocks=n_blocks)
        sampled = [values[start : start + BLOCK_LENGTH] for start in starts]
        boot = np.concatenate(sampled)[:n_obs]
        if float(np.mean(boot)) > 0.0:
            positive_count += 1
    return positive_count / N_BOOT


def _evaluate_window(
    window: str,
    payload: dict[str, Any],
    *,
    rng: np.random.Generator,
) -> WindowResult:
    corr_values = payload.get("corr_next")
    rank_values = payload.get("rank_next")
    if not isinstance(corr_values, list) or not isinstance(rank_values, list):
        return WindowResult(
            window=window,
            corr_next_sign_stability=None,
            rank_next_sign_stability=None,
            status="FAILED_TO_RUN_BOOTSTRAP",
            reason="missing corr_next/rank_next series",
        )
    corr = np.asarray(corr_values, dtype=float)
    rank = np.asarray(rank_values, dtype=float)
    if corr.size < BLOCK_LENGTH or rank.size < BLOCK_LENGTH:
        return WindowResult(
            window=window,
            corr_next_sign_stability=None,
            rank_next_sign_stability=None,
            status="FAILED_TO_RUN_BOOTSTRAP",
            reason="insufficient observations for block length",
        )
    corr_freq = _block_bootstrap_mean_sign_frequency(corr, rng=rng)
    rank_freq = _block_bootstrap_mean_sign_frequency(rank, rng=rng)
    status = "PASS" if corr_freq >= THRESHOLD and rank_freq >= THRESHOLD else "FAIL"
    return WindowResult(
        window=window,
        corr_next_sign_stability=float(corr_freq),
        rank_next_sign_stability=float(rank_freq),
        status=status,
        reason="",
    )


def _build_prereg_only_payload(
    config: BootstrapConfig,
    *,
    input_path: Path | None,
) -> dict[str, Any]:
    return {
        "mode": "prereg_skeleton_only",
        "status": "UNFILLED_INPUT_SOURCE",
        "reason": "input source for corr_next/rank_next window series is not wired in this step",
        "config": asdict(config),
        "input_path": str(input_path) if input_path is not None else None,
        "rules": {
            "double_test_required": True,
            "window_pass_rule": "corr_next_sign_stability>=0.60 and rank_next_sign_stability>=0.60",
            "failure_status": "FAILED_TO_RUN_BOOTSTRAP",
        },
    }


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase 0B bootstrap sign-stability prereg skeleton",
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        default=None,
        help="Optional JSON with window->{corr_next:[...], rank_next:[...]}",
    )
    args = parser.parse_args(argv)

    config = BootstrapConfig()
    if args.input_json is None:
        sys.stdout.write(json.dumps(_build_prereg_only_payload(config, input_path=None), indent=2))
        sys.stdout.write("\n")
        return 0

    if not args.input_json.exists():
        payload = _build_prereg_only_payload(config, input_path=args.input_json)
        payload["status"] = "FAILED_TO_RUN_BOOTSTRAP"
        payload["reason"] = "input json path does not exist"
        sys.stdout.write(json.dumps(payload, indent=2))
        sys.stdout.write("\n")
        return 0

    raw = json.loads(args.input_json.read_text())
    if not isinstance(raw, dict):
        payload = _build_prereg_only_payload(config, input_path=args.input_json)
        payload["status"] = "FAILED_TO_RUN_BOOTSTRAP"
        payload["reason"] = "input json must be an object mapping windows"
        sys.stdout.write(json.dumps(payload, indent=2))
        sys.stdout.write("\n")
        return 0

    rng = np.random.default_rng(SEED)
    window_results: list[dict[str, Any]] = []
    for window, window_payload in raw.items():
        if not isinstance(window_payload, dict):
            result = WindowResult(
                window=str(window),
                corr_next_sign_stability=None,
                rank_next_sign_stability=None,
                status="FAILED_TO_RUN_BOOTSTRAP",
                reason="window payload is not an object",
            )
            window_results.append(asdict(result))
            continue
        result = _evaluate_window(str(window), window_payload, rng=rng)
        window_results.append(asdict(result))

    output = {
        "mode": "bootstrap_sign_stability",
        "config": asdict(config),
        "window_results": window_results,
    }
    sys.stdout.write(json.dumps(output, indent=2))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
