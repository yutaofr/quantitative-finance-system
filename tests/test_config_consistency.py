from __future__ import annotations

from pathlib import Path
from typing import cast

import yaml


def _load_config(name: str) -> dict[str, object]:
    content = Path("configs", name).read_text(encoding="utf-8")
    return cast(dict[str, object], yaml.safe_load(content))


def test_frozen_constants_match_srd_v8_7() -> None:
    features = _load_config("features.yaml")
    law = _load_config("law.yaml")
    state = _load_config("state.yaml")
    decision = _load_config("decision.yaml")
    backtest = _load_config("backtest.yaml")

    assert features["hard_clip_bound"] == 5
    assert law["quantile_gap"] == 1.0e-4
    assert law["l2_alpha"] == 2.0
    assert law["tail_mult"] == 0.6
    assert state["states"] == 3
    assert decision["band"] == 7
    assert backtest["block_lengths"] == [52, 78]
    assert backtest["bootstrap_replications"] == 2000
    assert backtest["coverage_tol"] == 0.03
    assert backtest["crps_min_improve"] == 0.05
    assert backtest["ceq_floor"] == -0.005
    assert backtest["maxdd_tol"] == 0.03
    assert backtest["turnover_cap"] == 1.5
    assert backtest["blocked_cap"] == 0.15
    assert backtest["effective_strict_start"] == "2014-11-28"
