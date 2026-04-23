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
    panel = _load_config("panel.yaml")

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
    assert panel["panel_size"] == 3
    assert panel["minimum_viable_panel_size"] == 3
    assert panel["macro_feature_count"] == 7
    assert panel["micro_feature_count"] == 3
    assert panel["l2_alpha_macro"] == 2.0
    assert panel["l2_alpha_micro"] == 2.0
    assert panel["min_gap"] == 1.0e-4
    assert panel["tail_mult"] == 0.6
    assert panel["coverage_tol"] == 0.05
    assert panel["coverage_collapse"] == 0.08
    assert panel["crps_min_improve"] == 0.05
    assert panel["blocked_cap"] == 0.15
    assert panel["B"] == 2000
    assert panel["block_lengths"] == [52, 78]
    assert panel["min_train"] == 312
    assert panel["embargo_weeks"] == 53
    assert panel["label_anchor"] == "SPX"
    assert panel["hmm_vol_series"] == "VIXCLS"
