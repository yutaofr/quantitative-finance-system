# Rank-Scale Hybrid Final Decision

## 假说陈述

T5 提供可用的 ordinal / rank ordering，EGARCH(1,1)-Normal 提供可用的 cardinal / scale calibration。若该假说成立，则低自由度 `sigma_hybrid(t)` 应能同时继承 T5 的排序优势与 EGARCH-Normal 的尺度优势。

## 实验配置

- 主候选：`hybrid_quantile_map_zero_param`
- 低自由度对照：`hybrid_compressed_quantile_map_alpha_0_50`
- 比较基准：`T5_resid_persistence_M4` 与 `EGARCH(1,1)-Normal`
- 窗口：2017、2018、2020 三个 pilot 窗口
- 成功门槛：三窗口同时 `corr_next > 0`、`rank_next > 0`、`std(z) < 1.5`、`sigma_blowup = 0`、`pathology = 0`
- 运行产物：`artifacts/research/rank_scale_hybrid/rank_scale_hybrid_results.json`

## 结果摘要

| 候选 | 2017 | 2018 | 2020 | 最终裁决 |
|---|---|---|---|---|
| hybrid_quantile_map_zero_param | FULL_FAIL | FULL_FAIL | PARTIAL_FAIL_DIRECTION | FAIL |
| hybrid_compressed_quantile_map_alpha_0_50 | FULL_FAIL | FULL_FAIL | PARTIAL_FAIL_DIRECTION | FAIL |

主候选在 2017 方向反转，在 2018 未通过尺度与安全门槛，在 2020 尺度可控但 `corr_next` 未转正。低自由度保序压缩没有改变协议结论。

## 协议判定

`FAIL`

当前信息集下，T5 rank 与 EGARCH scale 不可在同一低自由度 `sigma_t` 中共存。

## 允许的下一步 / 终止结论

不允许进入 tail family / joint MLE 前的下一阶段。

本研究线按预注册规则终止。不得将失败包装成部分成功，不得更换 hybrid 函数族后继续试验，不得回到 trigger / crisis architecture 路线。

## 新旧线法理边界

旧 Phase 0B 已因 `trigger audit completed FAIL; research line archived` 关闭。rank-scale hybrid 是独立新假说，不是旧 Phase 0B 的恢复。即使本新线曾经成功，也只能授权独立新线的下一阶段；本次失败进一步确认当前证据链下没有恢复旧 crisis architecture 的合法路径。
