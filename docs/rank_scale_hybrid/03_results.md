# Rank-Scale Hybrid Results

> 文档性质  
> 本文记录 `src/research/run_rank_scale_hybrid_experiment.py` 的真实跑批结果。完整机器可读结果见 `artifacts/research/rank_scale_hybrid/rank_scale_hybrid_results.json`。

## 1. 实验输入

- T5 来源：当前仓库内恢复的 `T5_resid_persistence_M4`。
- Scale baseline：当前仓库内 EGARCH(1,1)-Normal benchmark fitter。
- 评估窗口：2017、2018、2020 三个 pilot 窗口。
- 结果口径：按 T5 与 EGARCH-Normal 共同可用日期对齐后计算诊断指标。

## 2. 完整对照表

| 窗口 | 模型 | mean(z) | std(z) | corr_next | rank_next | lag1_acf(z) | sigma_blowup | pathology | CRPS | 协议判定 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2017 | T5_resid_persistence_M4 | 1.743954 | 2.010464 | 0.629802 | 0.776303 | 0.923395 | 0 | 0 | 0.086197 | PARTIAL_FAIL_SCALE |
| 2017 | EGARCH(1,1)-Normal | -0.474190 | 1.702306 | 0.530067 | 0.504615 | 0.909491 | 0 | 0 | 0.061159 | PARTIAL_FAIL_SCALE |
| 2017 | hybrid_quantile_map_zero_param | -1.484761 | 2.851035 | -0.744516 | -0.742066 | 0.913215 | 0 | 0 | 0.067365 | FULL_FAIL |
| 2017 | hybrid_compressed_quantile_map_alpha_0_50 | -1.043258 | 2.249375 | -0.753675 | -0.742066 | 0.928914 | 0 | 0 | 0.064975 | FULL_FAIL |
| 2018 | T5_resid_persistence_M4 | -1.880406 | 1.976233 | 0.386938 | 0.535385 | 0.842405 | 0 | 0 | 0.067559 | PARTIAL_FAIL_SCALE |
| 2018 | EGARCH(1,1)-Normal | -2.507228 | 2.177269 | -0.485068 | -0.580769 | 0.939049 | 8 | 0 | 0.095610 | FULL_FAIL |
| 2018 | hybrid_quantile_map_zero_param | -1.086796 | 2.223179 | 0.606922 | 0.536923 | 0.786941 | 8 | 0 | 0.083995 | FULL_FAIL |
| 2018 | hybrid_compressed_quantile_map_alpha_0_50 | -1.448898 | 2.175198 | 0.629923 | 0.536923 | 0.854937 | 1 | 0 | 0.089628 | FULL_FAIL |
| 2020 | T5_resid_persistence_M4 | 1.057614 | 2.380050 | -0.033011 | 0.020000 | 0.757961 | 0 | 0 | 0.054682 | FULL_FAIL |
| 2020 | EGARCH(1,1)-Normal | 2.394477 | 0.820015 | 0.344931 | 0.314783 | 0.516473 | 0 | 0 | 0.158221 | SUCCESS |
| 2020 | hybrid_quantile_map_zero_param | 2.485995 | 0.948251 | -0.028351 | 0.095652 | 0.609329 | 0 | 0 | 0.160849 | PARTIAL_FAIL_DIRECTION |
| 2020 | hybrid_compressed_quantile_map_alpha_0_50 | 2.492664 | 0.875340 | -0.002492 | 0.095652 | 0.738982 | 0 | 0 | 0.161578 | PARTIAL_FAIL_DIRECTION |

## 3. Hybrid 候选摘要

| 候选 | 2017 | 2018 | 2020 | 候选级裁决 |
|---|---|---|---|---|
| hybrid_quantile_map_zero_param | FULL_FAIL | FULL_FAIL | PARTIAL_FAIL_DIRECTION | FAIL |
| hybrid_compressed_quantile_map_alpha_0_50 | FULL_FAIL | FULL_FAIL | PARTIAL_FAIL_DIRECTION | FAIL |

## 4. 结果解释

主候选在 2017 出现方向反转：`corr_next = -0.744516`，`rank_next = -0.742066`。在 2018，方向为正，但 `std(z) = 2.223179` 且 `sigma_blowup = 8`，未通过尺度与安全门槛。在 2020，尺度达标，`std(z) = 0.948251`，但 `corr_next = -0.028351`，未通过方向门槛。

保序压缩对照未改变结论。它在 2020 的 `std(z)` 降至 `0.875340`，但 `corr_next = -0.002492` 仍未转正；2017 仍出现方向反转，2018 仍未满足尺度与安全门槛。

因此，T5 rank 与 EGARCH-Normal scale 在本轮低自由度 hybrid 中没有形成三窗口共同可用的 `sigma_t`。
