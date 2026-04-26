# Rank-Scale Hybrid Sigma Preregistration

> 文档性质  
> 本文是在写入 hybrid 结果之前冻结的方法论预注册。本文定义候选模型、比较基准、成功标准、自由度边界和禁止事项。

## 1. 假说陈述

T5 提供可用的 ordinal / rank ordering，EGARCH(1,1)-Normal 提供可用的 cardinal / scale calibration。待验证的最小新假说是：

`sigma_hybrid(t) = EGARCH_scale(t) x f(T5_rank(t))`

其中 `f` 必须是零参数或极低参数的保序变换，不得依赖 trigger、regime classifier、override、hard switch 或 MoE。

## 2. 候选模型定义

### 2.1 主候选：无参同分位数映射

`hybrid_quantile_map_zero_param`：

1. 在每个评估窗口内计算 T5 sigma 的经验 rank percentile。
2. 将该 percentile 映射到同一窗口 EGARCH-Normal sigma 分布的同分位点。
3. 使用 EGARCH-Normal 的 `mu_hat` 与 hybrid sigma 计算 `z` 和 normal CRPS proxy。

该候选没有可调参数。它以 T5 决定时间排序，以 EGARCH-Normal 的窗口尺度分布决定绝对 sigma 水平。

### 2.2 低自由度对照：保序压缩同分位数映射

`hybrid_compressed_quantile_map_alpha_0_50`：

1. 先执行无参同分位数映射。
2. 在 log-sigma 空间围绕 EGARCH-Normal median 做固定压缩：`alpha = 0.50`。
3. 该变换保留 T5 rank ordering，不改变窗口内 sigma 的单调顺序。

该候选使用 1 个预先写死参数，不做拟合，不做结果后调参。

## 3. 比较基准

| 模型 | 角色 |
|---|---|
| `T5_resid_persistence_M4` | ordinal baseline |
| `EGARCH(1,1)-Normal` | scale baseline |
| `hybrid_quantile_map_zero_param` | primary hybrid candidate |
| `hybrid_compressed_quantile_map_alpha_0_50` | low-degree monotone comparator |

## 4. 成功标准

每个 hybrid 候选必须在 2017、2018、2020 三个窗口同时满足以下全部条件，才可判为 `SUCCESS`：

1. 三窗口同时 `corr_next > 0`。
2. 三窗口同时 `rank_next > 0`。
3. 三窗口同时 `std(z) < 1.5`。
4. 三窗口同时 `sigma_blowup = 0`。
5. 三窗口同时 `pathology = 0`。

窗口内协议标签：

| 条件 | 判定 |
|---|---|
| direction、scale、safety 全通过 | `SUCCESS` |
| direction 与 safety 通过，但 scale 不通过 | `PARTIAL_FAIL_SCALE` |
| scale 与 safety 通过，但 direction 不通过 | `PARTIAL_FAIL_DIRECTION` |
| 其他情况 | `FULL_FAIL` |

候选级最终判定：三个窗口全部为 `SUCCESS` 才是 `SUCCESS`；否则为 `FAIL`。

## 5. 自由度边界

1. hybrid 变换参数不超过 2 个。
2. 本轮只允许零参数同分位数映射与一个固定 `alpha = 0.50` 的保序压缩对照。
3. 不允许新增 trigger。
4. 不允许新增 regime classifier。
5. 不允许 MoE、override 或 hard switch。
6. 不允许新增数据源。
7. 不允许用 2008 结果选择或改写本轮候选。

## 6. 禁止事项

1. 不得在看到结果后更换 hybrid 函数族。
2. 不得在看到结果后调 `alpha`。
3. 不得将失败结果包装成“部分成功”来索取后续立项。
4. 不得将旧 Phase 0B trigger audit 的失败重新解释为未完成事项。
5. 不得把本研究线写成 crisis architecture 的恢复。

## 7. 结果裁决规则

若任一 hybrid 候选达成候选级 `SUCCESS`，则允许进入 tail family / joint MLE 前的下一阶段，但该下一阶段仍属于独立新研究线。

若所有 hybrid 候选均为 `FAIL`，则裁决为：当前信息集下，T5 rank 与 EGARCH scale 不可在同一 `sigma_t` 中共存；不得回到 trigger / crisis architecture 路线。
