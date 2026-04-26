# T5 Reproduction Note

> 文档性质  
> 本文记录本仓库内 T5 可复现性定位与 runner 执行结果。本文不是新架构文档，不修改协议，不替代 benchmark 结果。

## 1. 文档目的

本文件用于回答：`Stage A = T5` 是否能在当前仓库内通过可执行 runner 复现 2017 / 2018 / 2020 三个固定窗口的诊断表。

## 2. T5 仓库内来源定位结果

### 2.1 指定历史脚本定位

| 预期路径 | 是否存在 |
|---|---|
| `src/research/run_ndx_sigma_output_transform_pilot.py` | 否 |
| `src/research/run_ndx_sigma_output_refinement_pilot.py` | 否 |
| `src/research/run_ndx_sigma_output_monotone_family_scan.py` | 否 |

### 2.2 关键词定位

| 关键词 | 仓库内结果 |
|---|---|
| `T5` | 仅出现在协议 / Phase 0B 文档中，未找到可执行 T5 fit/predict 代码 |
| `_fit_t5` | 未找到 |
| `_predict_t5` | 未找到 |
| `residual persistence` | 未找到可执行实现 |
| `rank correction` | 未找到可执行实现 |
| `output transform` | 未找到可执行实现 |
| `M4` | 未找到可执行 T5 base 定义 |
| `R1` | 仅找到 R1 panel law / runner，与 T5 sigma output transform 来源未建立可执行连接 |

### 2.3 四项来源问题

| 问题 | 当前状态 | 证据 |
|---|---|---|
| T5 的 base 是什么 | UNRESOLVED_ORIGIN_DETAIL | 仓库内没有 T5 source 指明 base 是 M4、R1 或其他 |
| T5 correction 结构是什么 | UNRESOLVED_ORIGIN_DETAIL | 未找到 `_fit_t5` / `_predict_t5` / residual persistence 实现 |
| T5 训练窗与输出频率是什么 | UNRESOLVED_ORIGIN_DETAIL | 只有 benchmark lock 定义 416 周训练、53 周 embargo、周频 Friday 输出；没有 T5-specific source 确认 |
| T5 最终 `sigma_t` 在哪里生成 | UNRESOLVED_ORIGIN_DETAIL | 指定 T5 pilot 脚本缺失，未找到最终 sigma_t 生成函数 |

## 3. 实际调用路径

本轮新增 runner：

`src/research/run_phase0a_t5_reproduction.py`

实际执行路径：

1. 扫描指定历史 T5 pilot 脚本路径
2. 扫描仓库内 T5 / M4 / R1 / output transform / residual persistence / rank correction 关键词
3. 判断是否存在可执行 T5 fit/predict 与 sigma_t 生成路径
4. 输出结构化状态 `FAILED_TO_RUN_MISSING_T5_SOURCE`

该 runner 不硬编码历史 T5 数字，不从对话上下文读取数字，不伪造三窗口结果。

## 4. 训练窗与输出频率口径

| 项目 | 当前状态 |
|---|---|
| benchmark 口径 | rolling fixed-length window，训练窗 416 周，embargo 53 周，周频 Friday close |
| T5-specific 训练窗定义 | UNRESOLVED_ORIGIN_DETAIL |
| T5-specific 输出频率 | UNRESOLVED_ORIGIN_DETAIL |
| 是否可合法假定完全等同 benchmark | 否，仓库内缺少 T5-specific source 证明 |

## 5. 三窗口 T5 结果总表

| 模型 | 窗口 | mean(z) | std(z) | corr_next | rank_next | lag1_acf(z) | sigma_blowup | pathology | CRPS | 状态 |
|---|---|---|---|---|---|---|---|---|---|---|
| T5 | 2017-07-07 -> 2017-12-29 | FAILED_TO_RUN_MISSING_T5_SOURCE | FAILED_TO_RUN_MISSING_T5_SOURCE | FAILED_TO_RUN_MISSING_T5_SOURCE | FAILED_TO_RUN_MISSING_T5_SOURCE | FAILED_TO_RUN_MISSING_T5_SOURCE | FAILED_TO_RUN_MISSING_T5_SOURCE | FAILED_TO_RUN_MISSING_T5_SOURCE | FAILED_TO_RUN_MISSING_T5_SOURCE | FAILED_TO_RUN_MISSING_T5_SOURCE |
| T5 | 2018-07-06 -> 2018-12-28 | FAILED_TO_RUN_MISSING_T5_SOURCE | FAILED_TO_RUN_MISSING_T5_SOURCE | FAILED_TO_RUN_MISSING_T5_SOURCE | FAILED_TO_RUN_MISSING_T5_SOURCE | FAILED_TO_RUN_MISSING_T5_SOURCE | FAILED_TO_RUN_MISSING_T5_SOURCE | FAILED_TO_RUN_MISSING_T5_SOURCE | FAILED_TO_RUN_MISSING_T5_SOURCE | FAILED_TO_RUN_MISSING_T5_SOURCE |
| T5 | 2020-01-03 -> 2020-06-26 | FAILED_TO_RUN_MISSING_T5_SOURCE | FAILED_TO_RUN_MISSING_T5_SOURCE | FAILED_TO_RUN_MISSING_T5_SOURCE | FAILED_TO_RUN_MISSING_T5_SOURCE | FAILED_TO_RUN_MISSING_T5_SOURCE | FAILED_TO_RUN_MISSING_T5_SOURCE | FAILED_TO_RUN_MISSING_T5_SOURCE | FAILED_TO_RUN_MISSING_T5_SOURCE | FAILED_TO_RUN_MISSING_T5_SOURCE |

## 6. 与 `02_benchmark_results_filled.md` 的同步关系

`docs/phase0b/02_benchmark_results_filled.md` 继续保留 T5 三窗口为 `FAILED_TO_RUN_MISSING_T5_SOURCE`，并将 T5-vs-benchmark 依赖字段保留为不可判定。

Benchmark 已有四模型结果未修改。

## 7. 是否成功替换 `FAILED_TO_RUN`

未成功替换为数值结果。

原因为：当前仓库内缺少 T5 的可执行来源定义，无法在不猜测的前提下重建 T5 fit/predict 与最终 `sigma_t` 生成路径。

## 8. 未解决阻塞项

- 缺失函数：未找到 `_fit_t5`、`_predict_t5`
- 缺失 runner：未找到指定三条 historical T5 pilot runner
- 缺失 artifact：未找到可复现 T5 三窗口结果的仓库内 artifact
- 缺失训练窗定义：未找到 T5-specific train window source
- 缺失输出频率口径：未找到 T5-specific output frequency source
- 缺失输入数据定义：未找到 T5 correction 所需输入特征与最终 `sigma_t` 生成定义
