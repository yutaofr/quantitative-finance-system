# T5 Reproduction Note

> 文档性质  
> 本文记录原始 `T5_resid_persistence_M4` 从 `/tmp/qfs-sigma/repo` 回接到当前仓库后的可复现执行结果。本文不是新架构文档，不修改协议，不替代 benchmark family 结果。

## 1. 文档目的

本文件用于回答：原始 pilot 中的 `T5_resid_persistence_M4` 是否能在当前仓库内通过可执行 runner 复现 2017 / 2018 / 2020 三个固定窗口的诊断表。

## 2. 原始 T5 来源路径

| 来源文件 | 用途 |
|---|---|
| `/tmp/qfs-sigma/repo/src/research/run_ndx_sigma_output_transform_pilot.py` | 原始 `T5_resid_persistence_M4` 定义、`_fit_t5`、`_predict_t5`、sigma_t 生成路径、三窗口切片 |
| `/tmp/qfs-sigma/repo/src/research/run_ndx_sigma_output_refinement_pilot.py` | 后续 refinement 调用链参考；未用于替代原始 T5 定义 |
| `/tmp/qfs-sigma/repo/src/research/run_ndx_sigma_output_monotone_family_scan.py` | 后续 monotone Stage B 口径参考；未用于替代原始 T5 定义 |

## 3. 迁移的最小函数

迁移文件：`src/research/t5_recovered_source.py`

迁移范围只覆盖原始 `T5_resid_persistence_M4` 复现所需函数：

| 函数 / 对象 | 来源 | 当前用途 |
|---|---|---|
| `WINDOWS` | `run_ndx_sigma_output_transform_pilot.py` | 固定三窗口切片 |
| `_fit_t5` | `run_ndx_sigma_output_transform_pilot.py` | 训练 residual-persistence 系数 `c` |
| `_predict_t5` | `run_ndx_sigma_output_transform_pilot.py` | 生成 T5 corrected `sigma_t` |
| `_build_train_base` | `run_ndx_sigma_output_transform_pilot.py` | 构造 416 周训练窗、53 周 embargo、M4 base sigma train path |
| `_eval_original_t5_window` | 从原始 `_eval_transformed_window` 收窄 | 只执行 `T5_resid_persistence_M4` 分支 |
| M4 HAR + IV slopes helpers | 原始 T5 上游依赖 | 复现 `M4_har_plus_iv_slopes` base sigma |

未迁移 T1/T2/T3/T4/T6、G 系列、B 系列、trigger、bootstrap、override、hard switch、MoE。

## 4. 当前仓库内实际调用链

```text
src/research/run_phase0a_t5_reproduction.py
  -> research.t5_recovered_source.run_original_t5_reproduction()
    -> build_har_context(max_window_end)
    -> _eval_original_t5_window(context, start, end)
      -> _build_train_base(context, "M4_har_plus_iv_slopes", as_of)
      -> _fit_t5(train_base)
      -> _predict_t5(prev_abs_e, prev_sigma, base_sigma, fit)
      -> quantiles = mu_hat + sigma_hat * empirical standardized residual quantiles
      -> _window_metrics(...)
```

## 5. 训练窗与输出频率口径

| 项目 | 当前复现口径 |
|---|---|
| model | `T5_resid_persistence_M4` |
| base model | `M4_har_plus_iv_slopes` |
| train window | `R1_TRAIN_WINDOW = 416` |
| embargo | 53 周；`train_end = as_of - 53 weeks` |
| history frame | `R1_TRAIN_WINDOW + 54` |
| output frequency | 周频；`frame.feature_dates`，Friday-close aligned |
| windows | 2008 benchmark window + 2017 / 2018 / 2020 三个固定 pilot window |

## 6. 三窗口 T5 结果总表

| 模型 | 窗口 | mean(z) | std(z) | corr_next | rank_next | lag1_acf(z) | sigma_blowup | pathology | CRPS | 状态 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| T5_resid_persistence_M4 | 2008-01-04 -> 2008-12-26 | FAILED_TO_RUN_NONFINITE_STANDARDIZED_RESIDUALS | FAILED_TO_RUN_NONFINITE_STANDARDIZED_RESIDUALS | FAILED_TO_RUN_NONFINITE_STANDARDIZED_RESIDUALS | FAILED_TO_RUN_NONFINITE_STANDARDIZED_RESIDUALS | FAILED_TO_RUN_NONFINITE_STANDARDIZED_RESIDUALS | FAILED_TO_RUN_NONFINITE_STANDARDIZED_RESIDUALS | FAILED_TO_RUN_NONFINITE_STANDARDIZED_RESIDUALS | FAILED_TO_RUN_NONFINITE_STANDARDIZED_RESIDUALS | FAILED_TO_RUN_NONFINITE_STANDARDIZED_RESIDUALS |
| T5_resid_persistence_M4 | 2017-07-07 -> 2017-12-29 | 1.743954 | 2.010464 | 0.629802 | 0.776923 | 0.923395 | 0 | 0 | 0.086197 | PASS |
| T5_resid_persistence_M4 | 2018-07-06 -> 2018-12-28 | -1.795901 | 2.262279 | 0.363253 | 0.533846 | 0.754082 | 0 | 0 | 0.067886 | PASS |
| T5_resid_persistence_M4 | 2020-01-03 -> 2020-06-26 | 1.047204 | 2.489387 | -0.043688 | 0.013913 | 0.770902 | 0 | 0 | 0.055013 | PASS |

## 7. 与历史 checksum 的核对结果

历史 checksum 仅作为同源核对参考，未写入代码目标。

| 窗口 | 当前 std(z) | 历史参考 std(z) | 核对结果 |
|---|---:|---:|---|
| 2017 | 2.010464 | ≈ 2.016 | MATCH_WITHIN_TOLERANCE |
| 2018 | 2.262279 | ≈ 1.931 | MISMATCH |
| 2020 | 2.489387 | ≈ 2.361 | MISMATCH |

当前执行路径确认是原始 `T5_resid_persistence_M4`，但 2018 / 2020 的 `std(z)` 与历史 checksum 偏差明显。最小解释边界：当前仓库数据、依赖代码或缓存状态与 `/tmp/qfs-sigma/repo` 历史运行环境不完全一致；本文不使用历史数字覆盖当前真实跑批结果。


## 8. 2008 candidate-side result

| 项目 | 结果 |
|---|---|
| window | `2008-01-04 -> 2008-12-26` |
| benchmark 口径一致性 | YES: 周频 Friday-close aligned、rolling fixed-length、416 周训练窗、53 周 embargo、同一 recovered T5 数据路径 |
| actual call path | `run_phase0a_t5_reproduction.py -> run_original_t5_reproduction() -> _eval_original_t5_window()` |
| run status | `FAILED_TO_RUN_NONFINITE_STANDARDIZED_RESIDUALS` |
| failure point | 原始 T5 在 2008 首个 as_of 的训练窗中生成非有限 standardized residuals，无法合法计算 empirical standardized residual quantiles |
| 是否改口径重跑 | NO |

2008 完整指标均为 `FAILED_TO_RUN_NONFINITE_STANDARDIZED_RESIDUALS`，因此不能生成 2008 T5 `std(z)`、directionality 或 CRPS 数值。

## 9. 与 `02_benchmark_results_filled.md` 的同步关系

`docs/phase0b/02_benchmark_results_filled.md` 已用当前 runner 真实结果替换 T5 三窗口 `FAILED_TO_RUN_MISSING_T5_SOURCE`。

Benchmark 四模型结果未重跑、未修改。

## 10. 是否成功替换 `FAILED_TO_RUN`

已成功替换 2017 / 2018 / 2020 三窗口 candidate-side 结果。

2008 candidate-side 已按同口径尝试运行，但结果为 `FAILED_TO_RUN_NONFINITE_STANDARDIZED_RESIDUALS`。

## 11. 未解决阻塞项

- 2018 / 2020 `std(z)` 与历史 checksum 存在 mismatch。
- 2008 T5 candidate-side 已落盘为 `FAILED_TO_RUN_NONFINITE_STANDARDIZED_RESIDUALS`，无可用数值指标。
- Phase 0B 入口仍受 trigger audit `FAIL` 约束，不能因 T5 三窗口复现而打开。
