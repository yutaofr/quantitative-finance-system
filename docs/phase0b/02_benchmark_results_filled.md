# Benchmark 结果文档（已填充）

> 文档性质  
> 本文是 benchmark 结果本身，不是模板。本文只承接 `01_benchmark_lock.md` 已锁定协议下的真实跑批结果，并服务于 Phase 0A 的分支判定。

## 1. 文档目的

本文件用于记录 Phase 0A Benchmark Delivery 的真实结果，并回答以下问题：

1. T5 的失败是否为现架构特有缺陷。
2. 是否存在连续危机波动模型的共同上限。
3. T5 是否已经足够。
4. T5 是否虽然通过，但被更简单 benchmark 材料性支配。

## 2. Benchmark 锁定摘要

- benchmark 锁定文件：`docs/phase0b/01_benchmark_lock.md`
- candidate（Phase 0A）：`Stage A = T5`
- benchmark family：`EGARCH/GJR-GARCH(1,1) fixed benchmark family`
- 固定模型：
  - `EGARCH(1,1)-Normal`
  - `EGARCH(1,1)-Student-t`
  - `GJR-GARCH(1,1)-Normal`
  - `GJR-GARCH(1,1)-Student-t`
- 阶数锁定：`(1,1)`
- 外生变量：不允许
- CRPS 分布假设：与各模型 innovation distribution 一致
- 滚动方式：`rolling fixed-length window`，训练窗 416 周，embargo 53 周
- 预测频率 / 输出频率：周频，按 Friday close 对齐
- 数据源：`data/raw/nasdaq/NASDAQXNDX/close.parquet`

### 2.1 2008 分母值冻结

- `Benchmark_2008_std(z) = 3.2566245119291373`
- 该值取自锁定 benchmark family 中 2008 年 `std(z)` 最低者：`EGARCH(1,1)-Normal`
- 该值仅用于 Phase 0A 条件 D 的合法分母，不构成对 T5 的追溯性真正 OOS 证明

## 3. T5 三窗口结果

> 说明：本轮新增 `src/research/run_phase0a_t5_reproduction.py` 做仓库内 T5 来源定位与复现检查。结果为 `FAILED_TO_RUN_MISSING_T5_SOURCE`：仓库内未找到指定 historical T5 pilot runner、`_fit_t5` / `_predict_t5`、T5 correction 结构、T5-specific 训练窗与最终 `sigma_t` 生成路径。详见 `docs/phase0b/02_t5_reproduction_note.md`。

| 模型 | 窗口 | mean(z) | std(z) | corr_next | rank_next | lag1_acf(z) | sigma_blowup | pathology | CRPS | 状态 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| T5 | 2017-07-07 -> 2017-12-29 | FAILED_TO_RUN_MISSING_T5_SOURCE | FAILED_TO_RUN_MISSING_T5_SOURCE | FAILED_TO_RUN_MISSING_T5_SOURCE | FAILED_TO_RUN_MISSING_T5_SOURCE | FAILED_TO_RUN_MISSING_T5_SOURCE | FAILED_TO_RUN_MISSING_T5_SOURCE | FAILED_TO_RUN_MISSING_T5_SOURCE | FAILED_TO_RUN_MISSING_T5_SOURCE | FAILED_TO_RUN_MISSING_T5_SOURCE |
| T5 | 2018-07-06 -> 2018-12-28 | FAILED_TO_RUN_MISSING_T5_SOURCE | FAILED_TO_RUN_MISSING_T5_SOURCE | FAILED_TO_RUN_MISSING_T5_SOURCE | FAILED_TO_RUN_MISSING_T5_SOURCE | FAILED_TO_RUN_MISSING_T5_SOURCE | FAILED_TO_RUN_MISSING_T5_SOURCE | FAILED_TO_RUN_MISSING_T5_SOURCE | FAILED_TO_RUN_MISSING_T5_SOURCE | FAILED_TO_RUN_MISSING_T5_SOURCE |
| T5 | 2020-01-03 -> 2020-06-26 | FAILED_TO_RUN_MISSING_T5_SOURCE | FAILED_TO_RUN_MISSING_T5_SOURCE | FAILED_TO_RUN_MISSING_T5_SOURCE | FAILED_TO_RUN_MISSING_T5_SOURCE | FAILED_TO_RUN_MISSING_T5_SOURCE | FAILED_TO_RUN_MISSING_T5_SOURCE | FAILED_TO_RUN_MISSING_T5_SOURCE | FAILED_TO_RUN_MISSING_T5_SOURCE | FAILED_TO_RUN_MISSING_T5_SOURCE |

## 4. 四个 benchmark 的三窗口结果

### 4.1 EGARCH(1,1)-Normal

| 窗口 | mean(z) | std(z) | corr_same | corr_next | rank_same | rank_next | lag1_acf(z) | lag1_acf(|z|) | sigma_med | sigma_p10 | sigma_p90 | sigma_p99/med | sigma_blowup | pathology | CRPS | 状态 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2017-07-07 -> 2017-12-29 | -0.474190 | 1.702306 | 0.633365 | 0.530067 | 0.573846 | 0.504615 | 0.909491 | 0.642453 | 0.053934 | 0.030354 | 0.072267 | 1.493308 | 0 | 0 | 0.061159 | PASS |
| 2018-07-06 -> 2018-12-28 | -2.507228 | 2.177269 | -0.459346 | -0.485068 | -0.554615 | -0.580769 | 0.939049 | 0.921357 | 0.040607 | 0.032936 | 0.132835 | 4.312883 | 0 | 0 | 0.095610 | PASS |
| 2020-01-03 -> 2020-06-26 | 2.374020 | 0.810571 | 0.277092 | 0.412949 | 0.239231 | 0.362308 | 0.528656 | 0.528656 | 0.082234 | 0.063158 | 0.153788 | 2.026756 | 0 | 0 | 0.160771 | PASS |

### 4.2 EGARCH(1,1)-Student-t

| 窗口 | mean(z) | std(z) | corr_same | corr_next | rank_same | rank_next | lag1_acf(z) | lag1_acf(|z|) | sigma_med | sigma_p10 | sigma_p90 | sigma_p99/med | sigma_blowup | pathology | CRPS | 状态 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2017-07-07 -> 2017-12-29 | -0.482229 | 1.708624 | 0.627897 | 0.523920 | 0.573846 | 0.470769 | 0.907536 | 0.638565 | 0.054131 | 0.030680 | 0.072175 | 1.477618 | 0 | 0 | 0.061203 | PASS |
| 2018-07-06 -> 2018-12-28 | -2.521206 | 2.187687 | -0.460183 | -0.484285 | -0.554615 | -0.596154 | 0.940395 | 0.923229 | 0.040449 | 0.032886 | 0.132933 | 4.338914 | 0 | 0 | 0.095706 | PASS |
| 2020-01-03 -> 2020-06-26 | 2.372451 | 0.811930 | 0.276945 | 0.413064 | 0.239231 | 0.362308 | 0.528079 | 0.528079 | 0.082222 | 0.063251 | 0.154308 | 2.033745 | 0 | 0 | 0.160708 | PASS |

### 4.3 GJR-GARCH(1,1)-Normal

| 窗口 | mean(z) | std(z) | corr_same | corr_next | rank_same | rank_next | lag1_acf(z) | lag1_acf(|z|) | sigma_med | sigma_p10 | sigma_p90 | sigma_p99/med | sigma_blowup | pathology | CRPS | 状态 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2017-07-07 -> 2017-12-29 | -0.284116 | 1.691868 | 0.743452 | 0.639683 | 0.691538 | 0.600000 | 0.915229 | 0.610986 | 0.052715 | 0.032450 | 0.080247 | 1.574917 | 0 | 0 | 0.061707 | PASS |
| 2018-07-06 -> 2018-12-28 | -2.564035 | 2.212111 | -0.466053 | -0.489657 | -0.554615 | -0.596154 | 0.918876 | 0.901524 | 0.041884 | 0.030807 | 0.164528 | 5.567291 | 0 | 0 | 0.096803 | PASS |
| 2020-01-03 -> 2020-06-26 | 2.576858 | 1.341203 | 0.305748 | 0.423217 | 0.333846 | 0.441538 | 0.433940 | 0.433940 | 0.089641 | 0.050465 | 0.174818 | 2.429981 | 0 | 0 | 0.162728 | PASS |

### 4.4 GJR-GARCH(1,1)-Student-t

| 窗口 | mean(z) | std(z) | corr_same | corr_next | rank_same | rank_next | lag1_acf(z) | lag1_acf(|z|) | sigma_med | sigma_p10 | sigma_p90 | sigma_p99/med | sigma_blowup | pathology | CRPS | 状态 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2017-07-07 -> 2017-12-29 | -0.275378 | 1.687246 | 0.743968 | 0.640352 | 0.698462 | 0.605385 | 0.915201 | 0.610662 | 0.053367 | 0.032501 | 0.080464 | 1.571847 | 0 | 0 | 0.061692 | PASS |
| 2018-07-06 -> 2018-12-28 | -2.564250 | 2.214973 | -0.465579 | -0.495884 | -0.554615 | -0.599231 | 0.921021 | 0.904451 | 0.042446 | 0.030744 | 0.166077 | 5.543561 | 0 | 0 | 0.096782 | PASS |
| 2020-01-03 -> 2020-06-26 | 2.568222 | 1.272282 | 0.305540 | 0.418203 | 0.333846 | 0.417692 | 0.448194 | 0.448194 | 0.091801 | 0.050498 | 0.171143 | 2.299558 | 0 | 0 | 0.162732 | PASS |

## 5. 2008 benchmark run 结果摘要

### 5.1 2008 绝对表现总表

| 模型 | mean(z) | std(z) | corr_same | corr_next | rank_same | rank_next | lag1_acf(z) | lag1_acf(|z|) | sigma_med | sigma_p10 | sigma_p90 | sigma_p99/med | sigma_blowup | pathology | CRPS | 绝对表现判定 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| EGARCH(1,1)-Normal | -3.466627 | 3.256625 | -0.192856 | -0.133827 | -0.450781 | -0.411131 | 0.976584 | 0.970766 | 0.097345 | 0.050526 | 0.531314 | 6.017066 | 0 | 0 | 0.276818 | FAIL |
| EGARCH(1,1)-Student-t | -3.563075 | 3.351537 | -0.189251 | -0.131552 | -0.450013 | -0.410769 | 0.975640 | 0.970087 | 0.096819 | 0.048626 | 0.552751 | 6.355709 | 0 | 0 | 0.277963 | FAIL |
| GJR-GARCH(1,1)-Normal | -4.404689 | 4.510874 | -0.182995 | -0.118187 | -0.388799 | -0.337014 | 0.954775 | 0.949259 | 0.088346 | 0.038668 | 0.656670 | 7.878401 | 0 | 0 | 0.280241 | FAIL |
| GJR-GARCH(1,1)-Student-t | -4.456865 | 4.538592 | -0.182725 | -0.117787 | -0.391872 | -0.339819 | 0.955074 | 0.949373 | 0.087440 | 0.038185 | 0.632214 | 7.698369 | 0 | 0 | 0.279959 | FAIL |

### 5.2 2008 分母值

- `Benchmark_2008_std(z) = 3.2566245119291373`
- 采用 `EGARCH(1,1)-Normal` 作为 2008 分母值冻结基准，因为其在锁定 family 中的 2008 `std(z)` 最低

### 5.3 2008 结果解释边界

1. 2008 盲测在四个 benchmark 上均出现方向性倒置（`corr_next < 0` 且 `rank_next < 0`）。
2. 因此 benchmark family 在 2008 上不具最低绝对可用性。
3. 这不构成对 T5 的追溯性真正 OOS 证明。

## 6. T5 vs Benchmark 对照表

> 说明：
> - 2020 列用于验证 benchmark 是否对 T5 形成材料性支配。
> - T5 candidate-side 三窗口结果与 2008 结果均无法由仓库内 T5 source 复现，因此依赖 T5 candidate-side 的比较字段均标记为 `FAILED_TO_RUN_MISSING_T5_SOURCE`。
> - `Benchmark_2008_std(z)` 已冻结并可供后续 Phase 0A 条件 D 使用。

| 比较对象 | 2020 std(z) | 2020 是否优于 T5 至少 0.05 且 >=2% | 2008 是否超过 Benchmark_2008_std(z)+0.10 | 安全/方向性是否不劣于 T5 | 是否形成材料性支配 | 备注 |
|---|---:|---|---|---|---|---|
| T5 vs EGARCH(1,1)-Normal | 0.810571 | FAILED_TO_RUN_MISSING_T5_SOURCE | FAILED_TO_RUN_MISSING_T5_SOURCE | FAILED_TO_RUN_MISSING_T5_SOURCE | FAILED_TO_RUN_MISSING_T5_SOURCE | T5 source 缺失，不能合法比较 |
| T5 vs EGARCH(1,1)-Student-t | 0.811930 | FAILED_TO_RUN_MISSING_T5_SOURCE | FAILED_TO_RUN_MISSING_T5_SOURCE | FAILED_TO_RUN_MISSING_T5_SOURCE | FAILED_TO_RUN_MISSING_T5_SOURCE | T5 source 缺失，不能合法比较 |
| T5 vs GJR-GARCH(1,1)-Normal | 1.341203 | FAILED_TO_RUN_MISSING_T5_SOURCE | FAILED_TO_RUN_MISSING_T5_SOURCE | FAILED_TO_RUN_MISSING_T5_SOURCE | FAILED_TO_RUN_MISSING_T5_SOURCE | T5 source 缺失，不能合法比较 |
| T5 vs GJR-GARCH(1,1)-Student-t | 1.272282 | FAILED_TO_RUN_MISSING_T5_SOURCE | FAILED_TO_RUN_MISSING_T5_SOURCE | FAILED_TO_RUN_MISSING_T5_SOURCE | FAILED_TO_RUN_MISSING_T5_SOURCE | T5 source 缺失，不能合法比较 |

### 6.1 材料性支配判定

- 本次交付不能合法判定 benchmark 是否相对 T5 形成材料性支配，因为 T5 candidate-side 缺少仓库内可执行 source。
- benchmark 自身的 2020 表现已完整落盘，其中最低 2020 `std(z)` 来自 `EGARCH(1,1)-Normal`。
- `Benchmark_2008_std(z)` 已合法冻结，供后续补齐 candidate-side 对照使用。

## 7. Phase 0A 五分支的当前占位判断

| 分支 | 当前占位判断 | 依据 | 当前状态 |
|---|---|---|---|
| 分支一：T5 特有缺陷 | NO | 2008 上 benchmark family 未保持最低绝对可用性，故不能据此证明 T5 特有缺陷 | NO |
| 分支二：连续危机波动模型共同上限 | PENDING | T5 source 缺失，T5 的协议判定结果无法复现 | PENDING |
| 分支三：危机样本异质，不支持统一危机层 | YES | benchmark family 在 2020 与 2008 呈断裂式异质表现：2020 正方向且无 blowup，2008 方向性倒置 | YES |
| 分支四：T5 Success 但被 benchmark 材料性支配 | PENDING | T5 source 缺失，不能判定 T5 是否 Success 或是否被材料性支配 | PENDING |
| 分支五：T5 足够，无需继续修补 | PENDING | T5 source 缺失，不能判定 T5 是否 Success | PENDING |

### 7.1 当前最稳妥的占位判断

- 当前 benchmark delivery 能直接支持的，是 benchmark family 自身的 2020/2008 异质表现。
- 任何依赖 T5 candidate-side 的分支仍需补齐 T5 可执行 source；当前状态为 `FAILED_TO_RUN_MISSING_T5_SOURCE`。

## 8. 当前是否完成 Benchmark Delivery

### 8.1 完成状态

`YES`

### 8.2 仍然阻塞 Phase 0B 的项

- Trigger Audit Delivery：未完成
- Bootstrap Sign-Stability Preregistration：未完成

### 8.3 不能说的结论

1. 不能说 Phase 0B 已具立项资格。  
2. 不能把 2008 benchmark 结果夸大为 T5 的真正 OOS 证明。  
3. 不能把 benchmark family 在 2008 的失败掩盖为实现噪声。  
4. 不能在 trigger audit 与 bootstrap 预注册完成前宣布进入 Phase 0B。  

## 9. 阻塞项与证据边界

### 阻塞项

- Trigger audit 与 bootstrap prereg 仍未完成，故 Phase 0B 入口仍然关闭。

### 证据边界

- 本文只冻结 benchmark delivery 的实际结果。
- T5 candidate-side 三窗口与 2008 结果无法由仓库内 T5 source 复现，因此相关对照字段保留为 `FAILED_TO_RUN_MISSING_T5_SOURCE`。
- 本文不负责 trigger audit。
- 本文不负责 bootstrap sign-stability 阈值预注册。
- 本文不提供 crisis architecture 的立项批准。
