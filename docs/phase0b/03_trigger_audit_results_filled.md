# Trigger Audit 结果文档（已填充）

> 文档性质  
> 本文是 Phase 0A Trigger Audit Delivery 的真实跑批结果，不是新方法论文档。本文只承接 `03_trigger_audit_prereg.md` 已锁定的 trigger 清单、lead window、false positive 口径与准入规则。

## 1. 文档目的

本文件用于记录全量 trigger 候选的 lead-lag audit 结果，并回答哪些 trigger 具备进入未来 crisis architecture 实验的最低合法资格。

## 2. trigger audit prereg 摘要

- prereg 文件：`docs/phase0b/03_trigger_audit_prereg.md`
- 基础频率：日频
- 标的目标：`NASDAQXNDX` 日度收盘序列
- 主目标变量：下一交易日绝对收益 `|r_{t+1}|`
- 辅助目标变量：未来 5 交易日实现波动代理
- `L_max = 5` 个交易日
- 合法 lead window：`{-5, -4, -3, -2, -1}` 交易日
- `lag = 0` 无效，正滞后无效，早于 `-5` 的更长提前量不得单独赋予 trigger 准入资格
- false positive 定义：非危机窗口内，trigger 处于自身样本分布前 20% 极端区间，但下一交易日 `|r_{t+1}|` 未进入自身样本分布前 20%
- false positive 通过门槛：`<= 0.40`
- crisis consistency 通过口径：2000 / 2008 / 2020 三段危机样本中至少两段方向一致为正

## 3. 数据可用性摘要

| trigger 名称 | 数据状态 | 可用起点 | 可用终点 | 对齐样本数 | 2000 样本数 | 2008 样本数 | 2020 样本数 | 缺失输入 | 辅助目标状态 |
|---|---|---|---|---:|---:|---:|---:|---|---|
| VIX9D / VIX | FAILED_TO_RUN_DATA_MISSING | 2011-01-03 | 2026-04-17 | 3845 | 0 | 0 | 253 | none | PASS |
| VIX / VIX3M | FAILED_TO_RUN_DATA_MISSING | 2006-07-17 | 2026-04-17 | 4970 | 0 | 505 | 253 | none | PASS |
| ΔVIX9D | FAILED_TO_RUN_DATA_MISSING | 2011-01-04 | 2026-04-17 | 3844 | 0 | 0 | 253 | none | PASS |
| credit spread jump | PASS | 1986-01-03 | 2026-04-17 | 10150 | 752 | 505 | 253 | none | PASS |
| FRA-OIS widening | FAILED_TO_RUN_DATA_MISSING | NA | NA | NA | NA | NA | NA | missing:FRA_OIS | FAILED_TO_RUN_DATA_MISSING |
| liquidity stress signals | FAILED_TO_RUN_DATA_MISSING | NA | NA | NA | NA | NA | NA | missing:liquidity_stress | FAILED_TO_RUN_DATA_MISSING |
| lagged realized shock | PASS | 1985-10-03 | 2026-04-17 | 10213 | 752 | 505 | 253 | none | PASS |

## 4. 全 trigger 结果总表

| trigger 名称 | 数据状态 | 是否纳入本轮审计 | corr@-5 | corr@-4 | corr@-3 | corr@-2 | corr@-1 | rank_corr@-5 | rank_corr@-4 | rank_corr@-3 | rank_corr@-2 | rank_corr@-1 | false positive rate | 2000 一致性 | 2008 一致性 | 2020 一致性 | 最终状态 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| VIX9D / VIX | FAILED_TO_RUN_DATA_MISSING | 是 | 0.076476 | 0.081338 | 0.096725 | 0.110880 | 0.122691 | 0.001543 | -0.000289 | 0.016759 | 0.027210 | 0.034813 | 0.723227 | FAILED_TO_RUN_DATA_MISSING | FAILED_TO_RUN_DATA_MISSING | POSITIVE | FAILED_TO_RUN_DATA_MISSING |
| VIX / VIX3M | FAILED_TO_RUN_DATA_MISSING | 是 | 0.279423 | 0.287665 | 0.304808 | 0.321071 | 0.337083 | 0.137465 | 0.142547 | 0.161100 | 0.166951 | 0.187505 | 0.688019 | FAILED_TO_RUN_DATA_MISSING | POSITIVE | POSITIVE | FAILED_TO_RUN_DATA_MISSING |
| ΔVIX9D | FAILED_TO_RUN_DATA_MISSING | 是 | 0.067567 | -0.004294 | 0.048711 | 0.061873 | 0.076876 | -0.043765 | -0.077791 | -0.036038 | -0.064256 | -0.043452 | 0.817803 | FAILED_TO_RUN_DATA_MISSING | FAILED_TO_RUN_DATA_MISSING | POSITIVE | FAILED_TO_RUN_DATA_MISSING |
| credit spread jump | PASS | 是 | 0.041911 | 0.053824 | 0.074171 | 0.083421 | 0.083464 | 0.012283 | 0.025428 | 0.033514 | 0.051065 | 0.041680 | 0.767925 | POSITIVE | POSITIVE | POSITIVE | FAIL |
| FRA-OIS widening | FAILED_TO_RUN_DATA_MISSING | 是 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | FAILED_TO_RUN_DATA_MISSING | FAILED_TO_RUN_DATA_MISSING | FAILED_TO_RUN_DATA_MISSING | FAILED_TO_RUN_DATA_MISSING |
| liquidity stress signals | FAILED_TO_RUN_DATA_MISSING | 是 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | FAILED_TO_RUN_DATA_MISSING | FAILED_TO_RUN_DATA_MISSING | FAILED_TO_RUN_DATA_MISSING | FAILED_TO_RUN_DATA_MISSING |
| lagged realized shock | PASS | 是 | 0.269287 | 0.302392 | 0.313734 | 0.302908 | 0.300813 | 0.178850 | 0.201080 | 0.192950 | 0.203049 | 0.192905 | 0.685813 | POSITIVE | POSITIVE | POSITIVE | FAIL |

## 5. 各 trigger 的 lag 明细表

| trigger 名称 | lag | corr | rank_corr |
|---|---:|---:|---:|
| VIX9D / VIX | -5 | 0.076476 | 0.001543 |
| VIX9D / VIX | -4 | 0.081338 | -0.000289 |
| VIX9D / VIX | -3 | 0.096725 | 0.016759 |
| VIX9D / VIX | -2 | 0.110880 | 0.027210 |
| VIX9D / VIX | -1 | 0.122691 | 0.034813 |
| VIX / VIX3M | -5 | 0.279423 | 0.137465 |
| VIX / VIX3M | -4 | 0.287665 | 0.142547 |
| VIX / VIX3M | -3 | 0.304808 | 0.161100 |
| VIX / VIX3M | -2 | 0.321071 | 0.166951 |
| VIX / VIX3M | -1 | 0.337083 | 0.187505 |
| ΔVIX9D | -5 | 0.067567 | -0.043765 |
| ΔVIX9D | -4 | -0.004294 | -0.077791 |
| ΔVIX9D | -3 | 0.048711 | -0.036038 |
| ΔVIX9D | -2 | 0.061873 | -0.064256 |
| ΔVIX9D | -1 | 0.076876 | -0.043452 |
| credit spread jump | -5 | 0.041911 | 0.012283 |
| credit spread jump | -4 | 0.053824 | 0.025428 |
| credit spread jump | -3 | 0.074171 | 0.033514 |
| credit spread jump | -2 | 0.083421 | 0.051065 |
| credit spread jump | -1 | 0.083464 | 0.041680 |
| FRA-OIS widening | -5 | NA | NA |
| FRA-OIS widening | -4 | NA | NA |
| FRA-OIS widening | -3 | NA | NA |
| FRA-OIS widening | -2 | NA | NA |
| FRA-OIS widening | -1 | NA | NA |
| liquidity stress signals | -5 | NA | NA |
| liquidity stress signals | -4 | NA | NA |
| liquidity stress signals | -3 | NA | NA |
| liquidity stress signals | -2 | NA | NA |
| liquidity stress signals | -1 | NA | NA |
| lagged realized shock | -5 | 0.269287 | 0.178850 |
| lagged realized shock | -4 | 0.302392 | 0.201080 |
| lagged realized shock | -3 | 0.313734 | 0.192950 |
| lagged realized shock | -2 | 0.302908 | 0.203049 |
| lagged realized shock | -1 | 0.300813 | 0.192905 |

## 6. false positive 摘要

| trigger 名称 | false positive rate | false positive 门槛 | 是否满足 |
|---|---:|---:|---|
| VIX9D / VIX | 0.723227 | 0.40 | FAIL |
| VIX / VIX3M | 0.688019 | 0.40 | FAIL |
| ΔVIX9D | 0.817803 | 0.40 | FAIL |
| credit spread jump | 0.767925 | 0.40 | FAIL |
| FRA-OIS widening | NA | 0.40 | FAILED_TO_RUN_DATA_MISSING |
| liquidity stress signals | NA | 0.40 | FAILED_TO_RUN_DATA_MISSING |
| lagged realized shock | 0.685813 | 0.40 | FAIL |

## 7. 2000 / 2008 / 2020 一致性摘要

| trigger 名称 | 2000 一致性 | 2008 一致性 | 2020 一致性 | 正向危机段数 | 是否满足至少两段正向 |
|---|---|---|---|---:|---|
| VIX9D / VIX | FAILED_TO_RUN_DATA_MISSING | FAILED_TO_RUN_DATA_MISSING | POSITIVE | 1 | FAIL |
| VIX / VIX3M | FAILED_TO_RUN_DATA_MISSING | POSITIVE | POSITIVE | 2 | DATA_MISSING |
| ΔVIX9D | FAILED_TO_RUN_DATA_MISSING | FAILED_TO_RUN_DATA_MISSING | POSITIVE | 1 | FAIL |
| credit spread jump | POSITIVE | POSITIVE | POSITIVE | 3 | PASS |
| FRA-OIS widening | FAILED_TO_RUN_DATA_MISSING | FAILED_TO_RUN_DATA_MISSING | FAILED_TO_RUN_DATA_MISSING | 0 | FAILED_TO_RUN_DATA_MISSING |
| liquidity stress signals | FAILED_TO_RUN_DATA_MISSING | FAILED_TO_RUN_DATA_MISSING | FAILED_TO_RUN_DATA_MISSING | 0 | FAILED_TO_RUN_DATA_MISSING |
| lagged realized shock | POSITIVE | POSITIVE | POSITIVE | 3 | PASS |

## 8. 最终 pass/fail 清单

| trigger 名称 | 最终状态 | 直接原因 |
|---|---|---|
| VIX9D / VIX | FAILED_TO_RUN_DATA_MISSING | 2000 与 2008 危机段缺失，且 false positive rate 超过 0.40 |
| VIX / VIX3M | FAILED_TO_RUN_DATA_MISSING | 2000 危机段缺失，且 false positive rate 超过 0.40 |
| ΔVIX9D | FAILED_TO_RUN_DATA_MISSING | 2000 与 2008 危机段缺失，rank direction 不满足，且 false positive rate 超过 0.40 |
| credit spread jump | FAIL | lead window 与三段危机一致性为正，但 false positive rate = 0.767925，超过 0.40 |
| FRA-OIS widening | FAILED_TO_RUN_DATA_MISSING | 仓库内未找到可合法构造 FRA-OIS widening 的输入数据 |
| liquidity stress signals | FAILED_TO_RUN_DATA_MISSING | 仓库内未找到明确、可合法构造 liquidity stress signals 的输入数据 |
| lagged realized shock | FAIL | lead window 与三段危机一致性为正，但 false positive rate = 0.685813，超过 0.40 |

当前合法 `PASS` trigger 数量：`0`。

## 9. 当前是否完成 Trigger Audit Delivery

`YES`

Trigger Audit Delivery 已完成：prereg 已冻结，runner 已执行，结果已落盘。  
但审计结果未产生任何合法 `PASS` trigger，因此 Phase 0B 的 “至少存在一个合法 trigger” 条件未满足。

## 10. 阻塞项与证据边界

### 阻塞项

- 本轮 trigger audit 未产生任何合法 `PASS` trigger。
- Bootstrap Sign-Stability Preregistration 仍未完成。

### 证据边界

- 本文不授权 override、hard switch、MoE 或任何 crisis architecture。
- 数据缺失 trigger 未从审计清单删除，而是按 prereg 规则保留为 `FAILED_TO_RUN_DATA_MISSING`。
- 早于 `-5` 的更长提前量未被用于 trigger 准入。
- false positive 口径未在结果后改写。
