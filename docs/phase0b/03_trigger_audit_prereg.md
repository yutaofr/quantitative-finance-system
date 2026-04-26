# Trigger Audit 预注册文档

> 文档性质  
> 本文不是 trigger audit 结果文档。  
> 本文只定义 trigger audit 的合法输入、法定 lead window、false positive 口径与准入规则。

## 1. 文档目的

本文件用于在任何 crisis architecture 立项之前，先完成 trigger 的合法性预注册，回答：

1. 哪些外部信号具备最低 trigger 候选资格。
2. 哪些信号只是解释性变量，不能进入门控。
3. trigger 审计如何避免把滞后变量伪装成先导信号。
4. trigger 候选的纳入/排除如何避免选择性偏差。

## 2. 上游协议约束

本文件必须服从：

- `docs/PHASE0A_PROTOCOL_FREEZE.md`
- `docs/PHASE0B_PREP_CHECKLIST.md`

若本文件与上述冻结文件冲突，以冻结文件为准。

## 3. Trigger 审计对象清单

> 说明  
> 以下对象只是候选，不具自动准入资格。是否纳入本轮审计的决定，必须在**触碰任何原始数据之前**完成并写死。纳入决策一经锁定，不得因看到任何数据片段、描述性统计或预运行结果而修改。

### 3.1 纳入决策的治理规则

1. 纳入/排除的截止点是：**在触碰任何原始数据之前**。  
2. 默认规则是：要么全量送审，要么对每个被排除信号写出**先验机制性理由**。  
3. 排除理由不得引用任何数据相关表述，不得以“看起来可能没用”“以前效果不好”作为理由。  
4. 若未在截止点前完成纳入决策锁定，则整个 trigger audit 无效。

| Trigger 名称 | 是否纳入本轮审计 | 先验保留/排除理由 | 备注 | 状态 |
|---|---|---|---|---|
| VIX9D / VIX | 是 | 全量送审，以避免选择性纳入偏差 | 日频 trigger audit；数据缺失时保留失败状态，不从清单删除 | LOCKED |
| VIX / VIX3M | 是 | 全量送审，以避免选择性纳入偏差 | 日频 trigger audit；数据缺失时保留失败状态，不从清单删除 | LOCKED |
| ΔVIX9D | 是 | 全量送审，以避免选择性纳入偏差 | 日频 trigger audit；数据缺失时保留失败状态，不从清单删除 | LOCKED |
| credit spread jump | 是 | 全量送审，以避免选择性纳入偏差 | 日频 trigger audit；数据缺失时保留失败状态，不从清单删除 | LOCKED |
| FRA-OIS widening | 是 | 全量送审，以避免选择性纳入偏差 | 日频 trigger audit；数据缺失时保留失败状态，不从清单删除 | LOCKED |
| liquidity stress signals | 是 | 全量送审，以避免选择性纳入偏差 | 日频 trigger audit；数据缺失时保留失败状态，不从清单删除 | LOCKED |
| lagged realized shock | 是 | 全量送审，以避免选择性纳入偏差 | 日频 trigger audit；数据缺失时保留失败状态，不从清单删除 | LOCKED |

## 4. `L_max` 预注册区

### 裁决

在 trigger audit 开始之前，必须先预注册一个**与具体架构形式无关的保守执行延迟上限** `L_max`。若 `L_max` 未预注册，则整个 trigger audit 无效。

### 预注册字段

| 字段 | 取值 | 状态 | 说明 |
|---|---|---|---|
| `L_max`（交易日） | 5 | LOCKED | 未来任何架构的有效执行延迟不得超过此上限 |
| `L_max` 的治理理由 | 当前基准比较与 T5 输出口径是周频 Friday close 对齐；`L_max = 5` 代表一个交易周的保守执行延迟上限；未来任何架构若有效执行延迟大于 5 个交易日，不得继承本轮 trigger audit 合法性，必须重审 | LOCKED | 治理阈值在触碰 trigger audit 结果前写死 |
| 是否允许未来架构超过 `L_max` | 否 | LOCKED | 若超过，必须重新做 trigger audit |

## 5. 合法 lead window 定义

### 裁决

合法 lead window 固定为：

`{-5, -4, -3, -2, -1}` 交易日

其中：

- `lag = 0` 无效
- 正滞后无效
- 早于 `-L_max` 的更长提前量只能做描述性报告，不得单独赋予 trigger 准入资格

## 5.1 触发目标与频率口径

| 字段 | 取值 | 状态 | 说明 |
|---|---|---|---|
| 基础频率 | 日频 | LOCKED | 所有 trigger audit 以日频对齐 |
| 标的目标 | `NASDAQXNDX` 日度收盘序列 | LOCKED | 目标资产与 benchmark delivery 一致 |
| 主目标变量 | 下一交易日绝对收益 `|r_{t+1}|` | LOCKED | trigger 在 lead window 内必须先于该目标变量 |
| 辅助目标变量 | 未来 5 交易日实现波动代理 | LOCKED | 若无法稳定构造，结果文档记为 `FAILED_TO_RUN_SECONDARY_TARGET`，不影响主目标审计 |

### 说明

这一步的目的，是把 trigger 准入资格绑定到可执行的、可利用的时间窗口，而不是任意“负滞后”。

## 6. False Positive 计算规则

| 字段 | 取值 | 状态 | 说明 |
|---|---|---|---|
| 常态窗定义 | 排除 `2000-01-01` 到 `2002-12-31`、排除 `2008-01-01` 到 `2009-12-31`、排除 `2020-01-01` 到 `2020-12-31`；其余样本作为 calm regime | LOCKED | 与协议危机窗口边界一致 |
| false positive 事件定义 | 在非危机窗口内，trigger 处于自身样本分布的前 20% 极端区间，但下一交易日 `|r_{t+1}|` 没有进入其自身样本分布的前 20%，记为一次 false positive | LOCKED | 在看到结果前写死 |
| 统计频率 | 日频 | LOCKED | false positive 与 lead-lag audit 使用同一日频口径 |
| 通过门槛 | false positive rate `<= 0.40` | LOCKED | 本轮临时治理规则 |

### 法理要求

若 false positive 的定义未锁定，则 trigger audit 结果无效。

## 7. 审计输出格式

每个 trigger 必须输出以下表格，不得缺项：

| Trigger | lead window 内证据 | false positive rate | 2000 一致性 | 2008 一致性 | 2020 一致性 | pass/fail | 备注 |
|---|---|---:|---|---|---|---|---|
| 全量送审 trigger 清单 | 运行后填入 `03_trigger_audit_results_filled.md` | 运行后填入 | 运行后填入 | 运行后填入 | 运行后填入 | 运行后填入 | 预注册模板已锁定，结果不得事后改口径 |

### 最低报告内容

1. 合法 lead window 内的证据  
2. 窗口外 false positive  
3. 2000 / 2008 / 2020 的跨窗一致性  
4. 准入结论  

## 8. Trigger 准入判定规则

### 裁决

任一 trigger 只有在同时满足以下条件时，才具备进入未来架构实验的最低资格：

1. 在合法 lead window 内，至少一个 lag 上 `corr > 0` 且 `rank_corr > 0`
2. 在三段危机样本（2000、2008、2020）中，至少两段方向一致为正
3. false positive rate `<= 0.40`

否则记为 `FAIL`。如果数据缺失或无法合法构造该 trigger，记为 `FAILED_TO_RUN_DATA_MISSING`。如果代码实现失败，记为 `FAILED_TO_RUN_IMPLEMENTATION`。不得伪造 `PASS`。

### 通过状态模板

| Trigger | 是否满足 lead 要求 | 是否满足 false positive 要求 | 是否具备跨窗一致性 | 最终准入 | 说明 |
|---|---|---|---|---|---|
| VIX9D / VIX | 运行后填入 | 运行后填入 | 运行后填入 | 运行后填入 | LOCKED |
| VIX / VIX3M | 运行后填入 | 运行后填入 | 运行后填入 | 运行后填入 | LOCKED |
| ΔVIX9D | 运行后填入 | 运行后填入 | 运行后填入 | 运行后填入 | LOCKED |
| credit spread jump | 运行后填入 | 运行后填入 | 运行后填入 | 运行后填入 | LOCKED |
| FRA-OIS widening | 运行后填入 | 运行后填入 | 运行后填入 | 运行后填入 | LOCKED |
| liquidity stress signals | 运行后填入 | 运行后填入 | 运行后填入 | 运行后填入 | LOCKED |
| lagged realized shock | 运行后填入 | 运行后填入 | 运行后填入 | 运行后填入 | LOCKED |

## 9. 禁止事项

1. 未通过 lead-lag audit 的 trigger，不得进入任何架构实验。  
2. 不得因为某 trigger 在 2020 看起来“解释力很强”而绕过审计。  
3. 不得在看到结果后更改 `L_max`。  
4. 不得把落在 `-L_max` 之外的更长提前量包装成法定 lead 证据。  
5. 不得在 trigger 结果不利时重写 false positive 口径。  
6. 不得在触碰任何原始数据之后才决定哪些 trigger 纳入本轮审计。  

## 10. 运行前检查项

| 检查项 | 状态 | 备注 |
|---|---|---|
| trigger 清单是否锁定 | PASS | 七个候选全部纳入本轮审计并锁定 |
| 纳入/排除理由是否已写死 | PASS | 全量送审，以避免选择性纳入偏差 |
| `L_max` 是否预注册 | PASS | `L_max = 5` 个交易日 |
| 合法 lead window 是否由 `L_max` 唯一定义 | PASS | `{-5, -4, -3, -2, -1}` 交易日 |
| false positive 计算方式是否锁定 | PASS | calm regime、前 20% 极端区间与 `<= 0.40` 门槛均已锁定 |
| 输出表格模板是否完整 | PASS | 已创建 |
| 所有 `UNFILLED` 是否清空 | PASS | 预注册字段已清空；运行结果另见 `03_trigger_audit_results_filled.md` |
