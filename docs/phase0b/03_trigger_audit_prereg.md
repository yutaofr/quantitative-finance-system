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
| VIX9D / VIX | UNFILLED | UNFILLED | UNFILLED | UNFILLED |
| VIX / VIX3M | UNFILLED | UNFILLED | UNFILLED | UNFILLED |
| ΔVIX9D | UNFILLED | UNFILLED | UNFILLED | UNFILLED |
| credit spread jump | UNFILLED | UNFILLED | UNFILLED | UNFILLED |
| FRA-OIS widening | UNFILLED | UNFILLED | UNFILLED | UNFILLED |
| liquidity stress signals | UNFILLED | UNFILLED | UNFILLED | UNFILLED |
| lagged realized shock | UNFILLED | UNFILLED | UNFILLED | UNFILLED |

## 4. `L_max` 预注册区

### 裁决

在 trigger audit 开始之前，必须先预注册一个**与具体架构形式无关的保守执行延迟上限** `L_max`。若 `L_max` 未预注册，则整个 trigger audit 无效。

### 预注册字段

| 字段 | 取值 | 状态 | 说明 |
|---|---|---|---|
| `L_max`（交易日） | UNFILLED | UNFILLED | 未来任何架构的有效执行延迟不得超过此上限 |
| `L_max` 的治理理由 | UNFILLED | UNFILLED | 必须说明为什么是这个上限 |
| 是否允许未来架构超过 `L_max` | 否 | LOCKED | 若超过，必须重新做 trigger audit |

## 5. 合法 lead window 定义

### 裁决

合法 lead window 固定为：

`{-L_max, ..., -1}`

其中：

- `lag = 0` 无效
- 正滞后无效
- 早于 `-L_max` 的更长提前量只能做描述性报告，不得单独赋予 trigger 准入资格

### 说明

这一步的目的，是把 trigger 准入资格绑定到可执行的、可利用的时间窗口，而不是任意“负滞后”。

## 6. False Positive 计算规则

| 字段 | 取值 | 状态 | 说明 |
|---|---|---|---|
| 常态窗定义 | UNFILLED | UNFILLED | 必须与协议窗口口径一致 |
| false positive 事件定义 | UNFILLED | UNFILLED | 必须前置写死 |
| 统计频率 | UNFILLED | UNFILLED | 日频/周频必须明确 |
| 通过门槛 | UNFILLED | UNFILLED | 必须前置写死 |

### 法理要求

若 false positive 的定义未锁定，则 trigger audit 结果无效。

## 7. 审计输出格式

每个 trigger 必须输出以下表格，不得缺项：

| Trigger | lead window 内证据 | false positive rate | 2000 一致性 | 2008 一致性 | 2020 一致性 | pass/fail | 备注 |
|---|---|---:|---|---|---|---|---|
| UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED |

### 最低报告内容

1. 合法 lead window 内的证据  
2. 窗口外 false positive  
3. 2000 / 2008 / 2020 的跨窗一致性  
4. 准入结论  

## 8. Trigger 准入判定规则

### 裁决

任一 trigger 只有在同时满足以下条件时，才具备进入未来架构实验的最低资格：

1. 在合法 lead window 内存在证据  
2. false positive rate 不超过预注册门槛  
3. 不仅在单一窗口有效  
4. 未被识别为纯同期或滞后信号  

### 通过状态模板

| Trigger | 是否满足 lead 要求 | 是否满足 false positive 要求 | 是否具备跨窗一致性 | 最终准入 | 说明 |
|---|---|---|---|---|---|
| UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED |

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
| trigger 清单是否锁定 | UNFILLED | UNFILLED |
| 纳入/排除理由是否已写死 | UNFILLED | 若否，则 trigger audit 不得开始 |
| `L_max` 是否预注册 | UNFILLED | UNFILLED |
| 合法 lead window 是否由 `L_max` 唯一定义 | UNFILLED | UNFILLED |
| false positive 计算方式是否锁定 | UNFILLED | UNFILLED |
| 输出表格模板是否完整 | PASS | 已创建 |
| 所有 `UNFILLED` 是否清空 | UNFILLED | 若否，则 trigger audit 不得开始 |
