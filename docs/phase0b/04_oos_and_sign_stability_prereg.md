# OOS 隔离与 Bootstrap Sign-Stability 预注册文档

> 文档性质  
> 本文同时承担两项职责：  
>
> 1. 写死 OOS 样本隔离规则  
> 2. 预注册 bootstrap sign-stability 的审计字段  
> 本文不是结果报告，不得预填任何结果。

## 1. 文档目的

本文件用于在进入任何 Phase 0B 架构实验之前，先把以下两类东西写死：

1. 样本边界与证据边界  
2. directionality 稳健性审计方式  

## 2. 样本边界冻结声明

### 裁决

样本边界必须在看到任何新架构结果之前冻结，冻结后不得回填修改。

### 冻结表

| 样本区间 | 用途 | 是否允许训练 | 是否允许参数选择 | 是否允许作为唯一危机代理 | 状态 |
|---|---|---|---|---|---|
| 2008–2009 | blind holdout | 否 | 否 | 否 | LOCKED |
| 2000–2002 | 异质验证集 | 否 | 否 | 否 | LOCKED |
| 2020 | 可暴露危机样本 | 是 | 是 | 否 | LOCKED |

## 3. 2008 blind holdout 规则

1. 2008–2009 不参与训练。  
2. 2008–2009 不参与参数拟合。  
3. 2008–2009 不参与超参数选择。  
4. 不得看完 2008 结果后回填修改成功阈值。  
5. 不得用 2008 结果来调整 trigger。  

## 4. 2000–2002 异质验证规则

1. 2000–2002 用于识别仅对 2020 有效的过拟合设计。  
2. 2000–2002 不是补充训练样本。  
3. 若某设计只在 2020 有效而在 2000–2002 断裂，则不得包装成“统一危机层”。  

## 5. 2020 使用边界

1. 2020 可以作为暴露样本。  
2. 2020 不得作为唯一危机代理。  
3. 任何架构若只靠 2020 有效，必须被视为局部样本拟合嫌疑。  

## 6. Stage A 的证据边界声明

### 裁决

`Stage A = T5` 对 2008 不构成追溯性真正 OOS。

### 依据

T5 的选择过程已经暴露于后续危机样本信息，尤其 2020。  
因此：

- 2008 的 blind holdout 对未来候选 crisis architecture 与 benchmark 比较构成真正 OOS  
- 但不能把 2008 的结果夸大表述为“Stage A 的通用性已被 OOS 证明”

### 禁止说法

禁止出现以下表述：

- “T5 已经在 2008 上 OOS 验证通过”
- “T5 的泛化性已被 2008 证明”

## 7. Bootstrap Sign-Stability 预注册字段

> 说明  
> 这里先写字段与法理边界，不写结果。

| 字段 | 取值 | 状态 | 说明 |
|---|---|---|---|
| bootstrap 方案 | block bootstrap | LOCKED | 预注册后不得改为其他重采样方案 |
| block 长度 | 20 个观测点 | LOCKED | 预注册后不得更改 |
| 采样次数 | 400 | LOCKED | 预注册后不得更改 |
| 随机种子策略 | 固定随机种子 `20260426` | LOCKED | 预注册后不得更改 |
| 失败时如何处理数值异常 | 任一窗口任一对象出现采样运行失败或数值异常，记为 `FAILED_TO_RUN_BOOTSTRAP` | LOCKED | 不得以补跑结果覆盖失败记录 |

## 8. 检验对象锁定

### 裁决

在进入 Phase 0B 前，必须明确写死 sign-stability 的检验对象，禁止看到结果后再挑指标。

### 预注册表

| 检验对象 | 是否纳入 sign-stability | 通过阈值 | 状态 | 备注 |
|---|---|---|---|---|
| `corr_next` | 是 | `metric > 0` 的频率 `>= 0.60` | LOCKED | 必须与 `rank_next` 同时检验，不允许单检 |
| `rank_next` | 是 | `metric > 0` 的频率 `>= 0.60` | LOCKED | 必须与 `corr_next` 同时检验，不允许单检 |

### 法理要求

1. 若两者都纳入，必须分别预注册阈值。  
2. 若只纳入一个，必须在此文档中说明为什么另一个不构成独立稳健性要求。  
3. 不得在看到 bootstrap 结果后再改检验对象。  

## 9. 失败判定规则

| 项目 | 失败条件 | 状态 |
|---|---|---|
| OOS 使用 | 2008 被用于训练/调参/超参数选择 | LOCKED |
| sign-stability | 检验对象未预注册 | LOCKED |
| sign-stability | 阈值未预注册 | LOCKED |
| sign-stability | `corr_next > 0` 的频率 `< 0.60` | LOCKED |
| sign-stability | `rank_next > 0` 的频率 `< 0.60` | LOCKED |
| sign-stability | 任一窗口任一对象出现采样运行失败或数值异常，记为 `FAILED_TO_RUN_BOOTSTRAP` | LOCKED |
| sign-stability | 看结果后改 block 长度 | LOCKED |
| sign-stability | 看结果后改单检/双检对象 | LOCKED |

## 9.1 双检通过规则（窗口级）

窗口级通过必须同时满足：

1. `corr_next_sign_stability >= 0.60`
2. `rank_next_sign_stability >= 0.60`

任一条件不满足即窗口失败。

## 9.2 证据边界

1. Bootstrap sign-stability prereg 的完成，只意味着 directionality 稳健性审计规则已冻结。
2. 该完成状态不会覆盖 Trigger Audit Delivery 当前 `FAIL` 结论。
3. 该完成状态不授权进入 Phase 0B。
4. 只有四项前置交付都满足且 trigger 至少有一个合法 `PASS`，Phase 0B 才可能立项。

## 10. 禁止事项

1. 不得把 2008 用作训练反馈。  
2. 不得在看到 bootstrap 结果后再改 block 长度。  
3. 不得在看到 bootstrap 结果后再改检验对象。  
4. 不得把 T5 的 2008 结果误报为真正 OOS 证据。  
5. 不得在 2020 有利时弱化 2000–2002 与 2008 的约束地位。  

## 11. 启动前检查项

| 检查项 | 状态 | 备注 |
|---|---|---|
| 2008 blind holdout 是否冻结 | PASS | 已写死 |
| 2000–2002 是否冻结为异质验证集 | PASS | 已写死 |
| 2020 使用边界是否写死 | PASS | 已写死 |
| Stage A 的 OOS 证据边界是否写死 | PASS | 已写死 |
| bootstrap 方案字段是否完整 | PASS | block bootstrap / block=20 / n_boot=400 / seed=20260426 已写死 |
| `corr_next` / `rank_next` 检验对象是否锁定 | PASS | 双检对象与阈值 `>=0.60` 已写死 |
| 所有关键 `UNFILLED` 是否清空 | PASS | prereg 关键字段已清空；结果文档另行交付 |
