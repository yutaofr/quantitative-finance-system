# Phase 0B 执行准备清单

> 目的
>
> 本文不是新架构方案，也不授予任何 crisis architecture 立项资格。本文仅把 `PHASE0A_PROTOCOL_FREEZE.md` 中已经冻结的前置条件转写为可执行的准备清单，供进入 Phase 0B 之前逐项完成。

## 1. 总体裁决

在以下四项交付物全部完成并冻结前，Phase 0B 不得开始：

1. Benchmark Delivery
2. Trigger Audit Delivery
3. OOS Freeze Declaration
4. Bootstrap Sign-Stability Preregistration

任何缺失都意味着：不得进入 override / hard switch / MoE 或其他 crisis architecture 实验。

## 2. Benchmark Delivery

### 2.1 目标

在与当前协议相同的窗口定义下，输出锁定 benchmark family 的完整诊断表，用于回答：

- 2020 的失败是否为当前架构特有问题
- 还是连续危机波动模型的共同上限

### 2.2 必须锁定的内容

在运行 benchmark 之前，必须一次性写死：

- 模型清单
- 每个模型的阶数
- 误差分布设定
- **CRPS 计算所依赖的预测分布族假设**
- 是否允许 leverage / asymmetric term
- 是否允许外生变量
- 估计协议（MLE / 优化器 / 容差 / 收敛失败处理）
- 窗口定义与滚动方式

### 2.3 最低交付格式

每个 benchmark 至少输出以下三窗口结果：

- 2017-07-07 -> 2017-12-29
- 2018-07-06 -> 2018-12-28
- 2020-01-03 -> 2020-06-26

每窗口必须报告：

- mean(z)
- std(z)
- corr_next
- rank_next
- lag1_acf(z)
- sigma_blowup
- pathology
- CRPS

### 2.4 禁令

- 不得在看到结果后替换 benchmark
- 不得在看到结果后更改模型阶数
- 不得在看到结果后更改 CRPS 所依赖的残差分布族假设
- 不得用“更强 benchmark”或“更弱 benchmark”做事后调包

## 3. Trigger Audit Delivery

### 3.1 目标

回答：哪些 trigger 具备进入未来 crisis architecture 实验的最低合法资格。

### 3.2 前置锁定

在 trigger audit 开始之前，必须先预注册：

- **与具体架构形式无关的保守执行延迟上限** `L_max`（交易日）
- 合法 lead window = `{-L_max, ..., -1}`
- 常态窗 false positive 的计算方式

若 `L_max` 未预注册，则整个 trigger audit 无效。未来任何架构若其有效执行延迟大于 `L_max`，则不得继承本轮 trigger audit 的合法性，必须重新审计。

### 3.3 审计对象

可进入审计的候选信号包括但不限于：

- VIX9D / VIX
- VIX / VIX3M
- ΔVIX9D
- credit spread jump
- FRA-OIS widening
- liquidity stress signals
- lagged realized shock

### 3.4 最低交付格式

每个候选 trigger 至少必须输出：

- 合法 lead window 内的相关/审计结果
- 窗口外的 false positive rate
- 2000 / 2008 / 2020 的一致性说明
- 是否通过 trigger 准入

### 3.5 禁令

- 未通过 lead-lag audit 的 trigger，不得进入任何架构实验
- 不得因为某 trigger 在单一窗口（如 2020）看起来“有解释力”而绕过审计

## 4. OOS Freeze Declaration

### 4.1 目标

把未来架构实验的样本使用规则写成不可回退的硬隔离协议。

### 4.2 必须锁定的样本边界

- 2008–2009：blind holdout，不参与训练、参数拟合、超参数选择
- 2000–2002：异质验证集
- 2020：允许暴露，但不得作为唯一危机代理

### 4.3 必须声明的证据边界

- 2008 的 blind holdout 对未来候选 crisis architecture 与 benchmark 比较构成真正 OOS 约束
- 但 2008 不会追溯性地把 `Stage A = T5` 变成对 2008 的真正 OOS，因为 T5 的选择过程已经暴露于后来的危机样本信息

### 4.4 禁令

- 不得用 2008 调参数
- 不得看完 2008 结果后再修改成功阈值
- 不得把 2008 的通过夸大表述为“Stage A 的通用性已被 OOS 证明”

## 5. Bootstrap Sign-Stability 预注册

### 5.1 目标

补齐当前协议中尚未数值化的一条门槛：directionality 的稳健性不能只靠 point estimate 过零。

### 5.2 必须锁定的内容

在进入 Phase 0B 前，必须写死：

- bootstrap 方案
- block 长度或等价相关设置
- 采样次数
- **检验对象：`corr_next` 与 `rank_next` 是否都进入 sign-stability 审计，或分别采用何种通过阈值**
- sign-stability 的通过阈值
- 失败时的判定规则

若 `corr_next` 与 `rank_next` 的阈值不相同，必须分别预注册；若只对其中一个做审计，也必须预先说明为什么另一个不构成独立稳健性要求。

### 5.3 禁令

- 未预注册 sign-stability 阈值，不得进入 Phase 0B
- 不得在看到 bootstrap 结果后再改 block 长度、检验对象或通过阈值

## 6. Phase 0B 启动门槛

只有同时满足下面四条，Phase 0B 才具备立项资格：

1. benchmark family 已锁定并完成交付
2. trigger audit 已完成且存在至少一个合法 trigger
3. OOS 隔离方案已冻结
4. bootstrap sign-stability 阈值已预注册

若任一条件不满足，则 Phase 0B 自动延期，不得以“先做原型再补协议”的方式绕过。

## 7. 立项后仍禁止的事项

即使进入 Phase 0B，以下事项仍然禁止：

- 用未审计 trigger 直接驱动 override / MoE
- 用 2008 结果做任何参数回填
- 把 benchmark 当成可调对象
- 把单一窗口的局部改善包装成架构性成功

## 8. 最小执行顺序

建议按以下顺序推进：

1. 锁 benchmark family
2. 输出 benchmark diagnosis
3. 预注册 trigger audit 的 `L_max`
4. 完成 trigger audit
5. 冻结 OOS 样本边界
6. 预注册 bootstrap sign-stability
7. 复核是否具备 Phase 0B 立项资格

在第 7 步之前，不得开始任何新架构实验。
