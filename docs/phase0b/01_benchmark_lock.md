# Benchmark 锁定文档

> 文档性质  
> 本文是 Phase 0B 之前的 benchmark 运行前锁定表，不是结果报告，不构成任何架构立项批准。  
> 本文的作用是：在看到任何 benchmark 结果之前，把 benchmark family、估计协议、CRPS 分布假设、窗口与滚动方式一次性写死，阻断事后调包与结果导向替换。

## 1. 文档目的

本文件用于锁定 Phase 0A / Phase 0B 过渡阶段中 benchmark family 的全部关键定义，以回答以下问题：

1. 2020 的失败是否为当前 `Stage A = T5` 的特有缺陷。
2. 还是连续危机波动模型的共同上限。
3. 未来若进入 Phase 0B，新的候选 crisis architecture 是否真的优于简单连续 benchmark。

本文件不输出结果，不分析优劣，不讨论新架构。

## 2. 上游约束来源

本文件必须服从以下冻结文件：

- `docs/PHASE0A_PROTOCOL_FREEZE.md`
- `docs/PHASE0B_PREP_CHECKLIST.md`

若本文件与上述冻结文件冲突，以冻结文件为准。

## 3. Benchmark family 锁定表

| 字段 | 取值 | 状态 | 说明 |
|---|---|---|---|
| 评估阶段 | Phase 0A | LOCKED | 本轮 benchmark 仅服务于 Phase 0A：candidate = T5，benchmark = 预锁定连续模型 family |
| benchmark family 名称 | UNFILLED | UNFILLED | 例如：EGARCH / GJR-GARCH / EGARCH-GJR |
| 是否允许新增 benchmark | 否 | LOCKED | 结果出来后不得新增、替换、删减 |
| 是否允许结果后更改 family | 否 | LOCKED | benchmark 不得成为结果导向下的可调对象 |

## 4. 模型级配置表

> 说明  
> 下表每一行必须在运行前填写完整。任何 `UNFILLED` 若在运行时仍未填写，则 benchmark 结果无效。

| 模型名 | 是否启用 | 阶数(p,q) | 是否含 leverage/asymmetric | 是否允许外生变量 | 外生变量清单 | 误差分布 | 备注 |
|---|---:|---|---|---|---|---|---|
| EGARCH | UNFILLED | UNFILLED | UNFILLED | 否 | N/A | UNFILLED | UNFILLED |
| GJR-GARCH | UNFILLED | UNFILLED | UNFILLED | 否 | N/A | UNFILLED | UNFILLED |
| EGARCH-GJR 或等价备选 | UNFILLED | UNFILLED | UNFILLED | 否 | N/A | UNFILLED | UNFILLED |

### 模型锁定法理说明

1. 阶数必须在运行前写死。  
2. 不得在看到结果后把 `(1,1)` 改成其他阶数。  
3. 默认不允许外生变量。若要允许，必须在运行前完整列出变量清单。  
4. 不得在看到结果后临时增加“更强 benchmark”或删掉“较强 benchmark”。

## 5. CRPS 分布假设锁定

> 法理说明  
> CRPS 不是点预测指标，必须依赖预测分布。  
> 因此，任何 benchmark 的 CRPS 计算所依赖的残差分布族假设，都必须在运行前写死。

| 模型名 | 用于 CRPS 的预测分布族假设 | 是否与估计分布一致 | 若不一致，理由 | 状态 |
|---|---|---|---|---|
| EGARCH | UNFILLED | UNFILLED | UNFILLED | UNFILLED |
| GJR-GARCH | UNFILLED | UNFILLED | UNFILLED | UNFILLED |
| EGARCH-GJR 或等价备选 | UNFILLED | UNFILLED | UNFILLED | UNFILLED |

### 禁令

- 不得在看到结果后更改 CRPS 所依赖的预测分布族假设。  
- 不得在看到某模型 CRPS 不利时，切换其分布族来“修饰”结果。  

## 6. 估计协议锁定

| 字段 | 取值 | 状态 | 说明 |
|---|---|---|---|
| 估计方法 | UNFILLED | UNFILLED | 例如 MLE |
| 优化器 | UNFILLED | UNFILLED | 例如 BFGS / L-BFGS-B / SLSQP |
| 收敛容差 | UNFILLED | UNFILLED | 必须在运行前写死 |
| 最大迭代次数 | UNFILLED | UNFILLED | 必须在运行前写死 |
| 收敛失败处理 | UNFILLED | UNFILLED | 例如：记为失败 / 重试规则 / 不重试 |
| 数值异常处理 | UNFILLED | UNFILLED | 如 NaN / inf / 非正方差约束失败 |
| 随机种子策略 | UNFILLED | UNFILLED | 若适用，必须锁定 |

### 协议约束

1. 不得在看到结果后放宽优化器参数以“拯救”某个 benchmark。  
2. 收敛失败处理必须统一适用于所有 benchmark。  
3. 若某模型因估计协议而频繁失败，这本身是 benchmark 结果的一部分，不得事后淡化。

## 7. 窗口与滚动方式锁定

### 固定窗口

| 窗口名 | 起始日期 | 结束日期 | 状态 |
|---|---|---|---|
| Window_2017 | 2017-07-07 | 2017-12-29 | LOCKED |
| Window_2018 | 2018-07-06 | 2018-12-28 | LOCKED |
| Window_2020 | 2020-01-03 | 2020-06-26 | LOCKED |

### 训练/评估方式

| 字段 | 取值 | 状态 | 说明 |
|---|---|---|---|
| 滚动方式 | UNFILLED | UNFILLED | expanding / rolling / fixed 等 |
| 训练窗定义 | UNFILLED | UNFILLED | 必须写清楚 |
| 预测频率 | UNFILLED | UNFILLED | 日频 / 周频映射方式必须明确 |
| 输出频率 | UNFILLED | UNFILLED | 与 T5 比较口径必须一致 |
| 数据源 | UNFILLED | UNFILLED | 必须固定 |

### 约束

- 不得在看到结果后切换滚动方式。  
- benchmark 与 T5 的比较必须使用一致的窗口口径。  

## 8. 禁止事项

1. 不得在结果出来后增加 benchmark。  
2. 不得在结果出来后删除 benchmark。  
3. 不得在结果出来后更改阶数。  
4. 不得在结果出来后更改 CRPS 的分布族假设。  
5. 不得在结果出来后改滚动方式、优化器或容差。  
6. 不得把“更复杂”或“更弱”的 benchmark 拿来事后调包。  

## 9. 运行前检查项

| 检查项 | 状态 | 备注 |
|---|---|---|
| benchmark family 是否完整列出 | UNFILLED | UNFILLED |
| 每个模型的阶数是否写死 | UNFILLED | UNFILLED |
| 是否允许 leverage/asymmetric 是否写死 | UNFILLED | UNFILLED |
| 是否允许外生变量是否写死 | UNFILLED | UNFILLED |
| CRPS 分布族假设是否写死 | UNFILLED | UNFILLED |
| 优化器与容差是否写死 | UNFILLED | UNFILLED |
| 三个固定窗口是否一致 | PASS | 已冻结 |
| 滚动方式是否写死 | UNFILLED | UNFILLED |
| 所有 `UNFILLED` 是否已清空 | UNFILLED | 若否，则不得运行 benchmark |

## 10. 当前仍为 `UNFILLED` 的字段清单

> 本节必须在 benchmark 真正运行前清空。若仍保留 `UNFILLED`，则本文件不具执行资格。

- benchmark family 名称：UNFILLED
- 各模型是否启用：UNFILLED
- 各模型阶数：UNFILLED
- leverage/asymmetric 设定：UNFILLED
- 外生变量设定：UNFILLED
- 误差分布：UNFILLED
- CRPS 分布族假设：UNFILLED
- 估计方法：UNFILLED
- 优化器：UNFILLED
- 收敛容差：UNFILLED
- 最大迭代次数：UNFILLED
- 收敛失败处理：UNFILLED
- 数值异常处理：UNFILLED
- 滚动方式：UNFILLED
- 训练窗定义：UNFILLED
- 预测频率：UNFILLED
- 输出频率：UNFILLED
- 数据源：UNFILLED
