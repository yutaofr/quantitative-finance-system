# Benchmark 锁定文档

> 文档性质  
> 本文是 Phase 0B 之前的 benchmark 运行前锁定表，不是结果报告，不构成任何架构立项批准。  
> 本文的作用是：在看到任何 benchmark 结果之前，把 benchmark family、估计协议、CRPS 分布假设、窗口与滚动方式一次性写死，阻断事后调包与结果导向替换。

## 1. 文档目的

本文件用于锁定 Phase 0A / Phase 0B 过渡阶段中 benchmark family 的全部关键定义，以回答以下问题：

1. 2020 的失败是否为当前 `Stage A = T5` 的特有缺陷。
2. 还是连续危机波动模型的共同上限。
3. 未来若进入 Phase 0B，新的候选 crisis architecture 是否真的优于简单连续 benchmark。
4. 为 Phase 0A 第 4 节条件 D 提供合法的 `Benchmark_2008_std(z)` 分母值。

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
| benchmark family 名称 | EGARCH/GJR-GARCH(1,1) fixed benchmark family | LOCKED | 预锁定的连续 benchmark family |
| 是否允许新增 benchmark | 否 | LOCKED | 结果出来后不得新增、替换、删减 |
| 是否允许结果后更改 family | 否 | LOCKED | benchmark 不得成为结果导向下的可调对象 |

## 4. 模型级配置表

> 说明  
> 下表每一行必须在运行前填写完整。任何 `UNFILLED` 若在运行时仍未填写，则 benchmark 结果无效。

| 模型名 | 是否启用 | 阶数(p,q) | 是否含 leverage/asymmetric | 是否允许外生变量 | 外生变量清单 | 误差分布 | 备注 |
|---|---:|---|---|---|---|---|---|
| EGARCH(1,1)-Normal | 是 | (1,1) | 是 | 否 | N/A | Normal | 与估计一致 |
| EGARCH(1,1)-Student-t | 是 | (1,1) | 是 | 否 | N/A | Student-t | 与估计一致 |
| GJR-GARCH(1,1)-Normal | 是 | (1,1) | 是 | 否 | N/A | Normal | 与估计一致 |
| GJR-GARCH(1,1)-Student-t | 是 | (1,1) | 是 | 否 | N/A | Student-t | 与估计一致 |

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
| EGARCH(1,1)-Normal | Normal | 是 | 无 | LOCKED |
| EGARCH(1,1)-Student-t | Student-t | 是 | 无 | LOCKED |
| GJR-GARCH(1,1)-Normal | Normal | 是 | 无 | LOCKED |
| GJR-GARCH(1,1)-Student-t | Student-t | 是 | 无 | LOCKED |

### 禁令

- 不得在看到结果后更改 CRPS 所依赖的预测分布族假设。  
- 不得在看到某模型 CRPS 不利时，切换其分布族来“修饰”结果。  

## 6. 估计协议锁定

| 字段 | 取值 | 状态 | 说明 |
|---|---|---|---|
| 估计方法 | MLE | LOCKED | 逐窗口滚动估计 |
| 优化器 | L-BFGS-B | LOCKED | 统一适用于四个 benchmark |
| 收敛容差 | 1.0e-6 | LOCKED | 与 runner 一致 |
| 最大迭代次数 | 200 | LOCKED | 与 runner 一致 |
| 收敛失败处理 | 记为 FAILED_TO_RUN，不重试，不切换模型 | LOCKED | 若任一窗口失败，保留失败事实 |
| 数值异常处理 | 记为 FAILED_TO_RUN | LOCKED | NaN / inf / 非正方差约束失败均视为失败 |
| 随机种子策略 | 不适用（确定性优化，无随机性注入） | LOCKED | 不使用随机初始化 |

### 协议约束

1. 不得在看到结果后放宽优化器参数以“拯救”某个 benchmark。  
2. 收敛失败处理必须统一适用于所有 benchmark。  
3. 若某模型因估计协议而频繁失败，这本身是 benchmark 结果的一部分，不得事后淡化。

## 7. 窗口与滚动方式锁定

### 固定比较窗口

| 窗口名 | 起始日期 | 结束日期 | 用途 | 状态 |
|---|---|---|---|---|
| Window_2017 | 2017-07-07 | 2017-12-29 | Phase 0A 固定比较窗口 | LOCKED |
| Window_2018 | 2018-07-06 | 2018-12-28 | Phase 0A 固定比较窗口 | LOCKED |
| Window_2020 | 2020-01-03 | 2020-06-26 | Phase 0A 固定比较窗口 | LOCKED |
| Window_2008_Benchmark | 2008-01-04 | 2008-12-26 | 仅用于生成 `Benchmark_2008_std(z)` 与 benchmark 的 2008 绝对表现 | LOCKED |

### 训练/评估方式

| 字段 | 取值 | 状态 | 说明 |
|---|---|---|---|
| 滚动方式 | rolling fixed-length window | LOCKED | 每个 as_of 仅使用其前 416 周训练样本，并施加 53 周 embargo |
| 训练窗定义（2017/2018/2020） | as_of 往前回看 53 周后，再取更早的最后 416 周训练 | LOCKED | 仅使用 as_of 之前的历史周频样本 |
| 训练窗定义（2008 benchmark run） | 同一 rolling fixed-length 口径；评估窗为 2008-01-04 -> 2008-12-26，训练信息集不得依赖 2008 之后信息 | LOCKED | 与 2017/2018/2020 完全同口径 |
| 预测频率 | 周频 | LOCKED | 以 Friday close 对齐 |
| 输出频率 | 周频 | LOCKED | 与 T5 比较口径一致 |
| 数据源 | `data/raw/nasdaq/NASDAQXNDX/close.parquet` | LOCKED | 本地缓存的 NASDAQ XNDX 日度收盘数据 |

### 2008 benchmark run 的法理用途

1. `Window_2008_Benchmark` 的唯一用途是为 Phase 0A 第 4 节条件 D 提供 `Benchmark_2008_std(z)` 与 benchmark 的 2008 绝对表现。  
2. 该运行不构成对 T5 的追溯性 OOS 证明。  
3. 该运行允许 benchmark 在 2008 窗口进行**评估**，但其模型类、阶数、分布与估计协议必须完全遵守本文件的预锁定条款。  
4. 若 2008 benchmark run 的训练/预测信息集定义未在运行前写死，则 `Benchmark_2008_std(z)` 不具法律地位，不得进入任何结果表。

### 约束

- 不得在看到结果后切换滚动方式。  
- benchmark 与 T5 的比较必须使用一致的窗口口径。  
- 2008 benchmark run 的训练信息集必须在运行前写死。  

## 8. 禁止事项

1. 不得在结果出来后增加 benchmark。  
2. 不得在结果出来后删除 benchmark。  
3. 不得在结果出来后更改阶数。  
4. 不得在结果出来后更改 CRPS 的分布族假设。  
5. 不得在结果出来后改滚动方式、优化器或容差。  
6. 不得把“更复杂”或“更弱”的 benchmark 拿来事后调包。  
7. 不得在未锁定 2008 benchmark run 协议的情况下生成 `Benchmark_2008_std(z)`。  

## 9. 运行前检查项

| 检查项 | 状态 | 备注 |
|---|---|---|
| benchmark family 是否完整列出 | PASS | 四个模型已锁定 |
| 每个模型的阶数是否写死 | PASS | 均为 (1,1) |
| 是否允许 leverage/asymmetric 是否写死 | PASS | EGARCH/GJR 均已锁定 |
| 是否允许外生变量是否写死 | PASS | 不允许 |
| CRPS 分布族假设是否写死 | PASS | 与各模型 innovation distribution 一致 |
| 优化器与容差是否写死 | PASS | L-BFGS-B / 1.0e-6 / maxiter=200 |
| 四个固定窗口是否一致 | PASS | 已冻结 |
| 2008 benchmark run 的训练信息集是否写死 | PASS | 与 2017/2018/2020 相同的 rolling fixed-length 口径 |
| 滚动方式是否写死 | PASS | rolling fixed-length window |
| 所有 `UNFILLED` 是否已清空 | PASS | 已清空 |

## 10. 当前已冻结字段清单

> 本节在 benchmark 真正运行前必须为空；当前已全部冻结完成。

- benchmark family 名称：EGARCH/GJR-GARCH(1,1) fixed benchmark family
- 各模型是否启用：PASS
- 各模型阶数：PASS（均为 (1,1)）
- leverage/asymmetric 设定：PASS（EGARCH 与 GJR 均启用）
- 外生变量设定：PASS（不允许）
- 误差分布：PASS（Normal / Student-t）
- CRPS 分布族假设：PASS（与各模型 innovation distribution 一致）
- 估计方法：PASS（MLE）
- 优化器：PASS（L-BFGS-B）
- 收敛容差：PASS（1.0e-6）
- 最大迭代次数：PASS（200）
- 收敛失败处理：PASS（FAILED_TO_RUN，不重试，不替换模型）
- 数值异常处理：PASS（FAILED_TO_RUN）
- 滚动方式：PASS（rolling fixed-length window）
- 训练窗定义（2017/2018/2020）：PASS（as_of 前 53 周 embargo，再取更早最后 416 周）
- 训练窗定义（2008 benchmark run）：PASS（同一 rolling fixed-length 口径，不使用 2008 之后信息）
- 预测频率：PASS（周频）
- 输出频率：PASS（周频）
- 数据源：PASS（`data/raw/nasdaq/NASDAQXNDX/close.parquet`）
