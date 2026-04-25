# Benchmark 锁定文档

> 文档状态
>
> 本文是 Phase 0B 的运行前锁定文档，不是结果报告。本文只定义 benchmark 的运行边界与必填字段，不授予任何新架构立项资格。

## 1. 文档目的

本文用于在 benchmark 运行之前，一次性锁定 benchmark family、估计协议、CRPS 计算假设、窗口与滚动方式，防止在看到结果后替换 benchmark 或修改估计口径。

## 2. 上游约束来源

本文的唯一上游依据为：

- `docs/PHASE0A_PROTOCOL_FREEZE.md`
- `docs/PHASE0B_PREP_CHECKLIST.md`

若本文与上述两份冻结文档冲突，以上述两份为准。

## 3. Benchmark family 锁定表

| 字段 | 当前值 | 说明 |
|---|---|---|
| Benchmark family 名称 | UNFILLED | 例如 EGARCH / GJR-GARCH / EGARCH-GJR |
| 主 benchmark | UNFILLED | 用于 Phase 0A Section 4C 的主 benchmark |
| 是否允许多个 benchmark 并行 | UNFILLED | `YES` / `NO` |
| 是否允许运行后替换 benchmark | NO | 冻结后禁止替换 |

## 4. 模型级配置表

| 模型名 | 是否纳入 | 阶数 p | 阶数 q | leverage / asymmetric | 是否允许外生变量 | 备注 |
|---|---:|---:|---:|---|---|---|
| EGARCH | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED |
| GJR-GARCH | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED |
| EGARCH-GJR 或等价变体 | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED |

## 5. CRPS 分布假设锁定

> 说明
>
> CRPS 不是点预测评分，必须依赖预测分布族。该分布假设必须在运行 benchmark 前写死，不得在看到结果后替换。

| 模型名 | 误差分布设定 | CRPS 所依赖的预测分布族假设 | 是否允许结果后修改 |
|---|---|---|---|
| EGARCH | UNFILLED | UNFILLED | NO |
| GJR-GARCH | UNFILLED | UNFILLED | NO |
| EGARCH-GJR 或等价变体 | UNFILLED | UNFILLED | NO |

## 6. 估计协议锁定

| 字段 | 当前值 | 说明 |
|---|---|---|
| 估计方法 | UNFILLED | 例如 MLE |
| 优化器 | UNFILLED | 例如 BFGS / L-BFGS-B |
| 初值规则 | UNFILLED | 运行前必须写死 |
| 容差 | UNFILLED | 收敛标准 |
| 最大迭代次数 | UNFILLED | 运行前必须写死 |
| 收敛失败处理 | UNFILLED | 例如标记失败 / 跳过 / 终止 |
| 数值异常处理 | UNFILLED | 例如 NaN / inf 处理方式 |

## 7. 窗口与滚动方式锁定

### 7.1 固定窗口

| 窗口名称 | 日期范围 |
|---|---|
| Window 2017 | 2017-07-07 -> 2017-12-29 |
| Window 2018 | 2018-07-06 -> 2018-12-28 |
| Window 2020 | 2020-01-03 -> 2020-06-26 |

### 7.2 滚动方式

| 字段 | 当前值 | 说明 |
|---|---|---|
| 训练窗构造方式 | UNFILLED | expanding / rolling / fixed |
| 预测频率 | UNFILLED | 日频/周频映射规则 |
| 每窗是否重估参数 | UNFILLED | `YES` / `NO` |
| 数据对齐规则 | UNFILLED | 运行前必须写死 |

## 8. 禁止事项

1. 不得在看到结果后替换 benchmark。
2. 不得在看到结果后调整模型阶数。
3. 不得在看到结果后更改误差分布设定。
4. 不得在看到结果后更改 CRPS 所依赖的预测分布族假设。
5. 不得在看到结果后以“更强”或“更弱” benchmark 替换已锁定基准。
6. 未填写 `UNFILLED` 关键字段前，不得启动 benchmark 运行。

## 9. 运行前检查项

| 检查项 | 状态 |
|---|---|
| Benchmark family 已锁定 | UNFILLED |
| 主 benchmark 已指定 | UNFILLED |
| 模型阶数已全部填写 | UNFILLED |
| leverage / asymmetric 规则已锁定 | UNFILLED |
| 外生变量规则已锁定 | UNFILLED |
| 误差分布设定已锁定 | UNFILLED |
| CRPS 分布假设已锁定 | UNFILLED |
| 优化器与容差已锁定 | UNFILLED |
| 收敛失败处理已锁定 | UNFILLED |
| 三窗口与滚动方式已锁定 | UNFILLED |

> 结论
>
> 上表只要仍存在对运行有影响的 `UNFILLED` 字段，则 benchmark 运行无效。