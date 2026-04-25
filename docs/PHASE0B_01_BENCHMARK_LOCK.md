# Benchmark 锁定文档

## 1. 文档目的

本文用于在运行任何 benchmark 之前，锁定 Phase 0B 所需的 benchmark family、估计协议与输出要求。本文不是结果报告，不包含任何实验输出。任何在本文冻结后对 benchmark family 的替换、增删或弱化，都构成协议违规。

## 2. 上游约束来源

本文受以下冻结文件约束：

- `docs/PHASE0A_PROTOCOL_FREEZE.md`
- `docs/PHASE0B_PREP_CHECKLIST.md`

若本文与上述文件冲突，以冻结协议为准。

## 3. Benchmark family 锁定表

| 字段 | 状态 | 当前值 | 说明 |
|---|---|---:|---|
| Benchmark family 清单 | REQUIRED | UNFILLED | 必须在运行前一次性写死 |
| 主 benchmark | REQUIRED | UNFILLED | 用于 Phase 0A Section 4C 的相对阈值比较 |
| 是否允许新增 benchmark | FIXED | NO | 运行后禁止新增 |
| 是否允许删除 benchmark | FIXED | NO | 运行后禁止删除 |

建议候选 family 仅作为待填项，不构成批准：
- EGARCH
- GJR-GARCH
- EGARCH-GJR 或等价 asymmetric GARCH

## 4. 模型级配置表

| 模型名 | 阶数 p | 阶数 q | leverage/asymmetric | 外生变量 | 状态 |
|---|---:|---:|---|---|---|
| Model_1 | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED |
| Model_2 | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED |
| Model_3 | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED |

## 5. CRPS 分布假设锁定

| 模型名 | 条件均值处理 | 条件方差输出 | CRPS 分布族假设 | 是否允许结果后替换 |
|---|---|---|---|---|
| Model_1 | UNFILLED | UNFILLED | UNFILLED | NO |
| Model_2 | UNFILLED | UNFILLED | UNFILLED | NO |
| Model_3 | UNFILLED | UNFILLED | UNFILLED | NO |

说明：看到结果后更改分布族假设，视为 benchmark 漂移。

## 6. 估计协议锁定

| 字段 | 状态 | 当前值 | 说明 |
|---|---|---:|---|
| 估计方法 | REQUIRED | UNFILLED | 例如 MLE |
| 优化器 | REQUIRED | UNFILLED | 必须写死 |
| 收敛容差 | REQUIRED | UNFILLED | 必须写死 |
| 最大迭代次数 | REQUIRED | UNFILLED | 必须写死 |
| 收敛失败处理 | REQUIRED | UNFILLED | 例如 fail-fast / 记录并跳过 |
| 缺失数据处理 | REQUIRED | UNFILLED | 必须写死 |

## 7. 窗口与滚动方式锁定

以下三窗口必须固定：

- 2017-07-07 -> 2017-12-29
- 2018-07-06 -> 2018-12-28
- 2020-01-03 -> 2020-06-26

| 字段 | 状态 | 当前值 | 说明 |
|---|---|---:|---|
| 训练/估计窗口方式 | REQUIRED | UNFILLED | expanding / rolling / other |
| 每窗估计是否独立 | REQUIRED | UNFILLED | 必须写死 |
| 数据频率 | REQUIRED | UNFILLED | 例如 daily / weekly |
| 结果输出顺序 | FIXED | 2017 -> 2018 -> 2020 | 不得调换 |

## 8. 禁止事项

1. 不得在看到结果后替换 benchmark。
2. 不得在看到结果后调整模型阶数。
3. 不得在看到结果后更改 CRPS 分布族假设。
4. 不得以“更强”或“更弱” benchmark 为由事后调包。
5. 任何 `UNFILLED` 字段未完成前，不得运行 benchmark。

## 9. 运行前检查项

| 检查项 | 当前状态 |
|---|---|
| Benchmark family 已锁定 | UNFILLED |
| 主 benchmark 已指定 | UNFILLED |
| 每个模型阶数已填写 | UNFILLED |
| CRPS 分布族假设已填写 | UNFILLED |
| 估计协议已填写 | UNFILLED |
| 三窗口与滚动方式已填写 | UNFILLED |
| 所有关键字段不再含 UNFILLED | UNFILLED |
