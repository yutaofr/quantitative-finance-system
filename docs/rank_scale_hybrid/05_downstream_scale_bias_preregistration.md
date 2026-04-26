# Downstream Scale-Bias Preregistration

> 文档性质  
> 本文是在运行任何 downstream tail family 结果之前冻结的方法论预注册。本文只回答一个问题：T5 的 `std(z)≈2` 尺度偏差能否被下游单一稳定标量 `k` 吸收。

## 1. 固定输入

唯一输入为当前仓库内 `T5_resid_persistence_M4` 在 2017、2018、2020 三个窗口产生的 `z_t` 序列。T5 不修改、不替换、不重新校准。

2020 的方向性缺陷是已知风险：`corr_next = -0.033011`。本轮不修复该缺陷，只在 tail diagnostics 中显式记录其是否传导为 coverage 崩塌。

## 2. 候选 tail family

本轮只使用 Student-t scale-bias family：

`u_t = z_t / k`

`u_t` 服从均值为 0、方差为 1 的 Student-t innovation，`nu > 2`。联合估计参数为：

1. `k`：单一标量 scale-bias correction。
2. `nu`：Student-t 自由度。

`k` 不允许时变，不允许 regime-conditioned，不允许按窗口内子样本切换。

## 3. k 的估计方式

每个窗口内通过最大对数似然联合估计 `k` 和 `nu`。对 `z_t` 的 likelihood 为：

`log p(z_t | k, nu) = log p_std_t(z_t / k | nu) - log(k)`

其中 `p_std_t` 是方差标准化 Student-t 密度。

naive baseline 固定 `k = 1`，只估计 `nu`。

## 4. k 稳定性判定标准

以下三条在运行前写死，任一不满足即判 Option 1 不成立：

1. 三窗口估计的 `k` 全部落在 `[1.5, 2.75]` 区间内。
2. 三窗口 `k` 的极差 `max(k) - min(k) <= 0.60`。
3. 引入 `k` 后，相对 `k = 1` baseline 的总 log-likelihood 改善为正，且至少两个窗口 log-likelihood 改善为正。

## 5. Coverage 与 2020 方向性传导检验

使用估计出的 `k` 和 `nu`，将 `z_t` 转为 `u_t = z_t / k`。报告 central 80% 与 central 90% interval coverage。

预注册 coverage 崩塌标准：

1. 任一窗口 central 90% coverage 低于 `0.70`，记为 `COVERAGE_COLLAPSE`。
2. 2020 central 90% coverage 低于 `0.70`，或 2020 central 80% coverage 低于 `0.60`，记为 `DIRECTION_DEFECT_PROPAGATED`。

`DIRECTION_DEFECT_PROPAGATED` 不单独作为数学证明，但在本轮裁决中与 coverage 崩塌同等处理：若触发，则 Option 1 不成立。

## 6. 禁止事项

1. 不得在看到 `k` 估计值后修改区间 `[1.5, 2.75]`。
2. 不得在看到结果后把 `k` 改为时变或 regime-conditioned。
3. 不得把 2020 的方向性缺陷通过调整 tail family 参数来掩盖。
4. 不得因为 `k` 在某一窗口稳定就忽略其他窗口的漂移。
5. 不得在本轮新增 tail family 与 Student-t 竞争。

## 7. 成功与失败的对称定义

Option 1 成立，当且仅当：

1. 三窗口 `k` 全部落在 `[1.5, 2.75]`。
2. 三窗口 `k` 极差不超过 `0.60`。
3. scale-bias model 相对 `k = 1` baseline 的总 log-likelihood 改善为正，且至少两个窗口改善为正。
4. 未触发 `COVERAGE_COLLAPSE`。
5. 2020 未触发 `DIRECTION_DEFECT_PROPAGATED`。

Option 1 不成立，若任一条件失败。结论固定为：T5 的尺度偏差无法被单一稳定 `k` 吸收，回到选项二（联合估计）或选项三（终止）。
