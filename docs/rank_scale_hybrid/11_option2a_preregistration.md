# Option 2A Preregistration: Joint Location-Scale Minimum Falsification

> 文档性质
> 本文是在运行任何 Option 2A 实验结果之前冻结的方法论预注册。
> 目的只有一个：验证是否 μ_t 的位置修正能消除 tail family 实验中观察到的 quantile miscalibration。

## 1. 背景

Option 1 失败诊断（`09_tail_family_failure_diagnosis.md`）确认了失败主因为 location bias：
`z_t = (y_t − μ̂_t) / σ̂_t` 的条件均值随 market regime 系统性偏离零点，scalar k 无法修正。

本实验测试最小假说：**引入一步残差持久性的 location correction 后，quantile miscalibration 是否消失**。

## 2. 固定输入

与之前所有实验完全相同：T5_resid_persistence_M4 在 2017、2018、2020 三个窗口产生的
`(y_t, μ̂_t, σ̂_t, z_t)` 序列，不修改、不替换。

## 3. 模型形式（运行前写死）

**一步残差持久性位置修正（Tier 1）：**

```
z_t_corr = z_t − c · z_{t-1}
v_t      = z_t_corr / k_new
```

其中：
- `c ∈ (−0.99, 0.99)`：位置持久性系数，预注册约束范围
- `k_new > 0`：scale correction，联合估计
- `z_0` 约定：每个窗口的第一个 `z_{t-1}` 取 0（边界条件）

**Candidate A**：`v_t ~ N(0, 1)`，联合 MLE 估计 `(c, k_new)` per window

**Candidate B**：`v_t ~ StudentT_std(ν_fixed = 10)`，联合 MLE 估计 `(c, k_new)` per window
- `ν_fixed = 10` 预注册，不优化 ν，不允许事后修改

## 4. 估计方式

每个窗口内通过最大对数似然联合估计 `(c, k_new)`：

对 Candidate A（Gaussian）：

```
log L(c, k_new) = Σ_t [log φ(v_t) − log(k_new)]
```

其中 `φ` 是标准正态密度，`v_t = (z_t − c·z_{t-1}) / k_new`。

对 Candidate B（固定 ν = 10 Student-t）：

```
log L(c, k_new) = Σ_t [log p_std_t(v_t | ν=10) − log(k_new)]
```

## 5. 成功标准（运行前写死，三窗口同时满足）

以下 8 条全部通过，才允许 Candidate 成立：

1. `corr_next > 0`：`corr(σ̂_t_new[:-1], |e_{t+1}_new|[1:]) > 0`
2. `rank_next > 0`：`rank_corr(σ̂_t_new[:-1], |e_{t+1}_new|[1:]) > 0`
3. `|mean(v_t)| < 0.20`：位置偏差被吸收
4. `std(v_t) ∈ [0.80, 1.20]`：尺度校准合理
5. `cal_error(tau_0.25) < 0.08`
6. `cal_error(tau_0.75) < 0.08`
7. `sigma_blowup = 0`：窗口内无 `σ̂_t_new > 2 · median(σ̂_t_new)` 的周次
8. `pathology = 0`：优化器未失败，无 NaN/Inf

其中：
- `σ̂_t_new = k_new · σ̂_t_T5`（T5 的原始 σ̂_t 乘以联合估计的 k_new）
- `e_{t+1}_new = y_{t+1} − μ̂_{t+1} − c · (y_t − μ̂_t)`（新模型的残差）
  等价于：`e_{t+1}_new = σ̂_{t+1} · z_{t+1}_corr`

## 6. 校准指标计算

对每个窗口的 `v_t` 序列：

```
cal_error(τ) = |mean(v_t ≤ F^{-1}(τ | Candidate)) − τ|
```

其中 `F^{-1}` 是 Candidate A 或 Candidate B 的对应分位函数。

## 7. 裁决规则

**PASS_A**：Candidate A（Gaussian）在三窗口通过所有 8 条标准。

**PASS_B**：Candidate B（Student-t ν=10）在三窗口通过所有 8 条标准。

**FAIL**：Candidate A 和 Candidate B 均有任一窗口任一条件失败。

PASS_A 或 PASS_B 成立 → 允许进入 tail family 新预注册（引入 shape 参数）。
FAIL → 研究线终止，归档为负面结论，不做更复杂的连续模型修补。

## 8. 禁止事项

1. 不得在看到结果后修改成功门槛（0.20 / 1.20 / 0.80 / 0.08 等）。
2. 不得在看到结果后修改 `ν_fixed`（已锁定为 10）。
3. 不得引入 regime switch、MoE、trigger 机制。
4. 不得在 Candidate A/B 均失败之前引入更高自由度的 μ_t 模型（Tier 2）。
5. 不得因为某一窗口通过就忽略另一窗口的失败。
6. 2020 的方向性缺陷（corr_next = -0.033 在 T5 基础上）是已知风险；
   若新模型的 corr_next 在 2020 不满足 > 0，这不是可以豁免的条件。

## 9. 成功与失败的对称意义

- PASS：位置修正有效，μ_t 是主因，可以继续加 shape 层。
- FAIL：位置修正无效，问题不在 location 层，或 location + scale 联合后仍有深层缺陷。
  届时才有理由评估更复杂的 conditional shape 问题，或直接终止整条研究线。
