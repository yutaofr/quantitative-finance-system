# Tail Family Preregistration

> 文档性质
> 本文是在运行任何 tail family 实验结果之前冻结的方法论预注册。本文的唯一目的：在看到数据之前锁定所有判定标准。

## 1. 背景与前序结论

downstream scale-bias 实验（prereg-05）在三个窗口内验证了 per-window k 的稳定性：

| 窗口 | k | nu |
|---|---|---|
| 2017 | 2.682572 | 100.000000 |
| 2018 | 2.748825 | 100.000000 |
| 2020 | 2.604250 | 54.564819 |

k_range = 0.144575 < 0.60（预注册阈值），Option 1 = SUCCESS。

per-window k 确认稳定。本实验提出更严格的问题：**单一全局 k_fixed**（不随窗口变化）是否同样提供充分的分位数校准？

## 2. 固定输入

与 prereg-05 完全相同：T5_resid_persistence_M4 在 2017、2018、2020 三个窗口产生的 z_t 序列，不修改、不替换。

2020 的方向性缺陷（corr_next = -0.033011）是已知风险，在诊断表中显式记录，不修复。

## 3. 全局固定参数（运行前写死）

```
k_fixed = 2.682572
```

计算方式：对 prereg-05 产生的三个 per-window MLE 估计值取中位数：

```
median([2.604250, 2.682572, 2.748825]) = 2.682572
```

禁止：运行实验后修改 k_fixed；使用 per-window k；将 k 变为时变或 regime-conditioned。

## 4. Tail Family

对每个窗口计算：`u_t = z_t / k_fixed`

对 `u_t` 拟合方差标准化 Student-t：`u_t ~ StudentT_std(nu)`，联合参数为标量 `nu`（per-window）。

同时估计一个 pooled nu：将三个窗口的 `u_t` 合并，用一个全局 `nu_pooled` 拟合。

仅测试 Student-t。禁止在本轮引入 skewed-t、GED 或其他尾部族。若本实验失败，则在新的预注册中引入竞争族。

## 5. 判定标准（运行前写死）

### 5.1 分位数校准准则

对 τ ∈ {0.10, 0.25, 0.75, 0.90}，在每个窗口内：

```
calibration_error(τ) = |mean(u_t ≤ F_std_t^{-1}(τ | nu)) − τ|
```

**PASS 条件**：所有三个窗口、所有四个 τ 值的 calibration_error ≤ 0.05。

任一窗口任一 τ 超过 0.05 → 记为 `CALIBRATION_BREACH`，本实验 FAIL。

### 5.2 KS 准则

对每个窗口的 `u_t`，计算 Kolmogorov-Smirnov 检验统计量：

```
KS: u_t ~ StudentT_std(nu_window)
```

**PASS 条件**：所有三个窗口的 KS p-value ≥ 0.05。

任一窗口 p-value < 0.05 → 记为 `KS_REJECT`，本实验 FAIL。

### 5.3 Nu drift 准则

```
nu_drift = max(nu_2017, nu_2018, nu_2020) − min(nu_2017, nu_2018, nu_2020)
```

- nu_drift ≤ 30 → `NU_DRIFT_OK`：pooled nu 可接受，记录 `PASS_POOLED`。
- nu_drift > 30 → `NU_DRIFT`：per-window nu 差异过大，pooled nu 不可用。
  - `NU_DRIFT` 不直接导致 FAIL，但 pooled nu 的 PASS_POOLED 不成立。

### 5.4 2020 Coverage 传导准则

对 2020 窗口，用 per-window nu_2020 计算：

- central 90% coverage < 0.70 → `DIRECTION_DEFECT_PROPAGATED`，FAIL。
- central 80% coverage < 0.60 → `DIRECTION_DEFECT_PROPAGATED`，FAIL。

（与 prereg-05 §5 条件完全相同。）

### 5.5 Pooled Nu 额外校准

若 NU_DRIFT_OK 成立，对 pooled u_t 用 nu_pooled 也计算相同的校准准则（§5.1 + §5.2）。Pooled 的 calibration 与 KS 不影响 PASS/FAIL 主判定，只作为诊断输出。

## 6. 成功与失败的对称定义

### PASS_PER_WINDOW（主结论）

当且仅当：
1. 所有三个窗口 calibration_error(τ) ≤ 0.05（§5.1）
2. 所有三个窗口 KS p-value ≥ 0.05（§5.2）
3. 2020 未触发 DIRECTION_DEFECT_PROPAGATED（§5.4）

PASS_PER_WINDOW 成立：允许进入固定 k_fixed + per-window nu 的参数推荐阶段。

### PASS_POOLED（补充结论，需 NU_DRIFT_OK 为前提）

若 PASS_PER_WINDOW 成立且 NU_DRIFT_OK：pooled nu 可用于单参数下游配置。

### FAIL

任一主判定条件失败：
- 下游 tail family 建模以 Student-t + global k_fixed 不成立。
- 回到选项二（联合估计）或选项三（终止）。

## 7. 禁止事项

1. 运行实验后修改任何数值阈值（0.05、0.05、30、0.70、0.60）。
2. 运行实验后将 k_fixed 改为其他值。
3. 因为某一窗口通过就忽略另一窗口的失败。
4. 用调整 tail family 参数来掩盖 2020 的方向性缺陷。
5. 向本实验中新增 tail family 竞争模型（skewed-t 等）。
