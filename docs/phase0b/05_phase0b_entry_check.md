# Phase 0B 入场检查文档

> 文档性质  
> 本文不是结果结论书，而是是否具备进入 Phase 0B 资格的总闸门检查表。  
> 本文只回答一个问题：前置条件是否全部完成并冻结。

## 1. 文档目的

本文件用于在任何人提出进入 Phase 0B 之前，先统一检查：

1. benchmark 是否已锁定并交付  
2. trigger audit 是否已预注册并完成  
3. OOS 样本边界是否已冻结  
4. bootstrap sign-stability 是否已预注册  

若任一项不完整，则默认不得进入 Phase 0B。

## 2. Phase 0B 法定入场条件

根据已冻结协议，只有同时满足以下四条，Phase 0B 才具备立项资格：

1. Benchmark Delivery 已完成  
2. Trigger Audit Delivery 已完成且至少存在一个合法 trigger  
3. OOS Boundary Freeze 已冻结  
4. Bootstrap Sign-Stability Preregistration 已冻结  

## 3. 四项前置交付状态表

| 前置交付 | 文件路径 | 是否已落盘 | 是否已冻结 | 是否仍有 `UNFILLED` | 当前状态 |
|---|---|---|---|---|---|
| Benchmark Delivery | `docs/phase0b/01_benchmark_lock.md` + `docs/phase0b/02_benchmark_results_filled.md` | PASS | PASS | 否 | PASS |
| Trigger Audit Delivery | `docs/phase0b/03_trigger_audit_prereg.md` + `docs/phase0b/03_trigger_audit_results_filled.md` | PASS | PASS | 否 | FAIL |
| OOS Boundary Freeze | `docs/phase0b/04_oos_and_sign_stability_prereg.md` | PASS | PASS | 否 | PASS |
| Bootstrap Sign-Stability Preregistration | `docs/phase0b/04_oos_and_sign_stability_prereg.md` + `docs/phase0b/04_bootstrap_sign_stability_plan.md` | PASS | PASS | 否 | PASS |

## 4. Benchmark 完成状态

| 检查项 | 状态 | 备注 |
|---|---|---|
| benchmark family 是否锁定 | PASS | 已锁定四模型 family |
| 模型阶数是否锁定 | PASS | 均为 (1,1) |
| CRPS 分布族假设是否锁定 | PASS | 与各模型 innovation distribution 一致 |
| 2008 benchmark run 协议是否锁定 | PASS | rolling fixed-length，独立 2008 评估窗 |
| benchmark 结果表是否已填入真实结果 | PASS | 结果已落盘至 `02_benchmark_results_filled.md` |
| T5 candidate-side 是否可仓库内复现 | PARTIAL | 原始 `T5_resid_persistence_M4` 三窗口已复现；2008 candidate-side 同口径运行失败为 `FAILED_TO_RUN_NONFINITE_STANDARDIZED_RESIDUALS` |
| benchmark 结果是否仍含 `UNFILLED` | NO | 无关键 `UNFILLED` 残留 |
| 当前状态（PASS / FAIL / UNFILLED） | PASS | Benchmark Delivery 已完成；T5 三窗口可复现，2008 T5 failure 已落盘 |

## 5. Trigger audit 完成状态

| 检查项 | 状态 | 备注 |
|---|---|---|
| trigger 纳入/排除决策是否在触碰原始数据前锁定 | PASS | 七个候选全部纳入审计，状态已锁定 |
| `L_max` 是否预注册 | PASS | `L_max = 5` 个交易日 |
| 合法 lead window 是否写死 | PASS | `{-5, -4, -3, -2, -1}` 交易日 |
| false positive 口径是否锁定 | PASS | calm regime、前 20% 极端区间与 `<= 0.40` 门槛均已锁定 |
| trigger 审计结果是否已填入 | PASS | 结果已落盘至 `03_trigger_audit_results_filled.md` |
| 是否存在至少一个 pass 的合法 trigger | FAIL | 当前合法 `PASS` trigger 数量为 0 |
| 当前状态（PASS / FAIL / UNFILLED） | FAIL | Trigger Audit Delivery 已完成，但未产生合法 trigger |

## 6. OOS 边界冻结状态

| 检查项 | 状态 | 备注 |
|---|---|---|
| 2008 blind holdout 是否冻结 | PASS | 已在协议中写死 |
| 2000–2002 异质验证集是否冻结 | PASS | 已在协议中写死 |
| 2020 使用边界是否冻结 | PASS | 已在协议中写死 |
| Stage A 的 OOS 证据边界是否声明 | PASS | 已在协议中写死 |
| 当前状态（PASS / FAIL） | PASS | OOS 边界文本已冻结 |

## 7. Bootstrap prereg 完成状态

| 检查项 | 状态 | 备注 |
|---|---|---|
| bootstrap 方案是否锁定 | PASS | block bootstrap 已写死 |
| block 长度是否锁定 | PASS | 20 个观测点已写死 |
| 采样次数是否锁定 | PASS | 400 次已写死 |
| `corr_next` 是否纳入检验对象并写死阈值 | PASS | 纳入且阈值 `>= 0.60` 已写死 |
| `rank_next` 是否纳入检验对象并写死阈值 | PASS | 纳入且阈值 `>= 0.60` 已写死 |
| 是否仍含关键 `UNFILLED` | NO | prereg 字段已清空；计划文档结果区 `UNFILLED` 属预期占位 |
| 当前状态（PASS / FAIL / UNFILLED） | PASS | Bootstrap Sign-Stability Preregistration 已完成 |

## 8. 当前是否具备立项资格

> 这里只能填写以下三种标准结论之一，不得自由发挥。

### 标准结论模板

- `NO — prerequisite documents incomplete`
- `NO — documents present but still contain UNFILLED critical fields`
- `YES — all prerequisite conditions frozen`

### 当前结论

`NO — prerequisite documents incomplete`

## 9. 若不具备，缺失项清单

> 若第 8 节不是 YES，则必须逐项列出仍未完成的部分。

- Trigger Audit Delivery 已完成但未产生合法 trigger
- Bootstrap Sign-Stability Preregistration 已完成（结果填报仍待后续执行）
- 2008 T5 candidate-side 已同口径尝试运行并失败为 `FAILED_TO_RUN_NONFINITE_STANDARDIZED_RESIDUALS`；不影响 trigger audit 失败结论

## 10. 禁止绕过条款

1. 不得以“先做个原型再补协议”为由绕过本检查。  
2. 不得因为某项文档已落盘就视为已冻结。  
3. 不得因为单个结果好看就跳过其他前置交付。  
4. 不得在 `UNFILLED` 仍大量存在时宣布进入 Phase 0B。  

## 11. 当前四项前置交付默认状态

> 在你还没有实际填写和冻结之前，默认状态应是：

- Benchmark Delivery：`PASS`
- Trigger Audit Delivery：`FAIL`
- OOS Boundary Freeze：`PASS`
- Bootstrap Sign-Stability Preregistration：`PASS`

> 注意  
> OOS 边界冻结与 bootstrap 预注册是两个独立状态，不得合并解释。  
> 第 8 节仍然只能使用那三条标准模板结论。
