# Phase 0B 入场检查文档

> 文档性质  
> 本文不是结果结论书，而是是否具备进入 Phase 0B 资格的总闸门检查表。  
> 本文只回答一个问题：前置条件是否全部完成并冻结。

## 1. 文档目的

本文件用于在任何人提出进入 Phase 0B 之前，先统一检查：

1. benchmark 是否已锁定并交付  
2. trigger audit 是否已预注册并完成  
3. OOS freeze 是否已冻结  
4. bootstrap sign-stability 是否已预注册  

若任一项不完整，则默认不得进入 Phase 0B。

## 2. Phase 0B 法定入场条件

根据已冻结协议，只有同时满足以下四条，Phase 0B 才具备立项资格：

1. Benchmark Delivery 已完成  
2. Trigger Audit Delivery 已完成且至少存在一个合法 trigger  
3. OOS Freeze Declaration 已冻结  
4. Bootstrap Sign-Stability Preregistration 已冻结  

## 3. 四项前置交付状态表

| 前置交付 | 文件路径 | 是否已落盘 | 是否已冻结 | 是否仍有 `UNFILLED` | 当前状态 |
|---|---|---|---|---|---|
| Benchmark Delivery | `docs/phase0b/01_benchmark_lock.md` + `docs/phase0b/02_benchmark_results_template.md` | UNFILLED | UNFILLED | UNFILLED | UNFILLED |
| Trigger Audit Delivery | `docs/phase0b/03_trigger_audit_prereg.md` | UNFILLED | UNFILLED | UNFILLED | UNFILLED |
| OOS Freeze Declaration | `docs/phase0b/04_oos_and_sign_stability_prereg.md` | UNFILLED | UNFILLED | UNFILLED | UNFILLED |
| Bootstrap Sign-Stability Preregistration | `docs/phase0b/04_oos_and_sign_stability_prereg.md` | UNFILLED | UNFILLED | UNFILLED | UNFILLED |

## 4. Benchmark 完成状态

| 检查项 | 状态 | 备注 |
|---|---|---|
| benchmark family 是否锁定 | UNFILLED | UNFILLED |
| 模型阶数是否锁定 | UNFILLED | UNFILLED |
| CRPS 分布族假设是否锁定 | UNFILLED | UNFILLED |
| benchmark 结果表是否已填入真实结果 | UNFILLED | UNFILLED |
| benchmark 结果是否仍含 `UNFILLED` | UNFILLED | UNFILLED |
| 当前状态（PASS / FAIL / UNFILLED） | UNFILLED | UNFILLED |

## 5. Trigger audit 完成状态

| 检查项 | 状态 | 备注 |
|---|---|---|
| `L_max` 是否预注册 | UNFILLED | UNFILLED |
| 合法 lead window 是否写死 | UNFILLED | UNFILLED |
| false positive 口径是否锁定 | UNFILLED | UNFILLED |
| trigger 审计结果是否已填入 | UNFILLED | UNFILLED |
| 是否存在至少一个 pass 的合法 trigger | UNFILLED | UNFILLED |
| 当前状态（PASS / FAIL / UNFILLED） | UNFILLED | UNFILLED |

## 6. OOS freeze 完成状态

| 检查项 | 状态 | 备注 |
|---|---|---|
| 2008 blind holdout 是否冻结 | PASS | 已在协议中写死 |
| 2000–2002 异质验证集是否冻结 | PASS | 已在协议中写死 |
| 2020 使用边界是否冻结 | PASS | 已在协议中写死 |
| Stage A 的 OOS 证据边界是否声明 | PASS | 已在协议中写死 |
| 当前状态（PASS / FAIL / UNFILLED） | UNFILLED | UNFILLED |

## 7. Bootstrap prereg 完成状态

| 检查项 | 状态 | 备注 |
|---|---|---|
| bootstrap 方案是否锁定 | UNFILLED | UNFILLED |
| block 长度是否锁定 | UNFILLED | UNFILLED |
| 采样次数是否锁定 | UNFILLED | UNFILLED |
| `corr_next` 是否纳入检验对象并写死阈值 | UNFILLED | UNFILLED |
| `rank_next` 是否纳入检验对象并写死阈值 | UNFILLED | UNFILLED |
| 是否仍含关键 `UNFILLED` | UNFILLED | UNFILLED |
| 当前状态（PASS / FAIL / UNFILLED） | UNFILLED | UNFILLED |

## 8. 当前是否具备立项资格

> 这里只能填写以下三种标准结论之一，不得自由发挥。

### 标准结论模板

- `NO — prerequisite documents incomplete`
- `NO — documents present but still contain UNFILLED critical fields`
- `YES — all prerequisite conditions frozen`

### 当前结论

`UNFILLED`

## 9. 若不具备，缺失项清单

> 若第 8 节不是 YES，则必须逐项列出仍未完成的部分。

- UNFILLED

## 10. 禁止绕过条款

1. 不得以“先做个原型再补协议”为由绕过本检查。  
2. 不得因为某项文档已落盘就视为已冻结。  
3. 不得因为单个结果好看就跳过其他前置交付。  
4. 不得在 `UNFILLED` 仍大量存在时宣布进入 Phase 0B。  

## 11. 当前四项前置交付默认状态

> 在你还没有实际填写和冻结之前，默认状态应是：

- Benchmark Delivery：`UNFILLED`
- Trigger Audit Delivery：`UNFILLED`
- OOS Freeze Declaration：`PARTIAL`（已有边界文本，但整体交付未完成）
- Bootstrap Sign-Stability Preregistration：`UNFILLED`

> 注意  
> “PARTIAL” 只能作为说明，不能作为第 8 节的正式结论。  
> 第 8 节仍然只能使用那三条标准模板结论。
