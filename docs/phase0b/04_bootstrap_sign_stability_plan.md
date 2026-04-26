# Bootstrap Sign-Stability 执行计划文档

> 文档性质  
> 本文是 Bootstrap Sign-Stability Preregistration 的执行计划，不是结果文档。  
> 本文不提供任何 bootstrap 数值结果，也不授权进入 Phase 0B。

## 1. 文档目的

本文件用于固定 bootstrap sign-stability 的执行口径、参数和判定规则，确保后续运行不发生事后改口径。

## 2. prereg 摘要

- prereg 主文档：`docs/phase0b/04_oos_and_sign_stability_prereg.md`
- 预注册目标：冻结 directionality 稳健性审计规则
- 当前约束：Trigger Audit Delivery 仍为 `FAIL`，本步骤不改变该结论

## 3. bootstrap 参数锁定摘要

| 字段 | 锁定值 | 状态 |
|---|---|---|
| bootstrap 类型 | block bootstrap | LOCKED |
| block 长度 | 20 个观测点 | LOCKED |
| 采样次数 | 400 | LOCKED |
| 随机种子 | 20260426 | LOCKED |

## 4. 检验对象锁定摘要

| 检验对象 | 是否纳入 | 通过阈值 | 状态 |
|---|---|---|---|
| `corr_next` | 是 | `metric > 0` 频率 `>= 0.60` | LOCKED |
| `rank_next` | 是 | `metric > 0` 频率 `>= 0.60` | LOCKED |

双检必须同时成立，不允许单检替代。

## 5. 通过规则与失败规则

### 5.1 窗口级通过规则

窗口通过必须同时满足：

1. `corr_next_sign_stability >= 0.60`
2. `rank_next_sign_stability >= 0.60`

### 5.2 失败规则

任一窗口出现以下任一情况记为失败：

1. `corr_next > 0` 的频率 `< 0.60`
2. `rank_next > 0` 的频率 `< 0.60`
3. 采样运行失败或数值异常，记为 `FAILED_TO_RUN_BOOTSTRAP`

## 6. 结果填报占位区

> 本节本轮必须保持占位，不得伪造结果。

| 窗口 | corr_next_sign_stability | rank_next_sign_stability | 窗口状态 |
|---|---|---|---|
| Window_2017 | UNFILLED | UNFILLED | UNFILLED |
| Window_2018 | UNFILLED | UNFILLED | UNFILLED |
| Window_2020 | UNFILLED | UNFILLED | UNFILLED |
| Window_2008_Benchmark | UNFILLED | UNFILLED | UNFILLED |

## 7. 证据边界

1. Bootstrap sign-stability prereg 的完成，仅表示审计规则已冻结。  
2. 该完成状态不会覆盖 Trigger Audit Delivery 当前 `FAIL` 结论。  
3. 该完成状态不授权进入 Phase 0B。  
4. 只有四项前置交付都满足且 trigger 至少有一个合法 `PASS`，Phase 0B 才可能立项。  

## 8. 当前是否完成 Bootstrap Sign-Stability Preregistration

`YES — prereg and execution skeleton delivered; bootstrap numeric results remain UNFILLED by design`
