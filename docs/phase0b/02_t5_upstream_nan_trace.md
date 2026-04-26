# T5 2008 Upstream NaN Trace

> 文档性质  
> 本文只追踪源码对齐后的 2008 `T5_resid_persistence_M4` 中 `y_train = nan` 的上游来源。本文不修复数据，不修改协议，不引入新架构。

## 1. 文档目的

本文件回答：2008 第一处污染点 `1998-10-09` 的 `y_train` 如何生成，依赖哪些原始量，以及最早哪个原始量先变成 non-finite。

## 2. 目标变量定义

`y_train` 来自：

```text
train_frame.target_returns["NASDAQXNDX"]
```

其上游在 `build_panel_feature_block()` 中由 `_micro_history_for_asset()` 生成：

```text
target_returns = [_forward_52w_return(target_series, week) for week in weeks]
```

`_forward_52w_return()` 的定义为：

```text
current = value_at(target_series, week)
future = value_at(target_series, week + 52 weeks)
if current/future non-finite or <= 0: return nan
else return log(future / current)
```

## 3. 第一污染点

| 字段 | 值 |
|---|---|
| first polluted week | `1998-10-09` |
| target series first timestamp | `1999-03-10` |
| target series first value | `43.0743408203125` |
| first finite current target week | `1999-03-12` |
| first polluted week current | `nan` |
| first polluted week 52w future | `53.78361892700195` |
| first polluted week forward 52w return | `nan` |

## 4. 依赖原始量检查

| 依赖量 | `1998-10-09` 状态 | 说明 |
|---|---|---|
| `value_at(target_series, 1998-10-09)` | `nan` | NASDAQXNDX/QQQ target series begins on `1999-03-10` |
| `value_at(target_series, 1999-10-08)` | `53.78361892700195` | 52 周 future value 可用 |
| `_forward_52w_return(target_series, 1998-10-09)` | `nan` | current leg missing causes target nan |

## 5. 污染类别裁决

`UPSTREAM_DATA_EDGE_CASE_CONFIRMED`

理由：污染来自原始 target price series 的起始日期晚于 2008 训练窗最早周次。2008 candidate-side 按 416 周训练窗和 53 周 embargo 回看至 `1998-10-09`，但 target series 到 `1999-03-10` 才有首个价格，因此 `_forward_52w_return()` 的 current leg 为 non-finite。该污染先于 T5 residual-persistence 公式链发生。

## 6. 非裁决项

- 不是 `TARGET_CONSTRUCTION_EDGE_CASE_CONFIRMED`：target 公式按协议返回 `nan`，没有发现公式自身错误。
- 不是 `SLICING_WARMUP_BUG_CONFIRMED`：rolling 416 周 + 53 周 embargo 按锁定口径运行，问题是口径触及 target series 起点之前。
- 不是 `ORIGINAL_T5_INTERNAL_INSTABILITY_CONFIRMED`：第一污染点在 `y_train` 输入层，早于 `_fit_t5` / `_predict_t5`。
