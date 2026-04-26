# T5 Source Parity Audit

> 文档性质  
> 本文只审计当前仓库 `src/research/t5_recovered_source.py` 与历史来源 `/tmp/qfs-sigma/repo/src/research/run_ndx_sigma_output_transform_pilot.py` 中原始 `T5_resid_persistence_M4` 的源码等价性。本文不是新架构文档，不修改 Phase 0A/0B 协议。

## 1. 文档目的

本文件回答：当前仓库中的原始 `T5_resid_persistence_M4` 是否已经与 `/tmp/qfs-sigma/repo` 历史来源严格对齐。

## 2. 审计输入

| 类型 | 路径 |
|---|---|
| 当前仓库实现 | `src/research/t5_recovered_source.py` |
| 历史来源 | `/tmp/qfs-sigma/repo/src/research/run_ndx_sigma_output_transform_pilot.py` |
| 审计 runner | `src/research/run_phase0a_t5_2008_failure_audit.py` |

## 3. 源码对齐修复

| 项目 | 修复前当前值 | 历史值 | 修复后当前值 | 状态 |
|---|---:|---:|---:|---|
| `LOGSIG_PAD` | `0.75` | `0.35` | `0.35` | PATCHED_TO_MATCH |

`LOGSIG_PAD` 影响 `_safe_clip_sigma()` 的下界 padding。该差异在 2008 第一处失败之前尚未被执行到，因此不解释 2008 的第一处 non-finite，但它会影响成功窗口的 sigma clipping 与最终指标。已先修正到历史值后再重跑四窗口。

## 4. 逐项等价性审计

| 对象 | 当前位置 | 历史位置 | 状态 | 证据 |
|---|---|---|---|---|
| `_fit_t5` | `src/research/t5_recovered_source.py` | `/tmp/qfs-sigma/repo/src/research/run_ndx_sigma_output_transform_pilot.py` | MATCH | AST-normalized function body match |
| `_predict_t5` | 同上 | 同上 | MATCH | AST-normalized function body match |
| `_safe_clip_sigma` | 同上 | 同上 | MATCH | AST-normalized function body match after `LOGSIG_PAD` patch |
| `_build_train_base` | 同上 | 同上 | SEMANTIC_MATCH | 416 周 rolling train、53 周 embargo、M4 sigma fit、`R1_TRAIN_WINDOW + 54` history frame token 均一致；当前仓库仅有 typing / ndarray normalization 差异 |
| `WINDOWS` | 当前含 2008 + 2017/2018/2020 | 历史仅含 2017/2018/2020 | SEMANTIC_MATCH | 当前为 delivery 需要新增 2008 benchmark window；三个 pilot window 与历史一致 |
| `R1_TRAIN_WINDOW` | imported from `app.panel_runner` | imported from `app.panel_runner` | MATCH | 均使用 `R1_TRAIN_WINDOW = 416` |
| embargo logic | `as_of - timedelta(weeks=53)` | `as_of - timedelta(weeks=53)` | MATCH | token match |
| output frequency / week alignment | `frame.feature_dates`, weekly Friday-close aligned | `frame.feature_dates`, weekly Friday-close aligned | MATCH | runner 输出与文档均固定为 weekly feature dates |
| `CORR_CLIP` | `1.0` | `1.0` | MATCH | literal match |
| `SIGMA_EPS` | `1.0e-4` | imported same value in historical chain | MATCH | literal/imported value consistent |
| `SIGMA_MAX_MULT` | `3.0` | imported same value in historical chain | MATCH | literal/imported value consistent |
| empirical standardized residual eps | `np.maximum(train_base.sigma_train, SIGMA_EPS)` | same | MATCH | formula match |

## 5. 当前源码层结论

`SOURCE_EQUIVALENCE_CONFIRMED_AFTER_PATCH`

理由：发现并修正了唯一明确常量差异 `LOGSIG_PAD = 0.75 -> 0.35`。修正后，T5 公式链、clip 规则、sigma safety clip、训练窗、embargo、输出频率均与历史来源匹配或语义匹配。

## 6. 证据边界

- 本文只证明当前 recovered T5 与历史源码口径对齐。
- 本文不证明 T5 在 2008 可运行。
- 本文不授权 Phase 0B 入场。
