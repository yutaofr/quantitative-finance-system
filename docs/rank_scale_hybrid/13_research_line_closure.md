# Research Line Closure: Downstream T5 Patching Paradigm

> 本文档是所有 rank_scale_hybrid 子目录研究线的总结性归档。
> 写于 Option 2A 实验结果确认之后，不修改任何预注册判定。

## 1. 已关闭的研究线清单

| 研究线 | 判定文档 | 判定 | 主因 |
|---|---|---|---|
| Rank-Scale Hybrid (H0/H1) | `04_final_decision.md` | FAIL | T5 rank 与 EGARCH scale 无法在低自由度映射下共存 |
| Fixed T5 + downstream k + tail family (Option 1) | `10_option1_close.md` | FAIL | location bias；scalar k 不可吸收 regime-conditional 均值偏差 |
| Joint location-scale correction (Option 2A) | `12_option2a_decision.md` | FAIL | AR(1) location correction 近边界（c ≈ 0.94）仍无法消除 cal 误差；同时破坏 corr_next |

## 2. 两条独立失败链

**链一：固定 T5 σ_t → 下游尺度/形状修补**

```
T5 σ_t (scale bias ≈ 2×)
    → downstream scalar k (MLE per-window): SUCCESS（尺度稳定）
    → global k_fixed + tail family: FAIL（quantile miscalibration，位置偏差）
    → Option 2A joint (c, k_new): FAIL（c → 边界，corr_next 反转，cal 未收敛）
```

**链二：T5 rank + EGARCH scale → 低自由度 hybrid σ_t**

```
T5 ordinal signal + EGARCH-Normal cardinal signal
    → H0 无参 quantile map: FAIL（2017 方向反转，2018 scale 失败）
    → H1 alpha=0.50 保序压缩: FAIL（同上）
```

两链独立：H0/H1 从未依赖 downstream k；Option 2A 从未依赖 hybrid mapping。
两链共享同一方法论前提：**固定 T5 输出后做下游修补。**
两链均在该前提下失败。

## 3. 结论的法律边界

**已被证伪的命题：**

- 用无参同分位映射（H0）将 T5 rank 与 EGARCH scale 拼接为 σ_t 可行。
- 用 alpha=0.50 保序压缩映射（H1）实现该拼接可行。
- 全局 k_fixed = 2.683 配合固定 ν Student-t tail 给出校准分位数（Option 1）。
- 一步残差持久性 location correction z_corr = z − c·z_{t-1} 可消除 z_t 的 regime-conditional 均值偏差（Option 2A）。

**未被证伪的命题（不得外推）：**

- 所有可能的低自由度单调 hybrid family 均失败。（当前只测了 H0/H1 两个实例。）
- 更广义的 AR(p) 或多步 location correction 必然失败。（未测试。）
- 对 T5 σ_t 的任何下游修补均无效。（只测了 scalar k、(c, k) 两种。）

这两组命题的区分是认识论要求，不是为未来继续研究留口子。
终止决策基于**成本-信息比**，不基于穷举证明。

## 4. 诊断信号汇总

Option 2A 的两个结构性诊断值得记录：

1. **c → 边界（0.93–0.95）**：优化器在 AR(1) 框架内把持久性系数推到允许范围边缘，表明 AR(1) 这个函数族本身不足以表达数据中的位置依赖结构。

2. **corr_next 反转（2018: −0.18, 2020: −0.28）**：强 AR(1) 修正在吸收均值偏差的同时破坏了 T5 原有的方向性信号。这是一个方法论约束，不是参数调优问题。

Rank-Scale Hybrid 的结构性诊断：

3. **2017 方向反转**：T5 rank signal 在 bull regime 下与 EGARCH scale signal 在分位数空间内方向相反，无参映射强制拼接导致 corr_next < 0。
4. **2018 scale 失败**：EGARCH 在趋势下跌窗口的尺度校准与 T5 rank 在同一窗口产生系统性不相容。

以上四点合并为一个更高层次的诊断：**T5 和 EGARCH-Normal 在不同 regime 下各有局部优势，但这些局部优势在低自由度固定映射下无法被同时继承。**

## 5. 继续研究的条件（若有）

任何未来研究线必须同时满足以下两个条件，才不与当前关闭决定冲突：

**条件 A**：不依赖固定 T5 σ_t 输出做下游修补。
即：不以 T5 的 z_t 序列或 σ_t 序列作为输入，直接接受并尝试校正。

**条件 B**：不是当前失败实验的小变体。
即：不是新的 c 约束范围、新的 alpha 值、新的 ν_fixed，而是有独立理论动机的新假说。

满足条件 A 和 B 的例子（仅举示，不授权）：
- 从头 joint `(μ_t, σ_t, shape_t)` 条件密度估计，不继承 T5 任何组件
- 直接 conditional density 模型，基于完全不同的特征集

不满足条件 A 或 B 的例子（禁止继续）：
- AR(2) location correction
- H2: alpha = 0.30 或 alpha = 0.70
- 新的 ν_fixed 值配合相同 joint (c, k_new) 框架
- 任何形式的 trigger / crisis architecture 恢复

## 6. 最终档案状态

所有相关研究文件位于 `docs/rank_scale_hybrid/`，实验产物位于 `artifacts/research/`。

本研究线已正式关闭。Phase 0B 已于更早时间因独立原因关闭，此处关闭不改变那个状态，也不为其增加或减少任何合法性。

**归档结论：在当前信息集和当前模型类下，"固定 T5 输出 + 低自由度下游修补" 这一范式不可达到三窗口校准通过的预注册门槛。**
