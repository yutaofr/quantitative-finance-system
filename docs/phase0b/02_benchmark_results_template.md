# Benchmark 结果模板文档

> 文档性质  
> 本文是 benchmark 结果模板，不是结果本身。  
> 本文只提供结果填报骨架，禁止预填数字，禁止预设结论。

## 1. 文档目的

本文件用于承接 `01_benchmark_lock.md` 锁定后的真实 benchmark 跑批结果，并服务于 Phase 0A 的分支判定。

本文必须回答：

1. T5 的失败是否为现架构特有缺陷。
2. 是否存在连续危机波动模型的共同上限。
3. T5 是否已经足够。
4. T5 是否虽然通过，但被更简单 benchmark 材料性支配。

## 2. Benchmark 运行前锁定摘要

- benchmark 锁定文件：`docs/phase0b/01_benchmark_lock.md`
- candidate（Phase 0A）：`Stage A = T5`
- benchmark family：UNFILLED
- 阶数锁定：UNFILLED
- 分布假设锁定：UNFILLED
- 滚动方式锁定：UNFILLED

若以上任一项未冻结，则本结果模板不得转入正式结果文档。

## 3. 三窗口结果总表

| 模型 | 窗口 | mean(z) | std(z) | corr_next | rank_next | lag1_acf(z) | sigma_blowup | pathology | CRPS | 状态 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| T5 | 2017-07-07 -> 2017-12-29 | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED |
| T5 | 2018-07-06 -> 2018-12-28 | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED |
| T5 | 2020-01-03 -> 2020-06-26 | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED |
| Benchmark_1 | 2017-07-07 -> 2017-12-29 | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED |
| Benchmark_1 | 2018-07-06 -> 2018-12-28 | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED |
| Benchmark_1 | 2020-01-03 -> 2020-06-26 | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED |
| Benchmark_2 | 2017-07-07 -> 2017-12-29 | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED |
| Benchmark_2 | 2018-07-06 -> 2018-12-28 | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED |
| Benchmark_2 | 2020-01-03 -> 2020-06-26 | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED |

## 4. 单模型明细表

### 4.1 T5 明细

| 字段 | 2017 | 2018 | 2020 |
|---|---:|---:|---:|
| mean(z) | UNFILLED | UNFILLED | UNFILLED |
| std(z) | UNFILLED | UNFILLED | UNFILLED |
| corr_next | UNFILLED | UNFILLED | UNFILLED |
| rank_next | UNFILLED | UNFILLED | UNFILLED |
| lag1_acf(z) | UNFILLED | UNFILLED | UNFILLED |
| sigma_blowup | UNFILLED | UNFILLED | UNFILLED |
| pathology | UNFILLED | UNFILLED | UNFILLED |
| CRPS | UNFILLED | UNFILLED | UNFILLED |
| Phase 0A 判定（A/D/E/F） | UNFILLED | UNFILLED | UNFILLED |

### 4.2 Benchmark_1 明细

| 字段 | 2017 | 2018 | 2020 |
|---|---:|---:|---:|
| mean(z) | UNFILLED | UNFILLED | UNFILLED |
| std(z) | UNFILLED | UNFILLED | UNFILLED |
| corr_next | UNFILLED | UNFILLED | UNFILLED |
| rank_next | UNFILLED | UNFILLED | UNFILLED |
| lag1_acf(z) | UNFILLED | UNFILLED | UNFILLED |
| sigma_blowup | UNFILLED | UNFILLED | UNFILLED |
| pathology | UNFILLED | UNFILLED | UNFILLED |
| CRPS | UNFILLED | UNFILLED | UNFILLED |
| benchmark 自身绝对表现判定 | UNFILLED | UNFILLED | UNFILLED |

### 4.3 Benchmark_2 明细

| 字段 | 2017 | 2018 | 2020 |
|---|---:|---:|---:|
| mean(z) | UNFILLED | UNFILLED | UNFILLED |
| std(z) | UNFILLED | UNFILLED | UNFILLED |
| corr_next | UNFILLED | UNFILLED | UNFILLED |
| rank_next | UNFILLED | UNFILLED | UNFILLED |
| lag1_acf(z) | UNFILLED | UNFILLED | UNFILLED |
| sigma_blowup | UNFILLED | UNFILLED | UNFILLED |
| pathology | UNFILLED | UNFILLED | UNFILLED |
| CRPS | UNFILLED | UNFILLED | UNFILLED |
| benchmark 自身绝对表现判定 | UNFILLED | UNFILLED | UNFILLED |

## 5. T5 vs Benchmark 对照表

> 说明  
> 本表用于承接 Phase 0A 分支判定，不直接下结论。

| 比较对象 | 2020 std(z) | 2020 是否优于 benchmark 至少 0.05 且 >=2% | 2008 是否超过 Benchmark_2008_std(z)+0.10 | 安全/方向性是否不劣于 T5 | 是否形成材料性支配 | 备注 |
|---|---:|---|---|---|---|---|
| T5 vs Benchmark_1 | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED |
| T5 vs Benchmark_2 | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED | UNFILLED |

## 6. Phase 0A 分支判定占位区

> 这里只能填最终分支，不得写解释性散文。

| 分支 | 是否触发 | 依据 | 状态 |
|---|---|---|---|
| 分支一：T5 特有缺陷 | UNFILLED | UNFILLED | UNFILLED |
| 分支二：连续危机波动共同上限 | UNFILLED | UNFILLED | UNFILLED |
| 分支三：危机样本异质，不支持统一危机层 | UNFILLED | UNFILLED | UNFILLED |
| 分支四：T5 Success 但被 benchmark 材料性支配 | UNFILLED | UNFILLED | UNFILLED |
| 分支五：T5 足够，无需继续修补 | UNFILLED | UNFILLED | UNFILLED |

## 7. 当前结论占位区

### 7.1 当前是否允许进入 Phase 0B

`UNFILLED`

### 7.2 当前证据最支持的结论

`UNFILLED`

### 7.3 当前仍不能说的结论

`UNFILLED`

## 8. 未填写项与禁止解释说明

### 未填写项

所有 `UNFILLED` 项均表示：

- benchmark 尚未运行，或
- 结果尚未合法填入，或
- 分支判定尚未完成

### 禁止解释说明

1. 不得在 `UNFILLED` 状态下提前宣布分支。  
2. 不得用单一窗口的好看数字替代协议判定。  
3. 不得在 benchmark 未冻结前填结果。  
4. 不得在结果出来后反向修改 `01_benchmark_lock.md`。  
