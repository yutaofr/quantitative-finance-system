# AGENTS.md — AI Coding Agent Rulebook (Single Source of Truth)

> **适用对象**: Claude, Gemini, Codex, Cursor, Aider, Windsurf, 以及任何基于大模型的 coding agent。
> **等价文件**: `CLAUDE.md`、`GEMINI.md` 是本文件的**符号链接**。三者共享同一套规则，修改时**只改 `AGENTS.md`**。
> **配套文档**: 必须先读 `SRD_v8_7_final.md`（规格契约） + `ADD.md`（架构设计）。
> **冲突优先级**: `SRD` > `ADD` > `AGENTS.md`。
> **本文件存在的唯一理由**: 让 coding agent 在无人监督下产出可通过 CI 的代码，并且不偏离 SRD。

---

## 0. TL;DR — 十条戒律（读完这一节就可以开工，其余章节是展开）

1. **只实现 SRD 已定义的内容**。新增数学/常量/模块 → 拒绝，除非 SRD 升级。
2. **纯函数核心**：`src/features/**`、`src/state/**`、`src/law/**`、`src/decision/**`、`src/backtest/**`、`src/inference/**` 内 **零副作用**——不写文件、不发网络、不调 `print/logging`、不读 `datetime.now` / `os.environ` / `random.*`。随机性通过 `rng: np.random.Generator` 参数注入。
3. **禁止跨层反向 import**：domain 层不得 import adapter 层、app 层；`src/research/**` 不得被任何非 research 文件 import。
4. **禁止数据造假**：不生成假行情数据、不跨时间插值、不 silently fallback `strict→pseudo`。若缺数据 → 走 SRD §10 的 `DEGRADED/BLOCKED` 流程。
5. **禁止造测试轮子**：不在 `tests/` 重写 HMM / quantile / bootstrap / scaling 的参照实现。property 测试只断言性质，集成测试通过 `src/app/cli.py` 触发。
6. **所有常量从 `configs/*.yaml` 读取**，值与 SRD §18 完全一致；硬编码数字即违规（除了 1、0、1e-8 这类泛用数值守卫）。
7. **`.env` 是唯一凭据入口**：`FRED_API_KEY` 等必须通过 `pydantic-settings` 从环境变量注入；禁止代码中出现任何 API key 字面量。
8. **确定性**：同 `as_of` + 同 `uv.lock` + 同 `artifacts/training/**` → 同 `production_output.json` 字节一致。任何修改若破坏确定性必须在 PR 中声明。
9. **MUST 必须可测**：凡是 Agent 声明为 `MUST` 的规则，PR 同时提交测试文件。不能被测试捕获的规则写成 `SHOULD`。
10. **遇到不确定性停下来问人**：如果 SRD/ADD 没有明确答复，**不要猜**——在 PR 描述中列出不确定点并标 `[NEEDS-HUMAN]`。

---

## 1. 工作流程（Agent 启动后按这个顺序做）

```
读任务 → 读 SRD 相关章节 → 读 ADD 相关章节 → 读既有代码 → 写/改测试 →
写/改实现 → uv run ruff check/format → uv run mypy --strict →
uv run pytest (changed modules) → docker compose up engine (smoke) →
提交 PR (带模板答题)
```

### 1.1 开工前必做的三步

1. **定位 SRD 章节**：把任务映射到 SRD 的 `§x.y`，在 PR 描述顶部写 `closes SRD §x.y`。若找不到归属，该任务超出 SRD 范围，拒绝。
2. **读 ADD 的对应层**：新增文件落在 `src/` 哪一层？查 `ADD §3.1` 五层表。若层不确定，先写一条 `deviation-note` 放到 PR。
3. **检查是否已有同功能模块**：`rg 'def <拟新建函数名>' src/`。找到即复用，别造轮子。

### 1.2 Agent 的禁止动作清单

| 动作                                                     | 为什么禁     | 替代                                                            |
| -------------------------------------------------------- | ------------ | --------------------------------------------------------------- |
| 在测试里重新实现一份 HMM / QR / bootstrap 做「参照」    | 见 §5「同源」 | 用 hypothesis 写 property 断言；golden snapshot 用真历史数据     |
| 在纯核里 `import requests` / `open(...)`                 | 破坏 FP 边界  | 把 I/O 放到 `src/data_contract/` 或 `src/app/`，再把数据注入纯核 |
| 用 `datetime.now()` 决定业务逻辑                         | 非确定性      | 从 CLI `--as-of` 参数传入，或从 `FrozenConfig` 读取              |
| 直接 pip install 新依赖                                  | 破坏 lock     | `uv add <pkg>` + `uv lock` + 提交 `uv.lock` 变更                 |
| 给 `production_output.json` 增加字段                     | 破坏契约      | 改 SRD §11 → 升级版本 → 同步 ADD → 同步 schema                    |
| 引入随机性却不传 `rng`                                   | 不可复现      | 加 `rng: np.random.Generator` 参数                               |
| mock 纯函数                                              | 掩盖缺陷      | 只 mock `src/data_contract/**` 中真正的 HTTP adapter             |
| 把 `research/` 模块的结果写进生产 JSON                    | SRD §17-#10  | 写到 `research_report.json`                                      |
| 静默忽略 `cvxpy` 失败/`EM` 不收敛                         | SRD §8/§7   | 走 §10 降级流程并设 `model_status`                               |
| 在 `main` 分支直接 commit                                 | CI 被绕过     | 走 PR                                                            |

---

## 2. 代码风格（强制，由 ruff/mypy 自动判）

### 2.1 Python 版本与基础

- Python **3.11+**（`match/case`、`Self`、`StrEnum`、`tomllib` 可用）。
- `from __future__ import annotations` 是每个 `.py` 文件的第一行（允许 `"""docstring"""` 在其前）。
- 类型注解 **100% 覆盖** public API。`mypy --strict` 必过。
- `Any` 仅在 adapter 层与第三方库交互时允许，domain 层 `Any` 禁用。

### 2.2 命名

| 对象            | 规则                                                    |
| --------------- | ------------------------------------------------------- |
| 模块/文件       | `snake_case.py`                                         |
| 类 / TypeAlias  | `PascalCase`                                            |
| 函数 / 变量     | `snake_case`                                            |
| 常量 / 枚举值   | `UPPER_SNAKE`                                           |
| 私有            | `_leading_underscore`                                   |
| TypeVar         | 单大写字母，或 `T_<语义>`                                |
| 配置 YAML 键    | `snake_case`，与 SRD 术语对齐                           |

### 2.3 Dataclass 与不可变性

- 跨模块数据对象：`@dataclass(frozen=True, slots=True)` 或 `pydantic.BaseModel(frozen=True)`。
- `list`/`dict` 形参在 domain 层用 `Sequence`/`Mapping` 协议类型，不用可变的 `list`/`dict`。
- 返回时**新建对象**，不要修改入参。

### 2.4 错误处理

- 定义异常基类 `src/errors.py::LawEngineError`，各子模块继承：`VintageUnavailableError`、`BlockedModeError` 等。
- **不要捕获 `Exception`**。要么针对具体异常，要么让它向上冒泡到 `app.cli` 的顶层处理器。
- `assert` 只用在纯核的内部性质检查；对外部输入用显式 `raise`。

### 2.5 Numpy / 浮点

- `np.seterr(all='raise')`（在 `src/__init__.py` 中设置）。
- 不要用 `float('inf')` / `np.inf` 参与运算；用显式 clip。
- 除零、`log(0)`、`sqrt(-x)` 必须在调用点前显式守卫。

### 2.6 禁用的 stdlib

在 domain 层禁用：`datetime.now`, `time.time`, `os.environ`, `random`, `print`, `logging`, `threading`, `multiprocessing`, `subprocess`, `pickle` 的 `load`（仅允许 `dump`，且只能在 app 层）。

---

## 3. 函数式编程规则（这是架构的骨架，违反即 CI 红）

### 3.1 函数签名约定

```python
def f(x: FrozenIn, *, rng: np.random.Generator | None = None) -> FrozenOut:
    """pure. Given X and rng, returns Y. No side effects."""
```

- 第一行 docstring 必须以 `pure` 或 `io: <哪种>` 开头。
- 所有可选/带副作用参数 **用关键字传递**（`*`, 关键字分隔符）。
- 一个函数只做一件事；超过 40 行 → 拆。

### 3.2 禁止的模式

| 反模式                                                 | 正确写法                                               |
| ------------------------------------------------------ | ------------------------------------------------------ |
| `class Calculator: def fit(self): self.x = ...`        | 用 `@dataclass(frozen=True)` 存 fitted 参数，fit 返回它 |
| 全局可变 `_CACHE: dict = {}`                            | `@functools.lru_cache` 在纯函数上；或显式 state 传递    |
| `def f(x, config=Config())`（可变默认参数）              | `def f(x, config: Config | None = None)`                |
| `if rng is None: rng = np.random.default_rng()` 散落多处 | 只在 `app/` 层构造一次 `rng`，向下传递                    |
| `def f(x): x[0] = 0; return x`（原地改）                | `def f(x): return np.concatenate([[0], x[1:]])`         |
| Partial State via closure in domain                    | 纯函数 + 显式参数                                       |

### 3.3 组合模式

- 首选小函数 + `functools.reduce` / pandas pipe / 显式 pipeline。
- 用 `typing.Protocol` 描述「可替换算子」接口（如 `QuantileSolver`），而不是继承。
- Pipeline 在 `src/inference/weekly.py` 中组合，保持**单向数据流**；没有回调。

### 3.4 副作用明信化

I/O 只允许出现在：

- `src/data_contract/**`（拉数、写缓存）
- `src/app/**`（读 `.env`、CLI、写 artifacts、日志）
- `src/research/sidecar.py`（研究 sidecar 入口）

任何其他文件看到副作用 → CI 的 `test_fp_purity.py` 会红。

---

## 4. 测试规则（严格，禁止偷懒）

### 4.1 四层测试

| 层              | 用途                     | 工具                         | 运行时机     |
| --------------- | ------------------------ | ---------------------------- | ------------ |
| L1 Unit         | 单个纯函数               | pytest                       | push         |
| L2 Contract     | JSON schema、vintage 表   | pytest + jsonschema          | push         |
| L3 Property     | 性质不变                 | pytest + hypothesis          | push         |
| L4 Integration  | 端到端（真金样本）        | pytest + docker compose       | push         |
| L5 Acceptance   | §16 十一项                | pytest -m acceptance          | nightly / tag |

### 4.2 同源约束（再强调一次）

**测试代码 MUST NOT 重新实现 `src/` 下已有的算法**。

- 想测 quantile 非交叉 → 调 `src.law.fit_linear_quantiles`，断言 `is_monotone_non_decreasing(curve)`。
- **不要**在 `tests/` 写 `def _reference_quantile_regression(...)`。
- **不要** mock `src.state.ti_hmm_single.infer_hmm`；如果它依赖 HTTP，问题在 adapter 边界，去修 adapter。

CI 会 `ast-grep` 测试目录中的 `def _reference_*` 与 `def _my_*` 模式，命中即红。

### 4.3 Golden Snapshot 规则

- 位置：`tests/golden/as_of=YYYY-MM-DD/`
- 输入：**一次真实的 ALFRED 响应**保存为 parquet（**不允许合成**）。
- 期望输出：`production_output.json` 字节级比对。
- 变更：PR 标题必须含 `[snapshot-update]`，且人类 reviewer 审过 JSON diff。

### 4.4 Property 测试清单（最小集）

每个纯核模块至少一条 property：

```python
# tests/test_scaling.py (例)
from hypothesis import given, strategies as st

@given(arr=arrays_strategy())
def test_soft_squash_clip_bounded(arr):
    out = soft_squash_clip(arr)
    assert (np.abs(out) <= 5.0 + 1e-12).all()

@given(arr=arrays_strategy(min_size=10))
def test_soft_squash_clip_monotone(arr):
    sorted_in = np.sort(arr)
    out = soft_squash_clip(sorted_in)
    assert np.all(np.diff(out) >= -1e-12)
```

### 4.5 覆盖率与同源率

- 行覆盖率 `pytest --cov=src` ≥ **85%**（domain 层 ≥ 95%）。
- 同源率（见 ADD §7.6）≥ **0.95**。

---

## 5. 数据与 PIT 规则

### 5.1 五条红线

1. **strict 模式下**请求 `as_of < earliest_strict_pit` → `raise VintageUnavailableError`。不允许 silent fallback 到 pseudo。
2. ALFRED 返回为空 ≠ 可以用 revised；空就是空，走 missing 流程。
3. `data/raw/alfred/**` **append-only**。同一 `(series_id, as_of)` 二次写入必须字节一致；不一致即 raise `CachePoisonError`。
4. 禁止跨时间插值（`.interpolate(...)` 禁用）。同一周内 Friday 缺失可用 Thursday，周与周之间不补。
5. **绝不**将研究专属 series (`NFCI`, `STLFSI4`, BEA...) 读入 `src/features/**`。它们的白名单/黑名单都在 `src/data_contract/vintage_registry.py::FORBIDDEN_IN_PROD`。

### 5.2 日期与日历

- 周频 anchor = **Friday 16:00 America/New_York**；如果 Friday 是假日，用最近的前一个交易日 close。
- 所有 `as_of` 参数在 CLI 级别被归一化到最近的有效周 anchor；domain 层信任它是有效日。

---

## 6. 配置与常量规则

### 6.1 常量来源顺序

```
CLI 参数 > 环境变量 (.env) > configs/*.yaml > ADD 附录 C 默认 > SRD §18 冻结值
```

SRD §18 是**最终权威**；yaml 只是把它搬到运行时。`tests/test_config_consistency.py` 会比对两者。

### 6.2 不能改的常量（违反即 CI 红）

SRD §18 全部条目。示例：

- `hard_clip_bound = 5`（§5.2）
- `quantile_gap = 1e-4`（§8.1）
- `l2_alpha = 2.0`（§8.1）
- `tail_mult = 0.6`（§8.2）
- `band = 7`（§9.5）
- `K = 3`（§7.1）
- 等等——见 ADD 附录 C。

**想改其中任何一个** → 先 PR 修改 SRD（需要维护者批准）→ 升级至 v8.8 → 再改代码。

### 6.3 可调参（`configs/backtest.yaml` 等）

仅以下几项**允许**通过 config 覆盖（Agent 自主改）：

- `random_seed`（但任一变更必须能在 PR 描述里解释动机）
- `n_parallel_workers`（性能，不影响输出）
- `log_level`
- 测试用的合成数据生成参数（仅在 `tests/` 内）

---

## 7. 容器与部署规则

### 7.1 Agent 不要做

- **不要**修改 `Dockerfile` 的基础镜像 digest，除非在 PR 里说明原因 + 通过 acceptance.yml。
- **不要**在 `engine` 容器里新增写路径（`read_only: true` 的约束神圣）。
- **不要**给容器开放新的入站端口。
- **不要**在 `docker-compose.yml` 里放 API key 字面量；全部走 `env_file: .env`。

### 7.2 运行命令速查

```bash
cp .env.example .env && $EDITOR .env       # 首次填 API key
make build                                 # 构建镜像
make weekly                                # 周推断（容器内）
make backtest                              # 全量回测
make verify AS_OF=2024-12-27               # 重放并比对 sha256
make sidecar                               # 研究 sidecar
make test                                  # 全部 L1-L4 测试（容器内）
make acceptance                            # §16 十一项
```

---

## 8. PR 规则

### 8.1 PR 标题约定

```
feat(law): implement linear_quantiles closes SRD §8.1
fix(data_contract): VintageUnavailableError not raised in strict [bug #42]
chore(ci): upgrade ruff to 0.6.x
test(state): add property for label_map determinism
docs(add): clarify research firewall
[snapshot-update] 2024-06-28 rerun after law/linear_quantiles patch
```

前缀集合：`feat | fix | chore | test | docs | refactor | perf | [snapshot-update]`。

### 8.2 PR 正文模板（`.github/pull_request_template.md`）

Agent 必须填完每一项；留空即 CI 红。

```markdown
## 对应 SRD / ADD 条目
- SRD: §__
- ADD: §__
- Deviation from ADD (若有)：<deviation-note:>

## 变更内容
- [ ] 新函数
- [ ] 新测试
- [ ] 常量变更（需引用 SRD §18）
- [ ] 依赖变更（uv.lock diff）
- [ ] 合约/JSON schema 变更（需说明 srd_version bump）

## 确定性影响
- [ ] 无（同输入同输出）
- [ ] 有（附 diff）

## 测试证据
- `uv run pytest -q` 截图或 log 段
- 若改动 FP 纯核：`test_fp_purity` 通过截图

## 清单
- [ ] mypy --strict 通过
- [ ] ruff check/format 通过
- [ ] 无新增硬编码 secret
- [ ] 无 reference_/mirror_ 测试实现
- [ ] research firewall 测试通过
- [ ] 如需 snapshot 更新，PR 标题包含 [snapshot-update]
```

### 8.3 Commit 消息

`Conventional Commits`，例如：

```
feat(law): implement linear_quantiles per SRD §8.1

- cvxpy joint solve for τ ∈ {0.10, 0.25, 0.50, 0.75, 0.90}
- L2 α=2.0 on b_τ and c_τ; intercepts unpenalized
- non-crossing constraint gap=1e-4
- Chernozhukov rearrangement fallback + solver_status flag

Closes: SRD §8.1
Refs: ADD §5.5
```

---

## 9. 与其他 AI 的协作

此 repo 会被 Claude、Gemini、Codex 等并发工作。为避免撞车：

1. **领取任务前检查 GitHub Issues / Project** 是否已被 `@claude` / `@gemini` 认领。未领则在 Issue 下评论 `picked by claude` 后再开始。
2. **小步提交**，每个 PR 聚焦一个 SRD 章节。避免「实现整个 §8」这种合集 PR。
3. **不要合并自己的 PR**——除非 CI 全绿且有另一 agent 或人类审过。
4. **见到疑似错误的别人的代码**——开 Issue，引用具体 SRD/ADD 条目，不要静默改写。

---

## 10. 特殊场景手册

### 10.1 SRD 与 ADD 打架怎么办

```
1. 停止编码
2. 在 PR 描述里贴一段 ADD-vs-SRD-conflict 表
3. 提交一个 docs-only PR 改 ADD；代码 PR 阻塞
4. 人类 reviewer 决定哪边改
```

### 10.2 测试通过但黄金样本变了

- 哪怕仅 0.001 精度差异，也要视为「输出合约变更」。
- PR 标题加 `[snapshot-update]`，附：(a) 哪段代码改动引起；(b) 数值差是否在 SRD §16 容许范围；(c) 为什么这是**更正确**而非「漂移」。

### 10.3 需要引入新第三方依赖

决策流程：

```
1. 能用 numpy/scipy/pandas 做到吗？→ 用它们
2. 能用 cvxpy + ECOS 做到吗？→ 用它们（已在 deps）
3. 确实需要新库吗？→ 选项：statsmodels, scikit-learn, pyarrow, pydantic-settings
   其他库默认拒绝
4. uv add <pkg> && uv lock && 提交 uv.lock
5. PR 描述里写：「为什么 scipy 做不到」
```

### 10.4 性能瓶颈

- 先 profile（`py-spy record`），别凭感觉优化。
- 纯核优化优先向量化（numpy），其次 numba（只读纯函数、标注 `@njit(cache=True)`）。
- 禁止引入 C 扩展或 Cython 除非 profile 证据 + 人类审批。

### 10.5 遇到 SRD 明显的数学错误

- **不要**在代码里「修正」。
- 开一条 Issue：`[SRD-bug] §x.y: <问题>`。
- 等待 SRD 升版。在此之前按字面实现。

---

## 11. 自检清单（每次 commit 前默念）

- [ ] 我读了 SRD 对应章节？
- [ ] 我读了 ADD 对应章节？
- [ ] 我没在 domain 层 import IO？
- [ ] 我没在测试里重写参照实现？
- [ ] 我没新增硬编码 secret？
- [ ] 我没引入非确定性？
- [ ] 我的新常量有 SRD § 来源？
- [ ] 我的 PR 描述填完模板？
- [ ] `uv run ruff check && uv run mypy --strict && uv run pytest -q` 全绿？
- [ ] 若碰到模糊处，我标了 `[NEEDS-HUMAN]` 而不是猜测？

若十一项都是「是」，提交。否则回头改。

---

## 12. 符号链接机制

本仓库有三个 Agent rulebook：

```
AGENTS.md          # ← 真源
CLAUDE.md -> AGENTS.md
GEMINI.md -> AGENTS.md
```

在支持 symlink 的系统（macOS/Linux + Git）上，这是字节级共享。

Windows 或不支持 symlink 的 Git 客户端：

1. 使用 `git config core.symlinks true` 配合 Developer Mode / 管理员模式；
2. 或退化为每次更新 `AGENTS.md` 后手工 `cp AGENTS.md CLAUDE.md && cp AGENTS.md GEMINI.md`（CI 会比对三者哈希，不一致即红）。

CI 中 `tools/check_agent_symlinks.py` 验证三个文件 `sha256` 一致（或均为指向同文件的 symlink）。

---

## 13. 元规则（本文件自身的规则）

- 本文件的**任何规则**必须能由 `tests/` 或 CI workflow 强制；不可执行的规则写成 `SHOULD`，不要装作 `MUST`。
- 本文件**不含数学**；数学全在 SRD。
- 本文件**不含架构决策**；架构在 ADD。
- 本文件只讲「Agent 如何工作」。若某条规则更该属于 SRD/ADD，迁过去。
- 本文件版本号绑定 SRD 大版本：SRD 从 v8.7 → v8.8 时，AGENTS.md 顶部的版本号一起升。

---

**版本**: AGENTS.md v1.0（与 SRD v8.7、ADD v1.0 对齐）
**生效**: 仓库初始化即生效
**修订触发**: SRD 升级、ADD 升级、CI 工具链替换、出现新类型 AI agent
