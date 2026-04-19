# syntax=docker/dockerfile:1.7
# QQQ Cycle-Conditional Law Engine — multi-stage Dockerfile
# Aligned with ADD §6.3. Uses uv for deterministic dependency resolution.

ARG PYTHON_VERSION=3.11.9
ARG UV_VERSION=0.4.18

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uvbin

# ---------------------------------------------------------------------------
# Stage 1: base (shared by deps / src / test)
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim-bookworm AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_NO_CACHE=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    TZ=America/New_York

# 系统依赖：构建 cvxpy/ecos 需要 gcc；最终 runtime 会丢弃它们
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential \
      git \
      ca-certificates \
      tzdata \
      curl \
 && rm -rf /var/lib/apt/lists/*

# 安装 uv（pin 版本）
COPY --from=uvbin /uv /usr/local/bin/uv

WORKDIR /app

# ---------------------------------------------------------------------------
# Stage 2: deps — only lockfile & project metadata, high cache hit-rate
# ---------------------------------------------------------------------------
FROM base AS deps

COPY pyproject.toml uv.lock ./
# --frozen 保证使用 lock 文件；若 lock 与 pyproject 冲突，构建失败
RUN uv sync --frozen --no-dev --no-editable

# ---------------------------------------------------------------------------
# Stage 3: src — copy source, re-sync with project
# ---------------------------------------------------------------------------
FROM deps AS src

COPY src/ ./src/
COPY configs/ ./configs/
COPY README.md ./
RUN uv sync --frozen --no-dev

# ---------------------------------------------------------------------------
# Stage 4: test — dev extras for CI
# ---------------------------------------------------------------------------
FROM src AS test

RUN uv sync --frozen --extra dev
COPY tests/ ./tests/
COPY tools/ ./tools/

# ---------------------------------------------------------------------------
# Stage 5: runtime — lean, non-root, read-only friendly
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PATH="/app/.venv/bin:$PATH" \
    TZ=America/New_York \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    SRD_VERSION=8.7

RUN apt-get update \
 && apt-get install -y --no-install-recommends tzdata ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && groupadd -g 1000 app \
 && useradd -u 1000 -g 1000 -m -s /bin/bash app

WORKDIR /app

COPY --from=src --chown=app:app /app/.venv /app/.venv
COPY --from=src --chown=app:app /app/src /app/src
COPY --from=src --chown=app:app /app/configs /app/configs
COPY --from=src --chown=app:app /app/pyproject.toml /app/pyproject.toml

# 数据与产物挂载点
RUN mkdir -p /app/data /app/artifacts \
 && chown -R app:app /app/data /app/artifacts

USER app

# 健康检查：验证 CLI 可启动（不拉数据）
HEALTHCHECK --interval=60s --timeout=10s --start-period=10s --retries=3 \
  CMD python -m app.cli --health || exit 1

ENTRYPOINT ["python", "-m", "app.cli"]
CMD ["weekly", "--help"]
