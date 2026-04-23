SHELL := /bin/bash
.DEFAULT_GOAL := help

IMAGE_TAG ?= dev
IMAGE_NAME ?= qqq-law-engine
AS_OF ?= auto
START ?= 2001-02-02
END ?= auto
WINDOW ?= 312w
RUN_MODE ?= weekly
COMPOSE := docker compose
UV := uv

.PHONY: help
help: ## Show available targets.
	@awk 'BEGIN {FS = ":.*##"; printf "Usage: make <target>\n\nTargets:\n"} /^[a-zA-Z0-9_.-]+:.*##/ {printf "  %-22s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

.PHONY: init-env
init-env: ## Create .env from .env.example if it does not exist.
	@test -f .env || cp .env.example .env

.PHONY: sync
sync: ## Install runtime and dev dependencies from uv.lock.
	$(UV) sync --frozen --extra dev

.PHONY: lock
lock: ## Refresh uv.lock after an intentional pyproject.toml dependency change.
	$(UV) lock

.PHONY: build
build: ## Build the runtime Docker image.
	$(COMPOSE) build engine

.PHONY: build-test
build-test: ## Build the Docker test stage used by CI.
	docker build --target test -t $(IMAGE_NAME):test .

.PHONY: up
up: build ## Run the engine once with RUN_MODE and AS_OF.
	RUN_MODE=$(RUN_MODE) AS_OF=$(AS_OF) IMAGE_TAG=$(IMAGE_TAG) $(COMPOSE) up --abort-on-container-exit engine

.PHONY: weekly
weekly: build ## Run weekly inference for AS_OF=YYYY-MM-DD or auto.
	RUN_MODE=weekly AS_OF=$(AS_OF) IMAGE_TAG=$(IMAGE_TAG) $(COMPOSE) up --abort-on-container-exit engine

.PHONY: backtest
backtest: build ## Run the backtest command in the engine container.
	IMAGE_TAG=$(IMAGE_TAG) $(COMPOSE) run --rm engine backtest --start $(START) --end $(END)

.PHONY: panel-smoke
panel-smoke: build ## Run the one-year 2016 panel challenger smoke validation.
	IMAGE_TAG=$(IMAGE_TAG) $(COMPOSE) run --rm engine panel-smoke

.PHONY: panel-backtest
panel-backtest: build ## Run the full panel challenger backtest through 2024-12-27.
	IMAGE_TAG=$(IMAGE_TAG) $(COMPOSE) run --rm engine panel-backtest

.PHONY: train
train: build ## Train production artifacts with a deterministic window.
	IMAGE_TAG=$(IMAGE_TAG) $(COMPOSE) run --rm engine train --window $(WINDOW)

.PHONY: verify
verify: build ## Replay AS_OF and compare expected sha256 artifacts.
	IMAGE_TAG=$(IMAGE_TAG) $(COMPOSE) run --rm engine verify --as-of $(AS_OF)

.PHONY: sidecar
sidecar: build ## Run the research sidecar after engine completion.
	AS_OF=$(AS_OF) IMAGE_TAG=$(IMAGE_TAG) $(COMPOSE) --profile sidecar up --abort-on-container-exit engine research-sidecar

.PHONY: dev
dev: build ## Start the optional Jupyter profile.
	IMAGE_TAG=$(IMAGE_TAG) $(COMPOSE) --profile dev up jupyter

.PHONY: lint
lint: ## Run ruff lint checks.
	$(UV) run ruff check .

.PHONY: format
format: ## Format Python files with ruff.
	$(UV) run ruff format .

.PHONY: format-check
format-check: ## Check formatting without writing files.
	$(UV) run ruff format --check .

.PHONY: type
type: ## Run mypy in strict mode.
	$(UV) run mypy --strict src/ tests/

.PHONY: unit
unit: ## Run non-slow, non-integration, non-acceptance tests.
	$(UV) run pytest -q -m "not acceptance and not slow and not integration"

.PHONY: test
test: ## Run the default local test suite.
	$(UV) run pytest -q

.PHONY: purity
purity: ## Run FP purity, layer boundary, and anti-pattern checks.
	$(UV) run pytest -q tests/test_fp_purity.py tests/test_layer_boundaries.py tests/test_antipatterns.py
	PYTHONPATH=src $(UV) run lint-imports --config pyproject.toml

.PHONY: research-firewall
research-firewall: ## Run production/research isolation checks.
	$(UV) run pytest -q tests/test_research_isolation.py tests/test_output_schema_strict.py tests/test_research_output_isolation.py
	PYTHONPATH=src $(UV) run lint-imports --config pyproject.toml

.PHONY: acceptance
acceptance: ## Run SRD section 16 acceptance tests.
	$(UV) run pytest -q -m acceptance

.PHONY: weekly-twice-diff
weekly-twice-diff: ## Run determinism test that checks byte-identical weekly output.
	$(UV) run pytest -q tests/test_determinism.py

.PHONY: ci-local
ci-local: lint format-check type purity unit weekly-twice-diff ## Run the main local CI gates.

.PHONY: compose-ci
compose-ci: build ## Run the CI mock-fred compose path.
	$(COMPOSE) --profile ci up --abort-on-container-exit engine mock-fred

.PHONY: down
down: ## Stop compose services and remove anonymous volumes.
	$(COMPOSE) down -v

.PHONY: clean
clean: ## Remove local Python and test caches.
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage coverage.xml htmlcov
