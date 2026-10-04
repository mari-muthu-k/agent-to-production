SHELL := /bin/bash
COMPOSE := docker compose
RUN := $(COMPOSE) run --rm
# Reference notebooks live one level up in the course repo; mounted read-only for `make notebooks`.
REFERENCE_DIR ?= $(abspath ../reference)

.DEFAULT_GOAL := help
.PHONY: help env build up down test lint mock gateway notebooks exec-notebooks check-gateway lock logs clean

help:  ## List targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "} {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

env:
	@test -f .env || (cp .env.example .env && echo "created .env from .env.example (offline mode)")
	@mkdir -p logs data/papers

build: env  ## Build the runtime and dev images
	$(COMPOSE) build mock-llm tests

up: env  ## Start mock-llm (http://localhost:8100); notebooks themselves run in Google Colab
	$(COMPOSE) up -d --build --wait mock-llm

down:  ## Stop everything (keeps volumes)
	$(COMPOSE) --profile gateway --profile tools down

test: env  ## Run pytest against mock-llm (no API keys needed)
	$(RUN) tests

lint: env  ## ruff
	$(RUN) --no-deps tests ruff check .

mock: env  ## Start only mock-llm on http://localhost:8100
	$(COMPOSE) up -d --build --wait mock-llm

gateway: env  ## Start the LiteLLM gateway (rehearsal profile) on http://localhost:4000
	$(COMPOSE) --profile gateway up -d --wait gateway

notebooks: env  ## Regenerate notebooks (Day 1: byte-for-byte copies of the reference notebooks)
	@if [ -d "$(REFERENCE_DIR)" ]; then \
	  $(RUN) --no-deps -v "$(REFERENCE_DIR)":/reference:ro tests python tools/build_notebooks.py --reference /reference; \
	else \
	  $(RUN) --no-deps tests python tools/build_notebooks.py; \
	fi

exec-notebooks: env  ## Execute every notebook against mock-llm; fail on any unexpected error
	$(RUN) tests python tools/exec_notebooks.py

check-gateway: env  ## Probe the endpoint in .env (LLM_BASE_URL) for supported features
	$(RUN) --no-deps toolbox python tools/check_gateway.py $(ARGS)

lock:  ## Re-resolve requirements.lock / requirements-dev.lock (needs network)
	docker run --rm -v "$(CURDIR)":/w -w /w ghcr.io/astral-sh/uv:python3.11-trixie-slim sh -c '\
	  uv pip compile pyproject.toml --python-version 3.11 --universal -q -o requirements.lock && \
	  uv pip compile pyproject.toml --extra dev -c requirements.lock --python-version 3.11 --universal -q -o requirements-dev.lock'

logs:  ## Follow service logs
	$(COMPOSE) logs -f

clean:  ## Stop, remove containers, images and generated files
	$(COMPOSE) --profile gateway --profile tools down --rmi local --volumes --remove-orphans
	rm -rf build .pytest_cache .ruff_cache logs/*.log
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
