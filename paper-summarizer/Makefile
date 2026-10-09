SHELL := /bin/bash
COMPOSE := docker compose
RUN := $(COMPOSE) run --rm

.DEFAULT_GOAL := help
.PHONY: help env build up down sync-day1 test test-gateway lint mock gateway api keys eval eval-baseline notebooks exec-notebooks check-gateway lock logs clean

help:  ## List targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "} {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

env:
	@test -f .env || (cp .env.example .env && echo "created .env from .env.example (offline mode)")
	@mkdir -p logs data/papers

build: env  ## Build the runtime and dev images
	$(COMPOSE) build mock-llm tests

STACK := mock-llm postgres redis gateway api
up: env  ## Start the Day 4 stack: mock-llm, gateway (LiteLLM proxy), postgres, redis, api (http://localhost:8200)
	$(COMPOSE) up -d --build --wait $(STACK)

api: env  ## Start the api (and the gateway it needs); GATEWAY_MODE=proxy
	$(COMPOSE) up -d --build --wait api

down:  ## Stop everything (keeps volumes)
	$(COMPOSE) --profile tools --profile ui down

test: env  ## Run pytest against mock-llm (no API keys needed)
	$(RUN) tests

test-gateway: env  ## Proxy-mode tests: gateway + postgres + redis + api against mock-llm
	$(COMPOSE) up -d --build --wait $(STACK)
	$(RUN) --no-deps tests pytest -q -m gateway tests/test_day4_gateway.py

keys: env  ## Create demo virtual keys (alice, bob, eval-bot) on the gateway -> keys.local.json (gitignored)
	$(COMPOSE) up -d --wait gateway
	$(RUN) --no-deps tests python tools/make_keys.py

PROMPT_VERSION ?= v1
eval: env  ## Run the 20-question golden set against the api; exit 1 on a regression vs evals/baseline.json
	$(COMPOSE) up -d --build --wait $(STACK)
	$(RUN) --no-deps -e PROMPT_VERSION=$(PROMPT_VERSION) tests python -m paper_agent.evals.runner

eval-baseline: env  ## Run the golden set and write paper_agent/evals/baseline.json
	$(COMPOSE) up -d --build --wait $(STACK)
	$(RUN) --no-deps -e PROMPT_VERSION=$(PROMPT_VERSION) tests python -m paper_agent.evals.runner --write-baseline

lint: env  ## ruff
	$(RUN) --no-deps tests ruff check .

mock: env  ## Start only mock-llm on http://localhost:8100
	$(COMPOSE) up -d --build --wait mock-llm

gateway: env  ## Start the LiteLLM gateway (+ postgres, redis) on http://localhost:4000
	$(COMPOSE) up -d --wait gateway

DAYS ?= 3
notebooks: env  ## Regenerate notebooks from templates: make notebooks DAYS=3 (Day 1 is edited by hand, never touched)
	$(RUN) --no-deps tests python tools/build_notebooks.py $(DAYS)

sync-day1: env  ## Copy paper_agent/llm_client.py into the Day 1 Practice notebook's llm_client cells
	$(RUN) --no-deps tests python tools/sync_day1_llm_client.py

exec-notebooks: env  ## Execute every notebook against mock-llm (Day 4 demo: the full stack); fail on any unexpected error
	$(COMPOSE) up -d --build --wait $(STACK)
	$(RUN) --no-deps tests python tools/exec_notebooks.py $(NB)

check-gateway: env  ## Probe the endpoint in .env (LLM_BASE_URL) for supported features
	$(RUN) --no-deps toolbox python tools/check_gateway.py $(ARGS)

lock:  ## Re-resolve requirements.lock / requirements-dev.lock (needs network)
	docker run --rm -v "$(CURDIR)":/w -w /w ghcr.io/astral-sh/uv:python3.11-trixie-slim sh -c '\
	  uv pip compile pyproject.toml --extra mock --extra agent --extra rag --extra serve --python-version 3.11 --universal -q -o requirements.lock && \
	  uv pip compile pyproject.toml --extra dev -c requirements.lock --python-version 3.11 --universal -q -o requirements-dev.lock'

logs:  ## Follow service logs
	$(COMPOSE) logs -f

clean:  ## Stop, remove containers, images and generated files
	$(COMPOSE) --profile tools --profile ui down --rmi local --volumes --remove-orphans
	rm -rf build .pytest_cache .ruff_cache logs/*.log
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
