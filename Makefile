# Grounded Support Assistant: common commands. Run `make help` for the list.
PYTHON ?= .venv/bin/python
PORT ?= 8000
USERS ?= 50
DURATION ?= 60s
IMAGE ?= grounded-support-assistant

.PHONY: help setup setup-local-models setup-dev run run-offline run-llm validate eval test test-live test-local all serve serve-offline loadtest benchmark charts docker-build docker-run clean

help: ## list targets
	@grep -E "^[a-zA-Z_-]+:.*## " $(MAKEFILE_LIST) | awk -F ":.*## " '{printf "  %-20s %s\n", $$1, $$2}'

setup: ## create .venv and install core requirements (no torch)
	python3 -m venv .venv && $(PYTHON) -m pip install -q --upgrade pip && $(PYTHON) -m pip install -q -r requirements.txt

setup-local-models: setup ## + torch, transformers, sentence-transformers (optional)
	$(PYTHON) -m pip install -q -r requirements-local-models.txt

setup-dev: setup ## + locust and matplotlib (load test, charts)
	$(PYTHON) -m pip install -q -r requirements-dev.txt

run: ## full pipeline on the sample data → artifacts/ (LLM mode if a key is set)
	$(PYTHON) main.py

run-offline: ## full pipeline in deterministic fallback mode (ignores keys)
	$(PYTHON) main.py --no-llm

run-llm: ## LLM run into artifacts/llm_run/ (needs a key; small paid cost)
	$(PYTHON) main.py --artifacts artifacts/llm_run

validate: ## check inputs and generated artifacts
	$(PYTHON) validate.py

eval: ## score artifacts + robustness set
	$(PYTHON) evaluate.py --robustness

test: ## offline unit + API tests (no keys, no network)
	env -u OPENAI_API_KEY -u GEMINI_API_KEY -u ANTHROPIC_API_KEY -u OPENROUTER_API_KEY $(PYTHON) -m pytest -q

test-live: ## live LLM smoke test (uses your keys)
	$(PYTHON) -m pytest -q -m live

test-local: ## optional local-model test (BERT QA, sentence-transformers)
	USE_LOCAL_MODELS=1 $(PYTHON) -m pytest -q tests/test_local_models.py

all: run-offline validate test ## what a grader runs: deterministic pipeline, validation, tests

serve: ## API on http://127.0.0.1:$(PORT) (/docs, /redoc, /scalar)
	$(PYTHON) main.py --serve --port $(PORT)

serve-offline: ## API in deterministic mode (free to load test)
	DISABLE_LLM=1 $(PYTHON) main.py --serve --port $(PORT)

loadtest: ## headless Locust against a running server (make serve-offline first)
	mkdir -p artifacts/loadtest
	$(PYTHON) -m locust -f locustfile.py --host http://127.0.0.1:$(PORT) --headless -u $(USERS) -r 10 -t $(DURATION) --csv artifacts/loadtest/run --csv-full-history --only-summary

benchmark: ## per-mode step latency + tokens (LLM modes use your keys)
	$(PYTHON) scripts/benchmark.py

charts: ## render docs/images/*.png from benchmark + load test
	$(PYTHON) scripts/plot_charts.py

docker-build: ## build the core image
	docker build -t $(IMAGE) .

docker-run: ## run the image on :8000 (add --env-file .env for LLM mode)
	docker run --rm -p 8000:8000 $(IMAGE)

clean: ## remove caches
	rm -rf .pytest_cache __pycache__ */__pycache__
