# AGENTS.md

Rules for any coding agent working in this repo. Read this, then
[`docs/llm/INDEX.md`](docs/llm/INDEX.md), then only the pages its routing table names.

## Golden rules

1. **No secrets in git.** Keys live in `.env` (git-ignored). Never print, log or commit them.
   The repo is public.
2. **The no-key path must always work.** `python main.py --no-llm && python validate.py &&
   python -m pytest -q` must pass with no network. CI enforces this.
3. **Ask before spending.** LLM runs (`make run-llm`, `make benchmark`, `make test-live`) are
   billed. Load tests go against `make serve-offline`, never an LLM-mode server.
4. **Grounding is non-negotiable.** Any change to prompts, the gate or the guards must keep
   `eval/extra_questions.json` at 20/20 in deterministic mode, and report the LLM-mode result
   if keys are available.
5. **Required artifact shapes are fixed by the brief** (`docs/specs/problem-statement.md`).
   Extra fields go only in `run_log.jsonl` / `run_info.json`, never in `answers.json`.
6. Style: `black` (120) and `flake8`. Run `pre-commit run -a` before committing.

## Code map

| Path | Role |
|---|---|
| `main.py` | entry point: pipeline → artifacts, `--serve` for the API |
| `app/graph.py` | LangGraph pipeline, index lifecycle, async batch |
| `app/retrievers.py`, `app/indexer.py` | BM25 / TF-IDF / vector store, fusion, rerank |
| `app/guardrails.py` | input and output guards, relevance gate, citation verification |
| `app/llm_utils.py` | providers, tiers, prompts, retries |
| `app/qa_fallback.py`, `app/local_models.py` | deterministic answerer, optional local models |
| `app/tracing.py` | trace, cost table, metrics |
| `evaluate.py`, `validate.py` | evaluation, artifact validation |
