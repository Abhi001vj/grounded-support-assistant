> **Read when:** changing retrieval, the relevance gate, guardrails, generation or tracing
> **Describes:** `app/graph.py`, `app/retrievers.py`, `app/indexer.py`, `app/guardrails.py`, `app/qa_fallback.py`, `app/llm_utils.py`, `app/tracing.py`
> **Read next:** [GLOSSARY.md](GLOSSARY.md), [decisions/](decisions/)
> **Last verified:** 2026-11-02

# Answer pipeline

## What this is

A LangGraph state machine that turns a question into `{answer, citations,
insufficient_context}` using only the in-memory index of `kb/articles.json`. LLM mode and
deterministic mode share every node except `generate`.

## Key files

| File | Responsibility |
|---|---|
| `app/graph.py` (`SupportPipeline._build_graph`) | node wiring; `_timed` wraps every node with the timing middleware |
| `app/graph.py` (`Components`, `build_index`) | the index, retriever and answerer, swapped atomically on re-index |
| `app/retrievers.py` (`HybridRetriever.search`) | fusion (`w·bm25_norm + (1-w)·dense`), rerank, top 3 |
| `app/guardrails.py` (`relevance_gate`) | the refusal decision for off-topic questions |
| `app/guardrails.py` (`finalize`, `verify_citations`) | citation cleanup and refusal normalisation |
| `app/llm_utils.py` (`PROVIDERS`, `LLMRouter`) | tiers, parameter dropping, retries, provider chain |
| `app/tracing.py` | per-request `Trace` bound through a ContextVar; `DEFAULT_PRICES` |

## How it works

```text
input_guard → rephrase → retrieve → gate ─┬─ refuse ─────────────┐
     │ blocked                            └─ generate → output_guard ┴→ finalize
     └──────────────────────────────────────────────► refuse
```

- **retrieve:** BM25 over stemmed tokens of `title + chunk`, normalised by the best score for
  the query. Dense cosine is added in hybrid mode. Six candidates are kept, reranked by TF-IDF
  uni+bigram cosine (or a cross-encoder), and the top 3 returned.
- **gate:** looks at term coverage in the top chunk (see Invariants). LLM mode uses a lenient
  gate (any overlap, or dense ≥ 0.35) so the model can judge paraphrases.
- **generate:** `LLMRouter.answer`, then on exception `DeterministicAnswerer.answer`.
- **output_guard:** a failed regex check replaces an LLM answer with the extractive one.
- **finalize:** `verify_citations` keeps only KB ∩ retrieved, deduplicated, text-supported
  IDs. An answer left with none becomes a refusal.

## Invariants

- `answers.json` holds exactly the 4 required keys. Extra observability goes to the run log.
- A refusal always has `citations == []` and the fixed `REFUSAL` text.
- Citations ⊆ retrieved article IDs ⊆ KB IDs.
- Gate thresholds (`MIN_MATCHED_TERMS=2`, `MANY_MATCHED_TERMS=3`, `MIN_COVERAGE=0.4`,
  `STRONG_COVERAGE=0.75`, `DENSE_PASS=0.55`) were calibrated together on 25 questions.
  Change one, and re-run `python main.py --no-llm` and check robustness 20/20.
- The rephrased query is for retrieval only. Never pass it to the generator.
- Nodes read the index from `config["configurable"]["components"]`, not from `self`, so a
  concurrent `/index` never mixes two indexes within one request.

## Gotchas

- sklearn's stop-word list contains `once`, `only` and `first`. These are removed from the stop
  list (`_KEEP` in `data_utils.py`) because "exactly once" depends on them.
- The contact-request regex must not match "reset email". It matches `email address`,
  `direct email` and similar, not a bare `email`.
- Gemini's embeddings endpoint returns `index: null` and no usage block. Don't sort on
  `index`, and expect 0 embedding tokens in the trace.
- `gpt-5.5` rejects `temperature=0`. `LLMRouter._create` drops rejected parameters and
  remembers that per model.

## Related

- Tests: `tests/test_answers.py`, `tests/test_guardrails.py`, `tests/test_llm_path.py`, `tests/test_api.py`
- ADRs: 0001, 0002, 0003
