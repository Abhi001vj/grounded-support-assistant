# ADR-0003: Local models are opt-in; the default fallback is model-free

Status: Accepted
Date: 2026-11-02
Deciders: project owner
Tags: ops

## Context

The grader runs the repo from a clean checkout without keys. BERT QA, sentence-transformers and
a cross-encoder need torch plus about 0.6 GB of downloads, which can stall or fail on a locked-down
machine before the pipeline even starts.

## Decision

The default fallback uses only scikit-learn and numpy: BM25 retrieval, a TF-IDF reranker and
extractive sentence selection. Local models load only with `USE_LOCAL_MODELS=1` from
`requirements-local-models.txt`. Every loader fails soft, back to the lexical path.

## Alternatives considered

- **Local models by default**: rejected because of install and download risk for graders.
- **No local models at all**: rejected. The owner wanted a BERT QA / sentence-transformer
  fallback that adds "no answer" detection and semantic retrieval without keys.

## Consequences

Positive: `pip install -r requirements.txt` is light, and CI needs no model cache.
Negative: the default fallback misses paraphrases that share no vocabulary with the KB.

## References

- `app/local_models.py`, `app/qa_fallback.py`, `requirements-local-models.txt`
