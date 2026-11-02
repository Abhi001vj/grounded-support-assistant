# ADR-0002: One `openai` SDK client for OpenAI, Gemini, Anthropic and OpenRouter

Status: Accepted
Date: 2026-11-02
Deciders: project owner
Tags: orchestration

## Context

The owner asked for Claude, OpenAI, Gemini and OpenRouter keys to be supported. Four native
SDKs would mean four request/response shapes, four retry semantics and four sets of error
types, all in a 60-minute-scope project.

## Decision

Call all four through the `openai` SDK using their OpenAI-compatible endpoints. Providers differ
only in key, `base_url`, model tier and default parameters. Parameters a model rejects (HTTP 400
naming them) are dropped and remembered per model.

## Alternatives considered

- **Native SDKs (anthropic, google-genai)**: rejected for now. More code paths, and the compat
  layers cover chat + JSON, which is all this pipeline needs.
- **LangChain chat-model wrappers per provider**: rejected. Four extra packages for the same
  outcome. LangChain is used where it adds structure: prompts, `Document`, `BaseRetriever`,
  `InMemoryVectorStore`.

## Consequences

Positive: one retry/backoff path, one usage format for cost tracing, and new models need only a
string change.
Negative: compat layers lag native features (Anthropic ignores `response_format`, so JSON is
parsed leniently there; Gemini embeddings return no usage).
Accepted risks: the Anthropic and OpenRouter paths were not exercised live (no keys).

## References

- `app/llm_utils.py` (`PROVIDERS`, `LLMRouter._create`)
