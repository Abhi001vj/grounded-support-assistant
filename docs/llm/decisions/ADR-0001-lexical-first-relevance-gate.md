# ADR-0001: A deterministic relevance gate runs before any generator

Status: Accepted
Date: 2026-11-02
Deciders: project owner
Tags: agents, security

## Context

The brief's failure case (Q5, "CEO's direct email address") shares the word *email* with
ART-101, so similarity search always returns *something*. Refusal cannot depend only on an LLM,
because the no-key mode has none. And an LLM that is handed a nearly relevant context can still
produce plausible contact details.

## Decision

Run a deterministic gate after retrieval. It uses idf-weighted term coverage of the top chunk,
the number of matched terms, optional dense similarity, and a contact-detail guard. A failed
gate refuses without calling any generator. LLM mode uses a lenient version, so paraphrases
still reach the model.

## Alternatives considered

- **LLM-only refusal**: rejected. Unavailable without keys; costs a call per off-topic
  question; can be argued with by prompt injection.
- **A fixed similarity threshold on the retrieval score**: rejected. BM25 is normalised per
  query, so the top score is always 1.0 and carries no absolute meaning.
- **Hard-coded refusal topics (CEO, address…)**: rejected as overfitting to the sample.

## Consequences

Positive: Q5-style questions cost $0, the behaviour is unit-testable, and identical in both modes.
Negative: in fallback mode, paraphrases with no shared vocabulary are refused.
Accepted risks: thresholds were calibrated on 25 questions (5 sample + 20 held-out).

## Operational notes

A threshold change must keep `eval/extra_questions.json` at 20/20 (`python main.py --no-llm`).

## References

- `app/guardrails.py` (`relevance_gate`), `app/graph.py` (`_gate`)
- [../PIPELINE.md](../PIPELINE.md)
