# docs/llm: index

Load only the rows that match your task.

| Task | Read |
|---|---|
| Understand the brief and acceptance criteria | [../specs/problem-statement.md](../specs/problem-statement.md), [../specs/SPEC.md](../specs/SPEC.md) |
| Change retrieval, the gate, guardrails or the answerer | [PIPELINE.md](PIPELINE.md), [GLOSSARY.md](GLOSSARY.md) |
| Why lexical-first, one SDK, local models opt-in | [decisions/](decisions/) |
| Add a provider or model | [PIPELINE.md](PIPELINE.md) § LLM layer, [ADR-0002](decisions/ADR-0002-one-sdk-for-all-providers.md) |
| Calibrate thresholds | [PIPELINE.md](PIPELINE.md) § Invariants, `eval/extra_questions.json` |

## Decisions

| ADR | Decision |
|---|---|
| [ADR-0001](decisions/ADR-0001-lexical-first-relevance-gate.md) | A deterministic relevance gate runs before any generator |
| [ADR-0002](decisions/ADR-0002-one-sdk-for-all-providers.md) | One `openai` SDK client for OpenAI, Gemini, Anthropic and OpenRouter |
| [ADR-0003](decisions/ADR-0003-local-models-opt-in.md) | Local models (torch) are opt-in; the default fallback is model-free |
