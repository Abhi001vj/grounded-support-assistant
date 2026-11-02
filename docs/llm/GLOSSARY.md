# Glossary

| Term | Meaning in this repo |
|---|---|
| **Chunk** | A sentence window of an article (≤ 3 sentences, 1 overlapping). ID `ART-101-C1`. |
| **Coverage** | idf-weighted share of the question's content terms that appear in the top chunk. Terms not in the KB get the maximum idf. |
| **Matched terms** | distinct stemmed question terms present in the top chunk. |
| **Relevance gate** | the deterministic check that refuses before any generator runs (`guardrails.relevance_gate`). |
| **Deterministic / fallback mode** | no LLM: BM25 retrieval + extractive answer. `used_fallback: true`. |
| **LLM mode** | at least one provider key set. An answer produced after an LLM failure still has `used_fallback: true`. |
| **Tier** | `small` (cheap, default) or `best` model per provider (`LLM_TIER`). |
| **Rephrase** | an LLM-written search query used only for retrieval. |
| **Citation report** | `kept / invented / not_retrieved / duplicates / unsupported` from `verify_citations`. |
| **Trace** | per-request step timings, LLM/embedding calls, tokens per step and estimated cost. |
| **Components** | the index, retriever and answerer as one immutable snapshot, swapped on re-index. |
