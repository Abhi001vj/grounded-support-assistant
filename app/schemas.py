"""Structured records shared by the pipeline, the artifacts and the API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Article(BaseModel):
    id: str
    title: str
    product: str
    content: str


class Question(BaseModel):
    id: str
    question: str


class Chunk(BaseModel):
    """One retrieval record. This is exactly the shape written to index_preview.json."""

    chunk_id: str
    article_id: str
    title: str
    product: str
    text: str


class RetrievedChunk(BaseModel):
    chunk_id: str
    article_id: str
    score: float


class ScoredChunk(BaseModel):
    """A retrieved chunk plus the signals used to rank and gate it (internal)."""

    chunk: Chunk
    score: float  # final ranking score (after rerank), reported in retrieval_results.json
    lexical_score: float = 0.0  # BM25, normalised to [0, 1] within the query
    dense_score: float | None = None  # cosine similarity, when a dense embedder is used
    rerank_score: float | None = None

    def to_retrieved(self) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=self.chunk.chunk_id, article_id=self.chunk.article_id, score=round(self.score, 4)
        )


class RetrievalResult(BaseModel):
    question_id: str
    question: str
    retrieved_chunks: list[RetrievedChunk]


class Answer(BaseModel):
    question_id: str
    answer: str
    citations: list[str] = Field(default_factory=list)
    insufficient_context: bool


class RunLogRecord(BaseModel):
    question_id: str
    retrieval_chunk_ids: list[str]
    used_fallback: bool
    timestamp: str
    # Extra observability fields beyond the required four.
    provider: str | None = None
    retrieval_mode: str | None = None
    gate_passed: bool | None = None
    latency_ms: int | None = None
    errors: list[str] = Field(default_factory=list)
    # Tracing: per-step wall time and LLM/embedding usage for this question.
    step_timings_ms: dict[str, float] = Field(default_factory=dict)
    llm_calls: int = 0
    embedding_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    est_cost_usd: float | None = None
    tokens_by_step: dict[str, dict] = Field(default_factory=dict)
    events: list[dict] = Field(default_factory=list)


class GeneratedAnswer(BaseModel):
    """What a generator (LLM or deterministic) returns before guardrails run."""

    answer: str
    citations: list[str] = Field(default_factory=list)
    insufficient_context: bool = False
