"""Retrieval and reranking, done in code (never by the LLM).

score(chunk) = w * bm25_norm + (1 - w) * dense_cosine      (hybrid)
score(chunk) = bm25_norm                                    (lexical)

bm25_norm is BM25 divided by the best BM25 for that query, so it lies in [0, 1]. With
several queries (the original plus an LLM rephrase), each chunk keeps its best score.
Reranking then reorders the candidates:
  - cross-encoder (USE_LOCAL_MODELS=1), or
  - a model-free lexical reranker: mean of the fused score and TF-IDF (uni+bigram) cosine.
"""

from __future__ import annotations

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from app.data_utils import content_terms
from app.indexer import Index, index_text
from app.schemas import ScoredChunk


class HybridRetriever(BaseRetriever):
    """A LangChain retriever over the in-memory Index. search() returns scored chunks."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    index: Index
    top_k: int = 3
    candidate_k: int = 6
    lexical_weight: float = 0.5
    cross_encoder: object | None = None

    def _score_query(self, query: str, dense: dict[str, float]) -> dict[str, ScoredChunk]:
        bm25 = self.index.bm25.scores(content_terms(query))
        best = float(bm25.max()) if len(bm25) else 0.0
        out: dict[str, ScoredChunk] = {}
        for i, chunk in enumerate(self.index.chunks):
            lex = float(bm25[i] / best) if best > 0 else 0.0
            d = dense.get(chunk.chunk_id)
            score = lex if d is None else self.lexical_weight * lex + (1 - self.lexical_weight) * d
            out[chunk.chunk_id] = ScoredChunk(chunk=chunk, score=score, lexical_score=lex, dense_score=d)
        return out

    def candidates(self, queries: list[str]) -> list[ScoredChunk]:
        merged: dict[str, ScoredChunk] = {}
        dense_per_query = self.index.dense_scores_many(queries)
        for q, dense in zip(queries, dense_per_query):
            for cid, sc in self._score_query(q, dense).items():
                if cid not in merged or sc.score > merged[cid].score:
                    merged[cid] = sc
        ranked = sorted(merged.values(), key=lambda s: (-s.score, s.chunk.chunk_id))
        return ranked[: self.candidate_k]

    def rerank(self, question: str, candidates: list[ScoredChunk]) -> list[ScoredChunk]:
        if not candidates:
            return []
        texts = [index_text(c.chunk) for c in candidates]
        if self.cross_encoder is not None:
            import math

            logits = self.cross_encoder.predict([(question, t) for t in texts])
            rerank = [1 / (1 + math.exp(-float(x))) for x in logits]  # sigmoid → (0, 1)
            for sc, r in zip(candidates, rerank):
                sc.rerank_score = r
                sc.score = 0.5 * sc.score + 0.5 * r
        else:
            tfidf = self.index.tfidf_similarity(question, texts)
            for sc, r in zip(candidates, tfidf):
                sc.rerank_score = float(r)
                sc.score = 0.5 * sc.score + 0.5 * float(r)
        return sorted(candidates, key=lambda s: (-s.score, s.chunk.chunk_id))

    def search(self, question: str, extra_queries: list[str] | None = None) -> list[ScoredChunk]:
        queries = [question] + [q for q in (extra_queries or []) if q and q != question]
        return self.rerank(question, self.candidates(queries))[: self.top_k]

    # LangChain interface: lets the retriever be used in any LCEL chain.
    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        return [
            Document(
                page_content=s.chunk.text,
                metadata={**s.chunk.model_dump(exclude={"text"}), "score": s.score},
            )
            for s in self.search(query)
        ]
