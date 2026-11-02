"""In-memory index: BM25 (lexical), TF-IDF (sentence scoring) and an optional vector store.

Nothing is persisted. Rebuilding is cheap at this size, so /index simply builds a new Index.
"""

from __future__ import annotations

import logging
import math
from collections import Counter

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from sklearn.feature_extraction.text import TfidfVectorizer

from app.data_utils import content_terms
from app.schemas import Chunk

log = logging.getLogger(__name__)


def index_text(chunk: Chunk) -> str:
    """Text that gets indexed: the title adds topic words without changing the stored chunk text."""
    return f"{chunk.title}. {chunk.text}"


class BM25:
    """Okapi BM25 over pre-tokenised documents (k1=1.5, b=0.75)."""

    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tfs = [Counter(d) for d in docs]
        self.lens = np.array([len(d) for d in docs], dtype=float)
        self.avgdl = float(self.lens.mean()) if len(docs) else 0.0
        n = len(docs)
        df = Counter(t for d in docs for t in set(d))
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        # Unseen terms get the idf of a term that appears in no document.
        self.unseen_idf = math.log(1 + (n + 0.5) / 0.5) if n else 1.0

    def term_idf(self, term: str) -> float:
        return self.idf.get(term, self.unseen_idf)

    def scores(self, query_terms: list[str]) -> np.ndarray:
        out = np.zeros(len(self.tfs))
        for term in set(query_terms):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, tf in enumerate(self.tfs):
                f = tf.get(term, 0)
                if f:
                    norm = self.k1 * (1 - self.b + self.b * self.lens[i] / self.avgdl)
                    out[i] += idf * f * (self.k1 + 1) / (f + norm)
        return out


class Index:
    """Everything retrieval needs, held in memory."""

    def __init__(self, chunks: list[Chunk], embeddings: Embeddings | None = None):
        if not chunks:
            raise ValueError("cannot build an index with no chunks")
        self.chunks = chunks
        self.by_id = {c.chunk_id: c for c in chunks}
        self.article_ids = {c.article_id for c in chunks}
        self.doc_terms = [content_terms(index_text(c)) for c in chunks]
        self.term_sets = [set(t) for t in self.doc_terms]
        self.bm25 = BM25(self.doc_terms)
        # TF-IDF with unigrams + bigrams over stemmed terms; used to score sentences and rerank.
        self.tfidf = TfidfVectorizer(analyzer=lambda text: _uni_bi(content_terms(text)), sublinear_tf=True)
        self.tfidf.fit([index_text(c) for c in chunks])
        self.embeddings = embeddings
        self.embedder_name = getattr(embeddings, "name", None) if embeddings else None
        self.vector_store: InMemoryVectorStore | None = None
        if embeddings is not None:
            try:
                store = InMemoryVectorStore(embedding=embeddings)
                store.add_documents(
                    [Document(page_content=index_text(c), metadata={"chunk_id": c.chunk_id}) for c in chunks],
                    ids=[c.chunk_id for c in chunks],
                )
                self.vector_store = store
            except Exception as exc:  # noqa: BLE001 - embedding outage must not break indexing
                log.warning("dense indexing failed (%s); using lexical retrieval only", exc)
                self.embedder_name = None

    @property
    def mode(self) -> str:
        return "hybrid" if self.vector_store is not None else "lexical"

    def dense_scores(self, query: str) -> dict[str, float]:
        """Cosine similarity of the query to every chunk (empty if no vector store)."""
        return self.dense_scores_many([query])[0]

    def dense_scores_many(self, queries: list[str]) -> list[dict[str, float]]:
        """One embedding call for all queries (original + rephrased), then in-memory cosine search."""
        if self.vector_store is None or self.embeddings is None:
            return [{} for _ in queries]
        try:
            vectors = self.embeddings.embed_documents(queries)
        except Exception as exc:  # noqa: BLE001 - embedding outage → lexical-only for this query
            log.warning("query embedding failed (%s); lexical scores only", exc)
            return [{} for _ in queries]
        out = []
        for vec in vectors:
            hits = self.vector_store.similarity_search_with_score_by_vector(vec, k=len(self.chunks))
            out.append({doc.metadata["chunk_id"]: float(score) for doc, score in hits})
        return out

    def tfidf_similarity(self, query: str, texts: list[str]) -> np.ndarray:
        q = self.tfidf.transform([query])
        m = self.tfidf.transform(texts)
        return (m @ q.T).toarray().ravel()

    def preview(self) -> list[dict]:
        return [c.model_dump() for c in self.chunks]


def _uni_bi(terms: list[str]) -> list[str]:
    return terms + [f"{a} {b}" for a, b in zip(terms, terms[1:])]
