"""Deterministic answer generation, used when no LLM is configured or every provider failed.

Default (no models): the answer is built from the top-ranked chunk. We keep the sentences that
share content terms with the question, ordered by TF-IDF similarity and capped at
`max_sentences`, then put them back in their original order. The same input always gives
the same output.

USE_LOCAL_MODELS=1 adds a BERT SQuAD2 extractive-QA check. If the model finds no answer
span in any retrieved chunk, the answer is a refusal. Otherwise the sentence holding the
span is always kept, and sentences are ranked by sentence-transformer similarity.
"""

from __future__ import annotations

import numpy as np

from app.data_utils import content_terms, split_sentences
from app.indexer import Index
from app.schemas import GeneratedAnswer, ScoredChunk


class DeterministicAnswerer:
    name = "deterministic-lexical"

    def __init__(self, index: Index, sentence_model=None, qa_model=None, max_sentences: int = 3):
        self.index = index
        self.sentence_model = sentence_model
        self.qa_model = qa_model
        self.max_sentences = max_sentences
        if qa_model is not None:
            self.name = "deterministic-bert-qa"

    def _rank_sentences(self, question: str, sentences: list[str]) -> list[float]:
        if self.sentence_model is not None:
            emb = self.sentence_model.encode([question] + sentences, normalize_embeddings=True)
            return [float(x) for x in emb[1:] @ emb[0]]
        return [float(x) for x in self.index.tfidf_similarity(question, sentences)]

    def answer(self, question: str, top: list[ScoredChunk]) -> GeneratedAnswer:
        if not top:
            return GeneratedAnswer(answer="", insufficient_context=True)

        chosen = top[0]
        span_sentence: str | None = None
        if self.qa_model is not None:
            best = None
            for sc in top:
                span, margin = self.qa_model.best_span(question, sc.chunk.text)
                if span and (best is None or margin > best[2]):
                    best = (sc, span, margin)
            if best is None:  # SQuAD2 "no answer" for every retrieved chunk
                return GeneratedAnswer(answer="", insufficient_context=True)
            chosen, span, _ = best
            span_sentence = next((s for s in split_sentences(chosen.chunk.text) if span in s), None)

        sentences = split_sentences(chosen.chunk.text)
        q_terms = set(content_terms(question))
        overlapping = [i for i, s in enumerate(sentences) if q_terms & set(content_terms(s))]
        if span_sentence is not None:
            keep = {sentences.index(span_sentence)} if span_sentence in sentences else set()
            overlapping = sorted(set(overlapping) | keep)
        if not overlapping:
            return GeneratedAnswer(answer="", insufficient_context=True)

        scores = self._rank_sentences(question, [sentences[i] for i in overlapping])
        ranked = [overlapping[j] for j in np.argsort(scores)[::-1]][: self.max_sentences]
        if span_sentence is not None and span_sentence in sentences:
            must = sentences.index(span_sentence)
            if must not in ranked:
                ranked = ranked[:-1] + [must]
        text = " ".join(sentences[i] for i in sorted(ranked))
        return GeneratedAnswer(answer=text, citations=[chosen.chunk.article_id], insufficient_context=False)
