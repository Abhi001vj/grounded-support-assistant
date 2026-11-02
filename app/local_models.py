"""Optional local models, used only when USE_LOCAL_MODELS=1.

- sentence-transformers bi-encoder: dense vectors for hybrid retrieval without an API key,
  and question-sentence similarity in the deterministic answerer
- cross-encoder: reranker
- BERT extractive QA fine-tuned on SQuAD 2.0: it can say "no answer", which is a refusal signal

Each loader returns None if the libraries are missing or the model cannot be downloaded,
so the pipeline degrades to the lexical path instead of crashing.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from langchain_core.embeddings import Embeddings

log = logging.getLogger(__name__)


class LocalEmbeddings(Embeddings):
    def __init__(self, model):
        self.model = model
        self.name = "local:sentence-transformers"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode([text], normalize_embeddings=True)[0].tolist()


@lru_cache(maxsize=2)
def load_sentence_model(name: str):
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(name, device="cpu")
    except Exception as exc:  # noqa: BLE001
        log.warning("sentence-transformers unavailable (%s); continuing without it", exc)
        return None


@lru_cache(maxsize=2)
def load_cross_encoder(name: str):
    try:
        from sentence_transformers import CrossEncoder

        return CrossEncoder(name, device="cpu")
    except Exception as exc:  # noqa: BLE001
        log.warning("cross-encoder unavailable (%s); continuing without it", exc)
        return None


class BertQA:
    """Extractive QA with SQuAD2-style null-answer scoring, implemented without `pipeline()`."""

    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def best_span(self, question: str, context: str, max_answer_tokens: int = 30) -> tuple[str | None, float]:
        """Returns (answer text or None for "no answer", span score minus null score)."""
        import torch

        enc = self.tokenizer(
            question,
            context,
            return_tensors="pt",
            truncation="only_second",
            max_length=384,
            return_offsets_mapping=True,
        )
        offsets = enc.pop("offset_mapping")[0].tolist()
        seq_ids = enc.sequence_ids(0)
        with torch.no_grad():
            out = self.model(**enc)
        start = out.start_logits[0]
        end = out.end_logits[0]
        null_score = float(start[0] + end[0])
        ctx = [i for i, s in enumerate(seq_ids) if s == 1]
        best, best_score = None, float("-inf")
        top_starts = sorted(ctx, key=lambda i: float(start[i]), reverse=True)[:20]
        top_ends = sorted(ctx, key=lambda i: float(end[i]), reverse=True)[:20]
        for s in top_starts:
            for e in top_ends:
                if e < s or e - s + 1 > max_answer_tokens:
                    continue
                score = float(start[s] + end[e])
                if score > best_score:
                    best, best_score = (s, e), score
        if best is None:
            return None, float("-inf")
        margin = best_score - null_score
        if margin <= 0:
            return None, margin
        s, e = best
        return context[offsets[s][0] : offsets[e][1]], margin


@lru_cache(maxsize=2)
def load_bert_qa(name: str) -> BertQA | None:
    try:
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModelForQuestionAnswering.from_pretrained(name)
        model.eval()
        return BertQA(tokenizer, model)
    except Exception as exc:  # noqa: BLE001
        log.warning("BERT QA model unavailable (%s); continuing without it", exc)
        return None
