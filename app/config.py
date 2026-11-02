"""Runtime settings, read once from environment variables (and an optional local .env)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent

# A local .env (git-ignored) may hold API keys. It never overrides variables already set.
load_dotenv(ROOT / ".env", override=False)


def _flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    kb_path: Path = ROOT / "kb" / "articles.json"
    questions_path: Path = ROOT / "questions.json"
    artifacts_dir: Path = ROOT / "artifacts"

    # Chunking: sentence windows with overlap.
    chunk_max_sentences: int = 3
    chunk_overlap_sentences: int = 1

    # Retrieval
    top_k: int = 3
    candidate_k: int = 6  # candidates fetched before reranking
    # auto: hybrid when a dense embedder is available, else lexical (BM25)
    retrieval_mode: str = field(default_factory=lambda: os.getenv("RETRIEVAL_MODE", "auto").lower())
    hybrid_lexical_weight: float = 0.5

    # LLM
    use_llm: bool = field(default_factory=lambda: not _flag("DISABLE_LLM"))
    llm_provider: str | None = field(default_factory=lambda: os.getenv("LLM_PROVIDER") or None)
    # Retry/backoff: LLM_MAX_ATTEMPTS, LLM_BACKOFF_BASE_S, LLM_BACKOFF_MAX_S, LLM_MIN_INTERVAL_S (llm_utils.RetryPolicy)
    llm_timeout_s: float = 30.0

    # Optional local models (BERT QA, sentence-transformers, cross-encoder). Off by default.
    use_local_models: bool = field(default_factory=lambda: _flag("USE_LOCAL_MODELS"))
    qa_model: str = field(default_factory=lambda: os.getenv("QA_MODEL", "deepset/bert-base-cased-squad2"))
    sentence_model: str = field(
        default_factory=lambda: os.getenv("SENTENCE_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )
    cross_encoder_model: str = field(
        default_factory=lambda: os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    )

    # Concurrency for batch answering (API /answer/batch and main.py)
    max_concurrency: int = field(default_factory=lambda: int(os.getenv("MAX_CONCURRENCY", "4")))

    # Answer length cap (characters) for the user-facing response.
    max_answer_chars: int = 600


def get_settings(**overrides) -> Settings:
    settings = Settings()
    for key, value in overrides.items():
        if value is not None:
            setattr(settings, key, value)
    return settings
