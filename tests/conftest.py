import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.graph import SupportPipeline  # noqa: E402


def offline_settings(**overrides):
    """Deterministic settings: no LLM, lexical retrieval, no local models."""
    s = get_settings(**overrides)
    s.use_llm = False
    s.use_local_models = False
    s.retrieval_mode = "lexical"
    return s


@pytest.fixture(scope="session")
def pipeline() -> SupportPipeline:
    return SupportPipeline(offline_settings())


@pytest.fixture()
def fresh_pipeline() -> SupportPipeline:
    return SupportPipeline(offline_settings())


def has_any_llm_key() -> bool:
    return any(os.getenv(k) for k in ("OPENAI_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY"))
