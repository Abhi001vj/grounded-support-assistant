"""Optional local-model path (BERT QA + sentence-transformers + cross-encoder).

Runs only with USE_LOCAL_MODELS=1 and the extra requirements installed.
"""

import importlib.util
import os

import pytest

from tests.conftest import offline_settings

pytestmark = pytest.mark.skipif(
    os.getenv("USE_LOCAL_MODELS") != "1" or importlib.util.find_spec("transformers") is None,
    reason="set USE_LOCAL_MODELS=1 and install requirements-local-models.txt",
)


def test_local_models_answer_and_refuse():
    from app.graph import SupportPipeline

    s = offline_settings()
    s.use_local_models = True
    s.retrieval_mode = "auto"  # local sentence-transformer vectors → hybrid without an API key
    p = SupportPipeline(s)
    assert p.describe()["deterministic_answerer"] == "deterministic-bert-qa"
    assert p.describe()["retrieval_mode"] == "hybrid"
    ok = p.run("Q1", "My reset email came twice. Which link should I use and how long does it work?").answer
    assert ok.citations == ["ART-101"] and not ok.insufficient_context
    no = p.run("Q5", "Can you tell me the CEO's direct email address?").answer
    assert no.insufficient_context is True
