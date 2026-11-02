"""Live provider smoke tests. Skipped unless an API key is set (never needed for grading)."""

import json

import pytest

from app.config import get_settings
from app.graph import SupportPipeline
from tests.conftest import has_any_llm_key

pytestmark = [pytest.mark.live, pytest.mark.skipif(not has_any_llm_key(), reason="no LLM API key set")]


def test_live_llm_answers_sample_questions():
    settings = get_settings()
    settings.use_local_models = False
    p = SupportPipeline(settings)
    assert p.mode == "llm"
    questions = json.loads(settings.questions_path.read_text())
    expected = {"Q1": "ART-101", "Q2": "ART-102", "Q3": "ART-301", "Q4": "ART-302"}
    for q in questions:
        r = p.run(q["id"], q["question"])
        if q["id"] == "Q5":
            assert r.answer.insufficient_context is True
        elif q["id"] in expected:
            assert expected[q["id"]] in r.answer.citations, r.answer
    assert r.log.prompt_tokens >= 0
