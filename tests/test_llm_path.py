"""LLM path with a fake router: no network, deterministic."""

import pytest

from app.llm_utils import RetryPolicy, with_retries
from app.schemas import GeneratedAnswer


class FakeRouter:
    def __init__(self, answer=None, error=None):
        self.specs = []
        self._answer, self._error = answer, error
        self.calls = 0

    enabled = True

    def rephrase(self, question):
        return question + " synonyms", "fake", []

    def answer(self, question, chunks):
        self.calls += 1
        if self._error:
            raise self._error
        return self._answer, "fake", []


def test_llm_answer_with_invented_and_duplicate_citations_is_cleaned(fresh_pipeline):
    fresh_pipeline.llm = FakeRouter(
        GeneratedAnswer(
            answer="Only the latest reset link is valid, for 15 minutes.", citations=["ART-101", "ART-101", "ART-999"]
        )
    )
    result = fresh_pipeline.run("Q1", "My reset email came twice. Which link should I use and how long does it work?")
    assert result.answer.citations == ["ART-101"]
    assert result.log.used_fallback is False
    assert any(e["event"] == "citations_cleaned" for e in result.log.events)


def test_llm_failure_falls_back_to_deterministic(fresh_pipeline):
    fresh_pipeline.llm = FakeRouter(error=RuntimeError("all LLM providers failed"))
    result = fresh_pipeline.run("Q3", "What should my client do after hitting the API limit?")
    assert result.log.used_fallback is True
    assert result.answer.citations == ["ART-301"]


def test_llm_answer_with_made_up_number_is_replaced(fresh_pipeline):
    fresh_pipeline.llm = FakeRouter(GeneratedAnswer(answer="Bank withdrawals take 7 days.", citations=["ART-201"]))
    result = fresh_pipeline.run("X", "How long does a bank withdrawal take?")
    assert "7 days" not in result.answer.answer
    assert result.log.used_fallback is True
    assert any("ungrounded number" in e for e in result.log.errors)


def test_llm_is_not_called_for_unrelated_questions(fresh_pipeline):
    fake = FakeRouter(GeneratedAnswer(answer="ceo@corp.com", citations=["ART-101"]))
    fresh_pipeline.llm = fake
    result = fresh_pipeline.run("Q5", "Can you tell me the CEO's direct email address?")
    assert result.answer.insufficient_context is True
    assert fake.calls == 0


def test_with_retries_backs_off_then_succeeds():
    attempts = []

    def flaky():
        attempts.append(1)
        if len(attempts) < 3:
            raise TimeoutError("boom")
        return "ok"

    policy = RetryPolicy(attempts=3, base_delay_s=0.0, max_delay_s=0.0)
    assert with_retries(flaky, policy) == "ok" and len(attempts) == 3


def test_with_retries_gives_up_after_max_attempts():
    policy = RetryPolicy(attempts=2, base_delay_s=0.0, max_delay_s=0.0)
    with pytest.raises(TimeoutError):
        with_retries(lambda: (_ for _ in ()).throw(TimeoutError("x")), policy)


def test_backoff_is_exponential_and_capped():
    p = RetryPolicy(attempts=5, base_delay_s=1.0, max_delay_s=4.0)
    d = [p.delay(n) for n in (1, 2, 3, 4)]
    assert 1.0 <= d[0] <= 2.0 and 2.0 <= d[1] <= 3.0 and 4.0 <= d[2] <= 5.0 and 4.0 <= d[3] <= 5.0
