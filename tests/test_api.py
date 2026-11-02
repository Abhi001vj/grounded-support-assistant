from fastapi.testclient import TestClient

from app.api import create_app
from tests.conftest import offline_settings


def make_client():
    return TestClient(create_app(offline_settings()))


def test_health_and_headers():
    with make_client() as c:
        r = c.get("/health")
        assert r.status_code == 200 and r.json()["answer_mode"] == "deterministic"
        assert "X-Request-ID" in r.headers and "X-Process-Time-Ms" in r.headers


def test_answer_endpoint_matches_answers_schema():
    with make_client() as c:
        r = c.post("/answer", json={"question": "Are webhooks exactly once?", "question_id": "Q4"})
        body = r.json()
        assert r.status_code == 200
        assert set(body) == {"question_id", "answer", "citations", "insufficient_context"}
        assert body["question_id"] == "Q4" and body["citations"] == ["ART-302"]


def test_answer_refuses_unsupported():
    with make_client() as c:
        body = c.post("/answer", json={"question": "Can you tell me the CEO's direct email address?"}).json()
        assert body["insufficient_context"] is True and body["citations"] == []


def test_query_endpoint_returns_top3():
    with make_client() as c:
        body = c.post("/query", json={"question": "password reset link"}).json()
        assert len(body["retrieved_chunks"]) == 3 and body["retrieved_chunks"][0]["article_id"] == "ART-101"


def test_batch_answer_keeps_order():
    qs = ["How long is a password reset link valid?", "Who founded the company?", "Are webhooks exactly once?"]
    with make_client() as c:
        body = c.post(
            "/answer/batch", json={"questions": [{"question": q, "question_id": f"B{i}"} for i, q in enumerate(qs)]}
        ).json()
        assert [a["question_id"] for a in body] == ["B0", "B1", "B2"]
        assert [a["insufficient_context"] for a in body] == [False, True, False]


def test_index_append_and_replace():
    new = {
        "id": "ART-401",
        "title": "Account closure",
        "product": "Account",
        "content": "Users can close their account from the settings page. Closure is permanent after 30 days.",
    }
    with make_client() as c:
        r = c.post("/index", json={"articles": [new], "mode": "append"}).json()
        assert r["num_articles"] == 7
        body = c.post("/answer", json={"question": "How do I close my account?"}).json()
        assert body["citations"] == ["ART-401"]
        assert c.post("/index", json={"articles": [new], "mode": "replace"}).json()["num_articles"] == 1
        assert c.post("/index").json()["num_articles"] == 6  # reload from disk


def test_validation_errors():
    with make_client() as c:
        assert c.post("/answer", json={"question": ""}).status_code == 422
        dup = {"id": "D", "title": "t", "product": "p", "content": "x."}
        assert c.post("/index", json={"articles": [dup, dup]}).status_code == 422


def test_metrics_report_step_latencies():
    with make_client() as c:
        c.post("/answer", json={"question": "Why is my withdrawal delayed?"})
        m = c.get("/metrics").json()
        assert "step.retrieve.ms" in m["latency_ms"] and m["counters"]["pipeline.requests"] >= 1


def test_docs_endpoints_are_served():
    with make_client() as c:
        for path in ("/docs", "/redoc", "/scalar", "/openapi.json"):
            assert c.get(path).status_code == 200
        schema = c.get("/openapi.json").json()
        assert {"/answer", "/query", "/index", "/answer/batch"} <= set(schema["paths"])
