"""FastAPI service. All endpoints are async; pipeline work runs in worker threads.

    POST /index         build or extend the in-memory index (batch upsert of articles, or reload from disk)
    POST /query         retrieval only: top-3 chunks with scores
    POST /answer        one question → same shape as answers.json
    POST /answer/batch  many questions, answered concurrently (bounded), order kept
    GET  /health        mode, providers, index size
    GET  /metrics       rolling p50/p95 per pipeline step and HTTP route, token and cost counters

Docs: /docs (Swagger UI, "Try it out"), /redoc (ReDoc reference), /scalar (ReDoc-style
reference that can also send requests).

Every response carries X-Request-ID and X-Process-Time-Ms headers (timing middleware).
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.data_utils import load_articles
from app.guardrails import MAX_QUESTION_CHARS
from app.schemas import Answer, Article, RetrievalResult
from app.tracing import METRICS

log = logging.getLogger("app.api")


class IndexRequest(BaseModel):
    articles: list[Article] | None = Field(default=None, description="omit to reload kb/articles.json")
    mode: Literal["replace", "append"] = "replace"

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "mode": "append",
                    "articles": [
                        {
                            "id": "ART-401",
                            "title": "Account closure",
                            "product": "Account",
                            "content": (
                                "Users can close their account from the settings page. "
                                "Closure is permanent after 30 days."
                            ),
                        }
                    ],
                },
                {"mode": "replace"},
            ]
        }
    }


class IndexResponse(BaseModel):
    num_articles: int
    num_chunks: int
    retrieval_mode: str
    embedder: str | None


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    question_id: str | None = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "question_id": "Q1",
                    "question": "My reset email came twice. Which link should I use and how long does it work?",
                },
                {"question_id": "Q5", "question": "Can you tell me the CEO's direct email address?"},
            ]
        }
    }


class BatchRequest(BaseModel):
    questions: list[QuestionRequest] = Field(min_length=1, max_length=100)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "questions": [
                        {
                            "question_id": "Q1",
                            "question": "My reset email came twice. Which link should I use and how long does it work?",
                        },
                        {"question_id": "Q2", "question": "Can support disable my 2FA without verifying me first?"},
                        {"question_id": "Q3", "question": "What should my client do after hitting the API limit?"},
                        {"question_id": "Q4", "question": "Are webhooks exactly once?"},
                        {"question_id": "Q5", "question": "Can you tell me the CEO's direct email address?"},
                    ]
                }
            ]
        }
    }


SCALAR_HTML = """<!doctype html>
<html>
  <head>
    <title>Grounded Support Assistant · API reference</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
  </head>
  <body>
    <script id="api-reference" data-url="/openapi.json"></script>
    <script src="https://cdn.jsdelivr.net/npm/@scalar/api-reference"></script>
  </body>
</html>"""


def create_app(settings: Settings | None = None, pipeline=None) -> FastAPI:
    from app.graph import SupportPipeline

    settings = settings or get_settings()
    app = FastAPI(
        title="Grounded Support Assistant",
        version="1.0.0",
        description=(
            "Answers product-support questions only from the indexed knowledge base, with article "
            "citations, and refuses (`insufficient_context: true`) when the knowledge base does not "
            "support an answer. Try `/answer` with the prefilled examples."
        ),
    )
    app.state.pipeline = pipeline
    app.state.index_lock = asyncio.Lock()

    @app.middleware("http")
    async def timing_middleware(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        started = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
        finally:
            ms = (time.perf_counter() - started) * 1000
            METRICS.observe(f"http.{request.method} {request.url.path}.ms", ms)
            METRICS.inc(f"http.status.{status}")
            log.info("request_id=%s %s %s status=%s %.1fms", request_id, request.method, request.url.path, status, ms)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = f"{ms:.1f}"
        return response

    async def get_pipeline() -> SupportPipeline:
        if app.state.pipeline is None:
            app.state.pipeline = await asyncio.to_thread(SupportPipeline, settings)
        return app.state.pipeline

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", **(await get_pipeline()).describe()}

    @app.get("/scalar", include_in_schema=False)
    async def scalar() -> HTMLResponse:
        return HTMLResponse(SCALAR_HTML)

    @app.get("/metrics")
    async def metrics() -> dict:
        return METRICS.snapshot()

    @app.post("/index", response_model=IndexResponse)
    async def index(req: IndexRequest | None = None) -> IndexResponse:
        req = req or IndexRequest()
        p = await get_pipeline()
        async with app.state.index_lock:  # one rebuild at a time; reads continue on the old index
            try:
                if req.articles is None:
                    articles = await asyncio.to_thread(load_articles, settings.kb_path)
                    c = await asyncio.to_thread(p.build_index, articles)
                elif req.mode == "append":
                    c = await asyncio.to_thread(p.add_articles, req.articles)
                else:
                    c = await asyncio.to_thread(p.build_index, req.articles)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
        return IndexResponse(
            num_articles=len(c.articles),
            num_chunks=len(c.index.chunks),
            retrieval_mode=c.index.mode,
            embedder=c.index.embedder_name,
        )

    @app.post("/query", response_model=RetrievalResult)
    async def query(req: QuestionRequest) -> RetrievalResult:
        p = await get_pipeline()
        return await asyncio.to_thread(p.retrieve, req.question_id or _new_id(), req.question)

    @app.post("/answer", response_model=Answer)
    async def answer(req: QuestionRequest) -> Answer:
        p = await get_pipeline()
        return (await p.arun(req.question_id or _new_id(), req.question)).answer

    @app.post("/answer/batch", response_model=list[Answer])
    async def answer_batch(req: BatchRequest) -> list[Answer]:
        p = await get_pipeline()
        items = [(q.question_id or _new_id(), q.question) for q in req.questions]
        return [r.answer for r in await p.arun_many(items)]

    return app


def _new_id() -> str:
    return f"REQ-{uuid.uuid4().hex[:8]}"


app = create_app()  # `uvicorn app.api:app`; the pipeline is built lazily on first request
