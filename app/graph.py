"""The answer pipeline as a LangGraph state machine.

    input_guard ─┬─ blocked ─────────────────────────────────────────────┐
                 └─ rephrase → retrieve → gate ─┬─ refuse ────────────────┤
                                                └─ generate → output_guard ┴─► finalize

generate:     LLM (retries with backoff → next provider) → deterministic answerer on failure
output_guard: regex checks; an LLM answer that fails them is replaced by the deterministic one
finalize:     citation cleanup and refusal normalisation (guardrails.finalize)
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from app.config import Settings, get_settings
from app.data_utils import chunk_articles, content_terms, load_articles
from app.guardrails import (
    asks_for_contact_details,
    check_input,
    check_output,
    context_has_contact_details,
    finalize,
    relevance_gate,
)
from app.indexer import Index
from app.llm_utils import LLMRouter, api_embeddings
from app.qa_fallback import DeterministicAnswerer
from app.retrievers import HybridRetriever
from app.schemas import Answer, Article, GeneratedAnswer, RetrievalResult, RunLogRecord, ScoredChunk
from app.tracing import METRICS, Trace, bind_trace

log = logging.getLogger(__name__)

# LLM mode: refuse before calling the model only when the context is clearly unrelated.
LLM_MODE_MIN_DENSE = 0.35


class State(TypedDict, total=False):
    question_id: str
    question: str  # redacted question, used everywhere downstream
    input_flags: list[str]
    blocked_reason: str | None
    queries: list[str]
    retrieved: list[ScoredChunk]
    gate_passed: bool
    gate_reason: str
    generated: GeneratedAnswer
    provider: str | None
    used_fallback: bool
    errors: list[str]
    answer: Answer


@dataclass(frozen=True)
class Components:
    """Everything that depends on the indexed KB. Swapped atomically on re-index."""

    articles: tuple[Article, ...]
    index: Index
    retriever: HybridRetriever
    deterministic: DeterministicAnswerer


@dataclass
class PipelineResult:
    answer: Answer
    retrieval: RetrievalResult
    log: RunLogRecord
    debug: dict[str, Any]


def _components(config: RunnableConfig) -> Components:
    return config["configurable"]["components"]


def _trace(config: RunnableConfig) -> Trace:
    return config["configurable"]["trace"]


def _timed(name: str, fn):
    """Timing middleware for graph nodes: each step's wall time lands in the request trace."""

    def node(state: State, config: RunnableConfig) -> State:
        with _trace(config).span(name):
            return fn(state, config)

    node.__name__ = name
    return node


class SupportPipeline:
    def __init__(self, settings: Settings | None = None, articles: list[Article] | None = None):
        self.settings = settings or get_settings()
        s = self.settings
        self.llm = LLMRouter(s.llm_provider, timeout_s=s.llm_timeout_s) if s.use_llm else None
        if self.llm is not None and not self.llm.enabled:
            self.llm = None

        self.sentence_model = self.cross_encoder = self.qa_model = None
        if s.use_local_models:
            from app.local_models import load_bert_qa, load_cross_encoder, load_sentence_model

            self.sentence_model = load_sentence_model(s.sentence_model)
            self.cross_encoder = load_cross_encoder(s.cross_encoder_model)
            self.qa_model = load_bert_qa(s.qa_model)

        self._index_lock = threading.Lock()
        self.components: Components
        self.build_index(articles if articles is not None else load_articles(s.kb_path))
        self.graph = self._build_graph()

    # ------------------------------------------------------------------ indexing

    def _embedder(self):
        s = self.settings
        if s.retrieval_mode == "lexical":
            return None
        if self.llm is not None:
            emb = api_embeddings(s.llm_provider)
            if emb is not None:
                return emb
        if self.sentence_model is not None:
            from app.local_models import LocalEmbeddings

            return LocalEmbeddings(self.sentence_model)
        if s.retrieval_mode == "hybrid":
            log.warning("RETRIEVAL_MODE=hybrid but no embedder is available; using lexical (BM25)")
        return None

    def build_index(self, articles: list[Article]) -> Components:
        """Build a new in-memory index and swap it in. Requests in flight keep the old one."""
        s = self.settings
        ids = [a.id for a in articles]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate article ids")
        trace = Trace(request_id="index")
        with bind_trace(trace):
            with trace.span("index.chunk"):
                chunks = chunk_articles(articles, s.chunk_max_sentences, s.chunk_overlap_sentences)
            with trace.span("index.build"):  # BM25 + TF-IDF + document embeddings (if any)
                index = Index(chunks, self._embedder())
        self.index_trace = trace.summary()
        METRICS.record_index(trace)
        retriever = HybridRetriever(
            index=index,
            top_k=s.top_k,
            candidate_k=max(s.candidate_k, s.top_k),
            lexical_weight=s.hybrid_lexical_weight,
            cross_encoder=self.cross_encoder,
        )
        components = Components(
            tuple(articles), index, retriever, DeterministicAnswerer(index, self.sentence_model, self.qa_model)
        )
        with self._index_lock:
            self.components = components
        return components

    def add_articles(self, articles: list[Article]) -> Components:
        """Batch upsert: articles with an existing id replace the old version."""
        with self._index_lock:
            current = {a.id: a for a in self.components.articles}
        for a in articles:
            current[a.id] = a
        return self.build_index(list(current.values()))

    # convenience accessors
    @property
    def index(self) -> Index:
        return self.components.index

    @property
    def retriever(self) -> HybridRetriever:
        return self.components.retriever

    @property
    def articles(self) -> tuple[Article, ...]:
        return self.components.articles

    @property
    def chunks(self):
        return self.components.index.chunks

    @property
    def mode(self) -> str:
        return "llm" if self.llm is not None else "deterministic"

    def describe(self) -> dict[str, Any]:
        c = self.components
        return {
            "answer_mode": self.mode,
            "llm_providers": [p.name for p in self.llm.specs] if self.llm else [],
            "retrieval_mode": c.index.mode,
            "embedder": c.index.embedder_name,
            "reranker": "cross-encoder" if self.cross_encoder is not None else "lexical-tfidf",
            "deterministic_answerer": c.deterministic.name,
            "num_articles": len(c.articles),
            "num_chunks": len(c.index.chunks),
            "index_build": {
                "step_timings_ms": self.index_trace["step_timings_ms"],
                "embedding_tokens": self.index_trace["prompt_tokens"],
                "est_cost_usd": self.index_trace["est_cost_usd"],
            },
        }

    # ------------------------------------------------------------------ graph nodes

    def _input_guard(self, state: State, config: RunnableConfig) -> State:
        check = check_input(state["question"])
        return {
            "question": check.text,
            "input_flags": check.flags,
            "blocked_reason": check.reason if check.blocked else None,
        }

    def _rephrase(self, state: State, config: RunnableConfig) -> State:
        queries = [state["question"]]
        errors = list(state.get("errors", []))
        if self.llm is not None:
            try:
                rephrased, _, errs = self.llm.rephrase(state["question"])
                errors += errs
                queries.append(rephrased)
            except Exception as exc:  # noqa: BLE001 - rephrasing is optional
                errors.append(f"rephrase: {exc}")
        return {"queries": queries, "errors": errors}

    def _retrieve(self, state: State, config: RunnableConfig) -> State:
        retriever = _components(config).retriever
        return {"retrieved": retriever.search(state["question"], state["queries"][1:])}

    def _gate(self, state: State, config: RunnableConfig) -> State:
        index = _components(config).index
        question, top = state["question"], state["retrieved"]
        texts = [s.chunk.text for s in top]
        if asks_for_contact_details(question) and not context_has_contact_details(texts):
            return {"gate_passed": False, "gate_reason": "contact details requested but none in context"}
        gate = relevance_gate(index, question, top)
        if gate.passed or self.llm is None:
            return {"gate_passed": gate.passed, "gate_reason": gate.reason}
        # LLM mode: let the model judge paraphrases unless the context is plainly unrelated.
        q_terms = set(content_terms(question))
        any_overlap = any(q_terms & index.term_sets[index.chunks.index(s.chunk)] for s in top)
        semantic = gate.dense is not None and gate.dense >= LLM_MODE_MIN_DENSE
        passed = any_overlap or semantic
        return {"gate_passed": passed, "gate_reason": "llm-mode lenient gate" if passed else gate.reason}

    def _generate(self, state: State, config: RunnableConfig) -> State:
        deterministic = _components(config).deterministic
        chunks = [s.chunk for s in state["retrieved"]]
        errors = list(state.get("errors", []))
        if self.llm is not None:
            try:
                generated, provider, errs = self.llm.answer(state["question"], chunks)
                return {"generated": generated, "provider": provider, "used_fallback": False, "errors": errors + errs}
            except Exception as exc:  # noqa: BLE001 - fall through to the deterministic path
                errors.append(str(exc))
        generated = deterministic.answer(state["question"], state["retrieved"])
        return {"generated": generated, "provider": deterministic.name, "used_fallback": True, "errors": errors}

    def _output_guard(self, state: State, config: RunnableConfig) -> State:
        generated = state["generated"]
        if generated.insufficient_context:
            return {}
        texts = [s.chunk.text for s in state["retrieved"]]
        violations = check_output(generated.answer, texts, state["question"])
        if not violations:
            return {}
        errors = list(state.get("errors", [])) + [f"output guard: {v}" for v in violations]
        deterministic = _components(config).deterministic
        if state.get("provider") == deterministic.name:  # extractive text failed: refuse
            return {"generated": GeneratedAnswer(answer="", insufficient_context=True), "errors": errors}
        replacement = deterministic.answer(state["question"], state["retrieved"])
        if check_output(replacement.answer, texts, state["question"]):
            replacement = GeneratedAnswer(answer="", insufficient_context=True)
        return {"generated": replacement, "provider": deterministic.name, "used_fallback": True, "errors": errors}

    def _refuse(self, state: State, config: RunnableConfig) -> State:
        reason = state.get("blocked_reason") or state.get("gate_reason")
        return {
            "generated": GeneratedAnswer(answer="", insufficient_context=True),
            "provider": "input-guard" if state.get("blocked_reason") else "relevance-gate",
            "used_fallback": self.llm is None,
            "gate_reason": reason,
            "retrieved": state.get("retrieved", []),
        }

    def _finalize(self, state: State, config: RunnableConfig) -> State:
        c = _components(config)
        kb_ids = {a.id for a in c.articles}
        answer, report = finalize(
            state["question_id"], state["generated"], state.get("retrieved", []), kb_ids, self.settings.max_answer_chars
        )
        if report.changed:
            _trace(config).event("citations_cleaned", **report.as_dict())
        return {"answer": answer}

    def _build_graph(self):
        g = StateGraph(State)
        for name, fn in [
            ("input_guard", self._input_guard),
            ("rephrase", self._rephrase),
            ("retrieve", self._retrieve),
            ("gate", self._gate),
            ("generate", self._generate),
            ("output_guard", self._output_guard),
            ("refuse", self._refuse),
            ("finalize", self._finalize),
        ]:
            g.add_node(name, _timed(name, fn))
        g.add_edge(START, "input_guard")
        g.add_conditional_edges("input_guard", lambda s: "refuse" if s.get("blocked_reason") else "rephrase")
        g.add_edge("rephrase", "retrieve")
        g.add_edge("retrieve", "gate")
        g.add_conditional_edges("gate", lambda s: "generate" if s["gate_passed"] else "refuse")
        g.add_edge("generate", "output_guard")
        g.add_edge("output_guard", "finalize")
        g.add_edge("refuse", "finalize")
        g.add_edge("finalize", END)
        return g.compile()

    # ------------------------------------------------------------------ public API

    def retrieve(self, question_id: str, question: str) -> RetrievalResult:
        check = check_input(question)
        top = self.retriever.search(check.text) if check.text else []
        return RetrievalResult(
            question_id=question_id, question=check.text, retrieved_chunks=[s.to_retrieved() for s in top]
        )

    def run(self, question_id: str, question: str) -> PipelineResult:
        started = time.perf_counter()
        components = self.components  # one consistent snapshot for the whole request
        trace = Trace(request_id=question_id)
        with bind_trace(trace):
            state: State = self.graph.invoke(
                {"question_id": question_id, "question": question, "errors": [], "retrieved": []},
                config={"configurable": {"components": components, "trace": trace}},
            )
        trace_summary = trace.summary()
        top: list[ScoredChunk] = state.get("retrieved", [])
        record = RunLogRecord(
            question_id=question_id,
            retrieval_chunk_ids=[s.chunk.chunk_id for s in top],
            used_fallback=bool(state.get("used_fallback", True)),
            timestamp=datetime.now(timezone.utc).isoformat(),
            provider=state.get("provider"),
            retrieval_mode=components.index.mode,
            gate_passed=state.get("gate_passed"),
            latency_ms=int((time.perf_counter() - started) * 1000),
            errors=state.get("errors", []) + [f"input: {f}" for f in state.get("input_flags", [])],
            step_timings_ms=trace_summary["step_timings_ms"],
            llm_calls=trace_summary["llm_calls"],
            embedding_calls=trace_summary["embedding_calls"],
            prompt_tokens=trace_summary["prompt_tokens"],
            completion_tokens=trace_summary["completion_tokens"],
            est_cost_usd=trace_summary["est_cost_usd"],
            tokens_by_step=trace_summary["tokens_by_step"],
            events=trace_summary["events"],
        )
        METRICS.record_trace(trace, record.used_fallback, state["answer"].insufficient_context)
        METRICS.observe("pipeline.total.ms", record.latency_ms)
        retrieval = RetrievalResult(
            question_id=question_id, question=state["question"], retrieved_chunks=[s.to_retrieved() for s in top]
        )
        debug = {
            "trace": trace_summary,
            "queries": state.get("queries"),
            "gate_reason": state.get("gate_reason"),
            "scores": [
                {
                    "chunk_id": s.chunk.chunk_id,
                    "score": round(s.score, 4),
                    "lexical": round(s.lexical_score, 4),
                    "dense": None if s.dense_score is None else round(s.dense_score, 4),
                    "rerank": None if s.rerank_score is None else round(s.rerank_score, 4),
                }
                for s in top
            ],
        }
        return PipelineResult(answer=state["answer"], retrieval=retrieval, log=record, debug=debug)

    async def arun(self, question_id: str, question: str) -> PipelineResult:
        """Async wrapper: the pipeline is CPU/IO-bound sync code, so it runs in a worker thread."""
        return await asyncio.to_thread(self.run, question_id, question)

    async def arun_many(self, items: list[tuple[str, str]], max_concurrency: int | None = None) -> list[PipelineResult]:
        """Answer many questions concurrently (bounded), keeping input order."""
        sem = asyncio.Semaphore(max_concurrency or self.settings.max_concurrency)

        async def one(qid: str, q: str) -> PipelineResult:
            async with sem:
                return await self.arun(qid, q)

        return list(await asyncio.gather(*(one(qid, q) for qid, q in items)))
