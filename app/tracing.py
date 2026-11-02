"""Per-request tracing (step timings, LLM/embedding token usage and cost) and process metrics.

A Trace is bound to the current request through a ContextVar. Code deep in the stack, such
as an embedding call made by the vector store, can record usage without the trace being
passed through every function. asyncio.to_thread copies the context, so the API path keeps it.
"""

from __future__ import annotations

import json
import os
import statistics
import threading
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterator

# USD per 1M tokens (input, output), standard tier. ESTIMATES for cost reporting only:
# compiled 2026-11-02 from third-party pricing trackers (see README "Cost estimates"),
# not from the providers' own pages. Override or extend with
# LLM_PRICES_JSON='{"model-id": [input, output], ...}'. Unknown models report cost as null.
DEFAULT_PRICES: dict[str, tuple[float, float]] = {
    # OpenAI
    "gpt-5.5": (5.00, 30.00),
    "gpt-5.4-mini": (0.75, 4.50),
    "gpt-5.4-nano": (0.20, 1.25),
    "text-embedding-3-small": (0.02, 0.0),
    "text-embedding-3-large": (0.13, 0.0),
    # Google
    "gemini-3.1-pro-preview": (2.00, 12.00),
    "gemini-3.8-flash": (0.75, 3.75),  # introductory rate through 2026-12-31
    "gemini-3.5-flash-lite": (0.30, 2.50),
    # Anthropic
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}


def _prices() -> dict[str, tuple[float, float]]:
    prices = dict(DEFAULT_PRICES)
    raw = os.getenv("LLM_PRICES_JSON")
    if raw:
        try:
            prices.update({k: (float(v[0]), float(v[1])) for k, v in json.loads(raw).items()})
        except (ValueError, TypeError, IndexError, KeyError):
            pass
    return prices


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float | None:
    price = _prices().get(model.split("/")[-1].replace("claude-opus-5.5", "claude-opus-5-5"))
    if price is None:
        return None
    return round((prompt_tokens * price[0] + completion_tokens * price[1]) / 1_000_000, 8)


@dataclass
class LLMCall:
    kind: str  # "chat" | "embedding"
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    cost_usd: float | None
    ok: bool = True
    step: str | None = None  # pipeline step that made the call


@dataclass
class Trace:
    request_id: str
    steps: dict[str, float] = field(default_factory=dict)  # step name → ms
    calls: list[LLMCall] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    _stack: list[str] = field(default_factory=list)

    @contextmanager
    def span(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        self._stack.append(name)
        try:
            yield
        finally:
            self._stack.pop()
            self.steps[name] = round(self.steps.get(name, 0.0) + (time.perf_counter() - start) * 1000, 2)

    def record_call(self, call: LLMCall) -> None:
        if call.step is None and self._stack:
            call.step = self._stack[-1]
        self.calls.append(call)

    def tokens_by_step(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for c in self.calls:
            row = out.setdefault(
                c.step or "other", {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0, "est_cost_usd": 0.0}
            )
            row["prompt_tokens"] += c.prompt_tokens
            row["completion_tokens"] += c.completion_tokens
            row["calls"] += 1
            row["est_cost_usd"] = round(row["est_cost_usd"] + (c.cost_usd or 0.0), 8)
        return out

    def event(self, name: str, **data: Any) -> None:
        self.events.append({"event": name, **data})

    def summary(self) -> dict[str, Any]:
        costs = [c.cost_usd for c in self.calls if c.cost_usd is not None]
        return {
            "step_timings_ms": self.steps,
            "tokens_by_step": self.tokens_by_step(),
            "llm_calls": len([c for c in self.calls if c.kind == "chat"]),
            "embedding_calls": len([c for c in self.calls if c.kind == "embedding"]),
            "prompt_tokens": sum(c.prompt_tokens for c in self.calls),
            "completion_tokens": sum(c.completion_tokens for c in self.calls),
            "est_cost_usd": round(sum(costs), 8) if costs else (0.0 if not self.calls else None),
            "calls": [c.__dict__ for c in self.calls],
            "events": self.events,
        }


_current: ContextVar[Trace | None] = ContextVar("current_trace", default=None)


@contextmanager
def bind_trace(trace: Trace) -> Iterator[Trace]:
    token = _current.set(trace)
    try:
        yield trace
    finally:
        _current.reset(token)


def current_trace() -> Trace | None:
    return _current.get()


def record_usage(kind: str, provider: str, model: str, usage: Any, latency_ms: float, ok: bool = True) -> None:
    trace = current_trace()
    if trace is None:
        return
    prompt = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
    completion = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0
    trace.record_call(
        LLMCall(
            kind,
            provider,
            model,
            prompt,
            completion,
            round(latency_ms, 2),
            estimate_cost(model, prompt, completion),
            ok,
        )
    )


class Metrics:
    """Process-wide rolling metrics for /metrics (last N samples per series)."""

    def __init__(self, window: int = 2000):
        self._lock = threading.Lock()
        self._series: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=window))
        self._counters: dict[str, float] = defaultdict(float)

    def observe(self, name: str, value: float) -> None:
        with self._lock:
            self._series[name].append(value)

    def inc(self, name: str, value: float = 1.0) -> None:
        with self._lock:
            self._counters[name] += value

    def record_trace(self, trace: Trace, used_fallback: bool, refused: bool) -> None:
        s = trace.summary()
        for step, ms in s["step_timings_ms"].items():
            self.observe(f"step.{step}.ms", ms)
        self.inc("pipeline.requests")
        self.inc("pipeline.fallback", used_fallback)
        self.inc("pipeline.refusals", refused)
        for step, row in s["tokens_by_step"].items():
            self.inc(f"tokens.{step}", row["prompt_tokens"] + row["completion_tokens"])
        self.inc("tokens.prompt", s["prompt_tokens"])
        self.inc("tokens.completion", s["completion_tokens"])
        self.inc("cost.usd", s["est_cost_usd"] or 0.0)

    def record_index(self, trace: Trace) -> None:
        s = trace.summary()
        for step, ms in s["step_timings_ms"].items():
            self.observe(f"{step}.ms", ms)
        self.inc("index.builds")
        self.inc("tokens.embedding.index", s["prompt_tokens"])
        self.inc("cost.usd", s["est_cost_usd"] or 0.0)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            series = {k: list(v) for k, v in self._series.items()}
            counters = dict(self._counters)
        out: dict[str, Any] = {"counters": {k: round(v, 8) for k, v in counters.items()}, "latency_ms": {}}
        for name, values in sorted(series.items()):
            if not values:
                continue
            ordered = sorted(values)
            out["latency_ms"][name] = {
                "count": len(values),
                "p50": round(statistics.median(ordered), 2),
                "p95": round(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 2),
                "max": round(ordered[-1], 2),
            }
        return out


METRICS = Metrics()
