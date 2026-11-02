"""LLM and embedding clients for OpenAI, Gemini, Anthropic (Claude) and OpenRouter.

All four are called through the `openai` SDK: Gemini, Anthropic and OpenRouter expose
OpenAI-compatible endpoints, so providers differ only in key, base URL and model name.
Calls run with temperature 0, our own retry loop (exponential backoff with jitter) and a
provider fallback chain. When every provider fails, the caller uses the deterministic path.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

from langchain_core.embeddings import Embeddings
from langchain_core.messages import convert_to_openai_messages
from langchain_core.prompts import ChatPromptTemplate

from app.schemas import Chunk, GeneratedAnswer
from app.tracing import record_usage

log = logging.getLogger(__name__)
T = TypeVar("T")


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    key_env: str
    base_url: str | None
    models: dict[str, str]  # tier → chat model ("small": cheap and fast, "best": strongest)
    embedding_models: dict[str, str] | None  # tier → embedding model; None: no embeddings endpoint
    # Parameters that make output as deterministic as the model allows. Any parameter a model
    # rejects (HTTP 400 naming it) is dropped and the call retried once without it.
    chat_params: dict[str, Any] = field(default_factory=dict)

    @property
    def default_model(self) -> str:
        return self.models["small"]


# Model IDs checked against each provider's /models list on 2026-11-02 (OpenAI, Gemini).
# Anthropic and OpenRouter IDs follow their published naming but were not called here.
PROVIDERS: dict[str, ProviderSpec] = {
    "openai": ProviderSpec(
        "openai",
        "OPENAI_API_KEY",
        None,
        {"small": "gpt-5.4-nano", "best": "gpt-5.5"},
        {"small": "text-embedding-3-small", "best": "text-embedding-3-large"},
        {"temperature": 0, "reasoning_effort": "none"},
    ),
    "gemini": ProviderSpec(
        "gemini",
        "GEMINI_API_KEY",
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        {"small": "gemini-3.5-flash-lite", "best": "gemini-3.1-pro-preview"},
        {"small": "gemini-embedding-001", "best": "gemini-embedding-001"},
        {"temperature": 0, "reasoning_effort": "low"},
    ),
    "anthropic": ProviderSpec(
        "anthropic",
        "ANTHROPIC_API_KEY",
        "https://api.anthropic.com/v1/",
        {"small": "claude-haiku-5-5", "best": "claude-opus-5-5"},
        None,
        {"temperature": 0},
    ),
    "openrouter": ProviderSpec(
        "openrouter",
        "OPENROUTER_API_KEY",
        "https://openrouter.ai/api/v1",
        {"small": "openai/gpt-5.4-nano", "best": "anthropic/claude-opus-5.5"},
        None,
        {"temperature": 0},
    ),
}
DEFAULT_ORDER = ["openai", "gemini", "anthropic", "openrouter"]
_OPTIONAL_PARAMS = ("temperature", "reasoning_effort", "response_format")


def llm_tier() -> str:
    tier = os.getenv("LLM_TIER", "small").lower()
    return tier if tier in {"small", "best"} else "small"


def provider_model(spec: ProviderSpec) -> str:
    """<PROVIDER>_MODEL overrides the tier choice (LLM_TIER=small|best, default small)."""
    return os.getenv(f"{spec.name.upper()}_MODEL") or spec.models[llm_tier()]


def available_providers(preferred: str | None = None) -> list[ProviderSpec]:
    """Providers that have a key set. `preferred` (or LLM_PROVIDER) goes first."""
    order = list(DEFAULT_ORDER)
    if preferred:
        if preferred not in PROVIDERS:
            raise ValueError(f"unknown LLM_PROVIDER {preferred!r}; choose from {sorted(PROVIDERS)}")
        order.remove(preferred)
        order.insert(0, preferred)
    return [PROVIDERS[name] for name in order if os.getenv(PROVIDERS[name].key_env)]


def make_client(spec: ProviderSpec, timeout_s: float = 30.0):
    from openai import OpenAI  # imported lazily so the fallback path never needs it

    # max_retries=0: retries are handled by with_retries() so they are visible and bounded.
    return OpenAI(api_key=os.environ[spec.key_env], base_url=spec.base_url, timeout=timeout_s, max_retries=0)


@dataclass(frozen=True)
class RetryPolicy:
    """Exponential backoff with jitter: delay_n = min(max_delay, base * 2**(n-1)) + U(0, base).

    A server `Retry-After` header (429/503) replaces the computed delay, capped at max_delay.
    `min_interval_s` spaces out consecutive calls (a simple client-side rate limit).
    """

    attempts: int = field(default_factory=lambda: int(os.getenv("LLM_MAX_ATTEMPTS", "3")))
    base_delay_s: float = field(default_factory=lambda: float(os.getenv("LLM_BACKOFF_BASE_S", "0.5")))
    max_delay_s: float = field(default_factory=lambda: float(os.getenv("LLM_BACKOFF_MAX_S", "8")))
    min_interval_s: float = field(default_factory=lambda: float(os.getenv("LLM_MIN_INTERVAL_S", "0")))

    def delay(self, attempt: int, exc: Exception | None = None) -> float:
        retry_after = _retry_after_seconds(exc)
        if retry_after is not None:
            return min(self.max_delay_s, retry_after)
        return min(self.max_delay_s, self.base_delay_s * 2 ** (attempt - 1)) + random.uniform(0, self.base_delay_s)


def _retry_after_seconds(exc: Exception | None) -> float | None:
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if not headers:
        return None
    try:
        return float(headers.get("retry-after"))
    except (TypeError, ValueError):
        return None


def _is_retryable(exc: Exception) -> bool:
    """Auth, permission, not-found and bad-request errors will not fix themselves."""
    try:
        import openai

        fatal = (openai.AuthenticationError, openai.PermissionDeniedError, openai.NotFoundError, openai.BadRequestError)
        return not isinstance(exc, fatal)
    except ImportError:
        return True


class _Throttle:
    def __init__(self):
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self, min_interval_s: float) -> None:
        if min_interval_s <= 0:
            return
        with self._lock:
            sleep_for = self._last + min_interval_s - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            self._last = time.monotonic()


_THROTTLE = _Throttle()


def with_retries(fn: Callable[[], T], policy: RetryPolicy | None = None, label: str = "") -> T:
    """Run fn with throttling and exponential backoff. Re-raises the last error."""
    policy = policy or RetryPolicy()
    last: Exception | None = None
    for attempt in range(1, policy.attempts + 1):
        _THROTTLE.wait(policy.min_interval_s)
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - provider and parse errors
            last = exc
            log.warning("%s attempt %d/%d failed: %s", label, attempt, policy.attempts, exc)
            if attempt == policy.attempts or not _is_retryable(exc):
                break
            time.sleep(policy.delay(attempt, exc))
    assert last is not None
    raise last


# --------------------------------------------------------------------------- prompts

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a customer-support assistant. Answer ONLY from the numbered knowledge-base "
            "context provided by the user message.\n"
            "Rules:\n"
            "1. Do NOT use outside knowledge, assumptions or general advice. If a fact is not stated "
            "in the context, you do not know it.\n"
            "2. Never invent policies, people, names, contact details (emails, phone numbers, URLs), "
            "prices, timelines or steps.\n"
            "3. If the context does not contain the answer, set insufficient_context to true, set "
            "citations to [], and write a short message saying you cannot answer from the knowledge "
            "base and that a support agent can help.\n"
            "4. Otherwise write a short, direct answer (1-3 sentences) for the end user, and cite the "
            "article_id of every context passage you used.\n"
            'Return only JSON: {{"answer": string, "citations": [article_id, ...], '
            '"insufficient_context": boolean}}',
        ),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)

REPHRASE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the user's support question as one short search query for an internal support "
            "knowledge base. Keep the user's key terms, expand abbreviations, and add obvious "
            "synonyms. Do not answer the question and do not add facts. Return only JSON: "
            '{{"query": string}}',
        ),
        ("human", "{question}"),
    ]
)


def format_context(chunks: list[Chunk]) -> str:
    return "\n".join(
        f"[{i}] article_id={c.article_id} | title={c.title} | product={c.product}\n{c.text}"
        for i, c in enumerate(chunks, start=1)
    )


def _parse_json_object(text: str) -> dict:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError(f"no JSON object in model output: {text[:120]!r}")
        return json.loads(match.group(0))


# --------------------------------------------------------------------------- chat


class LLMRouter:
    """Calls the first provider that works; each provider gets `policy.attempts` tries."""

    def __init__(self, preferred: str | None = None, policy: RetryPolicy | None = None, timeout_s: float = 30.0):
        self.specs = available_providers(preferred)
        self.policy = policy or RetryPolicy()
        self.timeout_s = timeout_s
        self._clients: dict[str, object] = {}
        self._rejected: dict[str, set[str]] = {}

    @property
    def enabled(self) -> bool:
        return bool(self.specs)

    def _client(self, spec: ProviderSpec):
        if spec.name not in self._clients:
            self._clients[spec.name] = make_client(spec, self.timeout_s)
        return self._clients[spec.name]

    def _params(self, spec: ProviderSpec, model: str) -> dict[str, Any]:
        params = {k: v for k, v in spec.chat_params.items() if k not in self._rejected.get(model, set())}
        if spec.name != "anthropic" and "response_format" not in self._rejected.get(model, set()):
            params["response_format"] = {"type": "json_object"}  # Anthropic's compat layer ignores it
        return params

    def _create(self, spec: ProviderSpec, kwargs: dict[str, Any]):
        """Create a completion; drop parameters the model rejects (remembered per model)."""
        client = self._client(spec)
        for _ in range(len(_OPTIONAL_PARAMS) + 1):
            try:
                return client.chat.completions.create(**kwargs)
            except Exception as exc:  # noqa: BLE001
                message = str(exc).lower()
                bad = next((p for p in _OPTIONAL_PARAMS if p in kwargs and p in message), None)
                if bad is None or "400" not in message and "unsupported" not in message:
                    raise
                log.info("%s rejected %s; retrying without it", kwargs["model"], bad)
                self._rejected.setdefault(kwargs["model"], set()).add(bad)
                kwargs = {k: v for k, v in kwargs.items() if k != bad}
        return client.chat.completions.create(**kwargs)

    def _chat_json(self, spec: ProviderSpec, messages: list[dict]) -> dict:
        model = provider_model(spec)
        kwargs: dict[str, Any] = dict(model=model, messages=messages, **self._params(spec, model))
        started = time.perf_counter()
        try:
            resp = self._create(spec, kwargs)
        except Exception:
            record_usage("chat", spec.name, model, None, (time.perf_counter() - started) * 1000, ok=False)
            raise
        record_usage("chat", spec.name, kwargs["model"], resp.usage, (time.perf_counter() - started) * 1000)
        return _parse_json_object(resp.choices[0].message.content)

    def complete_json(
        self, prompt: ChatPromptTemplate, parse: Callable[[dict], T], **variables
    ) -> tuple[T, str, list[str]]:
        """Returns (parsed result, provider name, errors). Raises RuntimeError if all providers fail."""
        messages = convert_to_openai_messages(prompt.format_messages(**variables))
        errors: list[str] = []
        for spec in self.specs:
            try:
                result = with_retries(lambda: parse(self._chat_json(spec, messages)), self.policy, label=spec.name)
                return result, spec.name, errors
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{spec.name}: {type(exc).__name__}: {str(exc)[:160]}")
        raise RuntimeError("all LLM providers failed: " + "; ".join(errors))

    def answer(self, question: str, chunks: list[Chunk]) -> tuple[GeneratedAnswer, str, list[str]]:
        return self.complete_json(
            ANSWER_PROMPT, GeneratedAnswer.model_validate, question=question, context=format_context(chunks)
        )

    def rephrase(self, question: str) -> tuple[str, str, list[str]]:
        def parse(data: dict) -> str:
            query = str(data.get("query", "")).strip()
            if not query:
                raise ValueError("empty rephrased query")
            return query[:300]

        return self.complete_json(REPHRASE_PROMPT, parse, question=question)


# --------------------------------------------------------------------------- embeddings


class ApiEmbeddings(Embeddings):
    """LangChain Embeddings backed by an OpenAI-compatible /embeddings endpoint."""

    def __init__(self, spec: ProviderSpec, timeout_s: float = 30.0, batch_size: int = 64):
        if not spec.embedding_models:
            raise ValueError(f"{spec.name} has no embeddings endpoint")
        self.spec = spec
        self.model = os.getenv(f"{spec.name.upper()}_EMBEDDING_MODEL") or spec.embedding_models[llm_tier()]
        self.batch_size = batch_size
        self.policy = RetryPolicy()
        self.client = make_client(spec, timeout_s)
        self.name = f"{spec.name}:{self.model}"

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Embeds in batches of `batch_size` so large KBs stay within request limits."""
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            started = time.perf_counter()
            resp = with_retries(
                lambda: self.client.embeddings.create(model=self.model, input=batch),
                self.policy,
                label=f"{self.name} embeddings",
            )
            record_usage("embedding", self.spec.name, self.model, resp.usage, (time.perf_counter() - started) * 1000)
            data = list(resp.data)
            if all(d.index is not None for d in data):  # Gemini's compat endpoint returns index=None
                data.sort(key=lambda d: d.index)
            vectors += [d.embedding for d in data]
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]


def api_embeddings(preferred: str | None = None) -> ApiEmbeddings | None:
    """First keyed provider that offers embeddings (OpenAI, then Gemini), or None."""
    for spec in available_providers(preferred):
        if spec.embedding_models:
            return ApiEmbeddings(spec)
    return None
