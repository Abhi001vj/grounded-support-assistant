"""Checks that keep answers grounded: relevance gate, contact-detail guard, citation cleanup."""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.data_utils import content_terms
from app.indexer import Index
from app.schemas import Answer, GeneratedAnswer, ScoredChunk

REFUSAL = (
    "I can't answer that from the support knowledge base. " "Please contact the support team so an agent can help."
)

# Gate thresholds (see docs/specs/SPEC.md §3 and README "Calibration").
MIN_MATCHED_TERMS = 2  # distinct question terms that appear in the top chunk
MANY_MATCHED_TERMS = 3  # this many distinct matches passes on its own (long, wordy questions)
MIN_COVERAGE = 0.4  # idf-weighted share of question terms found in the top chunk
STRONG_COVERAGE = 0.75  # enough on its own (short questions such as "withdrawal times?")
DENSE_PASS = 0.55  # hybrid/local mode: semantic similarity strong enough to pass alone

_CONTACT_REQUEST = re.compile(
    r"\b(e-?mail address(es)?|(personal|direct|private|work) e-?mail|phone( number)?|telephone|"
    r"mobile number|contact (details|info|information|number)|whatsapp|"
    r"(home|office|postal|mailing) address|linkedin)\b",
    re.I,
)
_CONTACT_DATA = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+|\+?\d[\d\s().-]{7,}\d|https?://\S+")

# --------------------------------------------------------------------------- input guardrails (regex)

MAX_QUESTION_CHARS = 1000

_INJECTION = re.compile(
    r"(ignore|disregard|forget|override)\s+(all\s+|any\s+|the\s+|your\s+)?(previous|prior|above|earlier|system)?\s*"
    r"(instructions|rules|prompts?|context)"
    r"|\b(system|developer)\s+prompt\b|reveal\s+(your|the)\s+(prompt|instructions)"
    r"|\byou\s+are\s+now\b|\bjailbreak\b|\bdeveloper\s+mode\b|\bDAN\b"
    r"|<\s*/?\s*(system|assistant)\s*>",
    re.I,
)
# PII redacted before the question reaches retrieval, an LLM or the logs.
_PII = [
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")),
    ("API_KEY", re.compile(r"\b(sk|pk|rk|AIza|ghp|xox[bp])[-_A-Za-z0-9]{16,}\b")),
    ("CARD", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    ("PHONE", re.compile(r"(?<!\w)\+?\d[\d\s().-]{7,}\d(?!\w)")),
    ("IBAN", re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")),
]


@dataclass
class InputCheck:
    text: str  # redacted question used downstream
    blocked: bool
    reason: str | None
    flags: list[str]


def check_input(question: str) -> InputCheck:
    """Regex input guardrail: length, prompt-injection patterns, PII redaction."""
    text = " ".join((question or "").split())
    if not text:
        return InputCheck(text, True, "empty question", ["empty"])
    if len(text) > MAX_QUESTION_CHARS:
        return InputCheck(text[:MAX_QUESTION_CHARS], True, "question too long", ["too_long"])
    flags: list[str] = []
    for label, pattern in _PII:
        if pattern.search(text):
            text = pattern.sub(f"[{label}]", text)
            flags.append(f"redacted_{label.lower()}")
    if _INJECTION.search(text):
        return InputCheck(text, True, "prompt-injection pattern", flags + ["prompt_injection"])
    return InputCheck(text, False, None, flags)


# --------------------------------------------------------------------------- output guardrails (regex)

_NUMBER = re.compile(r"\b\d+(?:[.,]\d+)?\b")  # "2FA" is not a number (no word boundary)
_LEAK = re.compile(r"\b(system prompt|my instructions|as an ai language model)\b", re.I)


def check_output(answer: str, context_texts: list[str], question: str = "") -> list[str]:
    """Regex output guardrail. Returns violations; an empty list means the answer is OK.

    - contact details (email / phone / URL) that are not in the retrieved context
    - numbers that appear in neither the retrieved context nor the question (made-up limits, times)
    - prompt-leak phrases
    """
    context = " ".join(context_texts)
    violations: list[str] = []
    for m in _CONTACT_DATA.finditer(answer):
        if m.group(0) not in context:
            violations.append(f"ungrounded contact detail: {m.group(0)!r}")
    allowed = set(_NUMBER.findall(context)) | set(_NUMBER.findall(question))
    for n in _NUMBER.findall(answer):
        if n not in allowed:
            violations.append(f"ungrounded number: {n!r}")
    if _LEAK.search(answer):
        violations.append("prompt leak phrase")
    return violations


@dataclass
class GateResult:
    passed: bool
    matched_terms: list[str]
    coverage: float
    dense: float | None
    reason: str


def term_coverage(index: Index, question: str, chunk_text_terms: set[str]) -> tuple[list[str], float]:
    q_terms = list(dict.fromkeys(content_terms(question)))
    if not q_terms:
        return [], 0.0
    matched = [t for t in q_terms if t in chunk_text_terms]
    total = sum(index.bm25.term_idf(t) for t in q_terms)
    return matched, sum(index.bm25.term_idf(t) for t in matched) / total


def relevance_gate(index: Index, question: str, top: list[ScoredChunk]) -> GateResult:
    """Decide whether the retrieved context could plausibly answer the question at all."""
    if not top:
        return GateResult(False, [], 0.0, None, "nothing retrieved")
    best = top[0]
    pos = index.chunks.index(best.chunk)
    matched, coverage = term_coverage(index, question, index.term_sets[pos])
    dense = max((s.dense_score for s in top if s.dense_score is not None), default=None)
    if dense is not None and dense >= DENSE_PASS:
        return GateResult(True, matched, coverage, dense, "dense similarity")
    if coverage >= STRONG_COVERAGE:
        return GateResult(True, matched, coverage, dense, "strong term coverage")
    if len(matched) >= MANY_MATCHED_TERMS:
        return GateResult(True, matched, coverage, dense, "many matched terms")
    if len(matched) >= MIN_MATCHED_TERMS and coverage >= MIN_COVERAGE:
        return GateResult(True, matched, coverage, dense, "term coverage")
    return GateResult(False, matched, coverage, dense, "question terms not covered by the knowledge base")


def asks_for_contact_details(question: str) -> bool:
    return bool(_CONTACT_REQUEST.search(question))


def context_has_contact_details(texts: list[str]) -> bool:
    return any(_CONTACT_DATA.search(t) for t in texts)


_CITATION_ID = re.compile(r"[A-Za-z]+[-_ ]?\d+")


def normalize_citation(raw: object) -> str:
    """'[art-101]', 'ART 101', ' ART_101 ' → 'ART-101'. Anything else is returned stripped."""
    text = str(raw).strip().strip("[](){}.,;:'\"")
    m = _CITATION_ID.fullmatch(text)
    if not m:
        return text
    letters = re.match(r"[A-Za-z]+", text).group(0).upper()
    digits = re.search(r"\d+", text).group(0)
    return f"{letters}-{digits}"


@dataclass
class CitationReport:
    kept: list[str]
    invented: list[str]  # not an article in the knowledge base
    not_retrieved: list[str]  # real article, but not in the context the answer was built from
    duplicates: list[str]
    unsupported: list[str]  # retrieved, but the cited text shares no content with the answer

    @property
    def changed(self) -> bool:
        return bool(self.invented or self.not_retrieved or self.duplicates or self.unsupported)

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def verify_citations(
    citations: list[object],
    answer_text: str,
    retrieved: dict[str, str],
    kb_ids: set[str],
) -> CitationReport:
    """Clean a generator's citations.

    retrieved: article_id → concatenated text (title + chunks) of what was retrieved for it.
    Order of first appearance is kept. A citation survives only if it is a KB article,
    was retrieved for this question, is not a repeat, and its text overlaps the answer.
    """
    report = CitationReport([], [], [], [], [])
    answer_terms = set(content_terms(answer_text))
    need = min(2, len(answer_terms))
    seen: set[str] = set()
    for raw in citations:
        cid = normalize_citation(raw)
        if cid in seen:
            report.duplicates.append(cid)
            continue
        seen.add(cid)
        if cid not in kb_ids:
            report.invented.append(cid)
        elif cid not in retrieved:
            report.not_retrieved.append(cid)
        elif len(answer_terms & set(content_terms(retrieved[cid]))) < need:
            report.unsupported.append(cid)
        else:
            report.kept.append(cid)
    return report


def clean_citations(citations: list[str], allowed: list[str]) -> list[str]:
    """Keep IDs that were retrieved (and so exist in the KB), dedupe, keep first-seen order."""
    allowed_set = set(allowed)
    out: list[str] = []
    for c in map(normalize_citation, citations):
        if c in allowed_set and c not in out:
            out.append(c)
    return out


def finalize(
    question_id: str,
    generated: GeneratedAnswer,
    retrieved: list[ScoredChunk],
    kb_ids: set[str],
    max_chars: int,
) -> tuple[Answer, CitationReport]:
    """Normalise any generator output into a consistent Answer.

    - citations are verified (see verify_citations) and deduped
    - an answer with no surviving citation is not grounded, so it becomes a refusal
    - a refusal carries no citations and uses the fixed refusal text
    """
    by_article: dict[str, str] = {}
    for sc in retrieved:
        prev = by_article.get(sc.chunk.article_id, sc.chunk.title)
        by_article[sc.chunk.article_id] = f"{prev} {sc.chunk.text}"
    text = " ".join((generated.answer or "").split())
    report = verify_citations(generated.citations, text, by_article, kb_ids)
    if generated.insufficient_context or not report.kept or not text:
        return Answer(question_id=question_id, answer=REFUSAL, citations=[], insufficient_context=True), report
    if len(text) > max_chars:
        text = text[: max_chars - 1].rsplit(" ", 1)[0] + "…"
    return Answer(question_id=question_id, answer=text, citations=report.kept, insufficient_context=False), report
