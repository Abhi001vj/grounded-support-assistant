"""Loading the knowledge base and questions, sentence splitting, chunking and tokenising."""

from __future__ import annotations

import json
import re
from pathlib import Path

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from app.schemas import Article, Chunk, Question

# Splits after . ! ? when followed by whitespace and an uppercase letter or digit.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_TOKEN = re.compile(r"[a-z0-9]+")
# Extra words that carry no topic in support questions.
_EXTRA_STOP = {
    "tell",
    "know",
    "want",
    "need",
    "does",
    "did",
    "doing",
    "use",
    "used",
    "using",
    "just",
    "like",
    "came",
    "come",
    "comes",
    "work",
    "works",
    "get",
    "gets",
    "got",
    "make",
    "happen",
    "way",
    "please",
}
# Words that sklearn treats as stop words but that change meaning in support questions
# ("exactly once", "only the latest", "at least once").
_KEEP = {"once", "only", "least", "never", "always", "before", "after", "first", "last"}
STOP_WORDS = frozenset((ENGLISH_STOP_WORDS - _KEEP) | _EXTRA_STOP)
STEM_LEN = 5


def load_articles(path: Path) -> list[Article]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON list of articles")
    articles = [Article.model_validate(item) for item in data]
    ids = [a.id for a in articles]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate article ids")
    return articles


def load_questions(path: Path) -> list[Question]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON list of questions")
    return [Question.model_validate(item) for item in data]


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text.strip()) if s.strip()]


def chunk_article(article: Article, max_sentences: int = 3, overlap: int = 1) -> list[Chunk]:
    """Sentence windows of up to `max_sentences`, sharing `overlap` sentences with the previous window."""
    if max_sentences < 1 or not 0 <= overlap < max_sentences:
        raise ValueError("need max_sentences >= 1 and 0 <= overlap < max_sentences")
    sentences = split_sentences(article.content)
    if not sentences:
        return []
    step = max_sentences - overlap
    windows: list[list[str]] = []
    start = 0
    while True:
        windows.append(sentences[start : start + max_sentences])
        if start + max_sentences >= len(sentences):
            break
        start += step
    return [
        Chunk(
            chunk_id=f"{article.id}-C{n}",
            article_id=article.id,
            title=article.title,
            product=article.product,
            text=" ".join(window),
        )
        for n, window in enumerate(windows, start=1)
    ]


def chunk_articles(articles: list[Article], max_sentences: int = 3, overlap: int = 1) -> list[Chunk]:
    return [c for a in articles for c in chunk_article(a, max_sentences, overlap)]


def stem(token: str) -> str:
    """Truncation stemming: the first five characters. 'verifying', 'verification' -> 'verif'."""
    return token[:STEM_LEN]


def content_terms(text: str) -> list[str]:
    """Lower-cased, stop-word-free, stemmed tokens (order kept, duplicates kept)."""
    return [stem(t) for t in _TOKEN.findall(text.lower()) if t not in STOP_WORDS and (len(t) > 1 or t.isdigit())]
