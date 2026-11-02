from app.config import get_settings
from app.data_utils import chunk_article, chunk_articles, load_articles
from app.schemas import Article


def test_chunks_preserve_metadata_and_have_unique_ids():
    articles = load_articles(get_settings().kb_path)
    chunks = chunk_articles(articles)
    assert {c.article_id for c in chunks} == {a.id for a in articles}
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    by_id = {a.id: a for a in articles}
    for c in chunks:
        assert c.title == by_id[c.article_id].title
        assert c.product == by_id[c.article_id].product
        assert c.text and c.text in by_id[c.article_id].content


def test_sentence_window_overlap():
    a = Article(id="A-1", title="t", product="p", content="One. Two. Three. Four. Five.")
    chunks = chunk_article(a, max_sentences=3, overlap=1)
    assert [c.text for c in chunks] == ["One. Two. Three.", "Three. Four. Five."]


def test_password_reset_question_retrieves_password_article(pipeline):
    result = pipeline.retrieve("T1", "I forgot my password, how do I reset it?")
    assert result.retrieved_chunks[0].article_id == "ART-101"


def test_retrieval_returns_top3_sorted(pipeline):
    result = pipeline.retrieve("T2", "API rate limit exceeded")
    scores = [c.score for c in result.retrieved_chunks]
    assert len(scores) == 3
    assert scores == sorted(scores, reverse=True)
    assert result.retrieved_chunks[0].article_id == "ART-301"
