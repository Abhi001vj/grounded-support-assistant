"""Validate inputs and generated artifacts. Exit code 0 = all checks passed.

python validate.py              # checks artifacts/ produced by `python main.py`
python validate.py --run        # runs the pipeline first, then validates
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.config import get_settings
from app.output_utils import (
    ANSWERS,
    EVAL_SUMMARY,
    INDEX_PREVIEW,
    RETRIEVAL_RESULTS,
    RUN_LOG,
)

UNSUPPORTED_SAMPLE_TEXT = "can you tell me the ceo's direct email address?"


class Checker:
    def __init__(self):
        self.failures: list[str] = []
        self.passes = 0

    def check(self, ok: bool, message: str) -> bool:
        if ok:
            self.passes += 1
            print(f"  PASS  {message}")
        else:
            self.failures.append(message)
            print(f"  FAIL  {message}")
        return ok


def load(path: Path, c: Checker):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        c.check(False, f"{path.name} is valid JSON ({exc})")
        return None


def validate(settings) -> Checker:
    c = Checker()
    out = settings.artifacts_dir

    print("inputs")
    for p in (settings.kb_path, settings.questions_path):
        c.check(p.is_file(), f"input exists: {p.name}")
    kb = load(settings.kb_path, c) if settings.kb_path.is_file() else None
    questions = load(settings.questions_path, c) if settings.questions_path.is_file() else None
    kb_ids = {a["id"] for a in kb or []}
    c.check(
        bool(kb) and all({"id", "title", "product", "content"} <= set(a) for a in kb),
        "kb articles have id/title/product/content",
    )
    c.check(bool(questions) and all({"id", "question"} <= set(q) for q in questions), "questions have id/question")
    question_ids = [q["id"] for q in questions or []]

    print("artifacts")
    names = [INDEX_PREVIEW, RETRIEVAL_RESULTS, ANSWERS, EVAL_SUMMARY, RUN_LOG]
    for name in names:
        c.check((out / name).is_file(), f"artifact exists: {name}")
    data = {n: load(out / n, c) for n in names if n != RUN_LOG and (out / n).is_file()}
    for n, d in data.items():
        if d is not None:
            c.check(True, f"{n} is valid JSON")

    preview = data.get(INDEX_PREVIEW) or []
    keys = {"chunk_id", "article_id", "title", "product", "text"}
    c.check(
        bool(preview) and all(keys <= set(r) and all(isinstance(r[k], str) for k in keys) for r in preview),
        "index_preview records have chunk_id/article_id/title/product/text (strings)",
    )
    c.check(all(r["article_id"] in kb_ids for r in preview), "index_preview article_ids exist in KB")
    c.check(len({r["chunk_id"] for r in preview}) == len(preview), "chunk_ids are unique")

    retrievals = data.get(RETRIEVAL_RESULTS) or []
    c.check(
        [r.get("question_id") for r in retrievals] == question_ids,
        "retrieval_results has one entry per question, in order",
    )
    ok = all(
        isinstance(r.get("question"), str)
        and isinstance(r.get("retrieved_chunks"), list)
        and 0 < len(r["retrieved_chunks"]) <= 3
        and all(
            {"chunk_id", "article_id", "score"} <= set(ch) and isinstance(ch["score"], (int, float))
            for ch in r["retrieved_chunks"]
        )
        for r in retrievals
    )
    c.check(ok, "retrieval_results entries have question + 1..3 chunks with chunk_id/article_id/score")

    answers = data.get(ANSWERS) or []
    c.check([a.get("question_id") for a in answers] == question_ids, "answers has one entry per question, in order")
    c.check(
        all(
            isinstance(a.get("question_id"), str)
            and isinstance(a.get("answer"), str)
            and a["answer"].strip()
            and isinstance(a.get("citations"), list)
            and isinstance(a.get("insufficient_context"), bool)
            for a in answers
        ),
        "every answer has question_id/answer/citations/insufficient_context with correct types",
    )
    bad = [(a["question_id"], x) for a in answers for x in a.get("citations", []) if x not in kb_ids]
    c.check(not bad, f"every citation refers to a KB article{'' if not bad else f' (bad: {bad})'}")
    c.check(all(len(a["citations"]) == len(set(a["citations"])) for a in answers), "citations are unique per answer")
    c.check(all(a["citations"] == [] for a in answers if a["insufficient_context"]), "refusals carry no citations")
    c.check(
        all(a["citations"] for a in answers if not a["insufficient_context"]),
        "answered questions cite at least one article",
    )

    # Q5 by ID; also by its text in case the IDs were renumbered.
    q5 = [a for a in answers if a["question_id"] == "Q5"]
    by_text = {q["id"] for q in questions or [] if q["question"].strip().lower() == UNSUPPORTED_SAMPLE_TEXT}
    q5 += [a for a in answers if a["question_id"] in by_text and a not in q5]
    if q5:
        c.check(
            all(a["insufficient_context"] is True for a in q5),
            "unsupported sample question Q5 has insufficient_context: true",
        )
    else:
        print("  SKIP  Q5 not present in questions.json (questions were replaced)")

    if (out / RUN_LOG).is_file():
        lines = [line for line in (out / RUN_LOG).read_text(encoding="utf-8").splitlines() if line.strip()]
        try:
            records = [json.loads(line) for line in lines]
            c.check(True, "run_log.jsonl lines are valid JSON")
        except json.JSONDecodeError as exc:
            records = []
            c.check(False, f"run_log.jsonl lines are valid JSON ({exc})")
        c.check(
            sorted(r.get("question_id") for r in records) == sorted(question_ids), "run_log has one record per question"
        )
        c.check(
            all(
                isinstance(r.get("retrieval_chunk_ids"), list)
                and isinstance(r.get("used_fallback"), bool)
                and isinstance(r.get("timestamp"), str)
                and "T" in r["timestamp"]
                for r in records
            ),
            "run_log records have retrieval_chunk_ids/used_fallback/timestamp (ISO-8601)",
        )

    summary = data.get(EVAL_SUMMARY) or {}
    c.check(summary.get("all_passed") is True, "eval_summary: all labelled sample questions passed")
    return c


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument("--questions", type=Path)
    parser.add_argument("--kb", type=Path)
    parser.add_argument("--run", action="store_true", help="run the pipeline before validating")
    args = parser.parse_args()
    settings = get_settings(artifacts_dir=args.artifacts, questions_path=args.questions, kb_path=args.kb)
    if args.run:
        from main import run_pipeline

        run_pipeline(settings)
    c = validate(settings)
    print(f"\n{c.passes} passed, {len(c.failures)} failed")
    return 1 if c.failures else 0


if __name__ == "__main__":
    sys.exit(main())
