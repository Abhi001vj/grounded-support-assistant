"""Evaluation of the generated answers against expected sources.

    python evaluate.py                 # score artifacts/ from the last `python main.py`
    python evaluate.py --robustness    # also run the pipeline on eval/extra_questions.json

Expected sources are looked up by question text (normalised), then by question ID,
so renumbered or reworded question files still get scored where a label exists.
Unlabelled questions are reported, not failed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.config import ROOT
from app.output_utils import ANSWERS, EVAL_SUMMARY, RETRIEVAL_RESULTS, read_json, write_json

EXPECTED_PATH = ROOT / "eval" / "expected_sources.json"
EXTRA_PATH = ROOT / "eval" / "extra_questions.json"


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def load_expected(path: Path = EXPECTED_PATH) -> dict[str, object]:
    data = read_json(path)
    return {_norm(k): v for k, v in data.items() if not k.startswith("_")}


def score(
    questions: list[dict], answers: list[dict], retrievals: list[dict], expected: dict[str, object], kb_ids: set[str]
) -> dict:
    by_answer = {a["question_id"]: a for a in answers}
    by_retrieval = {r["question_id"]: r for r in retrievals}
    rows = []
    for q in questions:
        qid, text = q["id"], q["question"]
        label = expected.get(_norm(text), expected.get(_norm(qid)))
        ans = by_answer.get(qid)
        ret = by_retrieval.get(qid, {"retrieved_chunks": []})
        retrieved_articles = [c["article_id"] for c in ret["retrieved_chunks"]]
        row = {
            "question_id": qid,
            "question": text,
            "expected": label,
            "citations": ans["citations"] if ans else None,
            "insufficient_context": ans["insufficient_context"] if ans else None,
            "citations_valid": bool(ans) and all(c in kb_ids for c in ans["citations"]),
            "citations_unique": bool(ans) and len(ans["citations"]) == len(set(ans["citations"])),
        }
        if ans is None:
            row.update(type="missing", passed=False, reason="no answer produced")
        elif label is None:
            row.update(type="unlabelled", passed=None, reason="no expected label for this question")
        elif label == "unsupported":
            ok = ans["insufficient_context"] is True and ans["citations"] == []
            row.update(type="unsupported", passed=ok, reason="refused" if ok else "answered an unsupported question")
        else:
            hit = bool(set(label) & set(ans["citations"])) and not ans["insufficient_context"]
            row.update(
                type="supported",
                passed=hit and row["citations_valid"],
                retrieval_hit_at_3=bool(set(label) & set(retrieved_articles)),
                reason="expected source cited" if hit else "expected source not cited",
            )
        rows.append(row)

    def rate(kind: str, key: str = "passed") -> float | None:
        sel = [r for r in rows if r["type"] == kind]
        return round(sum(bool(r[key]) for r in sel) / len(sel), 3) if sel else None

    labelled = [r for r in rows if r["type"] in {"supported", "unsupported", "missing"}]
    return {
        "num_questions": len(rows),
        "num_labelled": len(labelled),
        "passed": sum(r["passed"] is True for r in labelled),
        "failed": sum(r["passed"] is False for r in labelled),
        "all_passed": all(r["passed"] for r in labelled) and bool(labelled),
        "metrics": {
            "supported_citation_hit_rate": rate("supported"),
            "supported_retrieval_recall_at_3": rate("supported", "retrieval_hit_at_3"),
            "unsupported_refusal_rate": rate("unsupported"),
            "citation_validity_rate": round(sum(r["citations_valid"] for r in rows) / len(rows), 3) if rows else None,
            "citation_uniqueness_rate": (
                round(sum(r["citations_unique"] for r in rows) / len(rows), 3) if rows else None
            ),
        },
        "results": rows,
    }


def run_robustness(pipeline) -> dict:
    """Run the live pipeline on held-out paraphrases and out-of-scope questions."""
    extra = read_json(EXTRA_PATH)
    questions = [{"id": x["id"], "question": x["question"]} for x in extra]
    expected = {_norm(x["question"]): x["expected"] for x in extra}
    results = [pipeline.run(q["id"], q["question"]) for q in questions]
    kb_ids = {a.id for a in pipeline.articles}
    return score(
        questions,
        [r.answer.model_dump() for r in results],
        [r.retrieval.model_dump() for r in results],
        expected,
        kb_ids,
    )


def evaluate_artifacts(artifacts_dir: Path, questions_path: Path, kb_path: Path) -> dict:
    questions = read_json(questions_path)
    kb_ids = {a["id"] for a in read_json(kb_path)}
    return score(
        questions,
        read_json(artifacts_dir / ANSWERS),
        read_json(artifacts_dir / RETRIEVAL_RESULTS),
        load_expected(),
        kb_ids,
    )


def main() -> int:
    from app.config import get_settings

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifacts", type=Path, default=None)
    parser.add_argument("--questions", type=Path, default=None)
    parser.add_argument("--kb", type=Path, default=None)
    parser.add_argument("--robustness", action="store_true", help="also run eval/extra_questions.json")
    args = parser.parse_args()
    s = get_settings(artifacts_dir=args.artifacts, questions_path=args.questions, kb_path=args.kb)

    summary = evaluate_artifacts(s.artifacts_dir, s.questions_path, s.kb_path)
    if args.robustness:
        from app.graph import SupportPipeline

        summary["robustness"] = run_robustness(SupportPipeline(s))
    write_json(s.artifacts_dir / EVAL_SUMMARY, summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in {"results", "robustness"}}, indent=2))
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
