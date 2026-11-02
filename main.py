"""Single entry point.

    python main.py                  # run the full pipeline on questions.json → artifacts/
    python main.py --no-llm         # force the deterministic fallback even if API keys exist
    python main.py --serve          # start the FastAPI service on http://127.0.0.1:8000

Mode selection is automatic: an LLM is used when any of OPENAI_API_KEY, GEMINI_API_KEY,
ANTHROPIC_API_KEY or OPENROUTER_API_KEY is set; otherwise the deterministic fallback runs.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from app.config import get_settings
from app.data_utils import load_questions
from app.output_utils import (
    ANSWERS,
    EVAL_SUMMARY,
    INDEX_PREVIEW,
    RETRIEVAL_RESULTS,
    RUN_INFO,
    RUN_LOG,
    write_json,
    write_jsonl,
)


def run_pipeline(settings, robustness: bool = True) -> dict:
    from app.graph import SupportPipeline
    from evaluate import evaluate_artifacts, run_robustness

    out = settings.artifacts_dir
    pipeline = SupportPipeline(settings)
    info = pipeline.describe()
    print(f"[mode] {json.dumps(info)}")

    # 1. index preview
    write_json(out / INDEX_PREVIEW, pipeline.index.preview())

    # 2-3. retrieval + grounded answers (questions run concurrently, results keep input order)
    questions = load_questions(settings.questions_path)
    results = asyncio.run(pipeline.arun_many([(q.id, q.question) for q in questions]))
    write_json(out / RETRIEVAL_RESULTS, [r.retrieval for r in results])
    write_json(out / ANSWERS, [r.answer for r in results])

    # 7. observability: one line per question
    write_jsonl(out / RUN_LOG, [r.log for r in results])
    write_json(out / RUN_INFO, {**info, "debug": {r.answer.question_id: r.debug for r in results}})

    # 4. evaluation
    summary = evaluate_artifacts(out, settings.questions_path, settings.kb_path)
    if robustness:
        summary["robustness"] = run_robustness(pipeline)
    summary["mode"] = info
    write_json(out / EVAL_SUMMARY, summary)

    for r in results:
        a = r.answer
        tag = "REFUSED " if a.insufficient_context else "ANSWERED"
        print(f"[{a.question_id}] {tag} {a.citations} via {r.log.provider}: {a.answer}")
    rob = summary.get("robustness")
    print(
        f"[eval] sample: {summary['passed']}/{summary['num_labelled']} passed"
        + (f" | robustness: {rob['passed']}/{rob['num_labelled']} passed" if rob else "")
    )
    print(f"[done] artifacts written to {out}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kb", type=Path, help="knowledge base JSON (default kb/articles.json)")
    parser.add_argument("--questions", type=Path, help="questions JSON (default questions.json)")
    parser.add_argument("--artifacts", type=Path, help="output directory (default artifacts/)")
    parser.add_argument("--no-llm", action="store_true", help="force the deterministic fallback")
    parser.add_argument("--provider", choices=["openai", "gemini", "anthropic", "openrouter"])
    parser.add_argument("--local-models", action="store_true", help="enable BERT QA / sentence-transformers")
    parser.add_argument("--retrieval", choices=["auto", "lexical", "hybrid"])
    parser.add_argument("--skip-robustness", action="store_true", help="skip eval/extra_questions.json")
    parser.add_argument("--serve", action="store_true", help="run the FastAPI service instead")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s"
    )

    settings = get_settings(
        kb_path=args.kb,
        questions_path=args.questions,
        artifacts_dir=args.artifacts,
        llm_provider=args.provider,
        retrieval_mode=args.retrieval,
    )
    if args.no_llm:
        settings.use_llm = False
    if args.local_models:
        settings.use_local_models = True

    if args.serve:
        import uvicorn

        from app.api import create_app

        uvicorn.run(create_app(settings), host=args.host, port=args.port)
        return 0

    summary = run_pipeline(settings, robustness=not args.skip_robustness)
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
