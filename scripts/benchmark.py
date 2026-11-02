"""Benchmark the pipeline per mode: step latency and tokens/cost per step.

    python scripts/benchmark.py                       # every mode whose keys are set
    python scripts/benchmark.py --modes deterministic # offline only

Writes artifacts/benchmarks/benchmark.json (consumed by scripts/plot_charts.py).
LLM modes make real, billed API calls (5 questions per mode, a few cents in total).
"""

from __future__ import annotations

import argparse

import os
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.data_utils import load_questions  # noqa: E402
from app.output_utils import write_json  # noqa: E402

MODES = {
    "deterministic": {"use_llm": False},
    "openai-small": {"provider": "openai", "tier": "small", "key": "OPENAI_API_KEY"},
    "openai-best": {"provider": "openai", "tier": "best", "key": "OPENAI_API_KEY"},
    "gemini-small": {"provider": "gemini", "tier": "small", "key": "GEMINI_API_KEY"},
    "gemini-best": {"provider": "gemini", "tier": "best", "key": "GEMINI_API_KEY"},
    "anthropic-small": {"provider": "anthropic", "tier": "small", "key": "ANTHROPIC_API_KEY"},
    "anthropic-best": {"provider": "anthropic", "tier": "best", "key": "ANTHROPIC_API_KEY"},
}
STEP_GROUPS = {"rephrase": ["rephrase"], "retrieve": ["retrieve"], "generate": ["generate"]}


def run_mode(name: str, cfg: dict) -> dict | None:
    from app.graph import SupportPipeline
    from app.llm_utils import provider_model, PROVIDERS

    if cfg.get("key") and not os.getenv(cfg["key"]):
        return None
    os.environ["LLM_TIER"] = cfg.get("tier", "small")
    s = get_settings(llm_provider=cfg.get("provider"))
    s.use_llm = cfg.get("use_llm", True)
    s.use_local_models = False
    p = SupportPipeline(s)
    if s.use_llm and (p.llm is None or p.llm.specs[0].name != cfg["provider"]):
        return None
    questions = load_questions(s.questions_path)
    results = [p.run(q.id, q.question) for q in questions]  # sequential: clean per-step timings

    steps: dict[str, list[float]] = {}
    for r in results:
        for step, ms in r.log.step_timings_ms.items():
            steps.setdefault(step, []).append(ms)
    tokens: dict[str, int] = {"indexing": p.index_trace["prompt_tokens"]}
    cost: dict[str, float] = {"indexing": p.index_trace["est_cost_usd"] or 0.0}
    for r in results:
        for step, row in r.log.tokens_by_step.items():
            key = {"retrieve": "retrieval", "generate": "answering"}.get(step, step)
            tokens[key] = tokens.get(key, 0) + row["prompt_tokens"] + row["completion_tokens"]
            cost[key] = cost.get(key, 0.0) + row["est_cost_usd"]
    model = provider_model(PROVIDERS[cfg["provider"]]) if s.use_llm else "none (extractive)"
    return {
        "mode": name,
        "model": model,
        "embedder": p.index.embedder_name,
        "retrieval_mode": p.index.mode,
        "questions": len(results),
        "correct": sum(
            (r.answer.insufficient_context if r.answer.question_id == "Q5" else not r.answer.insufficient_context)
            for r in results
        ),
        "step_ms_mean": {k: round(statistics.mean(v), 2) for k, v in steps.items()},
        "total_ms_mean": round(statistics.mean(r.log.latency_ms for r in results), 1),
        "tokens": tokens,
        "est_cost_usd": {k: round(v, 8) for k, v in cost.items()},
        "est_cost_usd_total": round(sum(cost.values()), 8),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="*", default=list(MODES))
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts" / "benchmarks" / "benchmark.json")
    args = ap.parse_args()
    rows = []
    for name in args.modes:
        row = run_mode(name, MODES[name])
        if row is None:
            print(f"[skip] {name}: no API key")
            continue
        rows.append(row)
        print(
            f"[{name}] {row['model']}: {row['correct']}/{row['questions']} correct, "
            f"{row['total_ms_mean']} ms/question, ${row['est_cost_usd_total']:.6f}"
        )
    write_json(args.out, rows)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
