"""Writing artifacts: JSON files and the JSONL run log."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel

INDEX_PREVIEW = "index_preview.json"
RETRIEVAL_RESULTS = "retrieval_results.json"
ANSWERS = "answers.json"
EVAL_SUMMARY = "eval_summary.json"
RUN_LOG = "run_log.jsonl"
RUN_INFO = "run_info.json"


def _plain(obj: Any) -> Any:
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    if isinstance(obj, (list, tuple)):
        return [_plain(o) for o in obj]
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    return obj


def write_json(path: Path, data: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_plain(data), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def write_jsonl(path: Path, records: Iterable[Any]) -> Path:
    """Overwrites: one run of the pipeline → one line per question."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(_plain(r), ensure_ascii=False) + "\n")
    return path


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))
