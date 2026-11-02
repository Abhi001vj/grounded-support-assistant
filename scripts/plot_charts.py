"""Render README charts (static PNGs) from benchmark and load-test outputs.

    python scripts/plot_charts.py

Inputs:  artifacts/benchmarks/benchmark.json, artifacts/loadtest/run_stats_history.csv
Outputs: docs/images/step_latency.png, tokens_by_step.png, loadtest_throughput.png, loadtest_latency.png

Palette: the dataviz reference categorical slots 1-4 (validated for CVD separation). Two slots
sit below 3:1 contrast, so every chart is paired with a value table in the README.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "images"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # fixed order, never cycled
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax, title: str, xlabel: str):
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def stacked_bars(rows, series, values, title, xlabel, path, fmt):
    labels = [f"{r['mode']}\n{r['model']}" for r in rows]
    fig, ax = plt.subplots(figsize=(9, 0.75 * len(rows) + 1.8), dpi=150)
    left = [0.0] * len(rows)
    for i, name in enumerate(series):
        vals = [values(r, name) for r in rows]
        ax.barh(
            labels, vals, left=left, height=0.5, color=SERIES[i], label=name, edgecolor=SURFACE, linewidth=2
        )  # 2px surface gap between segments
        left = [a + b for a, b in zip(left, vals)]
    for y, total in enumerate(left):  # direct label: the total only
        ax.text(total, y, f"  {fmt(total)}", va="center", ha="left", color=INK, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, max(left + [1]) * 1.18)
    _style(ax, title, xlabel)
    ax.legend(
        ncol=len(series),
        loc="upper left",
        bbox_to_anchor=(0, -0.12 - 0.02 * len(rows)),
        frameon=False,
        fontsize=9,
        labelcolor=INK2,
    )
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")


def benchmark_charts():
    path = ROOT / "artifacts" / "benchmarks" / "benchmark.json"
    if not path.exists():
        print("no benchmark.json; run `make benchmark` first")
        return
    rows = json.loads(path.read_text())

    def step_ms(r, name):
        s = r["step_ms_mean"]
        if name == "guards + gate":
            return sum(v for k, v in s.items() if k not in {"rephrase", "retrieve", "generate"})
        return s.get(name, 0.0)

    stacked_bars(
        rows,
        ["rephrase", "retrieve", "generate", "guards + gate"],
        step_ms,
        "Mean latency per question, by pipeline step",
        "milliseconds (sample questions, sequential)",
        OUT / "step_latency.png",
        lambda v: f"{v:,.0f} ms",
    )

    llm_rows = [r for r in rows if r["mode"] != "deterministic"]
    if llm_rows:
        stacked_bars(
            llm_rows,
            ["indexing", "rephrase", "retrieval", "answering"],
            lambda r, n: r["tokens"].get(n, 0),
            "Tokens per run of the 5 sample questions, by step",
            "tokens (prompt + completion; indexing = document embeddings)",
            OUT / "tokens_by_step.png",
            lambda v: f"{v:,.0f} tok",
        )


def loadtest_charts():
    path = ROOT / "artifacts" / "loadtest" / "run_stats_history.csv"
    if not path.exists():
        print("no load-test history; run `make loadtest` first")
        return
    rows = [r for r in csv.DictReader(path.open()) if r["Name"] == "Aggregated"]
    if not rows:
        return
    t0 = int(rows[0]["Timestamp"])
    t = [int(r["Timestamp"]) - t0 for r in rows]

    def num(r, k):
        try:
            return float(r[k])
        except (ValueError, KeyError):
            return float("nan")

    fig, ax = plt.subplots(figsize=(9, 3.4), dpi=150)
    ax.plot(t, [num(r, "Requests/s") for r in rows], color=SERIES[0], linewidth=2)
    ax2_users = [num(r, "User Count") for r in rows]
    _style(ax, f"Throughput (requests/s), ramping to {int(max(ax2_users))} users", "seconds since start")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(OUT / "loadtest_throughput.png", facecolor=SURFACE)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 3.4), dpi=150)
    for i, (col, label) in enumerate([("50%", "p50"), ("95%", "p95")]):
        ys = [num(r, col) for r in rows]
        ax.plot(t, ys, color=SERIES[i], linewidth=2, label=label)
        ax.text(t[-1], ys[-1], f" {label}", color=INK, fontsize=9, va="center")
    _style(ax, "Response time across all endpoints (ms)", "seconds since start")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "loadtest_latency.png", facecolor=SURFACE)
    plt.close(fig)
    print("wrote docs/images/loadtest_throughput.png, docs/images/loadtest_latency.png")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    benchmark_charts()
    loadtest_charts()
    return 0


if __name__ == "__main__":
    sys.exit(main())
