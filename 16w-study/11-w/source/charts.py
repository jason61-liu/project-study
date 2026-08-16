"""Render the three benchmark charts with matplotlib (headless Agg backend)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

# Palette shared with the fireworks diagrams (blue / orange / green / purple).
BLUE = "#3b82f6"
ORANGE = "#f97316"
GREEN = "#16a34a"
PURPLE = "#8b5cf6"
TEAL = "#14b8a6"
GRAY = "#6b7280"

ChartRow = dict[str, Any]


def _style_axes(ax: Any) -> None:
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _concurrency_rows(rows: list[ChartRow]) -> list[ChartRow]:
    return sorted(
        [r for r in rows if str(r["workload"]["name"]).startswith("concurrency-")],
        key=lambda r: r["workload"]["concurrency"],
    )


def plot_concurrency_throughput(rows: list[ChartRow], path: Path) -> None:
    sweep = _concurrency_rows(rows)
    xs = [r["workload"]["concurrency"] for r in sweep]
    ys = [r["throughput_tokens_s"] for r in sweep]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(xs, ys, marker="o", color=BLUE, linewidth=2.0, markersize=6)
    ax.set_xlabel("Concurrency (in-flight requests)")
    ax.set_ylabel("Throughput (output tokens / s)")
    ax.set_title("Concurrency → Throughput")
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_concurrency_tail_latency(rows: list[ChartRow], path: Path) -> None:
    sweep = _concurrency_rows(rows)
    xs = [r["workload"]["concurrency"] for r in sweep]
    p50 = [r["e2e_ms"]["p50"] for r in sweep]
    p95 = [r["e2e_ms"]["p95"] for r in sweep]
    p99 = [r["e2e_ms"]["p99"] for r in sweep]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(xs, p50, marker="o", color=TEAL, linewidth=2.0, label="P50")
    ax.plot(xs, p95, marker="s", color=ORANGE, linewidth=2.0, label="P95")
    ax.plot(xs, p99, marker="^", color=GRAY, linewidth=2.0, label="P99")
    ax.set_xlabel("Concurrency (in-flight requests)")
    ax.set_ylabel("End-to-end latency (ms)")
    ax.set_title("Concurrency → Tail Latency")
    ax.legend(frameon=False)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_quality_cost(agent_rows: list[ChartRow], path: Path) -> None:
    points = [
        r for r in agent_rows
        if r.get("success_rate") and r.get("cost_per_success_microusd") is not None
    ]
    xs = [r["success_rate"] for r in points]
    ys = [r["cost_per_success_microusd"] for r in points]
    labels = [r["config"]["output_budget"] for r in points]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(xs, ys, marker="o", color=GREEN, linewidth=2.0, markersize=7)
    for x, y, budget in zip(xs, ys, labels):
        ax.annotate(f"{budget}", (x, y), textcoords="offset points", xytext=(8, 4), color=GRAY, fontsize=9)
    ax.set_xlabel("Success rate (quality / goodput fraction)")
    ax.set_ylabel("Cost per successful task (microUSD)")
    ax.set_title("Quality → Cost per Successful Task")
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
