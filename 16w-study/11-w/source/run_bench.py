"""Run the week-11 async API benchmark end to end and write artifacts."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from agent_link import quality_sweep, run_agent_sweep
from benchmark import run_config
from charts import (
    plot_concurrency_tail_latency,
    plot_concurrency_throughput,
    plot_quality_cost,
)
from model_sim import LLMServer, ModelProfile, Pricing
from workload import concurrency_sweep, length_sweep

JSON = dict[str, Any]


def build_server(args: argparse.Namespace) -> LLMServer:
    profile = ModelProfile(
        prefill_ms_per_token=args.prefill_ms,
        decode_ms_per_token=args.decode_ms,
        prefill_concurrency=args.prefill_concurrency,
        server_concurrency=args.server_concurrency,
        jitter=args.jitter,
        seed=args.seed,
    )
    return LLMServer(profile, Pricing())


def _pct(summary: JSON, key: str) -> str:
    value = summary.get(key)
    return "-" if value is None else f"{value:.2f}"


def write_summary(output: Path, benchmark_rows: list[JSON], agent_rows: list[JSON]) -> None:
    lines = [
        "# 第 11 周 API Benchmark 结果摘要",
        "",
        "> 数据源：本地模拟 LLM 服务的异步压测（`run_bench.py`）。",
        "",
        "## 1. 并发扫描（固定 input=1024 / output=256）",
        "",
        "| 并发 | 吞吐 tokens/s | 吞吐 req/s | TTFT mean ms | E2E P50 ms | E2E P95 ms | E2E P99 ms | Decode Peak |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in benchmark_rows:
        if not str(row["workload"]["name"]).startswith("concurrency-"):
            continue
        lines.append(
            f"| {row['workload']['concurrency']} "
            f"| {row['throughput_tokens_s']:.1f} "
            f"| {row['throughput_req_s']:.2f} "
            f"| {_pct(row['ttft_ms'], 'mean')} "
            f"| {_pct(row['e2e_ms'], 'p50')} "
            f"| {_pct(row['e2e_ms'], 'p95')} "
            f"| {_pct(row['e2e_ms'], 'p99')} "
            f"| {row['server']['decode_peak']} |"
        )
    lines.extend([
        "",
        "## 2. 长度扫描（固定并发 = 4）",
        "",
        "| 名称 | 输入 tokens | 输出 tokens | TTFT mean ms | TPOT mean ms | E2E P95 ms | 总成本 microUSD |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in benchmark_rows:
        if str(row["workload"]["name"]).startswith("concurrency-"):
            continue
        lines.append(
            f"| {row['workload']['name']} "
            f"| {row['workload']['input_tokens']} "
            f"| {row['workload']['output_tokens']} "
            f"| {_pct(row['ttft_ms'], 'mean')} "
            f"| {_pct(row['tpot_ms'], 'mean')} "
            f"| {_pct(row['e2e_ms'], 'p95')} "
            f"| {row['total_cost_microusd']:.2f} |"
        )
    lines.extend([
        "",
        "## 3. Agent 端到端（质量 → 单位成功任务成本）",
        "",
        "| 输出预算 | 成功率 | 平均 Attempt | 每成功成本 microUSD | 每成功 tokens | 模型每次调用成本 microUSD |",
        "|---:|---:|---:|---:|---:|---:|",
    ])
    for row in agent_rows:
        cost = row["cost_per_success_microusd"]
        cost_str = "-" if cost is None else f"{cost:.2f}"
        lines.append(
            f"| {row['config']['output_budget']} "
            f"| {row['success_rate']:.1%} "
            f"| {row['mean_attempts']:.2f} "
            f"| {cost_str} "
            f"| {row['tokens_per_success'] if row['tokens_per_success'] is not None else '-'} "
            f"| {row['model_per_call_cost_microusd']:.2f} |"
        )
    lines.extend(["", "图表见 `charts/`；完整数据见 `benchmark.json` 与 `timelines.jsonl`。", ""])
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")


async def run(output: Path, args: argparse.Namespace) -> JSON:
    output.mkdir(parents=True, exist_ok=True)
    charts_dir = output / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    server = build_server(args)

    # 1. Model-call benchmark: concurrency sweep + length sweep.
    benchmark_rows: list[JSON] = []
    timeline_lines: list[JSON] = []
    for workload in concurrency_sweep() + length_sweep():
        server.reset_peaks()
        result = await run_config(server, workload)
        benchmark_rows.append(result.aggregate(server))
        for sample in result.samples:
            timeline_lines.append(
                {
                    "workload": workload.name,
                    "concurrency": workload.concurrency,
                    **sample.as_dict(server.pricing),
                }
            )

    # 2. Agent end-to-end link: model metrics -> success / cost per success.
    agent_rows = await run_agent_sweep(server, quality_sweep())

    # 3. Charts.
    plot_concurrency_throughput(benchmark_rows, charts_dir / "concurrency-throughput.png")
    plot_concurrency_tail_latency(benchmark_rows, charts_dir / "concurrency-tail-latency.png")
    plot_quality_cost(agent_rows, charts_dir / "quality-cost.png")

    report: JSON = {
        "schema_version": "1.0",
        "model_profile": {
            "prefill_ms_per_token": server.profile.prefill_ms_per_token,
            "decode_ms_per_token": server.profile.decode_ms_per_token,
            "prefill_concurrency": server.profile.prefill_concurrency,
            "server_concurrency": server.profile.server_concurrency,
            "jitter": server.profile.jitter,
        },
        "pricing": {
            "input_microusd_per_token": server.pricing.input_microusd_per_token,
            "output_microusd_per_token": server.pricing.output_microusd_per_token,
        },
        "benchmark": benchmark_rows,
        "agent_link": agent_rows,
    }

    (output / "benchmark.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "timelines.jsonl").write_text(
        "\n".join(json.dumps(line, ensure_ascii=False) for line in timeline_lines) + "\n",
        encoding="utf-8",
    )
    write_summary(output, benchmark_rows, agent_rows)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "artifacts")
    parser.add_argument("--prefill-ms", type=float, default=0.02)
    parser.add_argument("--decode-ms", type=float, default=1.0)
    parser.add_argument("--prefill-concurrency", type=int, default=4)
    parser.add_argument("--server-concurrency", type=int, default=8)
    parser.add_argument("--jitter", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    report = asyncio.run(run(args.output, args))
    print(json.dumps(
        {
            "benchmark_configs": len(report["benchmark"]),
            "agent_configs": len(report["agent_link"]),
            "output": str(args.output),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
