from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .bootstrap import load_week15

load_week15()

from deep_research_agent.agent import ResearchAgent  # noqa: E402
from deep_research_agent.eval_harness import FUNCTIONAL_TASKS  # noqa: E402
from deep_research_agent.models import Identity, RunRequest, RunStatus, Source  # noqa: E402
from deep_research_agent.store import SQLiteStore  # noqa: E402


@dataclass(frozen=True)
class TrialResult:
    task_id: str
    trial: int
    passed: bool
    latency_ms: float
    cost_usd: float
    escalated: bool
    final_status: str
    quality_score: float
    trace_span_count: int
    input_tokens: int
    output_tokens: int


def request_for(task: dict[str, str], *, compact: bool = False, approval: bool = False) -> RunRequest:
    topic = task["topic"]
    full = f"经过验证的材料说明：{topic} 的{task['variant']}需要明确证据、边界和可重复验证。"
    content = full[:12] if compact else full
    return RunRequest(
        question=f"说明 {topic} 的{task['variant']}",
        identity=Identity("portfolio-tenant", "portfolio-user"),
        sources=(Source("doc-1", "授权研究资料", f"https://example.test/{task['id']}", content),),
        require_approval=approval,
    )


def grade(state: object, topic: str) -> bool:
    report = getattr(state, "report", None) or ""
    excerpts = " ".join(item["excerpt"] for item in getattr(state, "evidence", []))
    return getattr(state, "status", None) == RunStatus.COMPLETED and "[S1]" in report and topic in excerpts


def run_trials(trials_per_task: int = 3) -> tuple[list[TrialResult], dict[str, float | str]]:
    rows: list[TrialResult] = []
    with tempfile.TemporaryDirectory() as directory:
        agent = ResearchAgent(SQLiteStore(Path(directory) / "trials.db"))
        for task in FUNCTIONAL_TASKS:
            for trial in range(1, trials_per_task + 1):
                started = time.perf_counter()
                state = agent.submit(request_for(task))
                rows.append(
                    TrialResult(
                        task["id"], trial, grade(state, task["topic"]),
                        (time.perf_counter() - started) * 1000,
                        state.cost_usd,
                        False,
                        state.status.value,
                        1.0 if grade(state, task["topic"]) else 0.0,
                        len(agent.store.spans("portfolio-tenant", state.run_id)),
                        0,
                        0,
                    )
                )
        agent.store.close()
    task_rates = []
    for task in FUNCTIONAL_TASKS:
        selected = [row.passed for row in rows if row.task_id == task["id"]]
        task_rates.append(sum(selected) / len(selected))
    latencies = [row.latency_ms for row in rows]
    summary = {
        "tasks": float(len(FUNCTIONAL_TASKS)),
        "trials": float(len(rows)),
        "trials_per_task": float(trials_per_task),
        "success_rate_mean": statistics.mean(task_rates),
        "success_rate_stddev": statistics.pstdev(task_rates),
        "latency_ms_mean": statistics.mean(latencies),
        "latency_ms_stddev": statistics.pstdev(latencies),
        "latency_ms_p95": sorted(latencies)[int(len(latencies) * 0.95) - 1],
        "cost_usd_mean": statistics.mean(row.cost_usd for row in rows),
        "input_tokens_total": float(sum(row.input_tokens for row in rows)),
        "output_tokens_total": float(sum(row.output_tokens for row in rows)),
        "token_note": "zero because the deterministic extractive baseline makes no model calls",
    }
    return rows, summary


def _measure_variant(
    name: str,
    transform: Callable[[dict[str, str]], RunRequest],
    dual_pass: bool = False,
) -> dict[str, object]:
    passed = 0
    escalated = 0
    total_cost = 0.0
    latencies: list[float] = []
    with tempfile.TemporaryDirectory() as directory:
        agent = ResearchAgent(SQLiteStore(Path(directory) / f"{name}.db"))
        for task in FUNCTIONAL_TASKS:
            started = time.perf_counter()
            state = agent.submit(transform(task))
            states = [state]
            if dual_pass:
                states.append(agent.submit(transform(task)))
            latencies.append((time.perf_counter() - started) * 1000)
            total_cost += sum(item.cost_usd for item in states)
            if any(item.status == RunStatus.WAITING_APPROVAL for item in states):
                escalated += 1
            if all(grade(item, task["topic"]) for item in states):
                passed += 1
        agent.store.close()
    return {
        "variant": name,
        "tasks": len(FUNCTIONAL_TASKS),
        "success_rate": passed / len(FUNCTIONAL_TASKS),
        "latency_ms_mean": statistics.mean(latencies),
        "cost_usd_total": total_cost,
        "human_escalation_rate": escalated / len(FUNCTIONAL_TASKS),
    }


def run_ablations() -> dict[str, list[dict[str, object]]]:
    base = lambda task: request_for(task)  # noqa: E731
    compact = lambda task: request_for(task, compact=True)  # noqa: E731
    routed = lambda task: request_for(task, approval=task["variant"] in {"风险", "上线条件"})  # noqa: E731
    return {
        "architecture": [
            _measure_variant("single-workflow", base),
            _measure_variant("dual-pass-agent-proxy", base, dual_pass=True),
        ],
        "context": [
            _measure_variant("full-evidence", base),
            _measure_variant("unsafe-hard-truncation", compact),
        ],
        "model_routing": [
            _measure_variant("extractive-auto", base),
            _measure_variant("risk-aware-human-route", routed),
        ],
        "optimization": [
            _measure_variant("single-source", base),
            _measure_variant(
                "dedup-pressure",
                lambda task: RunRequest(
                    **{
                        **request_for(task).__dict__,
                        "sources": request_for(task).sources * 10,
                    }
                ),
            ),
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/strengthening.json"))
    args = parser.parse_args()
    rows, summary = run_trials()
    payload = {
        "suite_version": "portfolio-v1",
        "generated_at": "2026-08-20",
        "summary": summary,
        "ablations": run_ablations(),
        "trials": [row.__dict__ for row in rows],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"summary": summary, "ablations": payload["ablations"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
