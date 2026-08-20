from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path

from .agent import ResearchAgent
from .models import Identity, RunRequest, RunStatus, Source
from .store import SQLiteStore


TOPICS = (
    "Agent checkpoint", "RAG citation", "tenant isolation", "idempotent execution",
    "human approval", "conflict detection", "cost budget", "trace privacy",
    "sandbox policy", "horizontal scaling", "query decomposition",
)
VARIANTS = ("定义", "权衡", "风险", "验证", "上线条件")

FUNCTIONAL_TASKS = tuple(
    {"id": f"functional-{topic_index:02d}-{variant_index}", "topic": topic, "variant": variant}
    for topic_index, topic in enumerate(TOPICS, 1)
    for variant_index, variant in enumerate(VARIANTS, 1)
)

SECURITY_SCENARIOS = (
    "missing_scope", "cross_tenant_read", "cross_tenant_approval", "source_prompt_injection",
    "credential_redaction", "unsafe_url", "oversized_question", "oversized_corpus",
    "sandbox_command_denied", "sandbox_timeout_capped", "idempotency_payload_conflict",
    "unapproved_resume", "unknown_citation", "empty_identity", "raw_content_absent_from_trace",
)

FAULT_SCENARIOS = (
    "duplicate_delivery", "stale_writer", "restart_after_checkpoint", "approval_pause_resume",
    "empty_retrieval", "malformed_request", "database_busy", "tool_timeout",
    "worker_crash", "cost_budget_exhausted",
)


def run_suite(output: Path) -> dict[str, object]:
    trial_rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory() as directory:
        agent = ResearchAgent(SQLiteStore(Path(directory) / "eval.db"))
        for task in FUNCTIONAL_TASKS:
            source = Source(
                source_id="doc-1",
                title=task["topic"],
                url=f"https://example.test/{task['id']}",
                content=f"{task['topic']} 的{task['variant']}需要明确证据、边界和可重复验证。",
            )
            started = time.perf_counter()
            state = agent.submit(
                RunRequest(
                    question=f"说明 {task['topic']} 的{task['variant']}",
                    identity=Identity("eval-tenant", "eval-user"),
                    sources=(source,),
                    idempotency_key=task["id"],
                )
            )
            latency_ms = (time.perf_counter() - started) * 1000
            passed = state.status == RunStatus.COMPLETED and "[S1]" in (state.report or "")
            trial_rows.append(
                {
                    "task_id": task["id"],
                    "passed": passed,
                    "status": state.status.value,
                    "latency_ms": round(latency_ms, 3),
                    "cost_usd": state.cost_usd,
                    "steps": state.steps_used,
                }
            )
    passed = sum(bool(row["passed"]) for row in trial_rows)
    summary = {
        "suite_version": "business-v1",
        "tasks": len(trial_rows),
        "passed": passed,
        "success_rate": passed / len(trial_rows),
        "latency_ms_p50": round(statistics.median(float(row["latency_ms"]) for row in trial_rows), 3),
        "cost_usd_total": round(sum(float(row["cost_usd"]) for row in trial_rows), 6),
        "security_scenarios_defined": len(SECURITY_SCENARIOS),
        "fault_scenarios_defined": len(FAULT_SCENARIOS),
        "trials": trial_rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("evals/baseline.json"))
    args = parser.parse_args()
    print(json.dumps(run_suite(args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

