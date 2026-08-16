"""Run all Week 10 failure injections, release drill and dashboard export."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import subprocess
import sys
import time

from budget import BudgetPolicy, ResourcePolicy
from dashboard import build_dashboard
from faults import Fault, FaultInjector
from release_drill import run_release_drill, verify_trace_version_combinations
from runtime import AgentRuntime, RuntimeConfig, hard_exit_process
from state_store import StateStore
from telemetry import Telemetry
from versioning import BASELINE_VERSIONS, CANDIDATE_VERSIONS


def config() -> RuntimeConfig:
    return RuntimeConfig(
        model=ResourcePolicy(timeout_s=0.15, concurrency=2),
        tool=ResourcePolicy(timeout_s=0.10, concurrency=1),
        global_budget=BudgetPolicy(
            wall_time_s=8.0,
            max_steps=8,
            max_tokens=2_000,
            max_cost_microusd=20_000,
        ),
        lease_s=0.08,
        max_attempts=4,
    )


def clean_owned_outputs(output: Path) -> None:
    for name in (
        "state.db", "state.db-shm", "state.db-wal", "traces.jsonl", "metrics.jsonl",
        "logs.jsonl", "dashboard.json", "dashboard.md", "fault-results.json",
        "release-drill.json", "trace-version-verification.json", "version-manifests.json",
    ):
        (output / name).unlink(missing_ok=True)


async def run_fault_scenario(
    runtime: AgentRuntime,
    *,
    message_id: str,
    fault_operation: str | None = None,
    fault: Fault | None = None,
) -> dict:
    if fault_operation and fault:
        runtime.faults.add(fault_operation, fault)
    task_id, inserted = runtime.submit(
        message_id=message_id,
        tenant_id="tenant-a",
        input_data={"request_type": "publish_report"},
        versions=CANDIDATE_VERSIONS,
    )
    await runtime.run_until_idle(limit=8)
    task = runtime.store.get_task(task_id)
    return {
        "task_id": task_id,
        "inserted": inserted,
        "state": task.state,
        "attempts": task.attempt,
        "last_error": task.last_error,
    }


async def parent_run(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    clean_owned_outputs(output)
    store = StateStore(output / "state.db")
    telemetry = Telemetry(output)
    runtime = AgentRuntime(store, telemetry, config=config())
    results: dict[str, dict] = {}

    results["429"] = await run_fault_scenario(
        runtime,
        message_id="fault-429",
        fault_operation="model",
        fault=Fault("429"),
    )
    results["network_timeout"] = await run_fault_scenario(
        runtime,
        message_id="fault-network-timeout",
        fault_operation="model",
        fault=Fault("timeout"),
    )
    results["tool_half_success"] = await run_fault_scenario(
        runtime,
        message_id="fault-half-success",
        fault_operation="tool",
        fault=Fault("half_success"),
    )

    duplicate_task, first_inserted = runtime.submit(
        message_id="duplicate-message",
        tenant_id="tenant-b",
        input_data={"request_type": "publish_report"},
        versions=BASELINE_VERSIONS,
    )
    duplicate_again, second_inserted = runtime.submit(
        message_id="duplicate-message",
        tenant_id="tenant-b",
        input_data={"request_type": "publish_report"},
        versions=BASELINE_VERSIONS,
    )
    await runtime.run_until_idle(limit=4)
    duplicate = store.get_task(duplicate_task)
    results["duplicate_message"] = {
        "task_id": duplicate_task,
        "same_task_id": duplicate_task == duplicate_again,
        "first_inserted": first_inserted,
        "second_inserted": second_inserted,
        "state": duplicate.state,
    }

    crash_task, _ = runtime.submit(
        message_id="fault-process-exit",
        tenant_id="tenant-a",
        input_data={"request_type": "publish_report"},
        versions=CANDIDATE_VERSIONS,
    )
    child = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--crash-child",
            "--output",
            str(output),
        ],
        check=False,
    )
    state_after_exit = store.get_task(crash_task)
    await asyncio.sleep(config().lease_s + 0.03)
    await runtime.run_until_idle(worker_id="worker-recovery", limit=4)
    recovered = store.get_task(crash_task)
    results["process_exit"] = {
        "task_id": crash_task,
        "child_exit_code": child.returncode,
        "state_after_exit": state_after_exit.state,
        "checkpoint_stage": state_after_exit.stage,
        "recovered_state": recovered.state,
        "attempts": recovered.attempt,
    }

    release = run_release_drill(telemetry, output / "release-drill.json")
    (output / "version-manifests.json").write_text(
        json.dumps(
            {
                "baseline": BASELINE_VERSIONS.as_dict(),
                "baseline_fingerprint": BASELINE_VERSIONS.fingerprint,
                "candidate": CANDIDATE_VERSIONS.as_dict(),
                "candidate_fingerprint": CANDIDATE_VERSIONS.fingerprint,
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    telemetry.force_flush()
    trace_check = verify_trace_version_combinations(telemetry.trace_path, release)
    (output / "trace-version-verification.json").write_text(
        json.dumps(trace_check, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    telemetry.shutdown()
    dashboard = build_dashboard(output, output)
    store.compact()

    fault_report = {
        "schema_version": "1.0",
        "faults": results,
        "all_faults_recovered": all(
            [
                results["429"]["state"] == "SUCCEEDED" and results["429"]["attempts"] == 2,
                results["network_timeout"]["state"] == "SUCCEEDED" and results["network_timeout"]["attempts"] == 2,
                results["tool_half_success"]["state"] == "SUCCEEDED",
                results["duplicate_message"]["same_task_id"] and not results["duplicate_message"]["second_inserted"],
                results["process_exit"]["child_exit_code"] == 73,
                results["process_exit"]["checkpoint_stage"] == "TOOL",
                results["process_exit"]["recovered_state"] == "SUCCEEDED",
            ]
        ),
        "release_rollback_verified": release["rollback_verified"],
        "trace_versions_verified": trace_check["passed"],
        "dashboard_groups": len(dashboard["groups"]),
    }
    (output / "fault-results.json").write_text(
        json.dumps(fault_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return fault_report


async def crash_child(output: Path) -> None:
    store = StateStore(output / "state.db")
    telemetry = Telemetry(output, service_version="10w-crash-child")
    faults = FaultInjector()
    faults.add("worker.after_model_checkpoint", Fault("process_exit"))
    runtime = AgentRuntime(
        store,
        telemetry,
        config=config(),
        faults=faults,
        hard_exit=hard_exit_process,
    )
    await runtime.process_next("worker-crash-child")
    raise RuntimeError("crash child did not terminate")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "artifacts")
    parser.add_argument("--crash-child", action="store_true")
    args = parser.parse_args()
    if args.crash_child:
        asyncio.run(crash_child(args.output))
    else:
        report = asyncio.run(parent_run(args.output))
        print(json.dumps(report, ensure_ascii=False, indent=2))
        raise SystemExit(0 if all([
            report["all_faults_recovered"],
            report["release_rollback_verified"],
            report["trace_versions_verified"],
        ]) else 1)


if __name__ == "__main__":
    main()
