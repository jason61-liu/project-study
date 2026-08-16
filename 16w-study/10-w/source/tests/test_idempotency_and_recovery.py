from __future__ import annotations

import asyncio

import pytest

from budget import BudgetPolicy, ResourcePolicy
from faults import Fault, FaultInjector, SimulatedProcessExit
from runtime import AgentRuntime, RuntimeConfig
from state_store import IdempotencyConflict, StateStore, StaleLease
from telemetry import Telemetry


CONFIG = RuntimeConfig(
    model=ResourcePolicy(0.2, 2),
    tool=ResourcePolicy(0.2, 1),
    global_budget=BudgetPolicy(3.0, 8, 2_000, 20_000),
    lease_s=0.03,
    max_attempts=3,
)


def test_half_success_is_recovered_via_result_query_and_effect_runs_once(tmp_path):
    async def scenario():
        faults = FaultInjector()
        faults.add("tool", Fault("half_success"))
        telemetry = Telemetry(tmp_path / "otel")
        runtime = AgentRuntime(StateStore(tmp_path / "state.db"), telemetry, config=CONFIG, faults=faults)
        task_id, _ = runtime.submit(
            message_id="half-success", tenant_id="tenant-a", input_data={"kind": "test"}
        )
        await runtime.process_next("worker")
        key = f"tenant-a:{task_id}:publish:v1"
        result = runtime.query_tool_result(tenant_id="tenant-a", idempotency_key=key)
        assert runtime.store.get_task(task_id).state == "SUCCEEDED"
        assert result is not None and result.status == "SUCCEEDED"
        assert result.execution_count == 1
        telemetry.shutdown()

    asyncio.run(scenario())


def test_duplicate_message_maps_to_one_task_and_one_effect(tmp_path):
    async def scenario():
        telemetry = Telemetry(tmp_path / "otel")
        runtime = AgentRuntime(StateStore(tmp_path / "state.db"), telemetry, config=CONFIG)
        first, inserted = runtime.submit(
            message_id="same", tenant_id="tenant-a", input_data={"kind": "test"}
        )
        second, inserted_again = runtime.submit(
            message_id="same", tenant_id="tenant-a", input_data={"kind": "test"}
        )
        assert first == second and inserted and not inserted_again
        await runtime.run_until_idle()
        effect = runtime.query_tool_result(
            tenant_id="tenant-a", idempotency_key=f"tenant-a:{first}:publish:v1"
        )
        assert effect is not None and effect.execution_count == 1
        assert runtime.query_tool_result(
            tenant_id="tenant-b", idempotency_key=f"tenant-a:{first}:publish:v1"
        ) is None
        with pytest.raises(IdempotencyConflict):
            runtime.submit(
                message_id="same", tenant_id="tenant-a",
                input_data={"kind": "different-retry-body"},
            )
        telemetry.shutdown()

    asyncio.run(scenario())


def test_checkpoint_resumes_after_worker_exit_and_fences_stale_owner(tmp_path):
    async def scenario():
        store = StateStore(tmp_path / "state.db")
        telemetry = Telemetry(tmp_path / "otel")
        faults = FaultInjector()
        faults.add("worker.after_model_checkpoint", Fault("process_exit"))
        runtime = AgentRuntime(store, telemetry, config=CONFIG, faults=faults)
        task_id, _ = runtime.submit(
            message_id="crash", tenant_id="tenant-a", input_data={"kind": "test"}
        )
        with pytest.raises(SimulatedProcessExit):
            await runtime.process_next("old-worker")
        crashed = store.get_task(task_id)
        assert crashed.state == "RUNNING" and crashed.stage == "TOOL"
        old_epoch = crashed.lease_epoch

        await asyncio.sleep(CONFIG.lease_s + 0.02)
        recovered_runtime = AgentRuntime(store, telemetry, config=CONFIG)
        await recovered_runtime.process_next("new-worker")
        recovered = store.get_task(task_id)
        assert recovered.state == "SUCCEEDED" and recovered.attempt == 2
        effect = store.get_effect(
            f"tenant-a:{task_id}:publish:v1", tenant_id="tenant-a"
        )
        assert effect is not None and effect.execution_count == 1
        with pytest.raises(StaleLease):
            store.checkpoint(
                task_id, owner="old-worker", epoch=old_epoch, stage="TOOL",
                checkpoint={}, budget={},
            )
        telemetry.shutdown()

    asyncio.run(scenario())
