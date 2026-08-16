from __future__ import annotations

import asyncio

from budget import BudgetPolicy, ResourcePolicy
from runtime import AgentRuntime, RuntimeConfig
from state_store import StateStore
from telemetry import Telemetry


def make_config(*, model_timeout=1.0, tool_timeout=1.0, model_concurrency=2, tool_concurrency=1, max_tokens=10_000):
    return RuntimeConfig(
        model=ResourcePolicy(model_timeout, model_concurrency),
        tool=ResourcePolicy(tool_timeout, tool_concurrency),
        global_budget=BudgetPolicy(5.0, 10, max_tokens, 100_000),
        lease_s=0.05,
        max_attempts=2,
    )


def test_model_and_tool_have_independent_concurrency_limits(tmp_path):
    async def scenario():
        telemetry = Telemetry(tmp_path / "otel")
        runtime = AgentRuntime(
            StateStore(tmp_path / "state.db"), telemetry,
            config=make_config(model_concurrency=2, tool_concurrency=1),
            model_latency_s=0.03, tool_latency_s=0.03,
        )
        for index in range(6):
            runtime.submit(
                message_id=f"message-{index}", tenant_id="tenant-a",
                input_data={"kind": "test"},
            )
        await asyncio.gather(*(runtime.process_next(f"worker-{index}") for index in range(6)))
        assert runtime.model_limiter.peak == 2
        assert runtime.tool_limiter.peak == 1
        assert all(task.state == "SUCCEEDED" for task in runtime.store.all_tasks())
        telemetry.shutdown()

    asyncio.run(scenario())


def test_model_and_tool_timeouts_are_independent(tmp_path):
    async def scenario():
        model_otel = Telemetry(tmp_path / "model-otel")
        model_runtime = AgentRuntime(
            StateStore(tmp_path / "model.db"), model_otel,
            config=make_config(model_timeout=0.005, tool_timeout=1.0),
            model_latency_s=0.05,
        )
        model_task, _ = model_runtime.submit(
            message_id="model-timeout", tenant_id="tenant-a", input_data={"kind": "test"}
        )
        await model_runtime.process_next("model-worker")
        assert model_runtime.store.get_task(model_task).state == "RETRY"
        assert model_runtime.store.get_task(model_task).stage == "MODEL"
        model_otel.shutdown()

        tool_otel = Telemetry(tmp_path / "tool-otel")
        tool_runtime = AgentRuntime(
            StateStore(tmp_path / "tool.db"), tool_otel,
            config=make_config(model_timeout=1.0, tool_timeout=0.005),
            model_latency_s=0.001, tool_latency_s=0.05,
        )
        tool_task, _ = tool_runtime.submit(
            message_id="tool-timeout", tenant_id="tenant-a", input_data={"kind": "test"}
        )
        await tool_runtime.process_next("tool-worker")
        assert tool_runtime.store.get_task(tool_task).state == "RETRY"
        assert tool_runtime.store.get_task(tool_task).stage == "TOOL"
        tool_otel.shutdown()

    asyncio.run(scenario())


def test_global_token_budget_blocks_paid_call_before_execution(tmp_path):
    async def scenario():
        telemetry = Telemetry(tmp_path / "otel")
        runtime = AgentRuntime(
            StateStore(tmp_path / "state.db"), telemetry,
            config=make_config(max_tokens=100),
        )
        task_id, _ = runtime.submit(
            message_id="budget", tenant_id="tenant-a", input_data={"kind": "test"}
        )
        await runtime.process_next("worker")
        task = runtime.store.get_task(task_id)
        assert task.state == "FAILED"
        assert task.last_error == "BudgetExceeded"
        assert runtime.store.get_effect(
            f"tenant-a:{task_id}:publish:v1", tenant_id="tenant-a"
        ) is None
        telemetry.shutdown()

    asyncio.run(scenario())
