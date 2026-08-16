from __future__ import annotations

import asyncio

import pytest

from agent_link import AgentConfig, run_agent_sweep
from model_sim import LLMServer, ModelProfile


def make_server():
    return LLMServer(ModelProfile(
        prefill_ms_per_token=0.0,
        decode_ms_per_token=0.05,
        prefill_concurrency=4,
        server_concurrency=8,
        jitter=0.0,
        seed=1,
    ))


def test_success_rate_is_monotonic_in_output_budget():
    async def scenario():
        server = make_server()
        configs = [
            AgentConfig(f"b{b}", b, 150, 350, num_tasks=50, max_attempts=3, seed=5)
            for b in (100, 200, 300, 400, 500)
        ]
        rows = await run_agent_sweep(server, configs)
        rates = [r["success_rate"] for r in rows]
        assert rates[-1] == 1.0  # budget above the whole difficulty range
        assert rates[0] < 1.0
        assert all(rates[i] <= rates[i + 1] + 1e-9 for i in range(len(rates) - 1))
    asyncio.run(scenario())


def test_low_budget_triggers_retry_amplification():
    async def scenario():
        server = make_server()
        rows = await run_agent_sweep(
            server,
            [AgentConfig("low", 100, 300, 700, num_tasks=40, max_attempts=4, seed=3)],
        )
        assert rows[0]["mean_attempts"] > 1.0
    asyncio.run(scenario())


def test_cost_per_success_undefined_when_no_success():
    async def scenario():
        server = make_server()
        rows = await run_agent_sweep(
            server,
            [AgentConfig("never", 0, 500, 600, num_tasks=10, max_attempts=1, seed=1)],
        )
        assert rows[0]["successful_tasks"] == 0
        assert rows[0]["cost_per_success_microusd"] is None
    asyncio.run(scenario())


def test_agent_metrics_are_linked_to_model_metrics():
    async def scenario():
        server = make_server()
        rows = await run_agent_sweep(
            server,
            [AgentConfig("link", 400, 150, 350, num_tasks=20, max_attempts=3, seed=2)],
        )
        row = rows[0]
        assert row["successful_tasks"] > 0
        assert row["total_cost_microusd"] > 0
        # The link: total cost = model per-call cost * total attempts.
        approx_total = (
            row["model_per_call_cost_microusd"]
            * row["mean_attempts"]
            * row["config"]["num_tasks"]
        )
        assert row["total_cost_microusd"] == pytest.approx(approx_total, rel=0.02)
    asyncio.run(scenario())
