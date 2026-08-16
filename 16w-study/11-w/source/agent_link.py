"""Link model-call metrics to Agent end-to-end success and cost.

An Agent task succeeds only when the model produces enough output tokens to
satisfy the task's difficulty; otherwise it retries (up to ``max_attempts``).
This reproduces the week-11 insight: retries inflate cost but never the
denominator, so cost-per-successful-task diverges from per-call token cost as
the success rate drops.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from typing import Any

from model_sim import LLMServer
from stats import mean

JSON = dict[str, Any]


@dataclass(frozen=True)
class AgentConfig:
    name: str
    output_budget: int          # tokens the model generates per attempt
    required_lo: int            # difficulty: min tokens needed to succeed
    required_hi: int
    input_tokens: int = 1024
    max_attempts: int = 3
    num_tasks: int = 20
    seed: int = 11


@dataclass
class AgentTaskResult:
    success: bool
    attempts: int
    total_cost_microusd: float
    total_latency_ms: float
    total_output_tokens: int


async def run_task(server: LLMServer, config: AgentConfig, required: int, rng: random.Random) -> AgentTaskResult:
    attempts = 0
    total_cost = 0.0
    total_latency = 0.0
    total_output = 0
    success = False
    while attempts < config.max_attempts and not success:
        attempts += 1
        sample = await server.stream(config.input_tokens, config.output_budget)
        total_cost += sample.cost_microusd(server.pricing)
        total_latency += sample.e2e_ms
        total_output += sample.output_tokens
        # Per-attempt success probability rises with the output budget relative
        # to the task's difficulty; a failed attempt is retried, which is what
        # makes retry amplification inflate cost without inflating success count.
        success_probability = min(1.0, config.output_budget / required)
        success = rng.random() < success_probability
    return AgentTaskResult(
        success=success,
        attempts=attempts,
        total_cost_microusd=total_cost,
        total_latency_ms=total_latency,
        total_output_tokens=total_output,
    )


async def run_agent_sweep(server: LLMServer, configs: list[AgentConfig]) -> list[JSON]:
    rows: list[JSON] = []
    for config in configs:
        tasks = []
        for i in range(config.num_tasks):
            # Per-task RNG keeps the difficulty and the success draws deterministic
            # across budget values, so success rate is monotonically comparable.
            task_rng = random.Random(config.seed * 10_000 + i)
            required = task_rng.randint(config.required_lo, config.required_hi)
            tasks.append(run_task(server, config, required, task_rng))
        results = await asyncio.gather(*tasks)

        successes = [r for r in results if r.success]
        total_cost = sum(r.total_cost_microusd for r in results)
        total_output = sum(r.total_output_tokens for r in results)
        # Model-call-level metrics (per attempt), for the explicit link below.
        per_call_cost = total_cost / sum(r.attempts for r in results) if results else 0.0
        per_call_latency = mean([r.total_latency_ms / r.attempts for r in results])

        rows.append({
            "config": {
                "name": config.name,
                "output_budget": config.output_budget,
                "required_range": [config.required_lo, config.required_hi],
                "max_attempts": config.max_attempts,
                "num_tasks": config.num_tasks,
            },
            # Agent end-to-end metrics.
            "successful_tasks": len(successes),
            "success_rate": round(len(successes) / len(results), 4),
            "mean_attempts": round(mean([r.attempts for r in results]), 3),
            "cost_per_success_microusd": round(total_cost / len(successes), 2) if successes else None,
            "tokens_per_success": round(total_output / len(successes), 2) if successes else None,
            "avg_task_latency_ms": round(mean([r.total_latency_ms for r in results]), 3),
            # Model-call-level metrics feeding the numbers above.
            "model_per_call_cost_microusd": round(per_call_cost, 2),
            "model_per_call_latency_ms": round(per_call_latency, 3),
            "total_cost_microusd": round(total_cost, 2),
        })
    return rows


def quality_sweep() -> list[AgentConfig]:
    """Output-budget sweep: a longer output budget raises success but costs more."""
    required_lo, required_hi = 150, 350
    return [
        AgentConfig(name=f"budget-{b}", output_budget=b, required_lo=required_lo, required_hi=required_hi)
        for b in (100, 200, 300, 400, 500)
    ]
