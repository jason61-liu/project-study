"""Async benchmark harness: run a workload against the server and aggregate.

Each trial launches ``concurrency`` in-flight requests together, records a full
request-level timeline per sample, and computes the batch makespan so throughput
can be derived. Trials are repeated so a latency distribution (P50/P95/P99) is
available instead of a single number.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from model_sim import LLMServer, StreamSample
from stats import confidence_interval, mean, summarize
from workload import WorkloadConfig

JSON = dict[str, Any]


@dataclass
class BenchmarkResult:
    workload: WorkloadConfig
    samples: list[StreamSample] = field(default_factory=list)
    trial_throughput_tokens_s: list[float] = field(default_factory=list)
    trial_throughput_req_s: list[float] = field(default_factory=list)

    def aggregate(self, server: LLMServer) -> JSON:
        ttft = [s.ttft_ms for s in self.samples]
        tpot = [s.tpot_ms for s in self.samples if s.output_tokens > 1]
        e2e = [s.e2e_ms for s in self.samples]
        queue = [s.queue_wait_ms for s in self.samples]
        total_input = sum(s.input_tokens for s in self.samples)
        total_output = sum(s.output_tokens for s in self.samples)
        total_cost = sum(s.cost_microusd(server.pricing) for s in self.samples)
        e2e_lo, e2e_hi = confidence_interval(e2e)
        return {
            "workload": {
                "name": self.workload.name,
                "input_tokens": self.workload.input_tokens,
                "output_tokens": self.workload.output_tokens,
                "concurrency": self.workload.concurrency,
                "streaming": self.workload.streaming,
                "trials": self.workload.trials,
            },
            "samples": len(self.samples),
            "ttft_ms": summarize(ttft),
            "tpot_ms": summarize(tpot),
            "e2e_ms": summarize(e2e),
            "e2e_ms_ci95": [round(e2e_lo, 3), round(e2e_hi, 3)],
            "queue_wait_ms": summarize(queue),
            "throughput_tokens_s": round(mean(self.trial_throughput_tokens_s), 3),
            "throughput_req_s": round(mean(self.trial_throughput_req_s), 3),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_cost_microusd": round(total_cost, 2),
            "server": {
                "decode_peak": server.decode_peak,
                "prefill_peak": server.prefill_peak,
            },
        }


async def run_trial(
    server: LLMServer,
    workload: WorkloadConfig,
) -> tuple[list[StreamSample], float, float]:
    """Launch one batch of ``concurrency`` requests and measure its makespan."""
    tasks = [
        server.stream(workload.input_tokens, workload.output_tokens)
        for _ in range(workload.concurrency)
    ]
    samples = await asyncio.gather(*tasks)

    t_min = min(s.t_sent for s in samples)
    t_max = max(s.t_last_token for s in samples)
    makespan_s = max(0.0, t_max - t_min)
    total_output = sum(s.output_tokens for s in samples)
    tput_tokens = total_output / makespan_s if makespan_s > 0 else 0.0
    tput_req = len(samples) / makespan_s if makespan_s > 0 else 0.0
    return samples, tput_tokens, tput_req


async def run_config(server: LLMServer, workload: WorkloadConfig) -> BenchmarkResult:
    result = BenchmarkResult(workload=workload)
    for _ in range(workload.trials):
        samples, tput_tokens, tput_req = await run_trial(server, workload)
        result.samples.extend(samples)
        result.trial_throughput_tokens_s.append(tput_tokens)
        result.trial_throughput_req_s.append(tput_req)
    return result


async def run_workloads(server: LLMServer, workloads: list[WorkloadConfig]) -> list[JSON]:
    rows: list[JSON] = []
    for workload in workloads:
        result = await run_config(server, workload)
        rows.append(result.aggregate(server))
    return rows
