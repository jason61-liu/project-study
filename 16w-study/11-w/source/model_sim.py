"""Simulated LLM serving layer with structurally realistic prefill/decode latency.

The server is not a real GPU, but it reproduces the two timing behaviours that
matter for benchmarking (see the week-11 documents):

* Prefill is compute-bound and linear in input tokens -> it dominates TTFT.
* Decode is memory-bound and linear in output tokens -> it dominates TPOT and
  total generation time.
* A finite number of prefill and decode slots -> queueing appears once
  concurrency exceeds the server's capacity, which is what the throughput and
  tail-latency curves are meant to expose.
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Any

JSON = dict[str, Any]


@dataclass(frozen=True)
class ModelProfile:
    """Timing and capacity knobs for the synthetic server."""

    prefill_ms_per_token: float = 0.02   # alpha: linear in input tokens
    decode_ms_per_token: float = 1.0     # beta: TPOT, linear in output tokens
    prefill_concurrency: int = 4         # concurrent prefill (compute) slots
    server_concurrency: int = 8          # concurrent decode (memory) slots
    jitter: float = 0.1                  # relative timing jitter; 0.0 -> deterministic
    seed: int = 7


@dataclass(frozen=True)
class Pricing:
    """Cost per token in microUSD (1e-6 USD)."""

    input_microusd_per_token: float = 0.5
    output_microusd_per_token: float = 1.5


@dataclass
class StreamSample:
    """Request-level timeline captured during one streaming model call.

    All *_ms fields are derived from the recorded timestamps so that a single
    sample can be dumped to a JSONL trace (requirement: request-level timeline)
    and aggregated into TTFT / TPOT / end-to-end percentiles.
    """

    input_tokens: int
    output_tokens: int
    t_sent: float
    queue_wait_ms: float
    t_first_token: float
    token_timestamps: list[float] = field(default_factory=list)
    t_last_token: float = 0.0

    @property
    def ttft_ms(self) -> float:
        return (self.t_first_token - self.t_sent) * 1000.0

    @property
    def e2e_ms(self) -> float:
        return (self.t_last_token - self.t_sent) * 1000.0

    @property
    def tpot_ms(self) -> float:
        # Average inter-token latency of the decode phase (excludes TTFT).
        if self.output_tokens <= 1:
            return 0.0
        return (self.t_last_token - self.t_first_token) * 1000.0 / (self.output_tokens - 1)

    def cost_microusd(self, pricing: Pricing) -> float:
        return (
            self.input_tokens * pricing.input_microusd_per_token
            + self.output_tokens * pricing.output_microusd_per_token
        )

    def as_dict(self, pricing: Pricing) -> JSON:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "queue_wait_ms": round(self.queue_wait_ms, 3),
            "ttft_ms": round(self.ttft_ms, 3),
            "tpot_ms": round(self.tpot_ms, 3),
            "e2e_ms": round(self.e2e_ms, 3),
            "cost_microusd": round(self.cost_microusd(pricing), 3),
        }


class LLMServer:
    """Streaming server with prefill/decode slots and observable saturation."""

    def __init__(self, profile: ModelProfile | None = None, pricing: Pricing | None = None) -> None:
        self.profile = profile or ModelProfile()
        self.pricing = pricing or Pricing()
        self._prefill_slots = asyncio.Semaphore(self.profile.prefill_concurrency)
        self._decode_slots = asyncio.Semaphore(self.profile.server_concurrency)
        self._rng = random.Random(self.profile.seed)

        # Observable saturation counters (mirrors the ResourceLimiter.peak idea).
        self.prefill_active = 0
        self.prefill_peak = 0
        self.decode_active = 0
        self.decode_peak = 0
        self.queue_wait_samples: list[float] = []

    def reset_peaks(self) -> None:
        """Clear saturation counters so each workload reports its own peak."""
        self.prefill_peak = 0
        self.decode_peak = 0
        self.queue_wait_samples.clear()

    async def stream(self, input_tokens: int, output_tokens: int) -> StreamSample:
        """Stream one completion and return its full request-level timeline."""
        t_sent = time.perf_counter()

        t_arrive = time.perf_counter()
        async with self._prefill_slots:
            t_acquire = time.perf_counter()
            self.prefill_active += 1
            self.prefill_peak = max(self.prefill_peak, self.prefill_active)
            try:
                await asyncio.sleep(self._jitter(self.profile.prefill_ms_per_token * input_tokens))
                t_first = time.perf_counter()
            finally:
                self.prefill_active -= 1
        queue_wait_ms = (t_acquire - t_arrive) * 1000.0
        self.queue_wait_samples.append(queue_wait_ms)

        token_timestamps: list[float] = []
        async with self._decode_slots:
            self.decode_active += 1
            self.decode_peak = max(self.decode_peak, self.decode_active)
            try:
                for _ in range(output_tokens):
                    await asyncio.sleep(self._jitter(self.profile.decode_ms_per_token))
                    token_timestamps.append(time.perf_counter())
            finally:
                self.decode_active -= 1

        t_last = token_timestamps[-1] if token_timestamps else t_first
        return StreamSample(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            t_sent=t_sent,
            queue_wait_ms=queue_wait_ms,
            t_first_token=t_first,
            token_timestamps=token_timestamps,
            t_last_token=t_last,
        )

    def _jitter(self, base_ms: float) -> float:
        if self.profile.jitter <= 0.0:
            return base_ms / 1000.0
        return base_ms / 1000.0 * self._rng.uniform(1.0 - self.profile.jitter, 1.0 + self.profile.jitter)
