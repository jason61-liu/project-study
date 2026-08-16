"""Independent resource policies plus one workflow-wide budget ledger."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
import time
from typing import AsyncIterator


class BudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class ResourcePolicy:
    timeout_s: float
    concurrency: int

    def __post_init__(self) -> None:
        if self.timeout_s <= 0 or self.concurrency < 1:
            raise ValueError("timeout_s and concurrency must be positive")


@dataclass(frozen=True)
class BudgetPolicy:
    wall_time_s: float
    max_steps: int
    max_tokens: int
    max_cost_microusd: int


@dataclass
class BudgetSnapshot:
    deadline_monotonic: float
    steps: int = 0
    tokens: int = 0
    cost_microusd: int = 0


class BudgetLedger:
    """Atomic per-workflow budget shared by model and tool operations."""

    def __init__(self, policy: BudgetPolicy, snapshot: BudgetSnapshot | None = None) -> None:
        self.policy = policy
        self.snapshot = snapshot or BudgetSnapshot(
            deadline_monotonic=time.monotonic() + policy.wall_time_s
        )
        self._lock = asyncio.Lock()

    def remaining_s(self) -> float:
        return max(0.0, self.snapshot.deadline_monotonic - time.monotonic())

    async def reserve_step(self) -> None:
        async with self._lock:
            self._ensure_time()
            if self.snapshot.steps + 1 > self.policy.max_steps:
                raise BudgetExceeded("global step budget exhausted")
            self.snapshot.steps += 1

    async def charge_model(self, *, tokens: int, cost_microusd: int) -> None:
        async with self._lock:
            self._ensure_time()
            if self.snapshot.tokens + tokens > self.policy.max_tokens:
                raise BudgetExceeded("global token budget exhausted")
            if self.snapshot.cost_microusd + cost_microusd > self.policy.max_cost_microusd:
                raise BudgetExceeded("global cost budget exhausted")
            self.snapshot.tokens += tokens
            self.snapshot.cost_microusd += cost_microusd

    async def ensure_model_capacity(self, *, tokens: int, cost_microusd: int) -> None:
        """Fail before the paid call when its declared maximum cannot fit."""

        async with self._lock:
            self._ensure_time()
            if self.snapshot.tokens + tokens > self.policy.max_tokens:
                raise BudgetExceeded("insufficient global token budget for model call")
            if self.snapshot.cost_microusd + cost_microusd > self.policy.max_cost_microusd:
                raise BudgetExceeded("insufficient global cost budget for model call")

    def operation_timeout(self, resource_timeout_s: float) -> float:
        remaining = self.remaining_s()
        if remaining <= 0:
            raise BudgetExceeded("global wall-time budget exhausted")
        return min(resource_timeout_s, remaining)

    def _ensure_time(self) -> None:
        if self.remaining_s() <= 0:
            raise BudgetExceeded("global wall-time budget exhausted")

    def as_dict(self) -> dict[str, float | int]:
        return {
            "steps": self.snapshot.steps,
            "tokens": self.snapshot.tokens,
            "cost_microusd": self.snapshot.cost_microusd,
            "remaining_ms": round(self.remaining_s() * 1000, 3),
        }


class ResourceLimiter:
    """A named concurrency gate with observable active/peak counts."""

    def __init__(self, name: str, policy: ResourcePolicy) -> None:
        self.name = name
        self.policy = policy
        self._semaphore = asyncio.Semaphore(policy.concurrency)
        self.active = 0
        self.peak = 0
        self._counter_lock = asyncio.Lock()

    @asynccontextmanager
    async def slot(self, ledger: BudgetLedger) -> AsyncIterator[float]:
        await ledger.reserve_step()
        resource_deadline = time.monotonic() + ledger.operation_timeout(self.policy.timeout_s)
        async with asyncio.timeout(max(0.0, resource_deadline - time.monotonic())):
            await self._semaphore.acquire()
        try:
            async with self._counter_lock:
                self.active += 1
                self.peak = max(self.peak, self.active)
            remaining = min(
                max(0.0, resource_deadline - time.monotonic()),
                ledger.remaining_s(),
            )
            if remaining <= 0:
                raise BudgetExceeded(f"{self.name} timeout consumed while waiting for concurrency slot")
            yield remaining
        finally:
            async with self._counter_lock:
                self.active -= 1
            self._semaphore.release()
