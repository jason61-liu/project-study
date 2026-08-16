"""Deterministic fault injection used by tests and the local drill."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass


class RateLimited(RuntimeError):
    def __init__(self, retry_after_s: float = 0.0) -> None:
        super().__init__("429 rate limited")
        self.retry_after_s = retry_after_s


class NetworkTimedOut(TimeoutError):
    pass


class HalfSucceeded(NetworkTimedOut):
    """The external effect committed but the response was lost."""


class SimulatedProcessExit(BaseException):
    pass


@dataclass(frozen=True)
class Fault:
    kind: str
    retry_after_s: float = 0.0


class FaultInjector:
    def __init__(self) -> None:
        self._plans: dict[str, deque[Fault]] = defaultdict(deque)

    def add(self, operation: str, *faults: Fault) -> None:
        self._plans[operation].extend(faults)

    def pop(self, operation: str) -> Fault | None:
        return self._plans[operation].popleft() if self._plans[operation] else None

    def raise_before(self, operation: str) -> None:
        fault = self.pop(operation)
        if fault is None or fault.kind == "ok":
            return
        if fault.kind == "429":
            raise RateLimited(fault.retry_after_s)
        if fault.kind == "timeout":
            raise NetworkTimedOut(f"network timeout at {operation}")
        if fault.kind == "process_exit":
            raise SimulatedProcessExit(f"process exited at {operation}")
        raise ValueError(f"unsupported pre-effect fault: {fault.kind}")

