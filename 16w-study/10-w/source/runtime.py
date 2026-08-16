"""Durable Agent worker with independent model/tool controls and fault recovery."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import hashlib
import os
import time
from typing import Any, Callable

from opentelemetry import trace

from budget import (
    BudgetExceeded,
    BudgetLedger,
    BudgetPolicy,
    BudgetSnapshot,
    ResourceLimiter,
    ResourcePolicy,
)
from faults import FaultInjector, HalfSucceeded, NetworkTimedOut, RateLimited, SimulatedProcessExit
from state_store import EffectRecord, StateStore, TaskRecord
from telemetry import Telemetry
from versioning import BASELINE_VERSIONS, VersionManifest


JSON = dict[str, Any]


@dataclass(frozen=True)
class RuntimeConfig:
    model: ResourcePolicy = ResourcePolicy(timeout_s=0.2, concurrency=2)
    tool: ResourcePolicy = ResourcePolicy(timeout_s=0.1, concurrency=1)
    global_budget: BudgetPolicy = BudgetPolicy(
        wall_time_s=10.0,
        max_steps=8,
        max_tokens=2_000,
        max_cost_microusd=20_000,
    )
    lease_s: float = 0.25
    max_attempts: int = 5


@dataclass(frozen=True)
class ModelResult:
    output_ref: str
    input_tokens: int
    output_tokens: int
    cost_microusd: int


class ModelService:
    def __init__(self, faults: FaultInjector, *, latency_s: float = 0.005) -> None:
        self.faults = faults
        self.latency_s = latency_s

    async def generate(self, task: TaskRecord) -> ModelResult:
        self.faults.raise_before("model")
        await asyncio.sleep(self.latency_s)
        digest = hashlib.sha256(
            f"{task.tenant_id}:{task.task_id}:{task.versions.prompt}".encode()
        ).hexdigest()[:16]
        return ModelResult(
            output_ref=f"model-output://{digest}",
            input_tokens=120,
            output_tokens=32,
            cost_microusd=760,
        )


class SideEffectTool:
    """A side-effect tool with durable idempotency and result lookup."""

    def __init__(self, store: StateStore, faults: FaultInjector, *, latency_s: float = 0.003) -> None:
        self.store = store
        self.faults = faults
        self.latency_s = latency_s

    async def execute(self, *, tenant_id: str, idempotency_key: str, request: JSON) -> EffectRecord:
        record = self.store.begin_effect(key=idempotency_key, tenant_id=tenant_id, request=request)
        if record.status == "SUCCEEDED":
            return record
        fault = self.faults.pop("tool")
        if fault and fault.kind == "429":
            raise RateLimited(fault.retry_after_s)
        if fault and fault.kind == "timeout":
            raise NetworkTimedOut("tool timed out before commit")
        await asyncio.sleep(self.latency_s)
        result = {
            "effect_id": f"effect_{hashlib.sha256(idempotency_key.encode()).hexdigest()[:12]}",
            "status": "published",
        }
        committed = self.store.commit_effect(key=idempotency_key, result=result)
        if fault and fault.kind == "half_success":
            raise HalfSucceeded("tool committed but response was lost")
        if fault and fault.kind == "process_exit":
            raise SimulatedProcessExit("process exited after tool commit")
        return committed

    def get_result(self, *, tenant_id: str, idempotency_key: str) -> EffectRecord | None:
        return self.store.get_effect(idempotency_key, tenant_id=tenant_id)


class AgentRuntime:
    def __init__(
        self,
        store: StateStore,
        telemetry: Telemetry,
        *,
        config: RuntimeConfig | None = None,
        faults: FaultInjector | None = None,
        model_latency_s: float = 0.005,
        tool_latency_s: float = 0.003,
        hard_exit: Callable[[int], Any] | None = None,
    ) -> None:
        self.store = store
        self.telemetry = telemetry
        self.config = config or RuntimeConfig()
        self.faults = faults or FaultInjector()
        self.model_limiter = ResourceLimiter("model", self.config.model)
        self.tool_limiter = ResourceLimiter("tool", self.config.tool)
        self.model = ModelService(self.faults, latency_s=model_latency_s)
        self.tool = SideEffectTool(store, self.faults, latency_s=tool_latency_s)
        self.hard_exit = hard_exit or (lambda code: (_ for _ in ()).throw(SimulatedProcessExit(str(code))))

    def submit(
        self,
        *,
        message_id: str,
        tenant_id: str,
        input_data: JSON,
        versions: VersionManifest = BASELINE_VERSIONS,
    ) -> tuple[str, bool]:
        now = time.time()
        return self.store.enqueue(
            message_id=message_id,
            tenant_id=tenant_id,
            input_data=input_data,
            versions=versions,
            deadline_epoch=now + self.config.global_budget.wall_time_s,
            budget={"steps": 0, "tokens": 0, "cost_microusd": 0},
        )

    async def process_next(self, worker_id: str) -> str | None:
        task = self.store.claim(worker_id, lease_s=self.config.lease_s)
        if task is None:
            return None
        await self._process(task, worker_id)
        return task.task_id

    async def run_until_idle(self, worker_id: str = "worker-main", *, limit: int = 100) -> list[str]:
        processed: list[str] = []
        for _ in range(limit):
            task_id = await self.process_next(worker_id)
            if task_id is None:
                break
            processed.append(task_id)
        return processed

    def query_tool_result(self, *, tenant_id: str, idempotency_key: str) -> EffectRecord | None:
        return self.tool.get_result(tenant_id=tenant_id, idempotency_key=idempotency_key)

    async def _process(self, task: TaskRecord, worker_id: str) -> None:
        ledger = self._restore_budget(task)
        started = time.perf_counter()
        attrs = {
            "gen_ai.operation.name": "invoke_workflow",
            "gen_ai.workflow.name": "durable_agent_job",
            "tenant.id": task.tenant_id,
            "task.id": task.task_id,
            "task.attempt": task.attempt,
            "app.operation.summary": True,
            **task.versions.trace_attributes(),
        }
        outcome = "error"
        input_tokens = int(task.checkpoint.get("input_tokens", 0))
        output_tokens = int(task.checkpoint.get("output_tokens", 0))
        try:
            with self.telemetry.span("invoke_workflow durable_agent_job", attributes=attrs) as workflow_span:
                try:
                    if task.stage == "MODEL":
                        model_result = await self._call_model(task, ledger)
                        input_tokens = model_result.input_tokens
                        output_tokens = model_result.output_tokens
                        checkpoint = {
                            "model_output_ref": model_result.output_ref,
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                        }
                        self.store.checkpoint(
                            task.task_id,
                            owner=worker_id,
                            epoch=task.lease_epoch,
                            stage="TOOL",
                            checkpoint=checkpoint,
                            budget=ledger.as_dict(),
                        )
                        fault = self.faults.pop("worker.after_model_checkpoint")
                        if fault and fault.kind == "process_exit":
                            self.telemetry.safe_log(
                                "worker.process_exit.injected",
                                {"task.id": task.task_id, "tenant.id": task.tenant_id, "lease.epoch": task.lease_epoch},
                            )
                            self.telemetry.force_flush()
                            self.hard_exit(73)
                            raise SimulatedProcessExit("hard exit returned unexpectedly")
                        task = self.store.get_task(task.task_id)

                    if task.stage == "TOOL":
                        effect = await self._call_tool(task, ledger)
                        result = {
                            "task_id": task.task_id,
                            "effect_id": (effect.result or {}).get("effect_id"),
                            "version_fingerprint": task.versions.fingerprint,
                        }
                        self.store.complete(
                            task.task_id,
                            owner=worker_id,
                            epoch=task.lease_epoch,
                            result=result,
                            budget=ledger.as_dict(),
                        )
                    outcome = "success"
                    workflow_span.set_attribute("app.outcome", outcome)
                    workflow_span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
                    workflow_span.set_attribute("gen_ai.usage.output_tokens", output_tokens)
                    workflow_span.set_attribute("app.cost.microusd", ledger.snapshot.cost_microusd)
                except (RateLimited, NetworkTimedOut) as exc:
                    self.store.checkpoint(
                        task.task_id,
                        owner=worker_id,
                        epoch=task.lease_epoch,
                        stage=task.stage,
                        checkpoint=task.checkpoint,
                        budget=ledger.as_dict(),
                    )
                    delay = exc.retry_after_s if isinstance(exc, RateLimited) else 0.0
                    if task.attempt >= self.config.max_attempts:
                        self.store.fail(
                            task.task_id, owner=worker_id, epoch=task.lease_epoch,
                            error=type(exc).__name__,
                        )
                    else:
                        self.store.retry(
                            task.task_id, owner=worker_id, epoch=task.lease_epoch,
                            delay_s=delay, error=type(exc).__name__,
                        )
                    workflow_span.set_attribute("app.outcome", "error")
                    workflow_span.set_attribute("error.type", type(exc).__name__)
                    workflow_span.set_status(trace.Status(trace.StatusCode.ERROR))
                except BudgetExceeded as exc:
                    self.store.fail(
                        task.task_id, owner=worker_id, epoch=task.lease_epoch,
                        error=type(exc).__name__,
                    )
                    workflow_span.set_attribute("app.outcome", "error")
                    workflow_span.set_attribute("error.type", "budget_exceeded")
                    workflow_span.set_status(trace.Status(trace.StatusCode.ERROR))
                except Exception as exc:
                    self.store.fail(
                        task.task_id, owner=worker_id, epoch=task.lease_epoch,
                        error=type(exc).__name__,
                    )
                    workflow_span.set_attribute("app.outcome", "error")
                    workflow_span.set_attribute("error.type", type(exc).__name__)
                    workflow_span.set_status(trace.Status(trace.StatusCode.ERROR))
                finally:
                    duration_ms = (time.perf_counter() - started) * 1000
                    self.telemetry.record_summary(
                        tenant_id=task.tenant_id,
                        versions=task.versions,
                        outcome=outcome,
                        duration_ms=duration_ms,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        cost_microusd=ledger.snapshot.cost_microusd,
                    )
                    self.telemetry.safe_log(
                        "workflow.attempt.finished",
                        {
                            "task.id": task.task_id,
                            "tenant.id": task.tenant_id,
                            "task.attempt": task.attempt,
                            "app.outcome": outcome,
                            "app.version.fingerprint": task.versions.fingerprint,
                        },
                    )
        except SimulatedProcessExit:
            raise

    async def _call_model(self, task: TaskRecord, ledger: BudgetLedger) -> ModelResult:
        await ledger.ensure_model_capacity(tokens=152, cost_microusd=760)
        async with self.model_limiter.slot(ledger) as timeout_s:
            with self.telemetry.span(
                f"chat {task.versions.model}",
                kind=trace.SpanKind.CLIENT,
                attributes={
                    "gen_ai.operation.name": "chat",
                    "gen_ai.provider.name": "local.test",
                    "gen_ai.request.model": task.versions.model,
                    "tenant.id": task.tenant_id,
                    **task.versions.trace_attributes(),
                },
            ) as span:
                try:
                    async with asyncio.timeout(timeout_s):
                        result = await self.model.generate(task)
                except RateLimited:
                    span.set_attribute("error.type", "rate_limited")
                    span.set_status(trace.Status(trace.StatusCode.ERROR))
                    raise
                except TimeoutError as exc:
                    span.set_attribute("error.type", "timeout")
                    span.set_status(trace.Status(trace.StatusCode.ERROR))
                    raise NetworkTimedOut("model timeout") from exc
                await ledger.charge_model(
                    tokens=result.input_tokens + result.output_tokens,
                    cost_microusd=result.cost_microusd,
                )
                span.set_attribute("gen_ai.usage.input_tokens", result.input_tokens)
                span.set_attribute("gen_ai.usage.output_tokens", result.output_tokens)
                span.set_attribute("app.cost.microusd", result.cost_microusd)
                return result

    async def _call_tool(self, task: TaskRecord, ledger: BudgetLedger) -> EffectRecord:
        idempotency_key = f"{task.tenant_id}:{task.task_id}:publish:v1"
        request = {"task_id": task.task_id, "operation": "publish_report"}
        async with self.tool_limiter.slot(ledger) as timeout_s:
            with self.telemetry.span(
                "execute_tool publish_report",
                attributes={
                    "gen_ai.operation.name": "execute_tool",
                    "gen_ai.tool.name": "publish_report",
                    "gen_ai.tool.type": "function",
                    "tenant.id": task.tenant_id,
                    "app.idempotency.key_hash": hashlib.sha256(idempotency_key.encode()).hexdigest()[:16],
                    **task.versions.trace_attributes(),
                },
            ) as span:
                try:
                    async with asyncio.timeout(timeout_s):
                        return await self.tool.execute(
                            tenant_id=task.tenant_id,
                            idempotency_key=idempotency_key,
                            request=request,
                        )
                except HalfSucceeded:
                    record = self.tool.get_result(
                        tenant_id=task.tenant_id,
                        idempotency_key=idempotency_key,
                    )
                    if record and record.status == "SUCCEEDED":
                        span.add_event("tool.result.recovered", {"effect.status": "SUCCEEDED"})
                        return record
                    raise
                except RateLimited:
                    span.set_attribute("error.type", "rate_limited")
                    span.set_status(trace.Status(trace.StatusCode.ERROR))
                    raise
                except TimeoutError as exc:
                    span.set_attribute("error.type", "timeout")
                    span.set_status(trace.Status(trace.StatusCode.ERROR))
                    raise NetworkTimedOut("tool timeout") from exc

    def _restore_budget(self, task: TaskRecord) -> BudgetLedger:
        remaining = max(0.0, task.deadline_epoch - time.time())
        snapshot = BudgetSnapshot(
            deadline_monotonic=time.monotonic() + remaining,
            steps=int(task.budget.get("steps", 0)),
            tokens=int(task.budget.get("tokens", 0)),
            cost_microusd=int(task.budget.get("cost_microusd", 0)),
        )
        return BudgetLedger(self.config.global_budget, snapshot)


def hard_exit_process(code: int) -> None:
    os._exit(code)
