from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from typing import Any

from .guardrails import PolicyViolation, validate_output, validate_request
from .models import Evidence, Identity, RunRequest, RunState, RunStatus
from .retrieval import decompose, detect_conflicts, retrieve
from .store import SQLiteStore
from .telemetry import Tracer


class BudgetExceeded(RuntimeError):
    pass


class ResearchAgent:
    STEPS = ("decompose", "retrieve", "verify", "report")

    def __init__(self, store: SQLiteStore) -> None:
        self.store = store

    def submit(self, request: RunRequest) -> RunState:
        validate_request(request)
        run_id = uuid.uuid4().hex
        payload = {
            "question": request.question,
            "identity": asdict(request.identity),
            "sources": [asdict(source) for source in request.sources],
            "budget": asdict(request.budget),
            "require_approval": request.require_approval,
        }
        state = RunState(
            run_id=run_id,
            tenant_id=request.identity.tenant_id,
            user_id=request.identity.user_id,
            question=request.question,
            max_steps=request.budget.max_steps,
            max_cost_usd=request.budget.max_cost_usd,
            require_approval=request.require_approval,
        )
        if request.idempotency_key:
            persisted = self.store.create_run_idempotent(
                state, request.idempotency_key, self.store.request_hash(payload)
            )
            if persisted.run_id != state.run_id:
                if persisted.status in {RunStatus.PENDING, RunStatus.RUNNING, RunStatus.FAILED}:
                    if persisted.evidence:
                        return self._resume_from_evidence(persisted)
                    return self._execute(
                        persisted, request.sources, request.budget.max_sources, request.require_approval
                    )
                return persisted
        else:
            self.store.create_run(state)
        return self._execute(state, request.sources, request.budget.max_sources, request.require_approval)

    def _save(self, state: RunState, step: str) -> None:
        expected = state.version
        state.current_step = step
        state.steps_used += 1
        self.store.save(state, expected)

    def _execute(
        self,
        state: RunState,
        sources: tuple[Any, ...],
        max_sources: int,
        require_approval: bool,
    ) -> RunState:
        trace_id = uuid.uuid4().hex
        tracer = Tracer(lambda span: self.store.record_span(state.tenant_id, state.run_id, span))
        try:
            state.status = RunStatus.RUNNING
            self._save(state, "running")
            with tracer.span(trace_id, "plan research-agent", **{"gen_ai.operation.name": "plan"}):
                state.subqueries = decompose(state.question)
            self._save(state, "decomposed")

            with tracer.span(
                trace_id,
                "retrieval supplied-corpus",
                **{"gen_ai.operation.name": "retrieval", "app.authz.tenant_filter": True},
            ):
                evidence = retrieve(state.question, state.subqueries, sources, max_sources)
                state.evidence = [asdict(item) for item in evidence]
            self._save(state, "evidence_collected")

            if not evidence:
                state.status = RunStatus.REFUSED
                state.refusal_reason = "没有找到足以支持回答的可信证据。"
                self._save(state, "refused_no_evidence")
                return state

            estimated_cost = sum(len(item.excerpt) for item in evidence) * 0.0000005
            if estimated_cost > state.max_cost_usd:
                state.status = RunStatus.REFUSED
                state.refusal_reason = "预计费用超过任务预算。"
                self._save(state, "refused_cost_budget")
                return state

            with tracer.span(trace_id, "verify evidence", **{"app.verifier": "deterministic-claims-v1"}):
                state.conflicts = detect_conflicts(evidence)
            self._save(state, "verified")

            if require_approval and not self.store.is_approved(state.tenant_id, state.run_id):
                state.status = RunStatus.WAITING_APPROVAL
                self._save(state, "waiting_approval")
                return state

            return self._finish(state, tracer, trace_id)
        except Exception:
            state.status = RunStatus.FAILED
            try:
                self._save(state, "failed")
            except Exception:
                pass
            raise

    def _finish(self, state: RunState, tracer: Tracer, trace_id: str) -> RunState:
        evidence = [Evidence(**item) for item in state.evidence]
        with tracer.span(trace_id, "invoke_agent report-writer", **{"gen_ai.agent.name": "report-writer"}):
            state.report = self._render_report(state.question, evidence, state.conflicts)
            validate_output(state.report, {item.evidence_id for item in evidence})
            # Deterministic cost proxy makes offline comparisons reproducible.
            state.cost_usd = round(sum(len(item.excerpt) for item in evidence) * 0.0000005, 6)
        state.status = RunStatus.COMPLETED
        self._save(state, "completed")
        return state

    @staticmethod
    def _render_report(question: str, evidence: list[Evidence], conflicts: list[dict[str, Any]]) -> str:
        findings = "\n".join(f"- {item.excerpt} [{item.evidence_id}]" for item in evidence)
        conflict_text = "未发现结构化声明冲突。"
        if conflicts:
            lines = []
            for conflict in conflicts:
                variants = "; ".join(
                    f"{variant['value']} ({', '.join(variant['sources'])})"
                    for variant in conflict["values"]
                )
                lines.append(f"- {conflict['claim']}: {variants}")
            conflict_text = "\n".join(lines)
        references = "\n".join(
            f"- [{item.evidence_id}] {item.title} — {item.url}" for item in evidence
        )
        return (
            f"# 研究报告\n\n## 问题\n\n{question}\n\n"
            f"## 证据摘要\n\n{findings}\n\n"
            f"## 冲突与不确定性\n\n{conflict_text}\n\n"
            f"## 来源\n\n{references}"
        )

    def approve_and_resume(self, tenant_id: str, run_id: str, approver: Identity) -> RunState:
        if approver.tenant_id != tenant_id or "research:approve" not in approver.scopes:
            raise PolicyViolation("approval is not authorized")
        state = self.store.load(tenant_id, run_id)
        if not state:
            raise KeyError(run_id)
        if state.status != RunStatus.WAITING_APPROVAL:
            return state
        self.store.approve(tenant_id, run_id, approver.user_id)
        state.status = RunStatus.RUNNING
        self._save(state, "approval_received")
        trace_id = uuid.uuid4().hex
        tracer = Tracer(lambda span: self.store.record_span(tenant_id, run_id, span))
        return self._finish(state, tracer, trace_id)

    def resume(self, run_id: str, identity: Identity) -> RunState:
        if "research:run" not in identity.scopes:
            raise PolicyViolation("resume is not authorized")
        state = self.store.load(identity.tenant_id, run_id)
        if not state:
            raise KeyError(run_id)
        if state.user_id != identity.user_id and "operator" not in identity.roles:
            raise PolicyViolation("only the owner or an operator may resume this run")
        if state.status in {RunStatus.COMPLETED, RunStatus.REFUSED, RunStatus.CANCELLED}:
            return state
        if state.status == RunStatus.WAITING_APPROVAL and not self.store.is_approved(state.tenant_id, run_id):
            return state
        if not state.evidence:
            raise RuntimeError("resume requires an evidence checkpoint; retry the original idempotent request")
        return self._resume_from_evidence(state)

    def _resume_from_evidence(self, state: RunState) -> RunState:
        state.status = RunStatus.RUNNING
        self._save(state, "resuming_from_evidence")
        evidence = [Evidence(**item) for item in state.evidence]
        state.conflicts = detect_conflicts(evidence)
        self._save(state, "verified_after_resume")
        if state.require_approval and not self.store.is_approved(state.tenant_id, state.run_id):
            state.status = RunStatus.WAITING_APPROVAL
            self._save(state, "waiting_approval")
            return state
        trace_id = uuid.uuid4().hex
        tracer = Tracer(lambda span: self.store.record_span(state.tenant_id, state.run_id, span))
        return self._finish(state, tracer, trace_id)

    def record(self, tenant_id: str, run_id: str) -> dict[str, Any] | None:
        state = self.store.load(tenant_id, run_id)
        if not state:
            return None
        return {"state": state.to_dict(), "trace": self.store.spans(tenant_id, run_id)}
