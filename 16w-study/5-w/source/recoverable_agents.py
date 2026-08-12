"""可恢复单 Agent 基线与 Manager + Specialist 消融实现。"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import time
import uuid
from typing import Any

from models import ResearchModel, ToolResult
from recovery_models import (
    FailureClass, FailureRecord, PlanItem, RecoveryRun, RecoveryTask,
    StructuredPlan, classify_failure,
)
from state_store import JsonPlanStore, StateConflict
from tools import ResearchToolRuntime


SYSTEM = """你是可审计的研究 Agent，只能依据 evidence 回答。
返回 JSON：{answer:string,citations:string[]}。citations 只能引用 evidence 中的文档 ID。
回答必须覆盖任务要求的术语；证据不足时明确说明，不得编造。"""


class SimulatedProcessInterrupt(RuntimeError):
    """只在 Checkpoint 完成后抛出，用于证明进程重启可恢复。"""


class RecoverableSingleAgent:
    """生产基线：一个 Agent 顺序推进计划，Runtime 掌握恢复与提交权。"""

    name = "recoverable_single_agent"

    def __init__(
        self, model: ResearchModel, store: JsonPlanStore, *, max_retries: int = 1,
    ) -> None:
        self.model = model
        self.store = store
        self.max_retries = max_retries
        self._input_tokens = 0
        self._output_tokens = 0
        self._model_calls = 0

    def create_plan(self, task: RecoveryTask, plan_id: str | None = None) -> StructuredPlan:
        """计划是可校验的数据结构；完成条件由执行器判断，不交给模型自报。"""

        plan = StructuredPlan(
            plan_id=plan_id or f"plan_{uuid.uuid4().hex}", task_id=task.id,
            architecture=self.name, revision=1, version=0, status="running",
            items=[
                PlanItem("search", "检索候选证据", "search", [], "至少返回一个候选文档",
                         {"query": task.search_query, "max_results": 6}),
                PlanItem("read", "读取检索命中文档", "read_hits", ["search"],
                         "至少读取一个证据，失败必须结构化记录"),
                PlanItem("synthesize", "基于证据形成带引用答案", "synthesize", ["read"],
                         "答案非空且引用是已读取证据的子集"),
            ],
        )
        self.store.create(plan)
        return plan

    def run(
        self, task: RecoveryTask, repetition: int, *, plan_id: str | None = None,
    ) -> RecoveryRun:
        started = time.perf_counter()
        runtime = ResearchToolRuntime(fault=self._runtime_fault(task.fault))
        plan = self.store.load(plan_id) if plan_id else self.create_plan(task)

        item_order = ("search", "read", "synthesize")
        cursor = 0
        while cursor < len(item_order):
            item_id = item_order[cursor]
            plan = self.store.load(plan.plan_id)
            item = plan.item(item_id)
            if item.status in {"completed", "partial"}:
                cursor += 1
                continue
            if any(plan.item(parent).status not in {"completed", "partial"} for parent in item.depends_on):
                item.status = "blocked"
                plan.status = "failed"
                plan = self._save(plan)
                break
            plan = self._execute_item(plan, item_id, task, runtime)
            if plan.status in {"failed", "waiting_human"}:
                break
            if task.fault == "process_interrupt" and item_id == "search":
                # 搜索结果已经原子落盘；重启后不得再次调用相同副作用。
                raise SimulatedProcessInterrupt(plan.plan_id)
            # 重规划会把 search/read 重新置为 pending，因此从 DAG 起点重新调度。
            cursor = 0 if plan.item("search").status == "pending" else cursor + 1

        plan = self.store.load(plan.plan_id)
        if plan.status == "running":
            statuses = {item.status for item in plan.items}
            plan.status = "partial" if "partial" in statuses else "completed"
            plan = self._save(plan)
        return self._result(task, repetition, plan, started)

    def _execute_item(
        self, plan: StructuredPlan, item_id: str, task: RecoveryTask,
        runtime: ResearchToolRuntime,
    ) -> StructuredPlan:
        item = plan.item(item_id)
        item.status = "in_progress"
        item.attempts += 1

        if item.action == "search":
            result, plan = self._tool_once(plan, runtime, item, "search_documents", item.arguments)
            if result.status == "error":
                return self._handle_tool_failure(plan, item_id, task, runtime, result)
            item = plan.item(item_id)
            item.result = result.data
            item.status = "completed" if result.data.get("hits") else "failed"
            if item.status == "failed":
                plan.status = "failed"

        elif item.action == "read_hits":
            hits = plan.item("search").result["hits"]
            evidence: list[dict[str, Any]] = []
            errors: list[str] = []
            for index, hit in enumerate(hits):
                if task.fault == "partial_success" and index == len(hits) - 1:
                    errors.append("tool_unavailable")
                    self._record_failure(plan, item, "tool_unavailable", "continue_with_partial_evidence")
                    continue
                result, plan = self._tool_once(
                    plan, runtime, plan.item(item_id), "read_document",
                    {"doc_id": hit["id"], "expected_version": runtime.corpus_version},
                    suffix=hit["id"],
                )
                item = plan.item(item_id)
                if result.error_type == "plan_invalidated":
                    return self._replan(plan, task)
                if result.status == "error" and result.retryable:
                    result, plan = self._tool_once(
                        plan, runtime, plan.item(item_id), "read_document",
                        {"doc_id": hit["id"], "expected_version": runtime.corpus_version},
                        suffix=f"{hit['id']}:retry",
                    )
                    item = plan.item(item_id)
                if result.status == "success":
                    evidence.append(result.data)
                else:
                    errors.append(result.error_type or "tool_error")
                    self._record_failure(plan, item, result.error_type or "tool_error", "continue_if_evidence_remains")
            item.result = {"evidence": evidence, "errors": errors}
            item.status = "partial" if evidence and errors else ("completed" if evidence else "failed")
            if item.status == "failed":
                plan.status = "failed"

        else:
            if task.fault == "context_lost" and not plan.context_rebuilds:
                # 模拟内存消息丢失；恢复只能读取 Checkpoint 中的结构化证据。
                plan.context_rebuilds += 1
                self._record_failure(plan, item, "context_lost", "rebuild_from_checkpoint")
            evidence = plan.item("read").result["evidence"]
            reply = self.model.complete_json(
                system=SYSTEM,
                user=json.dumps({
                    "question": task.question, "required_terms": task.required_terms,
                    "evidence": evidence,
                }, ensure_ascii=False),
                purpose="recoverable_synthesis",
            )
            self._model_calls += 1
            self._input_tokens += reply.input_tokens
            self._output_tokens += reply.output_tokens
            citations = [str(value) for value in reply.data.get("citations", [])]
            allowed = {document["id"] for document in evidence}
            answer = str(reply.data.get("answer", ""))
            item.result = {"answer": answer, "citations": [value for value in citations if value in allowed]}
            item.status = "completed" if answer and set(item.result["citations"]) <= allowed else "failed"
            if item.status == "failed":
                plan.status = "failed"
        return self._save(plan)

    def _handle_tool_failure(
        self, plan: StructuredPlan, item_id: str, task: RecoveryTask,
        runtime: ResearchToolRuntime, result: ToolResult,
    ) -> StructuredPlan:
        item = plan.item(item_id)
        category = classify_failure(result.error_type or "tool_error")
        if category is FailureClass.RETRYABLE and item.attempts <= self.max_retries:
            self._record_failure(plan, item, result.error_type or "tool_error", "retry")
            plan = self._save(plan)
            return self._execute_item(plan, item_id, task, runtime)
        if category is FailureClass.REPLAN_REQUIRED:
            return self._replan(plan, task)
        self._record_failure(plan, item, result.error_type or "tool_error", "stop")
        item.status = "blocked" if category is FailureClass.HUMAN_REQUIRED else "failed"
        plan.status = "waiting_human" if category is FailureClass.HUMAN_REQUIRED else "failed"
        return self._save(plan)

    def _replan(self, plan: StructuredPlan, task: RecoveryTask) -> StructuredPlan:
        """保留已验证产物，只重置受版本变化影响的节点。"""

        item = plan.item("read")
        self._record_failure(plan, item, "plan_invalidated", "replan")
        plan.revision += 1
        plan.item("search").status = "pending"
        plan.item("search").result = None
        plan.item("read").status = "pending"
        plan.item("read").result = None
        plan.item("synthesize").status = "pending"
        return self._save(plan)

    def _tool_once(
        self, plan: StructuredPlan, runtime: ResearchToolRuntime, item: PlanItem,
        name: str, arguments: dict[str, Any], *, suffix: str = "",
    ) -> tuple[ToolResult, StructuredPlan]:
        key = f"{plan.plan_id}:{plan.revision}:{item.id}:{item.attempts}:{name}:{suffix}"
        stored = plan.execution_records.get(key)
        if stored:
            return ToolResult(**stored), plan
        result = runtime.call(name, arguments)
        plan.execution_records[key] = result.to_dict()
        return result, plan

    def _record_failure(self, plan: StructuredPlan, item: PlanItem, error: str, action: str) -> None:
        category = classify_failure(error)
        item.error_type = error
        item.failure_class = category.value
        plan.failures.append(FailureRecord(item.id, error, category.value, item.attempts, action))

    def _save(self, plan: StructuredPlan) -> StructuredPlan:
        try:
            return self.store.save(plan, expected_version=plan.version)
        except StateConflict:
            # 不覆盖获胜写入；加载权威状态，调用者随后从未完成节点继续。
            return self.store.load(plan.plan_id)

    @staticmethod
    def _runtime_fault(fault: str) -> str:
        return fault if fault in {"retrieval_failure", "tool_failure", "plan_invalidated"} else "none"

    def _result(self, task: RecoveryTask, repetition: int, plan: StructuredPlan, started: float) -> RecoveryRun:
        final = plan.item("synthesize").result or {"answer": "", "citations": []}
        answer = final.get("answer", "")
        citations = final.get("citations", [])
        success = (
            plan.status in {"completed", "partial"}
            and all(term.lower() in answer.lower() for term in task.required_terms)
            and all(source in citations for source in task.required_sources)
        )
        counts = Counter(failure.failure_class for failure in plan.failures)
        return RecoveryRun(
            architecture=self.name, task_id=task.id, repetition=repetition,
            status=plan.status, success=success, plan_id=plan.plan_id,
            plan_revision=plan.revision,
            completed_items=sum(item.status in {"completed", "partial"} for item in plan.items),
            total_items=len(plan.items), tool_calls=len(plan.execution_records),
            model_calls=self._model_calls, input_tokens=self._input_tokens,
            output_tokens=self._output_tokens, latency_ms=(time.perf_counter() - started) * 1000,
            answer=answer, citations=citations, failure_counts=dict(counts),
            checkpoints=plan.checkpoints, context_rebuilds=plan.context_rebuilds,
            duplicate_delegations_suppressed=plan.duplicate_delegations_suppressed,
        )


class ManagerSpecialistAblation(RecoverableSingleAgent):
    """仅用于消融：Manager 保留控制权，Specialist 是可去重的有界调用。"""

    name = "manager_specialist_ablation"

    def _execute_item(
        self, plan: StructuredPlan, item_id: str, task: RecoveryTask,
        runtime: ResearchToolRuntime,
    ) -> StructuredPlan:
        delegation_key = f"{plan.plan_id}:{plan.revision}:{item_id}"
        if delegation_key in plan.delegations:
            plan.duplicate_delegations_suppressed += 1
            return self._save(plan)
        plan.delegations[delegation_key] = {
            "specialist": {"search": "retrieval", "read": "evidence", "synthesize": "writer"}[item_id],
            "status": "accepted",
        }
        plan = self._save(plan)
        result = super()._execute_item(plan, item_id, task, runtime)
        result.delegations[delegation_key]["status"] = "completed"
        return self._save(result)

    def invoke_duplicate_for_test(
        self, task: RecoveryTask, plan_id: str, item_id: str,
    ) -> StructuredPlan:
        """显式重放同一委派，验证 Manager 不会创建第二个 Specialist 工作单。"""

        plan = self.store.load(plan_id)
        key = f"{plan.plan_id}:{plan.revision}:{item_id}"
        if key in plan.delegations:
            plan.duplicate_delegations_suppressed += 1
            return self._save(plan)
        raise AssertionError("test requires an existing delegation")
