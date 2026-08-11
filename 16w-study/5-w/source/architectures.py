"""固定 Workflow、ReAct 与 Plan-and-Execute 的可比较实现。"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import time
import uuid
from typing import Any

from models import ModelReply, ResearchModel, RunMetrics, Scenario, TraceEvent
from tools import ResearchToolRuntime


SYSTEM_BASE = """你是严谨的技术研究 Agent。你只能使用给定工具结果作为事实依据。
最终 JSON 必须包含 answer 字符串和 citations 文档 ID 数组。结论要明确回答问题，说明架构选择、
安全边界、失败恢复与权衡。不得编造未读取的文档 ID。"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ArchitectureBase:
    """共享模型、工具 Schema、预算、Trace 和评分，控制策略由子类实现。"""

    name = "base"

    def __init__(self, model: ResearchModel, *, max_steps: int = 10) -> None:
        self.model = model
        self.max_steps = max_steps
        self.trace: list[TraceEvent] = []
        self.input_tokens = 0
        self.output_tokens = 0
        self.model_calls = 0
        self.failure_types: list[str] = []

    def _model(self, *, system: str, user: str, purpose: str) -> ModelReply:
        started_at = _now()
        started = time.perf_counter()
        try:
            reply = self.model.complete_json(system=system, user=user, purpose=purpose)
            status = "success"
        except Exception as exc:
            status = "failed"
            self.failure_types.append("model_error")
            self.trace.append(TraceEvent(
                kind="model", name=purpose, status=status, started_at=started_at, ended_at=_now(),
                latency_ms=(time.perf_counter()-started)*1000, detail={"error_type": type(exc).__name__},
            ))
            raise
        self.model_calls += 1
        self.input_tokens += reply.input_tokens
        self.output_tokens += reply.output_tokens
        self.trace.append(TraceEvent(
            kind="model", name=purpose, status=status, started_at=started_at, ended_at=_now(),
            latency_ms=reply.latency_ms, input_tokens=reply.input_tokens, output_tokens=reply.output_tokens,
            detail={"model": reply.model},
        ))
        return reply

    def _tool(self, runtime: ResearchToolRuntime, name: str, args: dict[str, Any]):
        started_at = _now()
        started = time.perf_counter()
        result = runtime.call(name, args)
        elapsed = (time.perf_counter() - started) * 1000
        if result.error_type:
            self.failure_types.append(result.error_type)
        self.trace.append(TraceEvent(
            kind="tool", name=name, status=result.status, started_at=started_at, ended_at=_now(),
            latency_ms=elapsed, call_id=result.call_id,
            detail={"error_type": result.error_type, "retryable": result.retryable,
                    "corpus_version": result.corpus_version},
        ))
        return result

    @staticmethod
    def _score(scenario: Scenario, answer: str, citations: list[str]) -> bool:
        terms_ok = all(term.lower() in answer.lower() for term in scenario.required_terms)
        sources_ok = all(source in citations for source in scenario.required_sources)
        return terms_ok and sources_ok

    def _finish(
        self, *, scenario: Scenario, repetition: int, started: float, steps: int,
        runtime: ResearchToolRuntime, answer: str, citations: list[str], status: str = "success",
    ) -> RunMetrics:
        success = status == "success" and self._score(scenario, answer, citations)
        final_status = "success" if success else (status if status != "success" else "failed")
        return RunMetrics(
            architecture=self.name, scenario_id=scenario.id, repetition=repetition,
            status=final_status, success=success, steps=steps, tool_calls=runtime.tool_calls,
            model_calls=self.model_calls, input_tokens=self.input_tokens, output_tokens=self.output_tokens,
            latency_ms=(time.perf_counter()-started)*1000, answer=answer, citations=citations,
            trace_id=f"trace_{uuid.uuid4().hex}", failure_types=list(dict.fromkeys(self.failure_types)),
            trace=self.trace,
        )

    def run(self, scenario: Scenario, repetition: int) -> RunMetrics:
        raise NotImplementedError


class FixedWorkflow(ArchitectureBase):
    """代码固定执行 search → read → synthesize，并用确定性策略恢复。"""

    name = "fixed_workflow"

    def run(self, scenario: Scenario, repetition: int) -> RunMetrics:
        started = time.perf_counter()
        runtime = ResearchToolRuntime(fault=scenario.fault)
        steps = 0
        search = self._tool(runtime, "search_documents", {"query": scenario.question, "max_results": 6})
        steps += 1
        if search.status == "error" and search.retryable:
            search = self._tool(runtime, "search_documents", {"query": scenario.question, "max_results": 6})
            steps += 1
        if search.status != "success":
            return self._finish(scenario=scenario, repetition=repetition, started=started, steps=steps,
                                runtime=runtime, answer="检索失败，无法形成可信结论。", citations=[], status="failed")

        evidence: list[dict[str, Any]] = []
        version = search.corpus_version
        for hit in search.data["hits"]:
            read = self._tool(runtime, "read_document", {"doc_id": hit["id"], "expected_version": version})
            steps += 1
            if read.error_type == "plan_invalidated":
                # 固定 Workflow 的恢复路径也是代码预定义：重新检索取得新版本后重读。
                refreshed = self._tool(runtime, "search_documents", {"query": scenario.question, "max_results": 6})
                steps += 1
                version = refreshed.corpus_version
                read = self._tool(runtime, "read_document", {"doc_id": hit["id"], "expected_version": version})
                steps += 1
            elif read.status == "error" and read.retryable:
                read = self._tool(runtime, "read_document", {"doc_id": hit["id"], "expected_version": version})
                steps += 1
            if read.status == "success":
                evidence.append(read.data)

        reply = self._model(
            system=SYSTEM_BASE,
            user=f"问题：{scenario.question}\n已读取证据：{json.dumps(evidence, ensure_ascii=False)}\n"
                 "输出 {answer,citations}。在答案中明确使用‘部分’、‘降级’、‘Trace’等故障语义（适用时）。",
            purpose="fixed_synthesis",
        )
        steps += 1
        return self._finish(
            scenario=scenario, repetition=repetition, started=started, steps=steps, runtime=runtime,
            answer=str(reply.data.get("answer", "")), citations=list(reply.data.get("citations", [])),
        )


class ReAct(ArchitectureBase):
    """模型逐步决定工具与终止；Runtime 仍负责预算和工具校验。"""

    name = "react"

    def run(self, scenario: Scenario, repetition: int) -> RunMetrics:
        started = time.perf_counter()
        runtime = ResearchToolRuntime(fault=scenario.fault)
        observations: list[dict[str, Any]] = []
        evidence: dict[str, dict[str, Any]] = {}
        for step in range(1, self.max_steps + 1):
            reply = self._model(
                system=SYSTEM_BASE + "\n你运行 ReAct。输出 type=tool 或 type=final。tool 时给出 tool_name 和 arguments；final 时给出 answer 和 citations。",
                user=json.dumps({
                    "question": scenario.question,
                    "tools": ResearchToolRuntime.schemas(),
                    "observations": observations[-8:],
                    "read_evidence_ids": list(evidence),
                    "remaining_steps": self.max_steps-step+1,
                    "required_behavior": "工具失败可重试；版本失效后用最新 corpus_version 重读；证据充分才 final",
                }, ensure_ascii=False),
                purpose="react_decision",
            )
            data = reply.data
            if data.get("type") == "final":
                return self._finish(
                    scenario=scenario, repetition=repetition, started=started, steps=step,
                    runtime=runtime, answer=str(data.get("answer", "")),
                    citations=list(data.get("citations", [])),
                )
            name = str(data.get("tool_name", ""))
            args = data.get("arguments") if isinstance(data.get("arguments"), dict) else {}
            result = self._tool(runtime, name, args)
            observations.append(result.to_dict())
            if result.status == "success" and name == "read_document" and isinstance(result.data, dict):
                evidence[result.data["id"]] = result.data

        return self._finish(
            scenario=scenario, repetition=repetition, started=started, steps=self.max_steps,
            runtime=runtime, answer="达到最大步骤，未满足证据完成条件。", citations=[],
            status="budget_exhausted",
        )


class PlanAndExecute(ArchitectureBase):
    """模型先生成研究计划；执行器按计划运行，版本失效时显式重规划。"""

    name = "plan_and_execute"

    def _plan(self, scenario: Scenario, corpus_version: int, reason: str = "initial") -> ModelReply:
        return self._model(
            system=SYSTEM_BASE + "\n你是 Planner。只输出 {queries:[string], read_limit:integer}；queries 需要覆盖控制权、安全、延迟、恢复与题目故障。",
            user=json.dumps({"question": scenario.question, "corpus_version": corpus_version,
                             "planning_reason": reason, "max_queries": 3}, ensure_ascii=False),
            purpose="plan" if reason == "initial" else "replan",
        )

    def run(self, scenario: Scenario, repetition: int) -> RunMetrics:
        started = time.perf_counter()
        runtime = ResearchToolRuntime(fault=scenario.fault)
        steps = 1
        plan = self._plan(scenario, runtime.corpus_version).data
        queries = [str(q) for q in plan.get("queries", [])][:3] or [scenario.question]
        read_limit = min(6, max(1, int(plan.get("read_limit", 6))))
        evidence: dict[str, dict[str, Any]] = {}
        replanned = False

        query_index = 0
        while query_index < len(queries):
            query = queries[query_index]
            search = self._tool(runtime, "search_documents", {"query": query, "max_results": read_limit})
            steps += 1
            if search.status == "error" and search.retryable:
                search = self._tool(runtime, "search_documents", {"query": query, "max_results": read_limit})
                steps += 1
            if search.status != "success":
                query_index += 1
                continue
            version = search.corpus_version
            restart_from_new_plan = False
            for hit in search.data["hits"]:
                if hit["id"] in evidence:
                    continue
                read = self._tool(runtime, "read_document", {"doc_id": hit["id"], "expected_version": version})
                steps += 1
                if read.error_type == "plan_invalidated" and not replanned:
                    replanned = True
                    updated_plan = self._plan(scenario, read.corpus_version, "corpus_version_changed")
                    steps += 1
                    # 版本变化后旧检索结果也可能过期，因此清除旧证据并从新计划重新搜索，
                    # 而不是只把 expected_version 改成新值后机械执行旧计划。
                    queries = [str(q) for q in updated_plan.data.get("queries", [])][:3] or [scenario.question]
                    evidence.clear()
                    query_index = 0
                    restart_from_new_plan = True
                    break
                elif read.status == "error" and read.retryable:
                    read = self._tool(runtime, "read_document", {"doc_id": hit["id"], "expected_version": version})
                    steps += 1
                if read.status == "success":
                    evidence[read.data["id"]] = read.data
            if restart_from_new_plan:
                continue
            query_index += 1

        reply = self._model(
            system=SYSTEM_BASE,
            user=f"问题：{scenario.question}\n计划执行后的证据：{json.dumps(list(evidence.values()), ensure_ascii=False)}\n"
                 f"是否发生重规划：{replanned}。输出 {{answer,citations}}，明确说明计划失效、部分失败、降级和 Trace（适用时）。",
            purpose="plan_synthesis",
        )
        steps += 1
        return self._finish(
            scenario=scenario, repetition=repetition, started=started, steps=steps, runtime=runtime,
            answer=str(reply.data.get("answer", "")), citations=list(reply.data.get("citations", [])),
        )
