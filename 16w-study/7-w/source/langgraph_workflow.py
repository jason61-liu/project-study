"""LangGraph 版本：SQLite Checkpoint + Interrupt + 幂等业务账本。"""

from __future__ import annotations

import sqlite3
from pathlib import Path
import time
from typing import Any, Literal, TypedDict
import uuid

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from common import (
    ArtifactLedger, ResearchTask, ResearchToolRuntime, RunReport,
    TraceRecorder, build_answer, score,
)


class GraphState(TypedDict, total=False):
    """只保存可序列化业务状态；Token、Client 和连接不进入 Checkpoint。"""

    task: dict[str, Any]
    run_id: str
    hits: list[dict[str, Any]]
    evidence: list[dict[str, Any]]
    answer: str
    citations: list[str]
    draft_id: str
    approval: str
    publication_id: str
    error_type: str


class LangGraphWorkflow:
    architecture = "langgraph"

    def __init__(self, root: Path, *, fault: str = "none") -> None:
        root.mkdir(parents=True, exist_ok=True)
        self.ledger = ArtifactLedger(root / "business.sqlite")
        self.connection = sqlite3.connect(root / "checkpoints.sqlite", check_same_thread=False)
        self.checkpointer = SqliteSaver(self.connection)
        self.fault = fault
        self._traces: dict[str, TraceRecorder] = {}
        self.graph = self._build_graph()

    def _trace(self, run_id: str) -> TraceRecorder:
        return self._traces.setdefault(run_id, TraceRecorder(self.architecture))

    def _runtime(self, run_id: str) -> ResearchToolRuntime:
        return ResearchToolRuntime(self.ledger, self._trace(run_id), fault=self.fault)

    def _build_graph(self):
        builder = StateGraph(GraphState)

        def search_node(state: GraphState) -> dict[str, Any]:
            result = self._runtime(state["run_id"]).call("search_documents", {
                "query": state["task"]["search_query"], "max_results": 4,
            })
            if result.status == "error":
                return {"error_type": result.error_type}
            return {"hits": result.data["hits"]}

        def read_node(state: GraphState) -> dict[str, Any]:
            runtime = self._runtime(state["run_id"])
            evidence = []
            for hit in state.get("hits", []):
                result = runtime.call("read_document", {"doc_id": hit["id"]})
                if result.status == "error" and result.retryable:
                    result = runtime.call("read_document", {"doc_id": hit["id"]})
                if result.status == "success":
                    evidence.append(result.data)
            return {"evidence": evidence, "error_type": "" if evidence else "no_evidence"}

        def draft_node(state: GraphState) -> dict[str, Any]:
            task = ResearchTask(**state["task"])
            answer, citations = build_answer(task, state["evidence"])
            result = self._runtime(state["run_id"]).call("save_draft", {
                "content": answer, "citations": citations,
                "idempotency_key": f"{state['run_id']}:draft:v1",
            })
            return {
                "answer": answer, "citations": citations,
                "draft_id": result.data["draft_id"],
            }

        def approval_node(state: GraphState) -> dict[str, Any]:
            started = time.perf_counter()
            decision = interrupt({
                "kind": "publish_approval",
                "run_id": state["run_id"],
                "draft_id": state["draft_id"],
                "allowed_actions": ["approve", "reject"],
            })
            value = decision.get("decision") if isinstance(decision, dict) else decision
            self._trace(state["run_id"]).record(
                kind="approval", name="human_decision", status=str(value), started=started,
            )
            return {"approval": str(value)}

        def publish_node(state: GraphState) -> dict[str, Any]:
            result = self._runtime(state["run_id"]).call("publish_report", {
                "draft_id": state["draft_id"], "approved": True,
                "idempotency_key": f"{state['run_id']}:publish:v1",
            })
            if result.status == "error":
                return {"error_type": result.error_type}
            return {"publication_id": result.data["publication_id"]}

        def after_read(state: GraphState) -> Literal["draft", "stop"]:
            return "draft" if state.get("evidence") else "stop"

        def after_approval(state: GraphState) -> Literal["publish", "stop"]:
            return "publish" if state.get("approval") == "approve" else "stop"

        builder.add_node("search", search_node)
        builder.add_node("read", read_node)
        builder.add_node("draft", draft_node)
        builder.add_node("approval", approval_node)
        builder.add_node("publish", publish_node)
        builder.add_edge(START, "search")
        builder.add_edge("search", "read")
        builder.add_conditional_edges("read", after_read, {"draft": "draft", "stop": END})
        builder.add_edge("draft", "approval")
        builder.add_conditional_edges(
            "approval", after_approval, {"publish": "publish", "stop": END},
        )
        builder.add_edge("publish", END)
        return builder.compile(checkpointer=self.checkpointer)

    @staticmethod
    def _config(run_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": run_id}}

    def start(self, task: ResearchTask, *, run_id: str | None = None) -> RunReport:
        started = time.perf_counter()
        run_id = run_id or f"graph_{uuid.uuid4().hex}"
        self._trace(run_id)
        result = self.graph.invoke(
            {"task": task.__dict__, "run_id": run_id}, self._config(run_id),
        )
        snapshot = self.graph.get_state(self._config(run_id))
        waiting = bool(snapshot.interrupts)
        status = "waiting_approval" if waiting else "failed"
        return self._report(task, run_id, status, result, started)

    def resume(self, run_id: str, *, decision: str, submission_id: str) -> RunReport:
        started = time.perf_counter()
        snapshot = self.graph.get_state(self._config(run_id))
        state = dict(snapshot.values)
        task = ResearchTask(**state["task"])
        first, cached = self.ledger.claim_submission(
            submission_id=submission_id, thread_id=run_id, decision=decision,
        )
        if not first:
            status = (cached or {}).get("status", "waiting_approval")
            report = self._report(task, run_id, status, state, started)
            report.duplicate_submissions = 1
            return report

        result = self.graph.invoke(
            Command(resume={"decision": decision}), self._config(run_id),
        )
        status = "completed" if result.get("publication_id") else (
            "rejected" if decision == "reject" else "failed"
        )
        self.ledger.finish_submission(submission_id, {"status": status})
        return self._report(task, run_id, status, result, started)

    def _report(
        self, task: ResearchTask, run_id: str, status: str,
        state: dict[str, Any], started: float,
    ) -> RunReport:
        trace = self._trace(run_id)
        metrics = trace.metrics()
        answer = state.get("answer", "")
        citations = state.get("citations", [])
        return RunReport(
            architecture=self.architecture, task_id=task.id, run_id=run_id,
            status=status, success=status == "completed" and score(task, answer, citations),
            answer=answer, citations=citations, trace_id=trace.trace_id,
            latency_ms=(time.perf_counter() - started) * 1000,
            error_type=state.get("error_type") or None, trace=trace.events, **metrics,
        )

