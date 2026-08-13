"""保留第 6 周思想的原生可恢复基线，不依赖任何 Agent 框架。"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any
import uuid

from common import (
    ArtifactLedger, ResearchTask, ResearchToolRuntime, RunReport,
    TraceEvent, TraceRecorder, build_answer, score,
)


class NativeBaseline:
    """用显式状态机展示框架隐藏的持久化、审批和恢复责任。"""

    architecture = "native"

    def __init__(self, root: Path, *, fault: str = "none") -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.ledger = ArtifactLedger(root / "business.sqlite")
        self.fault = fault

    def _path(self, run_id: str) -> Path:
        return self.root / f"{run_id}.json"

    def _save(self, run_id: str, state: dict[str, Any]) -> None:
        """原子替换避免写到一半留下损坏 JSON。"""

        path = self._path(run_id)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)

    def _load(self, run_id: str) -> dict[str, Any]:
        return json.loads(self._path(run_id).read_text(encoding="utf-8"))

    def start(self, task: ResearchTask, *, run_id: str | None = None) -> RunReport:
        started = time.perf_counter()
        run_id = run_id or f"native_{uuid.uuid4().hex}"
        trace = TraceRecorder(self.architecture)
        runtime = ResearchToolRuntime(self.ledger, trace, fault=self.fault)

        search = runtime.call("search_documents", {
            "query": task.search_query, "max_results": 4,
        })
        if search.status == "error":
            return self._report(task, run_id, "failed", "", [], trace, started,
                                error_type=search.error_type)

        evidence: list[dict[str, Any]] = []
        for hit in search.data["hits"]:
            result = runtime.call("read_document", {"doc_id": hit["id"]})
            if result.status == "error" and result.retryable:
                result = runtime.call("read_document", {"doc_id": hit["id"]})
            if result.status == "success":
                evidence.append(result.data)
        if not evidence:
            return self._report(task, run_id, "failed", "", [], trace, started,
                                error_type="no_evidence")

        answer, citations = build_answer(task, evidence)
        draft = runtime.call("save_draft", {
            "content": answer,
            "citations": citations,
            "idempotency_key": f"{run_id}:draft:v1",
        })
        state = {
            "task": task.__dict__, "answer": answer, "citations": citations,
            "draft_id": draft.data["draft_id"], "status": "waiting_approval",
            "trace_id": trace.trace_id,
            "trace": [event.__dict__ for event in trace.events],
            "started": started,
        }
        self._save(run_id, state)
        return self._report(task, run_id, "waiting_approval", answer, citations, trace, started)

    def resume(self, run_id: str, *, decision: str, submission_id: str) -> RunReport:
        state = self._load(run_id)
        task = ResearchTask(**state["task"])
        trace = TraceRecorder(self.architecture, state["trace_id"])
        # 恢复的不只是业务字段，还包括中断前轨迹。否则最终报告会只剩下
        # “审批 + 发布”，把搜索、读取和草稿阶段错误地丢掉。
        trace.events = [TraceEvent(**event) for event in state.get("trace", [])]
        started = time.perf_counter()
        first, cached = self.ledger.claim_submission(
            submission_id=submission_id, thread_id=run_id, decision=decision,
        )
        if not first:
            result = cached or {"status": state["status"]}
            return self._report(
                task, run_id, result["status"], state["answer"], state["citations"],
                trace, started, duplicate_submissions=1,
            )

        approval_started = time.perf_counter()
        if decision != "approve":
            state["status"] = "rejected"
            self._save(run_id, state)
            self.ledger.finish_submission(submission_id, {"status": "rejected"})
            trace.record(kind="approval", name="human_decision", status="rejected",
                         started=approval_started)
            return self._report(task, run_id, "rejected", state["answer"],
                                state["citations"], trace, started)

        trace.record(kind="approval", name="human_decision", status="approved",
                     started=approval_started)
        runtime = ResearchToolRuntime(self.ledger, trace, fault=self.fault)
        published = runtime.call("publish_report", {
            "draft_id": state["draft_id"], "approved": True,
            "idempotency_key": f"{run_id}:publish:v1",
        })
        state["status"] = "completed" if published.status == "success" else "failed"
        state["publication_id"] = (published.data or {}).get("publication_id")
        self._save(run_id, state)
        self.ledger.finish_submission(submission_id, {"status": state["status"]})
        return self._report(task, run_id, state["status"], state["answer"],
                            state["citations"], trace, started)

    def _report(
        self, task: ResearchTask, run_id: str, status: str, answer: str,
        citations: list[str], trace: TraceRecorder, started: float,
        *, error_type: str | None = None, duplicate_submissions: int = 0,
    ) -> RunReport:
        metrics = trace.metrics()
        return RunReport(
            architecture=self.architecture, task_id=task.id, run_id=run_id,
            status=status, success=status == "completed" and score(task, answer, citations),
            answer=answer, citations=citations, trace_id=trace.trace_id,
            latency_ms=(time.perf_counter() - started) * 1000,
            duplicate_submissions=duplicate_submissions, error_type=error_type,
            trace=trace.events, **metrics,
        )
