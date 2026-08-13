"""四种 Agent 架构共用的业务模型、工具运行时、轨迹和幂等账本。

实验刻意把这些对象放在同一模块中：如果每种框架各写一套工具、数据和
评分逻辑，最后测到的将是业务实现差异，而不是编排框架差异。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
import json
from pathlib import Path
import re
import sqlite3
import time
from typing import Any, Literal
import uuid


ROOT = Path(__file__).parent


@dataclass(frozen=True)
class ResearchTask:
    """四种实现接收的统一任务输入。"""

    id: str
    question: str
    search_query: str
    required_terms: list[str]
    required_sources: list[str]
    requires_approval: bool = True


@dataclass(frozen=True)
class ToolResult:
    """工具永远返回稳定结构，避免把任意异常文本交给模型猜测。"""

    status: Literal["success", "error", "partial"]
    tool: str
    call_id: str
    data: Any = None
    error_type: str | None = None
    retryable: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TraceEvent:
    """统一 Span；不记录 API Key、Access Token 或私有推理。"""

    trace_id: str
    span_id: str
    parent_span_id: str | None
    architecture: str
    kind: Literal["model", "tool", "control", "approval", "subagent"]
    name: str
    status: str
    started_at: str
    ended_at: str
    latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    call_id: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)


class TraceRecorder:
    """收集可横向比较的轨迹，同时提供 Token 和调用次数汇总。"""

    def __init__(self, architecture: str, trace_id: str | None = None) -> None:
        self.architecture = architecture
        self.trace_id = trace_id or f"trace_{uuid.uuid4().hex}"
        self.events: list[TraceEvent] = []

    def record(
        self,
        *,
        kind: TraceEvent.__annotations__["kind"],
        name: str,
        status: str,
        started: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
        call_id: str | None = None,
        parent_span_id: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        ended = time.perf_counter()
        now = datetime.now(UTC).isoformat()
        self.events.append(TraceEvent(
            trace_id=self.trace_id,
            span_id=f"span_{uuid.uuid4().hex[:16]}",
            parent_span_id=parent_span_id,
            architecture=self.architecture,
            kind=kind,
            name=name,
            status=status,
            started_at=now,
            ended_at=now,
            latency_ms=(ended - started) * 1000,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            call_id=call_id,
            detail=detail or {},
        ))

    def metrics(self) -> dict[str, int]:
        return {
            "steps": len(self.events),
            "tool_calls": sum(event.kind == "tool" for event in self.events),
            "model_calls": sum(event.kind == "model" for event in self.events),
            "input_tokens": sum(event.input_tokens for event in self.events),
            "output_tokens": sum(event.output_tokens for event in self.events),
        }


@dataclass
class RunReport:
    """所有架构最终写入相同结构，便于统一生成表格。"""

    architecture: str
    task_id: str
    run_id: str
    status: Literal["waiting_approval", "completed", "rejected", "failed"]
    success: bool
    answer: str
    citations: list[str]
    trace_id: str
    latency_ms: float
    steps: int
    tool_calls: int
    model_calls: int
    input_tokens: int
    output_tokens: int
    duplicate_submissions: int = 0
    error_type: str | None = None
    context_isolated: bool | None = None
    trace: list[TraceEvent] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["total_tokens"] = self.input_tokens + self.output_tokens
        return value


class ArtifactLedger:
    """保存草稿、发布收据和审批提交，承担业务层 exactly-once 效果。

    LangGraph Checkpoint 只知道图状态。即使图恢复后重复执行 publish Node，
    这里的 UNIQUE 约束仍保证同一幂等键只产生一个发布收据。
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS drafts (
                idempotency_key TEXT PRIMARY KEY,
                draft_id TEXT NOT NULL,
                content TEXT NOT NULL,
                citations_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS publications (
                idempotency_key TEXT PRIMARY KEY,
                publication_id TEXT NOT NULL,
                draft_id TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS approval_submissions (
                submission_id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL,
                decision TEXT NOT NULL,
                result_json TEXT
            );
        """)
        self.connection.commit()

    def save_draft(
        self, *, idempotency_key: str, content: str, citations: list[str],
    ) -> dict[str, Any]:
        row = self.connection.execute(
            "SELECT * FROM drafts WHERE idempotency_key = ?", (idempotency_key,),
        ).fetchone()
        if row:
            return {"draft_id": row["draft_id"], "replayed": True}
        draft_id = f"draft_{uuid.uuid4().hex[:12]}"
        self.connection.execute(
            "INSERT INTO drafts VALUES (?, ?, ?, ?)",
            (idempotency_key, draft_id, content, json.dumps(citations)),
        )
        self.connection.commit()
        return {"draft_id": draft_id, "replayed": False}

    def publish(self, *, idempotency_key: str, draft_id: str) -> dict[str, Any]:
        row = self.connection.execute(
            "SELECT * FROM publications WHERE idempotency_key = ?", (idempotency_key,),
        ).fetchone()
        if row:
            return {"publication_id": row["publication_id"], "replayed": True}
        publication_id = f"pub_{uuid.uuid4().hex[:12]}"
        self.connection.execute(
            "INSERT INTO publications VALUES (?, ?, ?)",
            (idempotency_key, publication_id, draft_id),
        )
        self.connection.commit()
        return {"publication_id": publication_id, "replayed": False}

    def claim_submission(
        self, *, submission_id: str, thread_id: str, decision: str,
    ) -> tuple[bool, dict[str, Any] | None]:
        """原子认领审批；重复 submission_id 不得再次恢复工作流。"""

        try:
            self.connection.execute(
                "INSERT INTO approval_submissions VALUES (?, ?, ?, NULL)",
                (submission_id, thread_id, decision),
            )
            self.connection.commit()
            return True, None
        except sqlite3.IntegrityError:
            row = self.connection.execute(
                "SELECT result_json FROM approval_submissions WHERE submission_id = ?",
                (submission_id,),
            ).fetchone()
            result = json.loads(row["result_json"]) if row and row["result_json"] else None
            return False, result

    def finish_submission(self, submission_id: str, result: dict[str, Any]) -> None:
        self.connection.execute(
            "UPDATE approval_submissions SET result_json = ? WHERE submission_id = ?",
            (json.dumps(result, ensure_ascii=False), submission_id),
        )
        self.connection.commit()

    def publication_count(self) -> int:
        row = self.connection.execute("SELECT COUNT(*) AS n FROM publications").fetchone()
        return int(row["n"])


def _terms(text: str) -> set[str]:
    """简单词法检索足以固定实验输入；这里不评估向量模型。"""

    lowered = text.lower()
    latin = set(re.findall(r"[a-z][a-z0-9-]+", lowered))
    chinese = "".join(re.findall(r"[\u4e00-\u9fff]", lowered))
    return latin | {chinese[i:i + 2] for i in range(max(0, len(chinese) - 1))}


class ResearchToolRuntime:
    """四种架构复用的确定性工具与一次性故障注入器。"""

    def __init__(
        self,
        ledger: ArtifactLedger,
        trace: TraceRecorder,
        *,
        fault: str = "none",
    ) -> None:
        self.ledger = ledger
        self.trace = trace
        self.fault = fault
        self._read_failed = False
        self.documents = json.loads(
            (ROOT / "data" / "corpus.json").read_text(encoding="utf-8")
        )
        self.by_id = {item["id"]: item for item in self.documents}

    def call(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        actor: str = "main",
    ) -> ToolResult:
        started = time.perf_counter()
        if name == "search_documents":
            result = self._search(arguments)
        elif name == "read_document":
            result = self._read(arguments)
        elif name == "save_draft":
            result = self._save(arguments)
        elif name == "publish_report":
            result = self._publish(arguments)
        else:
            result = ToolResult(
                status="error", tool=name, call_id=f"call_{uuid.uuid4().hex[:12]}",
                error_type="unknown_tool",
            )
        self.trace.record(
            kind="tool", name=name, status=result.status, started=started,
            call_id=result.call_id,
            detail={"actor": actor, "error_type": result.error_type},
        )
        return result

    @staticmethod
    def _error(tool: str, error_type: str, *, retryable: bool = False) -> ToolResult:
        return ToolResult(
            status="error", tool=tool, call_id=f"call_{uuid.uuid4().hex[:12]}",
            error_type=error_type, retryable=retryable,
        )

    def _search(self, arguments: dict[str, Any]) -> ToolResult:
        query = arguments.get("query")
        limit = arguments.get("max_results", 4)
        if not isinstance(query, str) or not query.strip() or not isinstance(limit, int):
            return self._error("search_documents", "invalid_arguments")
        query_terms = _terms(query)
        ranked = []
        for document in self.documents:
            score = len(query_terms & _terms(document["title"] + " " + document["content"]))
            if score:
                ranked.append((score, document))
        ranked.sort(key=lambda pair: (-pair[0], pair[1]["id"]))
        hits = [{"id": doc["id"], "title": doc["title"]} for _, doc in ranked[:limit]]
        return ToolResult(
            status="success", tool="search_documents",
            call_id=f"call_{uuid.uuid4().hex[:12]}", data={"hits": hits},
        )

    def _read(self, arguments: dict[str, Any]) -> ToolResult:
        doc_id = arguments.get("doc_id")
        if not isinstance(doc_id, str):
            return self._error("read_document", "invalid_arguments")
        if self.fault == "tool_exception" and not self._read_failed:
            self._read_failed = True
            return self._error("read_document", "tool_unavailable", retryable=True)
        document = self.by_id.get(doc_id)
        if document is None:
            return self._error("read_document", "not_found")
        return ToolResult(
            status="success", tool="read_document",
            call_id=f"call_{uuid.uuid4().hex[:12]}", data=document,
        )

    def _save(self, arguments: dict[str, Any]) -> ToolResult:
        content = arguments.get("content")
        citations = arguments.get("citations")
        key = arguments.get("idempotency_key")
        if not isinstance(content, str) or not isinstance(citations, list) or not isinstance(key, str):
            return self._error("save_draft", "invalid_arguments")
        data = self.ledger.save_draft(
            idempotency_key=key, content=content,
            citations=[str(value) for value in citations],
        )
        return ToolResult(
            status="success", tool="save_draft",
            call_id=f"call_{uuid.uuid4().hex[:12]}", data=data,
        )

    def _publish(self, arguments: dict[str, Any]) -> ToolResult:
        draft_id = arguments.get("draft_id")
        key = arguments.get("idempotency_key")
        approved = arguments.get("approved")
        if approved is not True:
            return self._error("publish_report", "approval_required")
        if not isinstance(draft_id, str) or not isinstance(key, str):
            return self._error("publish_report", "invalid_arguments")
        data = self.ledger.publish(idempotency_key=key, draft_id=draft_id)
        return ToolResult(
            status="success", tool="publish_report",
            call_id=f"call_{uuid.uuid4().hex[:12]}", data=data,
        )


def build_answer(task: ResearchTask, evidence: list[dict[str, Any]]) -> tuple[str, list[str]]:
    """确定性合成供恢复协议测试使用；真实基准会换成模型合成器。"""

    citations = [item["id"] for item in evidence]
    claims = "；".join(item["content"] for item in evidence)
    return f"{task.question}\n结论：{claims}", citations


def score(task: ResearchTask, answer: str, citations: list[str]) -> bool:
    """同一个完成谓词用于四种架构。"""

    return (
        bool(answer)
        and all(term.lower() in answer.lower() for term in task.required_terms)
        and all(source in citations for source in task.required_sources)
    )

