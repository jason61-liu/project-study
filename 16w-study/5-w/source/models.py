"""三种研究架构共享的数据契约与指标模型。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Protocol


Status = Literal["success", "failed", "partial", "budget_exhausted"]


@dataclass(frozen=True)
class ModelReply:
    """一次模型响应；Token 采用服务端 usage，不用字符数伪造。"""

    data: dict[str, Any]
    input_tokens: int
    output_tokens: int
    latency_ms: float
    model: str


class ResearchModel(Protocol):
    """所有架构共用的模型协议。"""

    model: str

    def complete_json(self, *, system: str, user: str, purpose: str) -> ModelReply: ...


@dataclass(frozen=True)
class ToolResult:
    """Tool Runtime 的稳定返回结构。"""

    status: Literal["success", "error", "partial"]
    tool: str
    call_id: str
    data: Any = None
    error_type: str | None = None
    retryable: bool = False
    corpus_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TraceEvent:
    """一条模型或工具 Span，不记录 API Key 和完整私有推理。"""

    kind: Literal["model", "tool", "control"]
    name: str
    status: str
    started_at: str
    ended_at: str
    latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    call_id: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunMetrics:
    architecture: str
    scenario_id: str
    repetition: int
    status: Status
    success: bool
    steps: int
    tool_calls: int
    model_calls: int
    input_tokens: int
    output_tokens: int
    latency_ms: float
    answer: str
    citations: list[str]
    trace_id: str
    failure_types: list[str]
    trace: list[TraceEvent]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["total_tokens"] = self.input_tokens + self.output_tokens
        return value


@dataclass(frozen=True)
class Scenario:
    id: str
    question: str
    fault: str
    required_terms: list[str]
    required_sources: list[str]

