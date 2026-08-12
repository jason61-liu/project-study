"""可恢复 Agent 实验使用的结构化计划、失败和持久化状态模型。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Literal


class FailureClass(str, Enum):
    """恢复决策使用的四类失败，而不是底层异常类型的简单复制。"""

    RETRYABLE = "retryable"
    REPLAN_REQUIRED = "replan_required"
    HUMAN_REQUIRED = "human_required"
    UNRECOVERABLE = "unrecoverable"


ERROR_CLASSIFICATION: dict[str, FailureClass] = {
    "retriever_unavailable": FailureClass.RETRYABLE,
    "tool_unavailable": FailureClass.RETRYABLE,
    "state_conflict": FailureClass.RETRYABLE,
    "context_lost": FailureClass.RETRYABLE,
    "plan_invalidated": FailureClass.REPLAN_REQUIRED,
    "policy_confirmation_required": FailureClass.HUMAN_REQUIRED,
    "permission_denied": FailureClass.HUMAN_REQUIRED,
    "invalid_arguments": FailureClass.UNRECOVERABLE,
    "not_found": FailureClass.UNRECOVERABLE,
    "unknown_tool": FailureClass.UNRECOVERABLE,
}


def classify_failure(error_type: str) -> FailureClass:
    """把实现层错误映射到稳定恢复语义；未知错误默认停止，避免盲重试。"""

    return ERROR_CLASSIFICATION.get(error_type, FailureClass.UNRECOVERABLE)


ItemStatus = Literal["pending", "in_progress", "completed", "partial", "failed", "blocked"]
PlanStatus = Literal["running", "completed", "partial", "failed", "waiting_human"]


@dataclass
class PlanItem:
    """一个可独立 Checkpoint 的计划节点。"""

    id: str
    objective: str
    action: Literal["search", "read_hits", "synthesize"]
    depends_on: list[str]
    completion_condition: str
    arguments: dict[str, Any] = field(default_factory=dict)
    status: ItemStatus = "pending"
    attempts: int = 0
    result: dict[str, Any] | None = None
    error_type: str | None = None
    failure_class: str | None = None


@dataclass
class FailureRecord:
    item_id: str
    error_type: str
    failure_class: str
    attempt: int
    action: str


@dataclass
class StructuredPlan:
    """磁盘中的权威恢复状态；version 每次成功提交都递增。"""

    plan_id: str
    task_id: str
    architecture: str
    revision: int
    version: int
    status: PlanStatus
    items: list[PlanItem]
    failures: list[FailureRecord] = field(default_factory=list)
    execution_records: dict[str, dict[str, Any]] = field(default_factory=dict)
    delegations: dict[str, dict[str, Any]] = field(default_factory=dict)
    checkpoints: int = 0
    context_rebuilds: int = 0
    duplicate_delegations_suppressed: int = 0

    def item(self, item_id: str) -> PlanItem:
        return next(item for item in self.items if item.id == item_id)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "StructuredPlan":
        return cls(
            plan_id=value["plan_id"], task_id=value["task_id"],
            architecture=value["architecture"], revision=value["revision"],
            version=value["version"], status=value["status"],
            items=[PlanItem(**item) for item in value["items"]],
            failures=[FailureRecord(**failure) for failure in value.get("failures", [])],
            execution_records=value.get("execution_records", {}),
            delegations=value.get("delegations", {}),
            checkpoints=value.get("checkpoints", 0),
            context_rebuilds=value.get("context_rebuilds", 0),
            duplicate_delegations_suppressed=value.get("duplicate_delegations_suppressed", 0),
        )


@dataclass(frozen=True)
class RecoveryTask:
    id: str
    question: str
    search_query: str
    required_terms: list[str]
    required_sources: list[str]
    fault: str = "none"


@dataclass
class RecoveryRun:
    architecture: str
    task_id: str
    repetition: int
    status: str
    success: bool
    plan_id: str
    plan_revision: int
    completed_items: int
    total_items: int
    tool_calls: int
    model_calls: int
    input_tokens: int
    output_tokens: int
    latency_ms: float
    answer: str
    citations: list[str]
    failure_counts: dict[str, int]
    checkpoints: int
    context_rebuilds: int
    duplicate_delegations_suppressed: int

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["total_tokens"] = self.input_tokens + self.output_tokens
        return value
