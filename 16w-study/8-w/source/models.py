"""第 8 周评测系统的统一数据合同。

Task 描述“要测什么”，Trial 保存“一次实际运行发生了什么”，GraderResult
解释“为什么通过或失败”。这些对象只依赖 Pydantic，不依赖某个 Agent 框架，
因此本地任务、τ³-bench 适配任务和未来的 BFCL 任务都能进入同一 Harness。
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TaskCategory(StrEnum):
    """任务切片；CI Gate 会分别约束高风险切片。"""

    NORMAL = "normal"
    BOUNDARY = "boundary"
    FAILURE = "failure"
    ADVERSARIAL = "adversarial"
    BENCHMARK = "benchmark"


class TrialStatus(StrEnum):
    COMPLETED = "completed"
    REFUSED = "refused"
    NEEDS_CLARIFICATION = "needs_clarification"
    TOOL_ERROR = "tool_error"
    FAILED = "failed"


class GraderKind(StrEnum):
    DETERMINISTIC = "deterministic"
    LLM_RUBRIC = "llm_rubric"


class TaskInput(BaseModel):
    """传给 Agent 的用户输入；arguments 是机器可执行输入而非答案。"""

    instruction: str = Field(min_length=1)
    operation: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)


class EnvironmentSpec(BaseModel):
    """每个 Trial 的初始环境。

    visible_context 可以交给 Agent；fixture 中的工具行为也属于真实环境现象。
    Gold answer 和 Grader 配置不放在这里，防止被测 Agent 看到评分答案。
    """

    fixture_id: str = Field(min_length=1)
    tenant_id: str = "tenant-a"
    timeout_ms: int = Field(default=2_000, ge=1)
    visible_context: dict[str, Any] = Field(default_factory=dict)
    tool_behavior: str = "ok"


class SuccessCondition(BaseModel):
    """人可读成功条件；实际执行逻辑由对应 GraderSpec 提供。"""

    id: str
    description: str
    hard_gate: bool = True


class GraderSpec(BaseModel):
    """版本化评分器配置。

    check 是稳定的实现名称，例如 ``status_equals``；config 保存期望值。
    同一个 Task 可以同时配置多个确定性 Grader 和一个 LLM Rubric Grader。
    """

    id: str
    version: str = "1.0.0"
    kind: GraderKind
    check: str
    config: dict[str, Any] = Field(default_factory=dict)
    hard_gate: bool = True


class SourceRef(BaseModel):
    """任务来源和版本，避免把本地任务与公开 Benchmark 混为一谈。"""

    name: str
    revision: str
    url: str | None = None
    original_task_id: str | None = None


class EvalTask(BaseModel):
    """统一 Task Schema。"""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]+$")
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    category: TaskCategory
    source: SourceRef
    input: TaskInput
    environment: EnvironmentSpec
    success_conditions: list[SuccessCondition] = Field(min_length=1)
    graders: list[GraderSpec] = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def condition_ids_are_graded(self) -> "EvalTask":
        """每个成功条件至少要有一个同 ID 的 Grader，防止只写不评。"""

        condition_ids = {item.id for item in self.success_conditions}
        grader_ids = {item.id for item in self.graders}
        missing = condition_ids - grader_ids
        if missing:
            raise ValueError(f"成功条件缺少 Grader: {sorted(missing)}")
        return self

    def agent_view(self) -> "AgentTaskView":
        """只向 Agent 暴露输入和初始环境，不泄漏成功条件与 Gold 数据。"""

        return AgentTaskView(
            task_id=self.id,
            category=self.category,
            input=self.input,
            environment=self.environment,
        )


class AgentTaskView(BaseModel):
    task_id: str
    category: TaskCategory
    input: TaskInput
    environment: EnvironmentSpec


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    call_id: str
    status: str = "success"


class AgentOutput(BaseModel):
    status: TrialStatus
    answer: str
    citations: list[str] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    actions: list[dict[str, Any]] = Field(default_factory=list)
    safety_flags: list[str] = Field(default_factory=list)
    input_tokens: int | None = None
    output_tokens: int | None = None


class TraceEvent(BaseModel):
    """Trace 中的单个 Span；detail 禁止写入 API Key 或 Access Token。"""

    trace_id: str
    span_id: str
    parent_span_id: str | None = None
    kind: str
    name: str
    status: str
    started_at: datetime
    ended_at: datetime
    latency_ms: float = Field(ge=0)
    detail: dict[str, Any] = Field(default_factory=dict)


class GraderResult(BaseModel):
    grader_id: str
    grader_version: str
    kind: GraderKind
    passed: bool | None
    score: float | None = Field(default=None, ge=0, le=1)
    hard_gate: bool
    reason: str
    evidence: dict[str, Any] = Field(default_factory=dict)
    status: str = "completed"


class TrialRecord(BaseModel):
    """可落盘的一次完整 Trial。"""

    schema_version: str = "1.0"
    run_id: str
    task_id: str
    task_version: str
    category: TaskCategory
    agent_version: str
    trace_id: str
    started_at: datetime
    ended_at: datetime
    latency_ms: float
    output: AgentOutput
    grades: list[GraderResult]
    strict_success: bool
    trace_path: str


class AggregateReport(BaseModel):
    schema_version: str = "1.0"
    run_id: str
    agent_version: str
    task_count: int
    trial_count: int
    strict_success_rate: float
    category_success_rate: dict[str, float]
    grader_pass_rate: dict[str, float]
    status_counts: dict[str, int]
    average_latency_ms: float
    total_input_tokens: int | None
    total_output_tokens: int | None
    judge_requested: bool = False
    judge_expected: int = 0
    judge_completed: int
    judge_errors: int = 0
    judge_input_tokens: int = 0
    judge_output_tokens: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class GateConfig(BaseModel):
    min_task_count: int = 50
    min_overall_success_rate: float = 0.95
    min_category_success_rate: dict[str, float] = Field(default_factory=dict)
    max_success_drop: float = 0.02
    max_llm_judge_misjudgment_rate: float = 0.10


class GateResult(BaseModel):
    passed: bool
    baseline_run_id: str
    candidate_run_id: str
    violations: list[str]
    checked_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
