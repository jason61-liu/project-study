"""OpenAI Agents SDK 版本：原生 Tool、Handoff、Guardrail 和审批恢复。

模型默认使用 DeepSeek 的 OpenAI-compatible Chat Completions API。SDK 自己负责
model -> tool -> observation -> model 循环，宿主应用仍负责身份、幂等和状态落盘。
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
from typing import Any
import uuid

from agents import (
    Agent, GuardrailFunctionOutput, ModelSettings, RunContextWrapper, RunHooks,
    RunConfig, RunState, Runner, function_tool, handoff, input_guardrail,
    output_guardrail,
)
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from agents.exceptions import AgentsException
from openai import AsyncOpenAI
from pydantic import BaseModel

from common import (
    ArtifactLedger, ResearchTask, ResearchToolRuntime, RunReport,
    TraceEvent, TraceRecorder, score,
)


class ReportOutput(BaseModel):
    answer: str
    citations: list[str]


@dataclass
class SDKContext:
    """只存在宿主进程；Token 和 Client 不会被序列化进模型输入。"""

    task: ResearchTask
    run_id: str
    runtime: ResearchToolRuntime
    ledger: ArtifactLedger
    trace: TraceRecorder
    draft_id: str | None = None


@input_guardrail(name="research_scope", run_in_parallel=False)
async def research_scope_guardrail(
    context: RunContextWrapper[SDKContext], agent: Agent, input_value: str | list[Any],
) -> GuardrailFunctionOutput:
    text = input_value if isinstance(input_value, str) else json.dumps(input_value, ensure_ascii=False)
    allowed = bool(text.strip()) and len(text) <= 8_000
    return GuardrailFunctionOutput(
        output_info={"allowed": allowed, "agent": agent.name},
        tripwire_triggered=not allowed,
    )


@output_guardrail(name="citation_boundary")
async def citation_guardrail(
    context: RunContextWrapper[SDKContext], agent: Agent, output: Any,
) -> GuardrailFunctionOutput:
    # DeepSeek 的兼容端点当前不接受 SDK 为 Pydantic output_type 发送的
    # response_format，因此模型返回 JSON 文本，宿主 Guardrail 再做严格解析。
    try:
        parsed = ReportOutput.model_validate_json(output) if isinstance(output, str) else output
    except Exception:
        parsed = None
    citations = list(getattr(parsed, "citations", [])) if parsed else []
    allowed = set(context.context.task.required_sources)
    has_required = allowed.issubset(set(citations))
    return GuardrailFunctionOutput(
        output_info={"required_sources_present": has_required, "agent": agent.name},
        tripwire_triggered=not has_required,
    )


class SDKHooks(RunHooks[SDKContext]):
    """把 SDK 生命周期投影到四架构共用的 TraceEvent。"""

    def __init__(
        self,
        trace: TraceRecorder,
        *,
        initial_input_tokens: int = 0,
        initial_output_tokens: int = 0,
    ) -> None:
        self.trace = trace
        self._started: dict[str, float] = {}
        # RunState 恢复后 Usage 仍是整个运行的累计值。因此新的 Hook 必须从
        # 中断前累计量开始做差，不能从 0 开始，否则恢复后的第一轮会重复计费。
        self._last_input_tokens = initial_input_tokens
        self._last_output_tokens = initial_output_tokens

    async def on_llm_start(self, context, agent, system_prompt, input_items) -> None:
        self._started[f"llm:{agent.name}"] = time.perf_counter()

    async def on_llm_end(self, context, agent, response) -> None:
        key = f"llm:{agent.name}"
        # context.usage 是整个 Run 的累计值；Span 必须记录相邻快照的增量，
        # 否则汇总每个 Span 时会重复计算前几轮 Token。
        input_tokens = context.usage.input_tokens - self._last_input_tokens
        output_tokens = context.usage.output_tokens - self._last_output_tokens
        self._last_input_tokens = context.usage.input_tokens
        self._last_output_tokens = context.usage.output_tokens
        self.trace.record(
            kind="model", name=agent.name, status="success",
            started=self._started.pop(key, time.perf_counter()),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    async def on_handoff(self, context, from_agent, to_agent) -> None:
        self.trace.record(
            kind="control", name="handoff", status="success",
            started=time.perf_counter(),
            detail={"from": from_agent.name, "to": to_agent.name},
        )


class AgentsSDKWorkflow:
    architecture = "agents_sdk"

    def __init__(self, root: Path, *, fault: str = "none") -> None:
        root.mkdir(parents=True, exist_ok=True)
        self.root = root
        self.ledger = ArtifactLedger(root / "business.sqlite")
        self.fault = fault
        self._contexts: dict[str, SDKContext] = {}
        self._agents: dict[str, Agent] = {}

    def _state_path(self, run_id: str) -> Path:
        return self.root / f"{run_id}.agents-state.json"

    def _metadata_path(self, run_id: str) -> Path:
        return self.root / f"{run_id}.metadata.json"

    @staticmethod
    def _model() -> OpenAIChatCompletionsModel:
        key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("缺少 DEEPSEEK_API_KEY/OPENAI_API_KEY")
        client = AsyncOpenAI(
            api_key=key,
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com"),
            timeout=90,
            max_retries=0,
        )
        return OpenAIChatCompletionsModel(
            model=os.getenv("AGENT_TEST_MODEL", "deepseek-v4-pro"),
            openai_client=client,
        )

    def _build_agent(self, context: SDKContext) -> Agent[SDKContext]:
        runtime = context.runtime

        @function_tool(timeout=20)
        async def search_documents(query: str, max_results: int = 4) -> str:
            """搜索研究语料，返回文档 ID 和标题。"""

            return json.dumps(runtime.call(
                "search_documents", {"query": query, "max_results": max_results},
            ).to_dict(), ensure_ascii=False)

        @function_tool(timeout=20)
        async def read_document(doc_id: str) -> str:
            """读取指定文档；瞬时错误允许模型再次调用一次。"""

            return json.dumps(runtime.call(
                "read_document", {"doc_id": doc_id},
            ).to_dict(), ensure_ascii=False)

        @function_tool(timeout=20)
        async def save_draft(answer: str, citations: list[str]) -> str:
            """按运行 ID 幂等保存草稿，并返回 draft_id。"""

            result = runtime.call("save_draft", {
                "content": answer, "citations": citations,
                "idempotency_key": f"{context.run_id}:draft:v1",
            })
            context.draft_id = result.data["draft_id"] if result.data else None
            return json.dumps(result.to_dict(), ensure_ascii=False)

        @function_tool(needs_approval=True, timeout=20)
        async def publish_report(draft_id: str) -> str:
            """发布已保存草稿；SDK 会在执行前产生可恢复审批中断。"""

            result = runtime.call("publish_report", {
                "draft_id": draft_id, "approved": True,
                "idempotency_key": f"{context.run_id}:publish:v1",
            })
            return json.dumps(result.to_dict(), ensure_ascii=False)

        compliance = Agent[SDKContext](
            name="compliance_specialist",
            handoff_description="研究完成且草稿保存后，接管并发布报告。",
            instructions=(
                "你是合规发布 Agent。检查答案必须包含 Runtime、Checkpoint、幂等、审批、"
                "Subagent、上下文。若已提供 draft_id，必须调用 publish_report；工具批准后，"
                "最终只输出 JSON，不要 Markdown：{answer:string,citations:string[]}。"
            ),
            model=self._model(),
            tools=[publish_report],
            output_guardrails=[citation_guardrail],
            model_settings=ModelSettings(temperature=0, max_tokens=900),
        )
        researcher = Agent[SDKContext](
            name="researcher",
            instructions=(
                "使用 search_documents 检索，逐个 read_document。工具若返回 retryable=true，"
                "最多重试一次。只能依据成功读取的文档写答案；答案必须含 Runtime、Checkpoint、"
                "幂等、审批、Subagent、上下文，并引用文档 ID runtime-boundary、"
                "durable-execution、human-approval、subagent-context。随后调用 save_draft，"
                "从结果取得 draft_id，再 Handoff 给 compliance_specialist。"
            ),
            model=self._model(),
            tools=[search_documents, read_document, save_draft],
            handoffs=[handoff(compliance)],
            input_guardrails=[research_scope_guardrail],
            model_settings=ModelSettings(temperature=0, max_tokens=900),
        )
        return researcher

    async def start(self, task: ResearchTask, *, run_id: str | None = None) -> RunReport:
        started = time.perf_counter()
        run_id = run_id or f"sdk_{uuid.uuid4().hex}"
        trace = TraceRecorder(self.architecture)
        context = SDKContext(
            task=task, run_id=run_id,
            runtime=ResearchToolRuntime(self.ledger, trace, fault=self.fault),
            ledger=self.ledger, trace=trace,
        )
        agent = self._build_agent(context)
        self._contexts[run_id] = context
        self._agents[run_id] = agent
        try:
            result = await Runner.run(
                agent, task.question, context=context, max_turns=12,
                hooks=SDKHooks(trace),
                run_config=RunConfig(tracing_disabled=True),
            )
        except AgentsException as error:
            # Guardrail、最大轮数和模型违反工具边界都是本轮业务失败，不能使
            # 整个批次崩溃。只记录错误类型，不把模型输出或异常正文写入 Trace。
            trace.record(
                kind="control", name="guardrail_tripwire", status="failed",
                started=time.perf_counter(),
                detail={"error_type": type(error).__name__},
            )
            return self._report(
                task, run_id, "failed", None, trace, started,
                error_type=type(error).__name__,
            )
        if not result.interruptions:
            return self._report(task, run_id, "failed", result.final_output, trace, started,
                                error_type="approval_not_requested")
        state_json = result.to_state().to_json(context_serializer=lambda _: {
            "run_id": run_id,
        })
        self._state_path(run_id).write_text(
            json.dumps(state_json, ensure_ascii=False), encoding="utf-8",
        )
        self._metadata_path(run_id).write_text(json.dumps({
            "task": task.__dict__,
            "trace_id": trace.trace_id,
            "draft_id": context.draft_id,
            "trace": [event.__dict__ for event in trace.events],
        }, ensure_ascii=False), encoding="utf-8")
        return self._report(task, run_id, "waiting_approval", result.final_output,
                            trace, started)

    async def resume(
        self, run_id: str, *, decision: str, submission_id: str,
    ) -> RunReport:
        started = time.perf_counter()
        context = self._contexts.get(run_id)
        agent = self._agents.get(run_id)
        if context is None or agent is None:
            # 模拟进程重启：凭磁盘元数据重建不可序列化的 Tool Runtime、Client
            # 和 Agent Definition，再把它们作为 context_override 注入 RunState。
            metadata = json.loads(self._metadata_path(run_id).read_text(encoding="utf-8"))
            task = ResearchTask(**metadata["task"])
            trace = TraceRecorder(self.architecture, metadata["trace_id"])
            trace.events = [TraceEvent(**event) for event in metadata.get("trace", [])]
            context = SDKContext(
                task=task, run_id=run_id,
                runtime=ResearchToolRuntime(self.ledger, trace, fault=self.fault),
                ledger=self.ledger, trace=trace, draft_id=metadata.get("draft_id"),
            )
            agent = self._build_agent(context)
            self._contexts[run_id] = context
            self._agents[run_id] = agent
        first, cached = self.ledger.claim_submission(
            submission_id=submission_id, thread_id=run_id, decision=decision,
        )
        if not first:
            report = self._report(
                context.task, run_id, (cached or {}).get("status", "waiting_approval"),
                None, context.trace, started,
            )
            report.duplicate_submissions = 1
            return report

        state_json = json.loads(self._state_path(run_id).read_text(encoding="utf-8"))
        state = await RunState.from_json(
            agent, state_json, context_override=context,
        )
        # 用已持久化 Model Span 求中断前累计量，避免依赖 SDK 的私有字段。
        # RunState 恢复后的 Usage 是累计值，新的 Hook 从该基线继续做差。
        prior_input_tokens = sum(event.input_tokens for event in context.trace.events)
        prior_output_tokens = sum(event.output_tokens for event in context.trace.events)
        interruptions = state.get_interruptions()
        if len(interruptions) != 1:
            raise RuntimeError(f"预期一个审批中断，实际为 {len(interruptions)}")
        approval_started = time.perf_counter()
        if decision == "approve":
            state.approve(interruptions[0])
            approval_status = "approved"
        else:
            state.reject(interruptions[0], rejection_message="用户拒绝发布；立即结束且不要再次调用工具。")
            approval_status = "rejected"
        context.trace.record(
            kind="approval", name="human_decision", status=approval_status,
            started=approval_started,
        )
        try:
            result = await Runner.run(
                agent, state, context=context, max_turns=12,
                hooks=SDKHooks(
                context.trace,
                    initial_input_tokens=prior_input_tokens,
                    initial_output_tokens=prior_output_tokens,
                ),
                run_config=RunConfig(tracing_disabled=True),
            )
        except AgentsException as error:
            context.trace.record(
                kind="control", name="guardrail_tripwire", status="failed",
                started=time.perf_counter(),
                detail={"error_type": type(error).__name__},
            )
            self.ledger.finish_submission(submission_id, {"status": "failed"})
            return self._report(
                context.task, run_id, "failed", None, context.trace, started,
                error_type=type(error).__name__,
            )
        status = "completed" if decision == "approve" else "rejected"
        self.ledger.finish_submission(submission_id, {"status": status})
        return self._report(context.task, run_id, status, result.final_output,
                            context.trace, started)

    def _report(
        self, task: ResearchTask, run_id: str, status: str, output: Any,
        trace: TraceRecorder, started: float, *, error_type: str | None = None,
    ) -> RunReport:
        parsed = None
        if isinstance(output, str):
            try:
                parsed = ReportOutput.model_validate_json(output)
            except Exception:
                parsed = None
        elif output is not None:
            parsed = output
        answer = str(getattr(parsed, "answer", ""))
        citations = list(getattr(parsed, "citations", [])) if parsed else []
        metrics = trace.metrics()
        return RunReport(
            architecture=self.architecture, task_id=task.id, run_id=run_id,
            status=status, success=status == "completed" and score(task, answer, citations),
            answer=answer, citations=citations, trace_id=trace.trace_id,
            latency_ms=(time.perf_counter() - started) * 1000,
            error_type=error_type, trace=trace.events, **metrics,
        )
