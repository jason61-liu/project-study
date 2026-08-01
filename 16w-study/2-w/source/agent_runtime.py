"""支持结构化输出、流式响应和工具循环的最小 Agent Runtime。

Transport 被设计成可注入协议：真实项目可以接 OpenAI SDK，测试则使用脚本化
Transport。Runtime 只信任通过 Schema、业务注册表和超时边界校验后的数据。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
import re
from time import monotonic
from typing import Any, Callable, Iterable, Mapping, Protocol
from uuid import uuid4


JSON = dict[str, Any]


class SchemaValidationError(ValueError):
    """JSON 值不满足本实验支持的 Schema 子集。"""


class ModelCallError(RuntimeError):
    """模型响应失败或流没有成功终止。"""


class ModelIncompleteError(ModelCallError):
    """模型响应因 Token 上限等原因未完整生成。"""


def validate_json(value: Any, schema: Mapping[str, Any], path: str = "$") -> None:
    """验证常用 JSON Schema 子集；失败时抛出带字段路径的异常。

    支持 type、required、properties、additionalProperties、enum、pattern、
    minimum、maximum、items、minItems 和 maxItems。生产系统应替换为完整的
    JSON Schema 实现，但业务与权限校验仍需单独保留。
    """

    # enum 与 type 是两个独立约束：即使类型正确，也不能接受枚举之外的值。
    if "enum" in schema and value not in schema["enum"]:
        raise SchemaValidationError(f"{path}: 必须是 {schema['enum']} 之一")

    # JSON Schema 的 type 可以是单个字符串，也可以是 ["string", "null"] 这类联合类型。
    expected = schema.get("type")
    allowed = expected if isinstance(expected, list) else [expected]
    if expected is not None and not any(_matches_type(value, kind) for kind in allowed):
        raise SchemaValidationError(f"{path}: 期望类型 {expected}，实际为 {type(value).__name__}")

    if value is None:
        return

    if isinstance(value, dict):
        properties = schema.get("properties", {})
        # required 只检查字段是否存在；字段值是否合法由下面的递归校验负责。
        for name in schema.get("required", []):
            if name not in value:
                raise SchemaValidationError(f"{path}.{name}: 缺少必填字段")
        # 工具参数建议关闭额外字段，防止拼写错误或模型生成的未知字段被静默接受。
        if schema.get("additionalProperties") is False:
            extras = sorted(set(value) - set(properties))
            if extras:
                raise SchemaValidationError(f"{path}: 存在未声明字段 {extras}")
        for name, child in value.items():
            if name in properties:
                validate_json(child, properties[name], f"{path}.{name}")

    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            raise SchemaValidationError(f"{path}: 数组元素过少")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            raise SchemaValidationError(f"{path}: 数组元素过多")
        if "items" in schema:
            for index, child in enumerate(value):
                validate_json(child, schema["items"], f"{path}[{index}]")

    if isinstance(value, str) and "pattern" in schema:
        if re.fullmatch(schema["pattern"], value) is None:
            raise SchemaValidationError(f"{path}: 不匹配 pattern={schema['pattern']!r}")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            raise SchemaValidationError(f"{path}: 小于 minimum={schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            raise SchemaValidationError(f"{path}: 大于 maximum={schema['maximum']}")


def _matches_type(value: Any, kind: str) -> bool:
    """判断 Python 值是否对应 JSON Schema 类型。

    Python 中 bool 是 int 的子类，所以 integer/number 必须显式排除 bool；否则
    ``True`` 会被错误地当成数字 1 接受。
    """

    checks: dict[str, Callable[[Any], bool]] = {
        "null": lambda item: item is None,
        "object": lambda item: isinstance(item, dict),
        "array": lambda item: isinstance(item, list),
        "string": lambda item: isinstance(item, str),
        "boolean": lambda item: isinstance(item, bool),
        "integer": lambda item: isinstance(item, int) and not isinstance(item, bool),
        "number": lambda item: isinstance(item, (int, float)) and not isinstance(item, bool),
    }
    if kind not in checks:
        raise SchemaValidationError(f"不支持的 Schema type: {kind}")
    return checks[kind](value)


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    name: str
    arguments: JSON


@dataclass(frozen=True)
class ModelReply:
    text: str = ""
    structured_output: JSON | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    refusal: str | None = None


@dataclass
class CallTrace:
    trace_id: str
    run_id: str
    kind: str
    name: str
    started_at: str
    ended_at: str | None = None
    status: str = "in_progress"
    result: JSON | None = None


class TraceRecorder:
    """记录一次 Agent Run 内所有模型调用和工具调用。

    ``run_id`` 关联整条任务链，``trace_id`` 标识单次实际调用。一次模型失败后
    如果发生重试，每次尝试都应拥有新的 trace_id，便于计算真实延迟和失败率。
    """

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.entries: list[CallTrace] = []

    def start(self, kind: str, name: str) -> CallTrace:
        # 先以 in_progress 落一条记录，保证调用过程中崩溃时仍能看到开始时间。
        entry = CallTrace(
            trace_id=uuid4().hex,
            run_id=self.run_id,
            kind=kind,
            name=name,
            started_at=_utc_now(),
        )
        self.entries.append(entry)
        return entry

    @staticmethod
    def finish(entry: CallTrace, status: str, result: JSON | None = None) -> None:
        # 状态和结束时间在同一个出口写入，避免出现 succeeded 但没有 ended_at 的记录。
        entry.ended_at = _utc_now()
        entry.status = status
        entry.result = result


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ModelTransport(Protocol):
    """ModelAdapter 所需的最小底层接口。

    Transport 只负责供应商 SDK/HTTP 协议；它不执行工具，也不决定 Agent 是否完成。
    ``timeout_s`` 是 Runtime 根据总体 Deadline 裁剪后的本次调用上限。
    """

    def complete(self, request: JSON, timeout_s: float) -> Mapping[str, Any]: ...

    def stream(self, request: JSON, timeout_s: float) -> Iterable[Mapping[str, Any]]: ...


class ModelAdapter:
    """把供应商响应归一化为 Runtime 可以消费的 ``ModelReply``。

    上层 Runtime 不需要知道供应商使用 Responses API 还是 Chat Completions，只处理
    文本、结构化对象、Tool Call 和拒答这四种语义结果。
    """

    def __init__(self, transport: ModelTransport, model: str) -> None:
        self.transport = transport
        self.model = model

    def call(
        self,
        input_items: list[JSON],
        tools: Iterable["ToolSpec"],
        recorder: TraceRecorder,
        *,
        response_schema: Mapping[str, Any] | None = None,
        stream: bool = False,
        timeout_s: float,
        tool_choice: str | None = None,
    ) -> ModelReply:
        """执行一次真实模型调用并记录完整 Trace 生命周期。"""

        trace = recorder.start("model", self.model)
        request = self._build_request(
            input_items,
            tools,
            response_schema,
            stream,
            tool_choice,
        )
        try:
            # 流式和非流式最终都必须收敛成同一种 ModelReply，避免 Agent Loop 写两套逻辑。
            if stream:
                reply = self._consume_stream(self.transport.stream(request, timeout_s))
            else:
                reply = self._parse_response(self.transport.complete(request, timeout_s))
            reply = self._validate_structured_output(reply, response_schema)
        except TimeoutError as exc:
            # 超时与普通失败分开记录，调用方才能决定是否安全重试。
            recorder.finish(trace, "timeout", {"error": str(exc)})
            raise
        except SchemaValidationError as exc:
            recorder.finish(trace, "invalid", {"error": str(exc)})
            raise
        except Exception as exc:
            recorder.finish(trace, "failed", {"error": str(exc)})
            raise

        # 拒答是模型正常返回的一种业务状态，不等同于网络或服务端失败。
        status = "refused" if reply.refusal else "succeeded"
        recorder.finish(
            trace,
            status,
            {
                "tool_call_count": len(reply.tool_calls),
                "has_text": bool(reply.text),
                "has_structured_output": reply.structured_output is not None,
            },
        )
        return reply

    def _build_request(
        self,
        input_items: list[JSON],
        tools: Iterable["ToolSpec"],
        response_schema: Mapping[str, Any] | None,
        stream: bool,
        tool_choice: str | None,
    ) -> JSON:
        """构造与 OpenAI Responses API 相近的供应商无关请求对象。"""

        request: JSON = {
            "model": self.model,
            "input": input_items,
            "tools": [tool.as_model_tool() for tool in tools],
            "stream": stream,
        }
        if response_schema is not None:
            # strict Schema 约束模型解码，但应用侧仍会再次调用 validate_json。
            request["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "agent_result",
                    "strict": True,
                    "schema": dict(response_schema),
                }
            }
        if tool_choice is not None:
            # Prompt 里的“必须调用工具”不是可靠控制边界；tool_choice 由 Runtime 确定。
            request["tool_choice"] = tool_choice
        return request

    def _parse_response(self, response: Mapping[str, Any]) -> ModelReply:
        """解析完整响应，并保留 message、function_call 与 refusal 的语义差异。"""

        status = response.get("status", "completed")
        if status == "failed":
            raise ModelCallError(str(response.get("error", "模型调用失败")))
        if status == "incomplete":
            raise ModelIncompleteError(str(response.get("incomplete_details", "响应不完整")))

        texts: list[str] = []
        refusals: list[str] = []
        calls: list[ToolCall] = []
        # 一个 Response 可以同时包含多个 Output Item，不能只读取 output[0]。
        for item in response.get("output", []):
            item_type = item.get("type")
            if item_type == "function_call":
                calls.append(self._tool_call_from_item(item))
            elif item_type == "message":
                for part in item.get("content", []):
                    if part.get("type") == "output_text":
                        texts.append(part.get("text", ""))
                    elif part.get("type") == "refusal":
                        refusals.append(part.get("refusal", "模型拒绝处理"))
        return ModelReply(text="".join(texts), tool_calls=tuple(calls), refusal="".join(refusals) or None)

    def _consume_stream(self, events: Iterable[Mapping[str, Any]]) -> ModelReply:
        """消费语义化流事件，直到明确收到成功终态。

        文本 Delta 和工具参数 Delta 都可能按任意字符边界切分。这里仅负责累积，
        工具参数必须等到流结束后才解析和执行。
        """

        text_parts: list[str] = []
        refusal_parts: list[str] = []
        calls: dict[int, JSON] = {}
        completed = False

        for event in events:
            event_type = event.get("type")
            if event_type == "response.output_item.added":
                item = dict(event.get("item", {}))
                if item.get("type") == "function_call":
                    calls[int(event.get("output_index", len(calls)))] = item
            elif event_type == "response.output_text.delta":
                text_parts.append(str(event.get("delta", "")))
            elif event_type == "response.refusal.delta":
                refusal_parts.append(str(event.get("delta", "")))
            elif event_type == "response.function_call_arguments.delta":
                index = int(event.get("output_index", 0))
                # output_index 用于区分并行 Tool Call 的参数缓冲区，避免参数串流。
                calls.setdefault(index, {"type": "function_call", "arguments": ""})
                calls[index]["arguments"] = calls[index].get("arguments", "") + str(event.get("delta", ""))
            elif event_type == "response.function_call_arguments.done":
                index = int(event.get("output_index", 0))
                calls.setdefault(index, {"type": "function_call"})
                if "arguments" in event:
                    calls[index]["arguments"] = event["arguments"]
            elif event_type == "response.output_item.done":
                item = event.get("item", {})
                if item.get("type") == "function_call":
                    index = int(event.get("output_index", 0))
                    merged = calls.setdefault(index, {"type": "function_call"})
                    for key in ("call_id", "name", "arguments"):
                        if key in item:
                            merged[key] = item[key]
            elif event_type == "response.completed":
                # 只有 completed 才能证明整条流成功结束；连接关闭本身不是成功信号。
                completed = True
            elif event_type == "response.incomplete":
                raise ModelIncompleteError(str(event.get("response", {}).get("incomplete_details", "响应不完整")))
            elif event_type in {"response.failed", "error"}:
                raise ModelCallError(str(event.get("error") or event.get("response", {}).get("error") or "流式响应失败"))

        if not completed:
            raise ModelCallError("流在 response.completed 前结束")

        tool_calls = tuple(self._tool_call_from_item(calls[index]) for index in sorted(calls))
        return ModelReply(
            text="".join(text_parts),
            tool_calls=tool_calls,
            refusal="".join(refusal_parts) or None,
        )

    def _tool_call_from_item(self, item: Mapping[str, Any]) -> ToolCall:
        """在完整参数到达后解析 Tool Call，并验证协议必需字段。"""

        try:
            arguments = json.loads(str(item.get("arguments", "")))
        except json.JSONDecodeError as exc:
            raise SchemaValidationError(f"工具参数不是完整 JSON: {exc.msg}") from exc
        if not isinstance(arguments, dict):
            raise SchemaValidationError("工具参数根节点必须是 object")
        call_id = str(item.get("call_id", ""))
        name = str(item.get("name", ""))
        # call_id 是工具结果回传时的因果关联键，不能用 output_index 替代。
        if not call_id or not name:
            raise SchemaValidationError("function_call 缺少 call_id 或 name")
        return ToolCall(call_id=call_id, name=name, arguments=arguments)

    @staticmethod
    def _validate_structured_output(
        reply: ModelReply,
        response_schema: Mapping[str, Any] | None,
    ) -> ModelReply:
        """对最终文本做第二次本地 JSON 解析和 Schema 校验。"""

        # Tool Call 和拒答不是最终结构化回答，不应强行按 final schema 解析。
        if response_schema is None or reply.tool_calls or reply.refusal:
            return reply
        try:
            value = json.loads(reply.text)
        except json.JSONDecodeError as exc:
            raise SchemaValidationError(f"结构化输出不是合法 JSON: {exc.msg}") from exc
        if not isinstance(value, dict):
            raise SchemaValidationError("结构化输出根节点必须是 object")
        validate_json(value, response_schema)
        return ModelReply(text=reply.text, structured_output=value)


@dataclass(frozen=True)
class ToolSpec:
    """工具公开协议与本地执行函数的绑定。"""

    name: str
    description: str
    parameters: Mapping[str, Any]
    handler: Callable[..., Any]

    def as_model_tool(self) -> JSON:
        # 这里只把名称、描述和参数 Schema 暴露给模型，handler 永远留在可信执行侧。
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": dict(self.parameters),
            "strict": True,
        }


@dataclass(frozen=True)
class AgentLimits:
    """Agent 的硬执行边界；局部超时始终受总体 Run Deadline 约束。"""

    max_steps: int = 6
    run_timeout_s: float = 30.0
    model_timeout_s: float = 15.0
    tool_timeout_s: float = 5.0

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ValueError("max_steps 必须大于 0")
        if min(self.run_timeout_s, self.model_timeout_s, self.tool_timeout_s) <= 0:
            raise ValueError("所有超时必须大于 0")


class RunStatus(str, Enum):
    COMPLETED = "completed"
    MAX_STEPS = "max_steps"
    EARLY_STOPPED = "early_stopped"
    DEADLINE = "deadline"
    MODEL_REFUSED = "model_refused"
    MODEL_TIMEOUT = "model_timeout"
    MODEL_ERROR = "model_error"
    INVALID_MODEL_OUTPUT = "invalid_model_output"


@dataclass
class AgentState:
    run_id: str
    input_items: list[JSON]
    observations: list[JSON] = field(default_factory=list)
    steps: int = 0


@dataclass(frozen=True)
class RunResult:
    run_id: str
    status: RunStatus
    steps: int
    text: str = ""
    structured_output: JSON | None = None
    observations: tuple[JSON, ...] = ()
    traces: tuple[CallTrace, ...] = ()
    error: str | None = None


StopCondition = Callable[[AgentState], bool]


class AgentRuntime:
    """执行有界的 model → tool → observation → model 循环。"""

    def __init__(
        self,
        model: ModelAdapter,
        tools: Iterable[ToolSpec] = (),
        *,
        limits: AgentLimits | None = None,
        response_schema: Mapping[str, Any] | None = None,
        stop_condition: StopCondition | None = None,
        force_first_tool_call: bool = False,
    ) -> None:
        self.model = model
        self.tools = {tool.name: tool for tool in tools}
        self.limits = limits or AgentLimits()
        self.response_schema = response_schema
        self.stop_condition = stop_condition
        self.force_first_tool_call = force_first_tool_call

    def run(self, user_input: str, *, stream: bool = False) -> RunResult:
        """运行一次 Agent 任务，返回显式终态、Observation 和完整 Trace。"""

        run_id = uuid4().hex
        recorder = TraceRecorder(run_id)
        state = AgentState(run_id=run_id, input_items=[{"role": "user", "content": user_input}])
        # 使用单调时钟计算绝对 Deadline，避免系统时间调整导致剩余时间跳变。
        deadline = monotonic() + self.limits.run_timeout_s

        for step in range(1, self.limits.max_steps + 1):
            # 显式停止条件在模型调用前检查，避免目标已满足后仍产生费用或副作用。
            if self._should_stop(state):
                return self._result(state, recorder, RunStatus.EARLY_STOPPED)
            remaining = deadline - monotonic()
            if remaining <= 0:
                return self._result(state, recorder, RunStatus.DEADLINE, error="Run deadline exceeded")

            state.steps = step
            try:
                # 单次模型超时不能超过整个 Run 的剩余时间。
                tool_choice = None
                if self.force_first_tool_call:
                    # 首轮由 API 强制模型选择工具；产生 Observation 后禁止再次调用。
                    # 这比依赖自然语言“只调用一次”可预测，也能防止模型跳过证据。
                    tool_choice = "required" if not state.observations else "none"
                # 工具决策阶段只约束 Tool Schema；拿到 Observation 后才约束最终回答。
                # 部分 OpenAI-compatible 服务无法同时正确处理 required Tool Call 和
                # response_format=json_schema，把两个阶段拆开也能让协议意图更清晰。
                response_schema = (
                    None if tool_choice == "required" else self.response_schema
                )
                reply = self.model.call(
                    state.input_items,
                    self.tools.values(),
                    recorder,
                    response_schema=response_schema,
                    stream=stream,
                    timeout_s=min(self.limits.model_timeout_s, remaining),
                    tool_choice=tool_choice,
                )
            except TimeoutError as exc:
                return self._result(state, recorder, RunStatus.MODEL_TIMEOUT, error=str(exc))
            except SchemaValidationError as exc:
                return self._result(state, recorder, RunStatus.INVALID_MODEL_OUTPUT, error=str(exc))
            except Exception as exc:
                return self._result(state, recorder, RunStatus.MODEL_ERROR, error=str(exc))

            if reply.refusal:
                # 拒答是明确终态，不进入工具循环，也不把它伪装成 completed。
                return self._result(
                    state,
                    recorder,
                    RunStatus.MODEL_REFUSED,
                    text=reply.refusal,
                    error=reply.refusal,
                )

            if not reply.tool_calls:
                # 没有 Tool Call 时，文本或结构化对象就是候选最终答案。
                if not reply.text and reply.structured_output is None:
                    return self._result(state, recorder, RunStatus.MODEL_ERROR, error="模型未返回文本或工具调用")
                return self._result(
                    state,
                    recorder,
                    RunStatus.COMPLETED,
                    text=reply.text,
                    structured_output=reply.structured_output,
                )

            for call in reply.tool_calls:
                # 保存模型原始 Action，下一轮模型需要看到自己提出过什么调用。
                state.input_items.append(
                    {
                        "type": "function_call",
                        "call_id": call.call_id,
                        "name": call.name,
                        "arguments": json.dumps(call.arguments, ensure_ascii=False),
                    }
                )
                observation = self._execute_tool(call, recorder, deadline)
                state.observations.append(observation)
                # 必须复用原 call_id，才能把 Observation 关联到正确的 Action。
                state.input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": call.call_id,
                        "output": json.dumps(observation, ensure_ascii=False),
                    }
                )

            # 工具结果可能直接满足业务停止谓词；此时不再发起下一次模型调用。
            if self._should_stop(state):
                return self._result(state, recorder, RunStatus.EARLY_STOPPED)

        return self._result(state, recorder, RunStatus.MAX_STEPS, error="达到最大 Agent Step")

    def _execute_tool(self, call: ToolCall, recorder: TraceRecorder, deadline: float) -> JSON:
        """校验并执行一个工具，把所有结果归一化为结构化 Observation。"""

        # 即使工具未知或参数非法，也创建 Trace，确保每个模型 Action 都可审计。
        trace = recorder.start("tool", call.name)
        spec = self.tools.get(call.name)
        if spec is None:
            result = _tool_error("UNKNOWN_TOOL", f"未注册工具: {call.name}", retryable=False)
            recorder.finish(trace, "unknown_tool", result)
            return result

        try:
            # Schema 校验在 handler 之前，非法参数绝不能触达真实业务函数。
            validate_json(call.arguments, spec.parameters)
        except SchemaValidationError as exc:
            result = _tool_error("INVALID_ARGUMENTS", str(exc), retryable=False)
            recorder.finish(trace, "invalid", result)
            return result

        # 子操作使用“局部上限与总体剩余时间的最小值”，防止突破 Run Deadline。
        timeout_s = min(self.limits.tool_timeout_s, deadline - monotonic())
        if timeout_s <= 0:
            result = _tool_error("TOOL_TIMEOUT", "Run deadline exceeded", retryable=True)
            recorder.finish(trace, "timeout", result)
            return result

        # Future 为同步 handler 提供等待超时。注意：Python 线程无法强杀已开始的函数，
        # 因此有副作用的工具仍必须自行实现幂等键、内部 Deadline 和状态查询。
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"tool-{call.name}")
        future = pool.submit(spec.handler, **call.arguments)
        try:
            value = future.result(timeout=timeout_s)
        except FutureTimeout:
            future.cancel()
            result = _tool_error("TOOL_TIMEOUT", f"工具超过 {timeout_s:.3f}s", retryable=True)
            recorder.finish(trace, "timeout", result)
            pool.shutdown(wait=False, cancel_futures=True)
            return result
        except Exception as exc:
            result = _tool_error("TOOL_EXCEPTION", str(exc), retryable=False)
            recorder.finish(trace, "failed", result)
            pool.shutdown(wait=True)
            return result

        pool.shutdown(wait=True)
        # 只把可序列化业务结果作为 Observation 回传；Trace 仅记录必要摘要。
        result = {"ok": True, "data": value}
        recorder.finish(trace, "succeeded", {"ok": True})
        return result

    def _should_stop(self, state: AgentState) -> bool:
        return self.stop_condition is not None and self.stop_condition(state)

    @staticmethod
    def _result(
        state: AgentState,
        recorder: TraceRecorder,
        status: RunStatus,
        *,
        text: str = "",
        structured_output: JSON | None = None,
        error: str | None = None,
    ) -> RunResult:
        return RunResult(
            run_id=state.run_id,
            status=status,
            steps=state.steps,
            text=text,
            structured_output=structured_output,
            observations=tuple(state.observations),
            traces=tuple(recorder.entries),
            error=error,
        )


def _tool_error(code: str, message: str, *, retryable: bool) -> JSON:
    return {
        "ok": False,
        "error": {"code": code, "message": message, "retryable": retryable},
    }
