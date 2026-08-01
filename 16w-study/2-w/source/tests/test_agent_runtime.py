from __future__ import annotations

import os
import time
import unittest

from openai import OpenAI

from agent_runtime import (
    AgentLimits,
    AgentRuntime,
    ModelAdapter,
    RunStatus,
    ToolCall,
    ToolSpec,
    TraceRecorder,
    validate_json,
)
from openai_transport import OpenAIChatCompletionsTransport


MODEL = os.getenv("AGENT_TEST_MODEL", "deepseek-ai/DeepSeek-V4-Pro")
HAS_CREDENTIALS = bool(os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_BASE_URL"))

ORDER_SCHEMA = {
    "type": "object",
    "properties": {"order_id": {"type": "string", "pattern": r"ORD-[0-9]{4}"}},
    "required": ["order_id"],
    "additionalProperties": False,
}

FINAL_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "resolved": {"type": "boolean"},
    },
    "required": ["answer", "resolved"],
    "additionalProperties": False,
}

RECOVERY_SCHEMA = {
    "type": "object",
    "properties": {
        "error_code": {
            "type": "string",
            "enum": ["TOOL_EXCEPTION", "TOOL_TIMEOUT"],
        },
        "recovered": {"type": "boolean"},
        "explanation": {"type": "string"},
    },
    "required": ["error_code", "recovered", "explanation"],
    "additionalProperties": False,
}


def live_runtime(
    *,
    tools: list[ToolSpec] | None = None,
    limits: AgentLimits | None = None,
    schema=None,
    stop_condition=None,
) -> AgentRuntime:
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"],
    )
    transport = OpenAIChatCompletionsTransport(client, temperature=0, max_tokens=256)
    return AgentRuntime(
        ModelAdapter(transport, MODEL),
        tools or [],
        limits=limits,
        response_schema=schema,
        stop_condition=stop_condition,
        # 有工具的测试明确验证 Tool Loop，不把“是否调用”交给模型随机决定。
        force_first_tool_call=bool(tools),
    )


@unittest.skipUnless(HAS_CREDENTIALS, "需要 OPENAI_API_KEY 和 OPENAI_BASE_URL")
class RealModelIntegrationTests(unittest.TestCase):
    def test_real_structured_output(self) -> None:
        result = live_runtime(schema=FINAL_SCHEMA).run(
            "返回一个 JSON 对象：answer 必须是 OK，resolved 必须是 true。"
        )

        self.assertEqual(result.status, RunStatus.COMPLETED)
        self.assertEqual(result.structured_output, {"answer": "OK", "resolved": True})
        self.assert_trace_complete(result.traces[0], "model", "succeeded")

    def test_real_streaming_structured_output(self) -> None:
        result = live_runtime(schema=FINAL_SCHEMA).run(
            "返回一个 JSON 对象：answer 必须是 STREAM，resolved 必须是 true。",
            stream=True,
        )

        self.assertEqual(result.status, RunStatus.COMPLETED)
        self.assertEqual(result.structured_output, {"answer": "STREAM", "resolved": True})

    def test_real_model_tool_observation_model_loop(self) -> None:
        tools = [
            ToolSpec(
                "get_order",
                "根据订单号查询订单；用户要求查询时必须调用",
                ORDER_SCHEMA,
                lambda order_id: {"order_id": order_id, "status": "shipped"},
            )
        ]
        result = live_runtime(tools=tools).run(
            "必须先调用且只调用一次 get_order 查询 ORD-1001；"
            "拿到结果后用中文回答状态，不要再次调用。"
        )

        self.assertEqual(result.status, RunStatus.COMPLETED)
        self.assertEqual(result.steps, 2)
        self.assertTrue(result.observations[0]["ok"])
        self.assertEqual([trace.kind for trace in result.traces], ["model", "tool", "model"])
        self.assertEqual(result.observations[0]["data"]["status"], "shipped")
        self.assertIn("ORD-1001", result.text)
        self.assertTrue(result.text.strip())

    def test_real_tool_exception_recovery(self) -> None:
        def broken(task: str) -> None:
            raise RuntimeError("database unavailable")

        schema = {
            "type": "object",
            "properties": {"task": {"type": "string"}},
            "required": ["task"],
            "additionalProperties": False,
        }
        tools = [ToolSpec("unstable_tool", "必须调用一次的测试工具", schema, broken)]
        result = live_runtime(tools=tools, schema=RECOVERY_SCHEMA).run(
            "必须先调用且只调用一次 unstable_tool，task 填 test；"
            "看到工具错误后不要重试；error_code 原样填写工具错误码，"
            "recovered 填 true，并简要解释。"
        )

        self.assertEqual(result.status, RunStatus.COMPLETED)
        self.assertEqual(result.observations[0]["error"]["code"], "TOOL_EXCEPTION")
        self.assertEqual(result.structured_output["error_code"], "TOOL_EXCEPTION")
        self.assertTrue(result.structured_output["recovered"])
        self.assertEqual(result.traces[1].status, "failed")

    def test_real_tool_timeout_recovery(self) -> None:
        def slow(task: str) -> str:
            time.sleep(0.05)
            return task

        schema = {
            "type": "object",
            "properties": {"task": {"type": "string"}},
            "required": ["task"],
            "additionalProperties": False,
        }
        tools = [ToolSpec("slow_tool", "必须调用一次的慢工具", schema, slow)]
        limits = AgentLimits(max_steps=3, tool_timeout_s=0.001, model_timeout_s=30)
        result = live_runtime(tools=tools, limits=limits, schema=RECOVERY_SCHEMA).run(
            "必须先调用且只调用一次 slow_tool，task 填 test；"
            "超时后不要重试；error_code 原样填写工具错误码，"
            "recovered 填 true，并简要解释。"
        )

        self.assertEqual(result.status, RunStatus.COMPLETED)
        self.assertEqual(result.observations[0]["error"]["code"], "TOOL_TIMEOUT")
        self.assertEqual(result.structured_output["error_code"], "TOOL_TIMEOUT")
        self.assertTrue(result.structured_output["recovered"])
        self.assertEqual(result.traces[1].status, "timeout")

    def test_real_max_steps(self) -> None:
        schema = {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
            "additionalProperties": False,
        }
        tools = [ToolSpec("ping", "每一轮都必须调用的工具", schema, lambda value: value)]
        limits = AgentLimits(max_steps=1, model_timeout_s=30)
        result = live_runtime(tools=tools, limits=limits).run(
            "必须调用 ping，value 填 again；不要直接输出文本。"
        )

        self.assertEqual(result.status, RunStatus.MAX_STEPS)
        self.assertEqual(result.steps, 1)
        self.assertEqual(len(result.observations), 1)

    def test_real_explicit_early_stop(self) -> None:
        tools = [
            ToolSpec(
                "get_order",
                "根据订单号查询订单；用户要求查询时必须调用",
                ORDER_SCHEMA,
                lambda order_id: {"order_id": order_id, "fatal": True},
            )
        ]
        stop = lambda state: any(
            observation.get("data", {}).get("fatal")
            for observation in state.observations
        )
        result = live_runtime(tools=tools, stop_condition=stop).run(
            "必须调用 get_order 查询 ORD-1001，不要直接回答。"
        )

        self.assertEqual(result.status, RunStatus.EARLY_STOPPED)
        self.assertEqual(result.steps, 1)
        self.assertEqual(len(result.traces), 2)

    def assert_trace_complete(self, trace, kind: str, status: str) -> None:
        self.assertEqual(trace.kind, kind)
        self.assertEqual(trace.status, status)
        self.assertTrue(trace.trace_id)
        self.assertTrue(trace.started_at)
        self.assertTrue(trace.ended_at)


@unittest.skipUnless(HAS_CREDENTIALS, "需要 OPENAI_API_KEY 和 OPENAI_BASE_URL")
class DeterministicBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runtime = live_runtime()
        self.recorder = TraceRecorder("boundary-test")

    def test_unknown_tool(self) -> None:
        result = self.runtime._execute_tool(
            ToolCall("call-unknown", "missing", {}),
            self.recorder,
            time.monotonic() + 1,
        )

        self.assertEqual(result["error"]["code"], "UNKNOWN_TOOL")
        self.assertEqual(self.recorder.entries[0].status, "unknown_tool")

    def test_invalid_tool_arguments(self) -> None:
        tool = ToolSpec("get_order", "查询订单", ORDER_SCHEMA, lambda order_id: {})
        self.runtime.tools[tool.name] = tool
        result = self.runtime._execute_tool(
            ToolCall("call-invalid", "get_order", {"order_id": "bad"}),
            self.recorder,
            time.monotonic() + 1,
        )

        self.assertEqual(result["error"]["code"], "INVALID_ARGUMENTS")
        self.assertEqual(self.recorder.entries[0].status, "invalid")

    def test_schema_rejects_additional_properties(self) -> None:
        with self.assertRaisesRegex(ValueError, "未声明字段"):
            validate_json({"order_id": "ORD-1001", "admin": True}, ORDER_SCHEMA)


if __name__ == "__main__":
    unittest.main()
