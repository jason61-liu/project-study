"""CMA Event Translator — 将 Strands Hook 事件和 Stream 事件翻译为 CMA 格式推送到 SSE Queue"""

from typing import Any

from strands.hooks.events import (
    AfterToolCallEvent,
    BeforeModelCallEvent,
    BeforeToolCallEvent,
)
from strands.types._events import TextStreamEvent

from poc_runner.mock_cma.sse_queue import SSEClient


class CmaEventTranslator:
    """Strands 事件 → CMA 事件翻译器"""

    def __init__(self, sse_queue: SSEClient):
        self.sse_queue = sse_queue

    def register(self, agent: "Agent") -> None:
        """在 Agent 上注册所有 Hook 回调"""
        agent.add_hook(self._on_before_tool_call, BeforeToolCallEvent)
        agent.add_hook(self._on_after_tool_call, AfterToolCallEvent)
        agent.add_hook(self._on_before_model_call, BeforeModelCallEvent)

    def _on_before_tool_call(self, event: BeforeToolCallEvent) -> None:
        """BeforeToolCallEvent → agent.tool_use"""
        tool_use = event.tool_use
        is_interrupting = getattr(event, "_interrupt_triggered", False)

        if event.cancel_tool:
            # TC-EVT-02: 拒绝时不发 tool_use
            return

        payload: dict[str, Any] = {
            "type": "agent.tool_use",
            "id": tool_use["toolUseId"],
            "name": tool_use["name"],
            "input": tool_use["input"],
        }

        if is_interrupting:
            # TC-EVT-03: interrupt 时加 status: "pending"
            payload["status"] = "pending"

        self.sse_queue.push_sync(payload)

    def _on_after_tool_call(self, event: AfterToolCallEvent) -> None:
        """AfterToolCallEvent → agent.tool_result"""
        result = event.result
        is_error = result.get("status") == "error"

        payload: dict[str, Any] = {
            "type": "agent.tool_result",
            "tool_use_id": event.tool_use["toolUseId"],
            "content": result.get("content", ""),
            "is_error": is_error,
        }

        if event.exception:
            payload["error_message"] = str(event.exception)

        if event.cancel_message:
            payload["cancel_message"] = event.cancel_message

        self.sse_queue.push_sync(payload)

    def _on_before_model_call(self, event: BeforeModelCallEvent) -> None:
        """BeforeModelCallEvent → 记录模型调用（可选）"""
        if event.cancel:
            return

    async def translate_stream_event(self, event: Any) -> dict | None:
        """将 stream_async 产生的 TypedEvent（作为 plain dict）翻译为 CMA 事件

        stream_async 产出的是 plain dict（TypedEvent.as_dict()），通过 dict key 判断事件类型。
        返回 None 表示该事件不需要翻译为 CMA 格式。
        """
        # stream_async 产出的是 dict，用 key 判断类型
        if not isinstance(event, dict):
            return None

        # TC-EVT-05: Text stream → agent.message.delta
        # 文本流事件有 "data" 和 "delta" 两个 key
        if "data" in event and "delta" in event:
            return {
                "type": "agent.message.delta",
                "content": [{"type": "text", "text": event["data"]}],
            }

        # TC-EVT-06: AgentResultEvent → session.status_idle
        # 最终事件有 "result" key
        if "result" in event:
            result = event["result"]
            stop_reason = "end_turn"
            if hasattr(result, 'stop_reason'):
                stop_reason = result.stop_reason
            return {
                "type": "session.status_idle",
                "stop_reason": stop_reason,
            }

        # EventLoopStopEvent → session.status_idle (alternative)
        if "stop_reason" in event:
            return {
                "type": "session.status_idle",
                "stop_reason": event.get("stop_reason", "end_turn"),
            }

        return None
