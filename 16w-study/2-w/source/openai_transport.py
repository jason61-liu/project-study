"""OpenAI Python SDK 到 ModelTransport 协议的薄适配。"""

from __future__ import annotations

from typing import Any, Iterable, Mapping


class OpenAIResponsesTransport:
    """把 ``client.responses.create`` 的对象转换成普通字典。

    客户端由调用方注入，因此本模块不读取 API Key，也不会在导入时创建网络连接。
    """

    def __init__(self, client: Any) -> None:
        self.client = client

    def complete(self, request: dict[str, Any], timeout_s: float) -> Mapping[str, Any]:
        # Responses SDK 返回 Pydantic 对象，转成普通字典后交给供应商无关的 ModelAdapter。
        response = self.client.responses.create(**request, timeout=timeout_s)
        return response.model_dump(mode="json")

    def stream(self, request: dict[str, Any], timeout_s: float) -> Iterable[Mapping[str, Any]]:
        events = self.client.responses.create(**request, timeout=timeout_s)
        for event in events:
            yield event.model_dump(mode="json")


class OpenAIChatCompletionsTransport:
    """将 OpenAI-compatible Chat Completions 归一化成 Responses 风格事件。

    SiliconFlow 当前为 DeepSeek 模型提供 Chat Completions。这个 Transport 隔离两套
    协议差异，使上层 ModelAdapter 和 AgentRuntime 无需增加供应商判断分支。
    """

    def __init__(self, client: Any, *, temperature: float = 0, max_tokens: int = 512) -> None:
        self.client = client
        self.temperature = temperature
        self.max_tokens = max_tokens

    def complete(self, request: dict[str, Any], timeout_s: float) -> Mapping[str, Any]:
        """把一个完整 ChatCompletion 转换成 Responses 风格 Output Item。"""

        response = self.client.chat.completions.create(
            **self._chat_request(request), timeout=timeout_s
        )
        choice = response.choices[0]
        # finish_reason=length 表示输出被截断，不能把半个 JSON 或参数当成成功结果。
        if choice.finish_reason == "length":
            return {"status": "incomplete", "incomplete_details": {"reason": "max_tokens"}}

        message = choice.message
        content: list[dict[str, Any]] = []
        if message.content:
            content.append({"type": "output_text", "text": message.content})
        refusal = getattr(message, "refusal", None)
        if refusal:
            content.append({"type": "refusal", "refusal": refusal})

        output: list[dict[str, Any]] = []
        if content:
            output.append({"type": "message", "role": "assistant", "content": content})
        # Chat Completions 的 tool_calls 嵌套在 assistant message 中；内部协议将其
        # 展平为独立 function_call Item，以便与 Responses API 使用同一解析器。
        for call in message.tool_calls or []:
            output.append(
                {
                    "type": "function_call",
                    "call_id": call.id,
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                }
            )
        return {"status": "completed", "output": output}

    def stream(self, request: dict[str, Any], timeout_s: float) -> Iterable[Mapping[str, Any]]:
        """把 ChatCompletionChunk 转换为语义化的 Responses 风格事件流。"""

        events = self.client.chat.completions.create(
            **self._chat_request(request), timeout=timeout_s
        )
        text_started = False
        # 每个并行 Tool Call 使用自己的 index 缓冲 name、call_id 和 arguments。
        calls: dict[int, dict[str, str]] = {}
        incomplete = False

        yield {"type": "response.created"}
        for chunk in events:
            # 某些兼容服务会发送只含 usage 的尾块，这类 Chunk 没有 choices。
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta
            if delta.content:
                if not text_started:
                    text_started = True
                    yield {
                        "type": "response.output_item.added",
                        "output_index": 0,
                        "item": {"type": "message", "role": "assistant"},
                    }
                yield {"type": "response.output_text.delta", "delta": delta.content}

            refusal = getattr(delta, "refusal", None)
            if refusal:
                yield {"type": "response.refusal.delta", "delta": refusal}

            for call_delta in delta.tool_calls or []:
                index = int(call_delta.index)
                current = calls.setdefault(
                    index, {"call_id": "", "name": "", "arguments": ""}
                )
                if call_delta.id:
                    current["call_id"] = call_delta.id
                if call_delta.function and call_delta.function.name:
                    current["name"] += call_delta.function.name
                arguments_delta = ""
                if call_delta.function and call_delta.function.arguments:
                    arguments_delta = call_delta.function.arguments
                    current["arguments"] += arguments_delta
                if arguments_delta:
                    # 参数 Delta 原样转发；ModelAdapter 会在 done 后统一 json.loads。
                    yield {
                        "type": "response.function_call_arguments.delta",
                        "output_index": index,
                        "delta": arguments_delta,
                    }

            if choice.finish_reason == "length":
                incomplete = True

        if incomplete:
            # 不再发送 completed，确保上层不会把截断流误判为成功。
            yield {
                "type": "response.incomplete",
                "response": {"incomplete_details": {"reason": "max_tokens"}},
            }
            return

        if text_started:
            yield {"type": "response.output_text.done"}
            yield {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {"type": "message"},
            }
        # Chat Chunk 没有 Responses 的 output_item.done，因此在流结束时合成边界事件。
        for index, call in sorted(calls.items()):
            item = {"type": "function_call", **call}
            yield {
                "type": "response.function_call_arguments.done",
                "output_index": index,
                "arguments": call["arguments"],
            }
            yield {
                "type": "response.output_item.done",
                "output_index": index,
                "item": item,
            }
        yield {"type": "response.completed"}

    def _chat_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """把内部 Responses 风格请求映射为 Chat Completions 参数。"""

        payload: dict[str, Any] = {
            "model": request["model"],
            "messages": self._messages(request.get("input", [])),
            "stream": bool(request.get("stream", False)),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        tools = request.get("tools", [])
        if tools:
            # Responses 风格工具字段是扁平的；Chat Completions 要求放到 function 下。
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool["parameters"],
                        "strict": tool.get("strict", True),
                    },
                }
                for tool in tools
            ]
            if request.get("tool_choice") is not None:
                # Chat Completions 接受 auto/none/required 或指定函数对象。
                payload["tool_choice"] = request["tool_choice"]

        text_format = request.get("text", {}).get("format")
        if text_format:
            # text.format 映射为 Chat Completions 的 response_format.json_schema。
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": text_format["name"],
                    "strict": text_format.get("strict", True),
                    "schema": text_format["schema"],
                },
            }
        return payload

    @staticmethod
    def _messages(input_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """重建 Chat 消息历史，尤其保证 Tool Call ID 前后一致。"""

        messages: list[dict[str, Any]] = []
        for item in input_items:
            item_type = item.get("type")
            if "role" in item:
                messages.append({"role": item["role"], "content": item.get("content", "")})
            elif item_type == "function_call":
                # 模型 Action 在 Chat 协议中属于 assistant.tool_calls。
                messages.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": item["call_id"],
                                "type": "function",
                                "function": {
                                    "name": item["name"],
                                    "arguments": item["arguments"],
                                },
                            }
                        ],
                    }
                )
            elif item_type == "function_call_output":
                # Observation 必须用 tool_call_id 指回相应 assistant Tool Call。
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": item["call_id"],
                        "content": item["output"],
                    }
                )
        return messages
