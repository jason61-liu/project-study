"""使用真实 DeepSeek 模型演示一条完整 Agent 轨迹。

运行前设置 OPENAI_API_KEY、OPENAI_BASE_URL；脚本不会读取或打印密钥。
"""

from __future__ import annotations

from datetime import datetime
import json
import os

from openai import OpenAI

from agent_runtime import AgentLimits, AgentRuntime, ModelAdapter, RunStatus, ToolSpec
from openai_transport import OpenAIChatCompletionsTransport


MODEL = os.getenv("AGENT_TEST_MODEL", "deepseek-ai/DeepSeek-V4-Pro")

ORDER_SCHEMA = {
    "type": "object",
    "properties": {
        "order_id": {"type": "string", "pattern": r"ORD-[0-9]{4}"},
    },
    "required": ["order_id"],
    "additionalProperties": False,
}

FINAL_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "order_id": {"type": "string", "pattern": r"ORD-[0-9]{4}"},
        "status": {"type": "string", "enum": ["shipped", "processing", "cancelled"]},
        "completed": {"type": "boolean"},
    },
    "required": ["answer", "order_id", "status", "completed"],
    "additionalProperties": False,
}


def get_order(order_id: str) -> dict[str, str]:
    """演示用确定性工具；真实项目应在这里访问订单服务。"""

    return {
        "order_id": order_id,
        "status": "shipped",
        "tracking_number": "SF-DEMO-1001",
    }


def build_agent() -> AgentRuntime:
    """组装真实模型 Transport、模型适配层、工具注册表与执行边界。"""

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    if not api_key or not base_url:
        raise SystemExit("请先设置 OPENAI_API_KEY 和 OPENAI_BASE_URL")

    client = OpenAI(api_key=api_key, base_url=base_url)
    transport = OpenAIChatCompletionsTransport(
        client,
        temperature=0,
        max_tokens=256,
    )
    adapter = ModelAdapter(transport, MODEL)
    tool = ToolSpec(
        name="get_order",
        description="根据订单号查询订单；用户要求查询订单时必须调用且只调用一次",
        parameters=ORDER_SCHEMA,
        handler=get_order,
    )
    return AgentRuntime(
        adapter,
        [tool],
        limits=AgentLimits(
            max_steps=3,
            run_timeout_s=60,
            model_timeout_s=30,
            tool_timeout_s=2,
        ),
        response_schema=FINAL_SCHEMA,
        # Prompt 中的“必须调用”可能被模型忽略，因此用协议参数强制首轮 Tool Call；
        # 一旦已有 Observation，Runtime 会把后续 tool_choice 设为 none。
        force_first_tool_call=True,
    )


def print_trace(result) -> None:
    """按发生顺序打印 Trace，便于演示时逐步讲解。"""

    print("\n=== 完整调用轨迹 ===")
    for index, trace in enumerate(result.traces, start=1):
        started = datetime.fromisoformat(trace.started_at)
        ended = datetime.fromisoformat(trace.ended_at) if trace.ended_at else started
        duration_ms = (ended - started).total_seconds() * 1000
        print(f"\n[{index}] {trace.kind.upper()} {trace.name}")
        print(f"    run_id:     {trace.run_id}")
        print(f"    trace_id:   {trace.trace_id}")
        print(f"    started_at: {trace.started_at}")
        print(f"    ended_at:   {trace.ended_at}")
        print(f"    duration:   {duration_ms:.1f} ms")
        print(f"    status:     {trace.status}")
        print("    result:     " + json.dumps(trace.result, ensure_ascii=False))


def main() -> None:
    agent = build_agent()
    prompt = (
        "必须先调用且只调用一次 get_order 查询 ORD-1001。"
        "收到工具结果后不要再次调用工具，按要求的 JSON Schema 返回最终结果；"
        "answer 用中文说明订单状态，completed 必须为 true。"
    )

    print("=== 最小 Agent 真实模型演示 ===")
    print(f"model: {MODEL}")
    print(f"user:  {prompt}")
    result = agent.run(prompt, stream=True)

    print("\n=== Agent 结果 ===")
    print(f"run_id: {result.run_id}")
    print(f"status: {result.status.value}")
    print(f"steps:  {result.steps}")
    print("observations: " + json.dumps(result.observations, ensure_ascii=False, indent=2))
    print("final: " + json.dumps(result.structured_output, ensure_ascii=False, indent=2))
    if result.error:
        print(f"error: {result.error}")

    print_trace(result)

    if result.status is not RunStatus.COMPLETED:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
