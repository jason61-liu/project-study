"""
示例 03: LangGraph 工具调用 Agent (Tool-calling Agent)

本示例演示如何使用 LangGraph 构建一个能调用外部工具的 Agent：
1. 使用 @tool 装饰器定义工具
2. 使用 ToolNode 自动处理工具调用
3. 使用 tools_condition 判断是否需要调用工具
4. 使用 llm.bind_tools() 将工具绑定到 LLM
5. 使用 MessagesState 管理对话消息

场景：一个助手 Agent，能查询天气和做数学计算。

运行方式: python examples/03_tool_agent.py
"""

import json
import urllib.request
from typing import Annotated

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_llm


# ============================================================
# 第一步：定义工具 (Tools)
# ============================================================
# 工具是 Agent 可以调用的外部函数。
# 使用 @tool 装饰器将普通函数标记为 LLM 可调用的工具。
# 函数的 docstring 会作为工具描述发送给 LLM，帮助它决定何时调用该工具。


@tool
def get_weather(city: str) -> str:
    """
    查询指定城市的天气信息

    Args:
        city: 城市名称，如"北京"、"上海"、"深圳"

    Returns:
        天气信息的文字描述
    """
    print(f"  [工具调用] 正在查询 {city} 的天气...")
    try:
        # 使用 wttr.in 免费天气 API（无需 API Key）
        url = f"https://wttr.in/{city}?format=j1&lang=zh"
        req = urllib.request.Request(url, headers={"User-Agent": "curl/7.68.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        current = data["current_condition"][0]
        temp = current["temp_C"]
        desc = current.get("lang_zh", [{}])[0].get("value", current["weatherDesc"][0]["value"])
        humidity = current["humidity"]
        return f"{city}当前天气: {desc}，温度 {temp}°C，湿度 {humidity}%"
    except Exception as e:
        return f"查询 {city} 天气失败: {str(e)}"


@tool
def calculate(expression: str) -> str:
    """
    计算数学表达式

    Args:
        expression: 数学表达式，如 "2 + 3 * 4" 或 "(10 + 5) / 3"

    Returns:
        计算结果
    """
    print(f"  [工具调用] 正在计算: {expression}")
    try:
        # 安全地计算数学表达式（只允许数字和基本运算符）
        allowed_chars = set("0123456789+-*/().% ")
        if not all(c in allowed_chars for c in expression):
            return "错误：表达式包含不允许的字符"
        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"


# 将所有工具放入列表，方便统一管理
tools = [get_weather, calculate]


# ============================================================
# 第二步：定义 Agent 节点
# ============================================================

# 获取绑定了工具的 LLM 实例
# bind_tools() 会将工具的描述信息发送给 LLM，让 LLM 知道有哪些工具可用
llm_with_tools = get_llm(temperature=0).bind_tools(tools)


def agent_node(state: MessagesState) -> dict:
    """
    Agent 主节点

    这个节点负责：
    1. 接收当前的消息历史
    2. 调用 LLM 决定下一步行动（回复用户 or 调用工具）
    3. 返回 LLM 的响应

    如果 LLM 决定调用工具，响应中会包含 tool_calls 字段，
    LangGraph 会自动路由到 ToolNode 执行工具。
    """
    # 调用 LLM，传入完整的消息历史
    response = llm_with_tools.invoke(state["messages"])
    print(f"  [agent] LLM 回复: {response.content[:100] if response.content else '(调用工具中...)'}")
    # 返回新的消息，追加到消息列表中
    return {"messages": [response]}


# ============================================================
# 第三步：构建 Agent 图
# ============================================================

def build_graph():
    """
    构建 Agent 图

    图的结构:
        START → agent ⇄ tools → END

    agent 和 tools 之间形成循环：
    - 如果 LLM 要调用工具 → 跳转到 tools 节点
    - 如果 LLM 直接回复 → 跳转到 END
    - tools 执行完后 → 回到 agent 继续处理
    """
    graph = StateGraph(MessagesState)

    # 添加 agent 节点
    graph.add_node("agent", agent_node)

    # ★ ToolNode 是 LangGraph 预置的工具执行节点 ★
    # 它会自动解析 LLM 返回的 tool_calls，调用对应的工具函数，
    # 然后将工具结果作为 ToolMessage 追加到消息列表中。
    graph.add_node("tools", ToolNode(tools))

    # 入口边
    graph.add_edge(START, "agent")

    # ★ 条件边：判断 LLM 是否需要调用工具 ★
    # tools_condition 是 LangGraph 预置的条件判断函数：
    #   - 如果 LLM 返回了 tool_calls → 返回 "tools"（跳转到工具节点）
    #   - 如果 LLM 直接回复文本 → 返回 END（结束流程）
    graph.add_conditional_edges("agent", tools_condition)

    # 工具执行完后，回到 agent 节点继续处理
    # 这样 agent 可以根据工具结果决定是否需要再次调用工具或直接回复
    graph.add_edge("tools", "agent")

    return graph.compile()


# ============================================================
# 第四步：运行 Agent
# ============================================================

def main():
    print("=" * 60)
    print("示例 03: LangGraph 工具调用 Agent")
    print("=" * 60)

    app = build_graph()

    # 测试用例 1：查询天气
    print("\n--- 测试 1: 查询天气 ---")
    result = app.invoke({
        "messages": [HumanMessage(content="请问深圳今天天气怎么样？")]
    })
    # 打印最终回复
    for msg in result["messages"]:
        if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            print(f"  Agent 回复: {msg.content}")

    # 测试用例 2：数学计算
    print("\n--- 测试 2: 数学计算 ---")
    result = app.invoke({
        "messages": [HumanMessage(content="帮我算一下 (15 + 27) * 3 等于多少？")]
    })
    for msg in result["messages"]:
        if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            print(f"  Agent 回复: {msg.content}")

    # 测试用例 3：综合（先查天气再计算）
    print("\n--- 测试 3: 综合问答 ---")
    result = app.invoke({
        "messages": [HumanMessage(content="帮我查一下北京天气，然后算一下 100 / 4 是多少")]
    })
    for msg in result["messages"]:
        if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            print(f"  Agent 回复: {msg.content}")


if __name__ == "__main__":
    main()
