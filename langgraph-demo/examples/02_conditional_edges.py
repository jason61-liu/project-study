"""
示例 02: LangGraph 条件边 (Conditional Edges)

本示例演示 LangGraph 的条件分支功能：
1. 使用 add_conditional_edges() 根据状态动态选择下一个节点
2. 路由函数 (Router) - 决定流程走向的逻辑
3. 多个不同路径的处理节点

场景：情感分析 —— 根据用户输入文本的情感（正面/负面/中性），
     走向不同的处理节点，给出不同的回复。

运行方式: python examples/02_conditional_edges.py
"""

from typing import TypedDict

from langgraph.graph import StateGraph, START, END

import sys
import os
# 将父目录加入路径，以便导入 config 模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_llm


# ============================================================
# 第一步：定义状态
# ============================================================

class State(TypedDict):
    """情感分析的状态"""
    # 用户输入的文本
    text: str
    # LLM 识别出的情感类别：positive / negative / neutral
    sentiment: str
    # 最终的回复内容
    response: str


# ============================================================
# 第二步：定义节点函数
# ============================================================

def analyze_sentiment(state: State) -> dict:
    """
    情感分析节点 - 使用 LLM 分析用户文本的情感

    这个节点调用 LLM，让模型判断输入文本是正面、负面还是中性情感。
    """
    text = state["text"]
    print(f"  [analyze] 正在分析文本: 「{text}」")

    # 获取 LLM 实例
    llm = get_llm(temperature=0)

    # 构造提示词，要求 LLM 只输出一个词：positive / negative / neutral
    prompt = (
        f"请分析以下文本的情感，只回复一个词（positive/negative/neutral）：\n\n"
        f"「{text}」"
    )

    # 调用 LLM
    result = llm.invoke(prompt)
    sentiment = result.content.strip().lower()

    print(f"  [analyze] LLM 分析结果: {sentiment}")

    return {"sentiment": sentiment}


def handle_positive(state: State) -> dict:
    """处理正面情感的节点"""
    print(f"  [positive] 检测到正面情感!")
    response = f"太好了！你的输入「{state['text']}」传递了积极正能量！继续保持！"
    return {"response": response}


def handle_negative(state: State) -> dict:
    """处理负面情感的节点"""
    print(f"  [negative] 检测到负面情感!")
    response = f"我理解你的感受。「{state['text']}」看起来有些消极。别担心，一切都会好起来的！"
    return {"response": response}


def handle_neutral(state: State) -> dict:
    """处理中性情感的节点"""
    print(f"  [neutral] 检测到中性情感!")
    response = f"收到！「{state['text']}」看起来是比较中性的表述。有什么我可以帮你的吗？"
    return {"response": response}


# ============================================================
# 第三步：定义路由函数
# ============================================================

def route_by_sentiment(state: State) -> str:
    """
    路由函数 - 根据状态中的 sentiment 字段决定下一个节点

    这是条件边的核心：返回值是下一个节点的名称（字符串）。
    LangGraph 会根据返回值跳转到对应的节点。

    如果返回 "positive" → 跳转到 handle_positive 节点
    如果返回 "negative" → 跳转到 handle_negative 节点
    如果返回 "neutral"  → 跳转到 handle_neutral 节点
    """
    sentiment = state["sentiment"]
    # 从 sentiment 中提取关键词
    if "positive" in sentiment:
        return "positive"
    elif "negative" in sentiment:
        return "negative"
    else:
        return "neutral"


# ============================================================
# 第四步：构建状态图
# ============================================================

def build_graph() -> StateGraph:
    """
    构建带条件边的状态图

    图的结构:
                          ┌─→ positive → END
        START → analyze ──┼─→ negative → END
                          └─→ neutral  → END

    analyze 节点之后通过条件边分支到三个不同的处理节点。
    """
    graph = StateGraph(State)

    # 添加节点
    graph.add_node("analyze", analyze_sentiment)
    graph.add_node("positive", handle_positive)
    graph.add_node("negative", handle_negative)
    graph.add_node("neutral", handle_neutral)

    # 添加起始边
    graph.add_edge(START, "analyze")

    # ★ 核心：添加条件边 ★
    # add_conditional_edges(source, router, {路由返回值: 目标节点, ...})
    # 当执行完 source 节点后，调用 router 函数，根据返回值跳转到对应节点
    graph.add_conditional_edges(
        "analyze",              # 源节点：条件从这个节点之后开始分支
        route_by_sentiment,     # 路由函数：决定走哪条边
        {                       # 路由映射：路由函数返回值 → 目标节点名
            "positive": "positive",
            "negative": "negative",
            "neutral": "neutral",
        }
    )

    # 三个处理节点都指向 END
    graph.add_edge("positive", END)
    graph.add_edge("negative", END)
    graph.add_edge("neutral", END)

    return graph.compile()


# ============================================================
# 第五步：运行图
# ============================================================

def main():
    print("=" * 60)
    print("示例 02: LangGraph 条件边 (情感分析)")
    print("=" * 60)

    app = build_graph()

    # 准备三个测试用例，分别对应不同情感
    test_cases = [
        {"text": "今天天气真好，心情特别愉快！", "sentiment": "", "response": ""},
        {"text": "这次考试没考好，感觉很失望", "sentiment": "", "response": ""},
        {"text": "明天下午三点开会", "sentiment": "", "response": ""},
    ]

    for i, initial_state in enumerate(test_cases, 1):
        print(f"\n--- 测试用例 {i}: 「{initial_state['text']}」 ---")
        result = app.invoke(initial_state)
        print(f"  最终回复: {result['response']}")


if __name__ == "__main__":
    main()
