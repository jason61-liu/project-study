"""
示例 01: LangGraph 基础状态图 (StateGraph)

本示例演示 LangGraph 最核心的概念：
1. 状态 (State) - 用 TypedDict 定义图的数据结构
2. 节点 (Node) - 处理状态的函数
3. 边 (Edge) - 节点之间的连接关系
4. Reducer - 状态更新的合并策略
5. 编译和执行图

运行方式: python examples/01_basic_stategraph.py
"""

import operator
from typing import TypedDict, Annotated

from langgraph.graph import StateGraph, START, END

# ============================================================
# 第一步：定义状态 (State)
# ============================================================
# 状态是图中所有节点共享的数据结构。
# 每个节点可以读取状态、修改状态，修改后的状态会传递给下一个节点。
# Annotated[type, reducer] 中的 reducer 定义了状态更新的合并策略：
#   - operator.add 表示将新值"追加/累加"到已有值上
#   - 如果不加 Annotated，则新值直接覆盖旧值


class State(TypedDict):
    """图的共享状态"""
    # name: 普通字符串，新值直接覆盖旧值
    name: str
    # messages: 使用 operator.add 作为 reducer，新消息会追加到列表末尾
    messages: Annotated[list[str], operator.add]
    # counter: 使用 operator.add 作为 reducer，新值会累加到计数器上
    counter: Annotated[int, operator.add]


# ============================================================
# 第二步：定义节点函数 (Node)
# ============================================================
# 节点是一个普通函数，接收当前状态作为参数，返回状态的局部更新。
# 返回值是一个字典，只包含需要更新的字段。


def greet_node(state: State) -> dict:
    """
    打招呼节点 - 第一个执行的节点

    根据状态中的 name 字段生成问候语，并将消息追加到 messages 列表中。
    """
    name = state["name"]
    message = f"你好，{name}！欢迎来到 LangGraph 的世界！"
    print(f"  [greet_node] 生成消息: {message}")

    # 返回状态更新：追加一条消息，计数器 +1
    return {"messages": [message], "counter": 1}


def process_node(state: State) -> dict:
    """
    处理节点 - 第二个执行的节点

    读取当前状态中的消息和计数器，进行加工处理。
    """
    # 查看状态中累积的所有消息
    all_messages = state["messages"]
    current_count = state["counter"]

    message = f"处理完成！共处理了 {len(all_messages)} 条消息，计数器当前值为 {current_count}。"
    print(f"  [process_node] 当前状态: messages={all_messages}, counter={current_count}")
    print(f"  [process_node] 生成消息: {message}")

    # 返回状态更新：追加消息，计数器再 +1
    return {"messages": [message], "counter": 1}


def summary_node(state: State) -> dict:
    """
    总结节点 - 最后一个执行的节点

    汇总整个流程的状态，输出最终总结。
    """
    all_messages = state["messages"]
    final_count = state["counter"]

    message = f"=== 流程总结 === 共 {len(all_messages)} 条消息，最终计数: {final_count}"
    print(f"  [summary_node] {message}")

    return {"messages": [message], "counter": 0}


# ============================================================
# 第三步：构建状态图 (StateGraph)
# ============================================================

def build_graph() -> StateGraph:
    """
    构建并编译状态图

    图的结构:
        START → greet → process → summary → END

    这是一个最简单的线性流程图。
    """
    # 1. 创建 StateGraph，传入状态类型
    graph = StateGraph(State)

    # 2. 添加节点：每个节点关联一个处理函数
    graph.add_node("greet", greet_node)
    graph.add_node("process", process_node)
    graph.add_node("summary", summary_node)

    # 3. 添加边：定义节点之间的执行顺序
    # START 是特殊的起始标记，表示图的入口
    graph.add_edge(START, "greet")
    # 从 greet 节点指向 process 节点
    graph.add_edge("greet", "process")
    # 从 process 节点指向 summary 节点
    graph.add_edge("process", "summary")
    # END 是特殊的结束标记，表示图的出口
    graph.add_edge("summary", END)

    # 4. 编译图：验证图的完整性并生成可执行的对象
    return graph.compile()


# ============================================================
# 第四步：运行图
# ============================================================

def main():
    print("=" * 60)
    print("示例 01: LangGraph 基础状态图")
    print("=" * 60)

    # 编译图
    app = build_graph()

    # 初始化状态
    # 注意：counter 初始值为 0，messages 初始为空列表
    initial_state = {
        "name": "小明",
        "messages": [],
        "counter": 0,
    }

    print(f"\n初始状态: {initial_state}")
    print("-" * 60)

    # 调用图，传入初始状态
    # invoke() 会执行整个图并返回最终状态
    result = app.invoke(initial_state)

    print("-" * 60)
    print(f"\n最终状态:")
    print(f"  name: {result['name']}")
    print(f"  counter: {result['counter']}")
    print(f"  messages:")
    for i, msg in enumerate(result["messages"], 1):
        print(f"    {i}. {msg}")


if __name__ == "__main__":
    main()
