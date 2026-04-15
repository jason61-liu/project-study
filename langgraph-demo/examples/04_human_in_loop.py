"""
示例 04: LangGraph 人机交互 (Human-in-the-Loop)

本示例演示如何在 LangGraph 中实现人机交互：
1. 使用 MemorySaver 做内存检查点，保存图的执行状态
2. 使用 interrupt() 暂停图的执行，等待人工输入
3. 使用 Command(resume=...) 恢复图的执行并传入人工审批结果
4. 实现审批/打回的循环流程

场景：文档审核 —— LLM 生成文档摘要后，暂停等待人工审批。
     - 如果审批通过 → 输出最终结果
     - 如果打回修改 → LLM 重新生成并再次等待审批

运行方式: python examples/04_human_in_loop.py
"""

from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_llm


# ============================================================
# 第一步：定义状态
# ============================================================

class State(TypedDict):
    """文档审核流程的状态"""
    # 原始文档内容
    document: str
    # LLM 生成的摘要
    summary: str
    # 人工审批结果: "approved" 或 "rejected"
    approval: str
    # 打回修改的意见
    feedback: str
    # 修改次数
    revision_count: int
    # 最终输出
    final_output: str


# ============================================================
# 第二步：定义节点函数
# ============================================================

def generate_summary(state: State) -> dict:
    """
    生成摘要节点 - LLM 根据文档内容生成摘要

    如果是首次生成，直接对文档做摘要。
    如果被打回，则根据反馈意见修改摘要。
    """
    document = state["document"]
    feedback = state.get("feedback", "")
    revision_count = state.get("revision_count", 0)

    llm = get_llm(temperature=0.3)

    if revision_count == 0:
        # 首次生成
        print(f"  [generate] 首次生成摘要...")
        prompt = f"请用简洁的中文总结以下文档内容（不超过3句话）：\n\n{document}"
    else:
        # 根据反馈修改
        print(f"  [generate] 根据反馈修改摘要（第 {revision_count} 次修改）...")
        print(f"  [generate] 反馈意见: {feedback}")
        prompt = (
            f"之前的摘要:\n{state['summary']}\n\n"
            f"审核意见:\n{feedback}\n\n"
            f"请根据审核意见修改摘要。原始文档:\n{document}"
        )

    result = llm.invoke(prompt)
    summary = result.content.strip()
    print(f"  [generate] 生成的摘要: {summary[:80]}...")

    return {"summary": summary}


def human_review(state: State) -> dict:
    """
    ★ 人工审核节点 - 图的执行会在此暂停 ★

    interrupt() 是 LangGraph 的人机交互机制：
    1. 调用 interrupt(value) 时，图的执行会暂停
    2. interrupt 的参数会作为图当前状态的快照保存
    3. 外部可以通过 Command(resume=数据) 恢复执行
    4. resume 传入的数据就是 interrupt() 的返回值

    这样就可以在图的执行过程中插入人工决策。
    """
    print(f"\n  [review] ===== 等待人工审核 =====")
    print(f"  [review] 文档: {state['document'][:60]}...")
    print(f"  [review] 摘要: {state['summary']}")

    # ★ 调用 interrupt 暂停执行 ★
    # 参数会传递给外部调用者，帮助他们做决策
    # 返回值是 Command(resume=...) 中传入的数据
    decision = interrupt({
        "message": "请审核以上摘要",
        "summary": state["summary"],
        "revision_count": state.get("revision_count", 0),
    })

    print(f"  [review] 收到审核结果: {decision}")

    # decision 是外部传入的审批结果
    return {
        "approval": decision["approval"],
        "feedback": decision.get("feedback", ""),
        "revision_count": state.get("revision_count", 0) + 1,
    }


def finalize(state: State) -> dict:
    """完成节点 - 审批通过后的最终处理"""
    output = (
        f"=== 文档审核通过 ===\n"
        f"原始文档: {state['document'][:100]}...\n"
        f"最终摘要: {state['summary']}\n"
        f"修改次数: {state['revision_count']}"
    )
    print(f"  [finalize] {output}")
    return {"final_output": output}


# ============================================================
# 第三步：定义路由函数
# ============================================================

def route_after_review(state: State) -> str:
    """
    审核后的路由函数

    根据人工审批结果决定下一步：
    - "approved" → 跳转到 finalize 节点，结束流程
    - "rejected" → 跳转回 generate_summary 节点，重新生成摘要
    """
    if state["approval"] == "approved":
        return "finalize"
    else:
        return "generate"


# ============================================================
# 第四步：构建状态图
# ============================================================

def build_graph():
    """
    构建人机交互图

    图的结构:
        START → generate → review ─┬→ finalize → END
                    ↑               │
                    └─── generate ←─┘  (打回修改循环)

    注意：使用 MemorySaver 作为检查点存储，这是人机交互的必要条件。
    MemorySaver 将图的状态保存在内存中，使得暂停后可以恢复执行。
    """
    graph = StateGraph(State)

    # 添加节点
    graph.add_node("generate", generate_summary)
    graph.add_node("review", human_review)
    graph.add_node("finalize", finalize)

    # 添加边
    graph.add_edge(START, "generate")
    graph.add_edge("generate", "review")

    # 条件边：根据审批结果决定走向
    graph.add_conditional_edges(
        "review",
        route_after_review,
        {
            "finalize": "finalize",  # 审批通过 → 结束
            "generate": "generate",  # 打回修改 → 重新生成
        }
    )

    graph.add_edge("finalize", END)

    # ★ 编译图时传入 checkpointer ★
    # MemorySaver 是内存检查点存储，保存图的执行状态。
    # 对于人机交互，需要持久化状态以便暂停后恢复。
    # 生产环境可以使用 SqliteSaver 或 PostgresSaver 做持久化存储。
    checkpointer = MemorySaver()

    return graph.compile(checkpointer=checkpointer)


# ============================================================
# 第五步：运行图（模拟人机交互）
# ============================================================

def main():
    print("=" * 60)
    print("示例 04: LangGraph 人机交互 (文档审核)")
    print("=" * 60)

    app = build_graph()

    # thread_id 用于标识一次会话，确保中断和恢复对应同一个流程
    config = {"configurable": {"thread_id": "doc-review-001"}}

    # 原始文档
    document = (
        "人工智能（AI）是计算机科学的一个分支，致力于创建能够执行"
        "通常需要人类智能的任务的系统。这些任务包括学习、推理、"
        "问题解决、感知和语言理解。近年来，深度学习的突破推动了"
        "AI技术的快速发展，特别是在自然语言处理、计算机视觉和"
        "语音识别等领域。"
    )

    print(f"\n原始文档: {document}")
    print("-" * 60)

    # --- 第一次执行：生成摘要后会暂停在 human_review 节点 ---
    print("\n>>> 第一步：启动流程，生成摘要并暂停等待审核\n")
    result = app.invoke(
        {"document": document, "summary": "", "approval": "", "feedback": "",
         "revision_count": 0, "final_output": ""},
        config=config,
    )

    # 此时图已暂停，检查暂停时的 interrupt 信息
    # 如果执行到 interrupt()，会抛出 GraphInterrupt 异常并被 invoke 捕获
    # 我们可以通过读取检查点来确认状态
    print("\n>>> 图已暂停，正在模拟人工审核...")

    # --- 模拟第一次审核：打回修改 ---
    print("\n>>> 第二步：模拟人工审核 - 打回修改\n")
    result = app.invoke(
        # ★ 使用 Command(resume=...) 恢复执行 ★
        # resume 的数据会成为 interrupt() 的返回值
        Command(resume={
            "approval": "rejected",
            "feedback": "摘要太简短了，请加入深度学习相关内容",
        }),
        config=config,
    )

    # --- 模拟第二次审核：通过 ---
    print("\n>>> 第三步：模拟人工审核 - 通过\n")
    result = app.invoke(
        Command(resume={
            "approval": "approved",
            "feedback": "",
        }),
        config=config,
    )

    print("\n" + "=" * 60)
    print("流程完成！")
    if result.get("final_output"):
        print(result["final_output"])


if __name__ == "__main__":
    main()
