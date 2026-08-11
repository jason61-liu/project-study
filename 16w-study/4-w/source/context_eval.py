"""全历史、摘要历史和检索式记忆三种上下文策略及统一评测。"""

from __future__ import annotations

from dataclasses import dataclass
import re
import time
from typing import Protocol

try:
    import tiktoken
except ImportError:  # pragma: no cover - requirements 中已声明，仅作为易懂的降级路径。
    tiktoken = None

from memory_backends import AuthScope, VectorMemory
from model_client import DeepSeekModel


_TOKEN_ENCODING = None
_TOKEN_ENCODING_PROBED = False


@dataclass(frozen=True)
class Turn:
    """一条可持久化的会话消息。"""

    role: str
    content: str


@dataclass(frozen=True)
class MultiTurnCase:
    """上下文策略共享的同一组多轮任务。"""

    id: str
    history: tuple[Turn, ...]
    query: str
    expected_terms: tuple[str, ...]


class ContextStrategy(Protocol):
    """三种策略只需实现 compose，便于统一测量。"""

    name: str

    def compose(self, case: MultiTurnCase) -> str: ...


class FullHistory:
    """保留所有消息，Recall 高但 Token 随轮数线性增长。"""

    name = "full_history"

    def compose(self, case: MultiTurnCase) -> str:
        return "\n".join(f"{turn.role}: {turn.content}" for turn in case.history) + f"\nuser: {case.query}"


class SummaryHistory:
    """把旧消息压成事实槽位；保留最后两轮以减少摘要丢失。"""

    name = "summary_history"

    def compose(self, case: MultiTurnCase) -> str:
        older, recent = case.history[:-2], case.history[-2:]
        facts: list[str] = []
        for turn in older:
            # 教学语料中的事实以自然语言陈述；按句保留包含关键系词/标识的内容。
            for sentence in re.split(r"[。！？\n]", turn.content):
                sentence = sentence.strip()
                if sentence and any(marker in sentence for marker in ("是", "使用", "偏好", "截止", "包含", "部署")):
                    facts.append(sentence)
        deduplicated = list(dict.fromkeys(facts))
        summary = "摘要事实：" + "；".join(deduplicated[-8:])
        tail = "\n".join(f"{turn.role}: {turn.content}" for turn in recent)
        return f"{summary}\n{tail}\nuser: {case.query}"


class RetrievalMemoryContext:
    """将历史消息写入租户隔离的向量记忆，只召回与当前问题相关的 K 条。"""

    name = "retrieval_memory"

    def __init__(self, *, top_k: int = 3) -> None:
        self.top_k = top_k

    def compose(self, case: MultiTurnCase) -> str:
        memory = VectorMemory()
        scope = AuthScope("eval-tenant", case.id)
        for index, turn in enumerate(case.history):
            memory.add(scope, turn.content, memory_id=f"{case.id}-{index}")
        hits = memory.search(scope, case.query, top_k=self.top_k)
        recalled = "\n".join(f"memory: {hit.text}" for hit in hits)
        return f"{recalled}\nuser: {case.query}"


def build_context_cases() -> list[MultiTurnCase]:
    """构造包含干扰轮次的同一组多轮任务，避免只比较单轮提示。"""

    topics = [
        ("Phoenix", "项目 Phoenix 使用 Python 3.12。", "它使用什么语言版本？", ("Python 3.12",)),
        ("deploy", "项目 Phoenix 部署在新加坡。", "这个项目部署在哪里？", ("新加坡",)),
        ("docs", "用户偏好中文技术文档。", "用户偏好什么语言的文档？", ("中文",)),
        ("deadline", "周报截止时间是周五 17:00。", "周报什么时候截止？", ("周五 17:00",)),
        ("report", "周报必须包含风险和下周计划。", "周报要包含哪些项目？", ("风险", "下周计划")),
        ("alerts", "当前告警渠道是 Slack。", "告警发到哪里？", ("Slack",)),
        ("oncall", "严重故障升级给 oncall-lead。", "严重故障找谁？", ("oncall-lead",)),
        ("editor", "用户的编辑器是 VS Code。", "用户使用哪个编辑器？", ("VS Code",)),
    ]
    filler = (
        Turn("assistant", "已记录，下面讨论无关的会议安排。"),
        Turn("user", "今天午餐改到十二点半。"),
        Turn("assistant", "收到，午餐信息不影响项目配置。"),
        Turn("user", "请继续保留此前的关键事实。"),
    )
    return [
        MultiTurnCase(key, (Turn("user", fact),) + filler, query, expected)
        for key, fact, query, expected in topics
    ]


def count_tokens(text: str) -> int:
    """使用 cl100k_base 作为可复现估算器；它不是 DeepSeek 的官方计费值。"""

    global _TOKEN_ENCODING, _TOKEN_ENCODING_PROBED
    if tiktoken is None:
        return max(1, len(text) // 3)
    if not _TOKEN_ENCODING_PROBED:
        _TOKEN_ENCODING_PROBED = True
        try:
            _TOKEN_ENCODING = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _TOKEN_ENCODING = None
    if _TOKEN_ENCODING is not None:
        return len(_TOKEN_ENCODING.encode(text))
    # 首次使用 tiktoken 可能需要下载词表。离线环境不能因为计数器失败而阻断
    # 主实验，因此降级为明确标注的字符近似，而不是伪造精确模型 Token。
    return max(1, len(text) // 3)


def evaluate_context_strategies(
    cases: list[MultiTurnCase] | None = None,
    *,
    input_usd_per_million: float = 1.0,
    model: DeepSeekModel | None = None,
) -> dict:
    """对相同任务记录正确率、Token、物理延迟和单位成本。

    `model=None` 用于确定性边界测试；注入 DeepSeekModel 时，正确率来自真实回答，
    Token/费用来自 API usage，而不是本地估算。
    """

    cases = cases or build_context_cases()
    strategies: list[ContextStrategy] = [FullHistory(), SummaryHistory(), RetrievalMemoryContext()]
    result: dict[str, dict] = {}
    for strategy in strategies:
        rows = []
        for case in cases:
            started = time.perf_counter()
            context = strategy.compose(case)
            compose_latency_ms = (time.perf_counter() - started) * 1000
            if model is None:
                answer = context
                tokens = count_tokens(context)
                output_tokens = 0
                model_latency_ms = 0.0
                cost = tokens / 1_000_000 * input_usd_per_million
                usage = None
            else:
                generated = model.complete(
                    "根据提供的 CONTEXT 回答最后一个用户问题。只使用上下文中的事实；"
                    "上下文不足就回答不知道。回答要简洁，不要解释上下文策略。",
                    f"CONTEXT:\n{context}",
                    max_tokens=100,
                )
                answer = generated.text
                tokens = generated.input_tokens
                output_tokens = generated.output_tokens
                model_latency_ms = generated.latency_ms
                cost = generated.estimated_cost_usd
                usage = generated.to_dict()
            correct = all(term.lower() in answer.lower() for term in case.expected_terms)
            rows.append(
                {
                    "case_id": case.id,
                    "correct": correct,
                    "input_tokens": tokens,
                    "output_tokens": output_tokens,
                    "compose_latency_ms": compose_latency_ms,
                    "model_latency_ms": model_latency_ms,
                    "latency_ms": compose_latency_ms + model_latency_ms,
                    "estimated_cost_usd": cost,
                    "answer": answer if model is not None else None,
                    "model_usage": usage,
                }
            )
        result[strategy.name] = {
            "accuracy": sum(row["correct"] for row in rows) / len(rows),
            "average_input_tokens": sum(row["input_tokens"] for row in rows) / len(rows),
            "average_output_tokens": sum(row["output_tokens"] for row in rows) / len(rows),
            "average_latency_ms": sum(row["latency_ms"] for row in rows) / len(rows),
            "unit_task_cost_usd": sum(row["estimated_cost_usd"] for row in rows) / len(rows),
            "cost_basis": (
                "DeepSeek API usage with official cache-hit/cache-miss/output prices"
                if model is not None
                else f"estimated input only at ${input_usd_per_million}/1M tokens; not a provider invoice"
            ),
            "rows": rows,
        }
    return result
