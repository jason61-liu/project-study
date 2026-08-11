"""共享研究工具、检索器和可重复故障注入。"""

from __future__ import annotations

import json
from pathlib import Path
import re
import time
import uuid
from typing import Any

from models import ToolResult


ROOT = Path(__file__).parent


def _terms(text: str) -> set[str]:
    """同时提取英文词和中文二元组，避免依赖外部向量服务。"""

    lowered = text.lower()
    latin = set(re.findall(r"[a-z][a-z0-9-]+", lowered))
    chinese = "".join(re.findall(r"[\u4e00-\u9fff]", lowered))
    bigrams = {chinese[i : i + 2] for i in range(max(0, len(chinese) - 1))}
    return latin | bigrams


class ResearchToolRuntime:
    """每条轨迹创建独立实例，确保故障注入可重复且互不污染。"""

    def __init__(self, *, fault: str = "none") -> None:
        self.documents: list[dict[str, Any]] = json.loads(
            (ROOT / "data" / "corpus.json").read_text(encoding="utf-8")
        )
        self.by_id = {document["id"]: document for document in self.documents}
        self.fault = fault
        self.corpus_version = 1
        self.tool_calls = 0
        self._search_failures = 0
        self._read_failures = 0
        self._plan_invalidated = False

    @staticmethod
    def schemas() -> dict[str, dict[str, Any]]:
        """模型看到的统一工具定义；三种架构不得各自改描述。"""

        return {
            "search_documents": {
                "description": "按主题搜索研究资料，返回文档 ID、标题、摘要和当前语料版本。",
                "parameters": {"query": "string", "max_results": "integer 1..6"},
            },
            "read_document": {
                "description": "按稳定文档 ID 读取全文；expected_version 用于检测计划失效。",
                "parameters": {"doc_id": "string", "expected_version": "integer"},
            },
        }

    def _result(self, *, tool: str, status: str, data: Any = None, error_type: str | None = None, retryable: bool = False) -> ToolResult:
        return ToolResult(
            status=status, tool=tool, call_id=f"call_{uuid.uuid4().hex[:12]}", data=data,
            error_type=error_type, retryable=retryable, corpus_version=self.corpus_version,
        )

    def call(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """校验并执行工具；故障以结构化结果返回，不抛出含噪异常。"""

        self.tool_calls += 1
        started = time.perf_counter()
        if name == "search_documents":
            result = self._search(arguments)
        elif name == "read_document":
            result = self._read(arguments)
        else:
            result = self._result(tool=name, status="error", error_type="unknown_tool")
        # 保留最小真实墙钟耗时，使 Trace 的工具 Span 不为负或未定义。
        _ = (time.perf_counter() - started) * 1000
        return result

    def _search(self, arguments: dict[str, Any]) -> ToolResult:
        query = arguments.get("query")
        limit = arguments.get("max_results", 4)
        if not isinstance(query, str) or not query.strip():
            return self._result(tool="search_documents", status="error", error_type="invalid_arguments")
        if not isinstance(limit, int) or not 1 <= limit <= 6:
            return self._result(tool="search_documents", status="error", error_type="invalid_arguments")
        if self.fault == "retrieval_failure" and self._search_failures == 0:
            self._search_failures += 1
            return self._result(
                tool="search_documents", status="error", error_type="retriever_unavailable", retryable=True
            )

        query_terms = _terms(query)
        ranked = []
        for document in self.documents:
            haystack = " ".join([document["title"], document["content"], *document["tags"]])
            score = len(query_terms & _terms(haystack))
            if score:
                ranked.append((score, document))
        ranked.sort(key=lambda item: (-item[0], item[1]["id"]))
        hits = [
            {"id": doc["id"], "title": doc["title"], "snippet": doc["content"][:90]}
            for _, doc in ranked[:limit]
        ]
        return self._result(tool="search_documents", status="success", data={"hits": hits})

    def _read(self, arguments: dict[str, Any]) -> ToolResult:
        doc_id = arguments.get("doc_id")
        expected = arguments.get("expected_version")
        if not isinstance(doc_id, str) or not isinstance(expected, int):
            return self._result(tool="read_document", status="error", error_type="invalid_arguments")

        if self.fault == "plan_invalidated" and not self._plan_invalidated:
            self.corpus_version += 1
            self._plan_invalidated = True
        if expected != self.corpus_version:
            return self._result(
                tool="read_document", status="error", error_type="plan_invalidated", retryable=True
            )
        if self.fault == "tool_failure" and self._read_failures == 0:
            self._read_failures += 1
            return self._result(tool="read_document", status="error", error_type="tool_unavailable", retryable=True)
        document = self.by_id.get(doc_id)
        if document is None:
            return self._result(tool="read_document", status="error", error_type="not_found")
        return self._result(tool="read_document", status="success", data=document)

