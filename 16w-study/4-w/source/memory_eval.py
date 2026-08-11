"""用同一组多轮事实任务比较自建向量记忆与 Mem0。"""

from __future__ import annotations

import json
from pathlib import Path
import statistics
import time

from context_eval import count_tokens
from memory_backends import AuthScope, MemoryBackend, VectorMemory, choose_mem0_backend


DATA_PATH = Path(__file__).parent / "data" / "memory_tasks.json"


def load_memory_tasks(path: Path = DATA_PATH) -> list[dict]:
    """加载包含同 user 跨 tenant 和同 tenant 跨 user 的多轮任务。"""

    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_memory_backend(backend: MemoryBackend, tasks: list[dict] | None = None, *, top_k: int = 3) -> dict:
    """记录写入、召回、错误记忆、隔离、更新/删除、延迟和 Token。

    上游把每条 `writes` 当作已经抽取的原子事实，同时传给两个后端。这使实验比较
    存储/检索/生命周期能力，而不是把某个云 LLM 的抽取波动混入后端差异。
    """

    tasks = tasks or load_memory_tasks()
    created: list[tuple[AuthScope, str, str]] = []
    write_latencies: list[float] = []
    search_latencies: list[float] = []
    token_count = 0
    write_correct = 0
    write_total = 0
    query_rows: list[dict] = []

    try:
        for task in tasks:
            scope = AuthScope(task["tenant_id"], task["user_id"])
            for index, fact in enumerate(task["writes"]):
                token_count += count_tokens(fact)
                started = time.perf_counter()
                memory_id = backend.add(scope, fact, memory_id=f"{task['id']}-{index}")
                write_latencies.append((time.perf_counter() - started) * 1000)
                created.append((scope, memory_id, fact))
                write_total += 1
                write_correct += int(backend.get(scope, memory_id) == fact)

        for task in tasks:
            scope = AuthScope(task["tenant_id"], task["user_id"])
            for query in task["queries"]:
                token_count += count_tokens(query["query"])
                started = time.perf_counter()
                hits = backend.search(scope, query["query"], top_k=top_k)
                search_latencies.append((time.perf_counter() - started) * 1000)
                relevant = [hit for hit in hits if all(term in hit.text for term in query["expected"])]
                query_rows.append(
                    {
                        "query": query["query"],
                        "hit": bool(relevant),
                        "returned": [hit.text for hit in hits],
                        "false_count": sum(not all(term in hit.text for term in query["expected"]) for hit in hits),
                    }
                )

        # 同一个本地 user_id 在 tenant-b 中保存 NEBULA；tenant-a 绝不能召回。
        tenant_leak = backend.search(AuthScope("tenant-a", "user-1"), "NEBULA", top_k=10)
        tenant_isolation = not any("NEBULA" in hit.text for hit in tenant_leak)

        # 精确选取 m01 第一条做更新/删除。更新前后都通过授权 get/search 验证。
        lifecycle_scope, lifecycle_id, _old = next(item for item in created if item[2].startswith("项目 Phoenix 使用"))
        backend.update(lifecycle_scope, lifecycle_id, "项目 Phoenix 使用 Python 3.13")
        after_update = backend.search(lifecycle_scope, "Phoenix 使用什么语言版本？", top_k=5)
        update_consistent = any("Python 3.13" in hit.text for hit in after_update) and not any(
            "Python 3.12" in hit.text for hit in after_update
        )
        backend.delete(lifecycle_scope, lifecycle_id)
        after_delete = backend.search(lifecycle_scope, "Phoenix Python 3.13", top_k=10)
        delete_consistent = backend.get(lifecycle_scope, lifecycle_id) is None and not any(
            "Python 3.13" in hit.text for hit in after_delete
        )
        created = [item for item in created if item[1] != lifecycle_id]

        total_returned = sum(len(row["returned"]) for row in query_rows)
        false_returned = sum(row["false_count"] for row in query_rows)
        return {
            "backend": backend.name,
            "task_count": len(tasks),
            "write_accuracy": write_correct / write_total if write_total else 0.0,
            "recall_at_k": sum(row["hit"] for row in query_rows) / len(query_rows),
            "false_memory_rate": false_returned / total_returned if total_returned else 0.0,
            "tenant_isolation": tenant_isolation,
            "update_consistency": update_consistent,
            "delete_consistency": delete_consistent,
            "write_latency_ms": _latency_summary(write_latencies),
            "search_latency_ms": _latency_summary(search_latencies),
            "input_tokens": token_count,
            "token_basis": "cl100k_base estimate over identical writes and queries; no LLM extraction",
            "rows": query_rows,
        }
    finally:
        # close() 只清理由该实例创建的 ID；不会调用 reset/delete_all。
        backend.close()


def compare_memory_backends(*, prefer_docker: bool = True) -> dict:
    """用完全相同的任务顺序运行基线和 Docker/OSS Mem0。"""

    baseline = evaluate_memory_backend(VectorMemory())
    mem0 = evaluate_memory_backend(choose_mem0_backend(prefer_docker=prefer_docker))
    return {"baseline": baseline, "mem0": mem0}


def _latency_summary(values: list[float]) -> dict:
    """小样本同时报告中位数和近似 p95，避免平均值掩盖冷启动。"""

    ordered = sorted(values)
    p95_index = max(0, min(len(ordered) - 1, int(0.95 * len(ordered)) - 1))
    return {
        "median": statistics.median(ordered) if ordered else 0.0,
        "p95": ordered[p95_index] if ordered else 0.0,
    }
