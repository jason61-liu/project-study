"""运行第四周完整实验并生成 JSON 与 Markdown 报告。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from context_eval import evaluate_context_strategies
from memory_backends import DockerMem0Memory, Mem0CloudMemory, Mem0OssMemory, VectorMemory
from memory_eval import evaluate_memory_backend
from model_client import DeepSeekModel
from rag_chain import DocumentStore, HybridRetriever, MinimalRAGChain, evaluate_retrieval


ROOT = Path(__file__).parent
ARTIFACTS = ROOT / "artifacts"


def select_mem0(mode: str):
    """选择 Mem0 后端并返回可审计的选择原因。"""

    cloud_configured = bool(os.getenv("MEM0_API_KEY"))
    cloud_backend = None
    cloud_ready = False
    if cloud_configured and mode in {"auto", "cloud"}:
        try:
            cloud_backend = Mem0CloudMemory()
            cloud_ready = True
        except ValueError:
            cloud_backend = None
    docker_running = DockerMem0Memory.available()
    docker_ready = False
    if docker_running and mode in {"auto", "docker"}:
        docker_ready = DockerMem0Memory().ready()

    if mode == "cloud":
        if not cloud_configured:
            raise SystemExit("指定 cloud，但缺少 MEM0_API_KEY")
        if not cloud_ready:
            raise SystemExit("指定 cloud，但 MEM0_API_KEY 验证失败")
        return cloud_backend, docker_running, docker_ready, cloud_ready, "forced Mem0 Cloud"
    if mode == "docker":
        if not docker_running:
            raise SystemExit("指定 docker，但 mem0-dev-mem0-1 未运行")
        if not docker_ready:
            raise SystemExit("指定 docker，但 Mem0 Provider 健康探测失败")
        return DockerMem0Memory(), docker_running, docker_ready, cloud_ready, "forced docker"
    if mode == "local":
        return Mem0OssMemory(), docker_running, docker_ready, cloud_ready, "forced process-local OSS"
    if cloud_ready:
        return cloud_backend, docker_running, docker_ready, cloud_ready, "auto selected validated Mem0 Cloud"
    if docker_ready:
        return DockerMem0Memory(), docker_running, docker_ready, cloud_ready, "auto selected healthy Docker service"
    reason = "Cloud key invalid and Docker unavailable/unhealthy; fell back to process-local OSS"
    if not cloud_configured:
        reason = "Cloud not configured and Docker unavailable/unhealthy; fell back to process-local OSS"
    return Mem0OssMemory(), docker_running, docker_ready, cloud_ready, reason


def run(mode: str, input_price: float, *, real_model: bool) -> dict:
    """顺序运行检索、上下文和记忆实验，避免并发互相污染延迟。"""

    model = DeepSeekModel() if real_model else None
    chain = MinimalRAGChain(HybridRetriever(DocumentStore.from_json()), model=model)
    retrieval = evaluate_retrieval(chain)
    contexts = evaluate_context_strategies(input_usd_per_million=input_price, model=model)
    mem0_backend, docker_running, docker_ready, cloud_ready, reason = select_mem0(mode)
    memories = {
        "baseline": evaluate_memory_backend(VectorMemory()),
        "mem0": evaluate_memory_backend(mem0_backend),
    }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "requested_mem0_mode": mode,
            "docker_container_running": docker_running,
            "docker_provider_ready": docker_ready,
            "selected_mem0_backend": memories["mem0"]["backend"],
            "selection_reason": reason,
            "cloud_mem0_key_present": bool(os.getenv("MEM0_API_KEY")),
            "cloud_mem0_key_valid": cloud_ready,
            "real_model_used": real_model,
            "model": model.model if model else None,
            "model_base_url": os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com") if model else None,
        },
        "retrieval": retrieval,
        "context_strategies": contexts,
        "memory_comparison": memories,
    }


def write_reports(report: dict, output_dir: Path = ARTIFACTS) -> tuple[Path, Path]:
    """保存机器可读明细和适合学习复盘的 Markdown 摘要。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "experiment-report.json"
    markdown_path = output_dir / "experiment-report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    retrieval = report["retrieval"]
    contexts = report["context_strategies"]
    baseline = report["memory_comparison"]["baseline"]
    mem0 = report["memory_comparison"]["mem0"]
    lines = [
        "# 第四周检索与记忆实验报告",
        "",
        f"> 生成时间：{report['generated_at']}",
        f"> Mem0 后端：`{report['environment']['selected_mem0_backend']}`；{report['environment']['selection_reason']}",
        "",
        "## 文档检索链",
        "",
        "| 问题数 | 正确率 | Recall@K | 无答案正确率 | 平均延迟 ms |",
        "|---:|---:|---:|---:|---:|",
        f"| {retrieval['question_count']} | {retrieval['accuracy']:.3f} | {retrieval['recall_at_k']:.3f} | {retrieval['no_answer_accuracy']:.3f} | {retrieval['average_latency_ms']:.3f} |",
        "",
        f"模型输入/输出 Token：`{retrieval['model_input_tokens']}/{retrieval['model_output_tokens']}`；估算费用：`${retrieval['model_cost_usd']:.8f}`。",
        "",
        "## 三种上下文策略",
        "",
        "| 策略 | 正确率 | 平均输入 Token | 平均输出 Token | 平均总延迟 ms | 单位任务成本 USD |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in contexts.items():
        lines.append(
            f"| {name} | {metrics['accuracy']:.3f} | {metrics['average_input_tokens']:.2f} | "
            f"{metrics['average_output_tokens']:.2f} | {metrics['average_latency_ms']:.3f} | "
            f"{metrics['unit_task_cost_usd']:.8f} |"
        )
    lines.extend(
        [
            "",
            "> 真实模型模式使用 DeepSeek API usage；无模型模式才使用 cl100k_base/字符近似。费用按官方 DeepSeek-V4-Pro 缓存命中、未命中和输出单价估算。",
            "",
            "## 自建向量记忆与 Mem0",
            "",
            "| 后端 | 写入正确率 | Recall@K | 错误记忆率 | 租户隔离 | 更新一致 | 删除一致 | 写入 p50 ms | 检索 p50 ms | Token |",
            "|---|---:|---:|---:|---|---|---|---:|---:|---:|",
        ]
    )
    for metrics in (baseline, mem0):
        lines.append(
            f"| {metrics['backend']} | {metrics['write_accuracy']:.3f} | {metrics['recall_at_k']:.3f} | "
            f"{metrics['false_memory_rate']:.3f} | {metrics['tenant_isolation']} | "
            f"{metrics['update_consistency']} | {metrics['delete_consistency']} | "
            f"{metrics['write_latency_ms']['median']:.3f} | {metrics['search_latency_ms']['median']:.3f} | "
            f"{metrics['input_tokens']} |"
        )
    lines.extend(
        [
            "",
            "## 环境与解释",
            "",
            f"- Docker 容器运行：`{report['environment']['docker_container_running']}`",
            f"- Docker Provider 可用：`{report['environment']['docker_provider_ready']}`",
            f"- Mem0 Cloud Key：`{report['environment']['cloud_mem0_key_present']}`",
            f"- Mem0 Cloud Key 验证成功：`{report['environment']['cloud_mem0_key_valid']}`",
            f"- 真实模型：`{report['environment']['real_model_used']}`；`{report['environment']['model']}`",
            "- 两个记忆后端接收相同的原子事实；本轮比较存储、检索和生命周期，不比较云 LLM 事实抽取质量。",
            "- 错误记忆率按 top-K 返回中不含 Gold 事实的条目比例计算；它与 Recall@K 是互补指标。",
            "",
        ]
    )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, markdown_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mem0", choices=("auto", "cloud", "docker", "local"), default="auto")
    parser.add_argument("--input-usd-per-million", type=float, default=1.0)
    parser.add_argument("--real-model", action="store_true", help="调用 deepseek-v4-pro 并记录 API usage")
    args = parser.parse_args()
    paths = write_reports(run(args.mem0, args.input_usd_per_million, real_model=args.real_model))
    print("generated:", *(str(path) for path in paths))


if __name__ == "__main__":
    main()
