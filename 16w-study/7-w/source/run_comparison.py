"""运行四种架构并输出统一 JSON/Markdown 轨迹指标。

默认只运行完整、正常审批路径。故障与重复提交由 pytest 精确验证，避免把
故障注入和真实模型随机性混为一谈。
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import statistics
import time
from typing import Iterable

from agents_sdk_workflow import AgentsSDKWorkflow
from common import ResearchTask, RunReport
from deep_agents_workflow import DeepAgentsWorkflow
from langgraph_workflow import LangGraphWorkflow
from native_baseline import NativeBaseline


ROOT = Path(__file__).parent


def load_task() -> ResearchTask:
    values = json.loads((ROOT / "tasks.json").read_text(encoding="utf-8"))
    return ResearchTask(**values[0])


async def run_all(output_dir: Path, *, repeats: int = 1) -> list[RunReport]:
    """四种实现使用同一任务；每次运行使用独立存储，避免缓存污染。"""

    if repeats < 1:
        raise ValueError("repeats 必须至少为 1")
    if (output_dir / "runs").exists():
        raise FileExistsError(
            f"{output_dir / 'runs'} 已存在；请使用新的 --output，避免 Checkpoint/幂等账本污染实验"
        )
    task = load_task()
    reports: list[RunReport] = []
    for repetition in range(1, repeats + 1):
        run_root = output_dir / "runs" / str(repetition)

        native = NativeBaseline(run_root / "native")
        started = time.perf_counter()
        pending = native.start(task)
        completed = native.resume(
            pending.run_id, decision="approve",
            submission_id=f"{pending.run_id}:approval",
        )
        completed.latency_ms = (time.perf_counter() - started) * 1000
        reports.append(completed)

        graph = LangGraphWorkflow(run_root / "langgraph")
        started = time.perf_counter()
        pending = graph.start(task)
        completed = graph.resume(
            pending.run_id, decision="approve",
            submission_id=f"{pending.run_id}:approval",
        )
        completed.latency_ms = (time.perf_counter() - started) * 1000
        reports.append(completed)

        sdk = AgentsSDKWorkflow(run_root / "agents-sdk")
        started = time.perf_counter()
        pending = await sdk.start(task)
        if pending.status == "waiting_approval":
            # 丢弃原对象并从 SQLite + RunState JSON 重建，真实覆盖进程重启路径。
            sdk = AgentsSDKWorkflow(run_root / "agents-sdk")
            completed = await sdk.resume(
                pending.run_id, decision="approve",
                submission_id=f"{pending.run_id}:approval",
            )
        else:
            # Guardrail 失败是本轮有效实验结果，后续架构仍应继续执行。
            completed = pending
        completed.latency_ms = (time.perf_counter() - started) * 1000
        reports.append(completed)

        harness = DeepAgentsWorkflow(run_root / "deep-agents")
        reports.append(harness.run(task))
    return reports


def summarize(reports: Iterable[RunReport]) -> dict[str, dict]:
    grouped: dict[str, list[RunReport]] = {}
    for report in reports:
        grouped.setdefault(report.architecture, []).append(report)
    summary = {}
    for name, items in grouped.items():
        latencies = [item.latency_ms for item in items]
        total_tokens = [item.input_tokens + item.output_tokens for item in items]
        summary[name] = {
            "runs": len(items),
            "success_rate": sum(item.success for item in items) / len(items),
            "average_steps": statistics.fmean(item.steps for item in items),
            "average_tool_calls": statistics.fmean(item.tool_calls for item in items),
            "average_model_calls": statistics.fmean(item.model_calls for item in items),
            "average_input_tokens": statistics.fmean(item.input_tokens for item in items),
            "average_output_tokens": statistics.fmean(item.output_tokens for item in items),
            "average_latency_ms": statistics.fmean(item.latency_ms for item in items),
            "token_stddev": statistics.pstdev(total_tokens),
            "latency_stddev_ms": statistics.pstdev(latencies),
            "failure_types": sorted({
                item.error_type or "completion_predicate_failed"
                for item in items if not item.success
            }),
        }
    return summary


def write_reports(reports: list[RunReport], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(reports)
    payload = {
        "method": "same task, corpus, tools, completion predicate and metric schema",
        "runs": [report.to_dict() for report in reports],
        "summary": summary,
    }
    json_path = output_dir / "architecture-comparison.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 第 7 周四架构实验结果", "",
        "四种实现共享任务、语料、工具返回结构、完成谓词和指标字段。", "",
        "| 架构 | 成功率 | 平均步骤 | 平均工具数 | 平均模型数 | 总 Token（均值±σ） | 延迟 ms（均值±σ） |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, value in summary.items():
        lines.append(
            f"| {name} | {value['success_rate']:.2%} | {value['average_steps']:.2f} | "
            f"{value['average_tool_calls']:.2f} | {value['average_model_calls']:.2f} | "
            f"{value['average_input_tokens'] + value['average_output_tokens']:.0f}"
            f"±{value['token_stddev']:.0f} | "
            f"{value['average_latency_ms']:.2f}±{value['latency_stddev_ms']:.2f} |"
        )
    failures = [report for report in reports if not report.success]
    lines += ["", "## 失败案例", ""]
    if failures:
        for report in failures:
            lines.append(
                f"- `{report.architecture}/{report.run_id}`："
                f"`{report.error_type or 'completion_predicate_failed'}`。"
            )
    else:
        lines.append("- 三轮实验没有失败样本。")
    lines += [
        "", "## 解读边界", "",
        "- 原生和 LangGraph 的合成步骤使用确定性函数，因此 Token 为 0；它们用于比较持久化与恢复机制。",
        "- Agents SDK 和 Deep Agents 调用真实模型，Token 取服务端或框架可观测值。",
        "- Deep Agents 只验证子 Agent 和上下文隔离，不重复实现审批恢复。",
        "- 不应把一次样本的延迟差异解释为稳定性能结论；正式报告至少运行三次。",
    ]
    markdown_path = output_dir / "architecture-comparison.md"
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, markdown_path


async def async_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--repeats", type=int, default=1)
    arguments = parser.parse_args()
    reports = await run_all(arguments.output, repeats=arguments.repeats)
    paths = write_reports(reports, arguments.output)
    print("\n".join(str(path) for path in paths))


if __name__ == "__main__":
    asyncio.run(async_main())
