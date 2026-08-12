"""运行可恢复单 Agent 与 Manager + Specialist 的 20 任务消融实验。"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import tempfile
from typing import Iterable

from model_client import DeepSeekResearchModel
from models import ResearchModel
from recoverable_agents import (
    ManagerSpecialistAblation, RecoverableSingleAgent, SimulatedProcessInterrupt,
)
from recovery_models import RecoveryRun, RecoveryTask
from state_store import JsonPlanStore


ROOT = Path(__file__).parent
ARTIFACTS = ROOT / "artifacts"


def load_tasks() -> list[RecoveryTask]:
    values = json.loads((ROOT / "data" / "recovery_tasks.json").read_text(encoding="utf-8"))
    tasks = [RecoveryTask(**value) for value in values]
    if len(tasks) < 20:
        raise ValueError("消融实验至少需要 20 条任务")
    return tasks


def run_ablation(
    model: ResearchModel, *, repeats: int = 3, checkpoint_root: Path | None = None,
) -> dict:
    if repeats < 3:
        raise ValueError("消融实验每条任务至少重复 3 次")
    root = checkpoint_root or Path(tempfile.mkdtemp(prefix="agent-recovery-"))
    architecture_types = [RecoverableSingleAgent, ManagerSpecialistAblation]
    runs: list[RecoveryRun] = []

    for task in load_tasks():
        for repetition in range(1, repeats + 1):
            # 轮换顺序，降低远端模型负载随时间变化造成的顺序偏差。
            ordered = architecture_types if repetition % 2 else list(reversed(architecture_types))
            for architecture_type in ordered:
                store = JsonPlanStore(root / architecture_type.name / task.id / str(repetition))
                agent = architecture_type(model, store)
                try:
                    result = agent.run(task, repetition)
                except SimulatedProcessInterrupt as interrupted:
                    # 创建新 Agent 实例模拟旧进程内存全部消失，只凭 plan_id 恢复。
                    restarted = architecture_type(model, store)
                    result = restarted.run(task, repetition, plan_id=str(interrupted))
                runs.append(result)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": model.model,
        "task_count": len(load_tasks()),
        "repeats_per_task": repeats,
        "architectures": [item.name for item in architecture_types],
        "baseline": RecoverableSingleAgent.name,
        "ablation_only": ManagerSpecialistAblation.name,
        "comparability": {
            "same_model": True, "same_tools": True, "same_tasks": True,
            "same_failure_policy": True, "rotated_order": True,
        },
        "runs": [run.to_dict() for run in runs],
        "summary": summarize(runs),
    }


def summarize(runs: Iterable[RecoveryRun]) -> dict[str, dict]:
    grouped: dict[str, list[RecoveryRun]] = defaultdict(list)
    for run in runs:
        grouped[run.architecture].append(run)
    output = {}
    for architecture, items in sorted(grouped.items()):
        output[architecture] = {
            "runs": len(items),
            "success_rate": sum(item.success for item in items) / len(items),
            "partial_rate": sum(item.status == "partial" for item in items) / len(items),
            "average_tool_calls": statistics.fmean(item.tool_calls for item in items),
            "average_model_calls": statistics.fmean(item.model_calls for item in items),
            "average_total_tokens": statistics.fmean(item.input_tokens + item.output_tokens for item in items),
            "average_latency_ms": statistics.fmean(item.latency_ms for item in items),
            "average_checkpoints": statistics.fmean(item.checkpoints for item in items),
            "context_rebuilds": sum(item.context_rebuilds for item in items),
            "duplicates_suppressed": sum(item.duplicate_delegations_suppressed for item in items),
            "failure_counts": _sum_counts(item.failure_counts for item in items),
        }
    return output


def _sum_counts(counts: Iterable[dict[str, int]]) -> dict[str, int]:
    total: dict[str, int] = defaultdict(int)
    for value in counts:
        for key, count in value.items():
            total[key] += count
    return dict(sorted(total.items()))


def write_report(report: dict, output_dir: Path = ARTIFACTS) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "recovery-ablation.json"
    md_path = output_dir / "recovery-ablation.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 可恢复 Agent 编排消融实验", "",
        f"> 模型：`{report['model']}`；任务：`{report['task_count']}`；每任务重复：`{report['repeats_per_task']}`", "",
        "`recoverable_single_agent` 是生产基线；`manager_specialist_ablation` 只用于测量委派控制面的增益与开销。", "",
        "| 架构 | 运行数 | 成功率 | 部分成功率 | 平均工具数 | 平均模型数 | 平均 Token | 平均延迟 ms | 平均 Checkpoint |", 
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, value in report["summary"].items():
        lines.append(
            f"| {name} | {value['runs']} | {value['success_rate']:.1%} | {value['partial_rate']:.1%} | "
            f"{value['average_tool_calls']:.2f} | {value['average_model_calls']:.2f} | "
            f"{value['average_total_tokens']:.1f} | {value['average_latency_ms']:.1f} | "
            f"{value['average_checkpoints']:.2f} |"
        )
    lines.extend([
        "", "## 解释约束", "",
        "两组共享模型、工具、任务、失败分类与评分器，唯一变量是 Manager 的委派控制面。",
        "若 Specialist 没有带来成功率提升，其额外状态、委派和上下文成本应被视为负收益，而不是默认扩展点。",
        "逐次结果、计划 revision、四类失败计数和恢复指标见 JSON。", "",
    ])
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=ARTIFACTS)
    parser.add_argument("--checkpoint-dir", type=Path, default=ARTIFACTS / "checkpoints")
    arguments = parser.parse_args()
    report = run_ablation(
        DeepSeekResearchModel(), repeats=arguments.repeats,
        checkpoint_root=arguments.checkpoint_dir,
    )
    print("generated:", *write_report(report, arguments.output_dir))


if __name__ == "__main__":
    main()
