"""运行三种研究架构的真实模型对比实验并生成报告。"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
from typing import Iterable

from architectures import FixedWorkflow, PlanAndExecute, ReAct
from model_client import DeepSeekResearchModel
from models import ResearchModel, RunMetrics, Scenario


ROOT = Path(__file__).parent
ARTIFACTS = ROOT / "artifacts"


def load_scenarios() -> list[Scenario]:
    raw = json.loads((ROOT / "data" / "scenarios.json").read_text(encoding="utf-8"))
    return [Scenario(**item) for item in raw]


def run_benchmark(model: ResearchModel, *, repeats: int = 3) -> dict:
    """对所有场景重复运行；每个 repetition 轮换架构顺序，降低时间漂移偏差。"""

    if repeats < 3:
        raise ValueError("正式对比要求每种架构至少运行 3 次")
    architecture_types = [FixedWorkflow, ReAct, PlanAndExecute]
    runs: list[RunMetrics] = []
    for scenario in load_scenarios():
        for repetition in range(1, repeats + 1):
            offset = (repetition - 1) % len(architecture_types)
            ordered = architecture_types[offset:] + architecture_types[:offset]
            for architecture_type in ordered:
                architecture = architecture_type(model)
                runs.append(architecture.run(scenario, repetition))
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": model.model,
        "repeats_per_scenario": repeats,
        "scenario_count": len(load_scenarios()),
        "comparability": {
            "same_model": True,
            "same_tools": True,
            "same_test_set": True,
            "same_max_step_budget": True,
            "execution_order": "rotated by repetition",
        },
        "runs": [run.to_dict() for run in runs],
        "summary": summarize(runs),
    }


def summarize(runs: Iterable[RunMetrics]) -> dict[str, dict]:
    grouped: dict[str, list[RunMetrics]] = defaultdict(list)
    for run in runs:
        grouped[run.architecture].append(run)
    summary: dict[str, dict] = {}
    for architecture, items in sorted(grouped.items()):
        success_count = sum(item.success for item in items)
        summary[architecture] = {
            "runs": len(items),
            "success_rate": success_count / len(items),
            "average_steps": statistics.fmean(item.steps for item in items),
            "average_tool_calls": statistics.fmean(item.tool_calls for item in items),
            "average_model_calls": statistics.fmean(item.model_calls for item in items),
            "average_input_tokens": statistics.fmean(item.input_tokens for item in items),
            "average_output_tokens": statistics.fmean(item.output_tokens for item in items),
            "average_total_tokens": statistics.fmean(item.input_tokens + item.output_tokens for item in items),
            "average_latency_ms": statistics.fmean(item.latency_ms for item in items),
            "p95_latency_ms": _percentile([item.latency_ms for item in items], 0.95),
            "failure_counts": _failure_counts(items),
            "scenario_success": {
                scenario: sum(x.success for x in items if x.scenario_id == scenario)
                / sum(1 for x in items if x.scenario_id == scenario)
                for scenario in sorted({x.scenario_id for x in items})
            },
        }
    return summary


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered)-1, max(0, round((len(ordered)-1)*quantile)))
    return ordered[index]


def _failure_counts(items: list[RunMetrics]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for item in items:
        for failure in item.failure_types:
            counts[failure] += 1
    return dict(sorted(counts.items()))


def decision_tree(best_architecture: str) -> str:
    """决策树使用任务结构做硬判断，实验赢家只作为同类任务的叶节点建议。"""

    return f"""```text
任务路径是否可预定义？
├─ 是
│  ├─ 单次调用达到质量 SLO？ ─ 是 → Single Call
│  └─ 否 → Fixed Workflow
└─ 否
   ├─ 是否存在可靠、频繁的环境反馈？
   │  ├─ 否 → 先补证据工具；不要使用自治 Agent
   │  └─ 是
   │     ├─ 是否需要全局依赖/里程碑？
   │     │  ├─ 是 → Plan-and-Execute / Rolling Horizon
   │     │  └─ 否 → Bounded ReAct
   │     └─ 不可逆动作 → Runtime 确认、幂等、Checkpoint
   └─ 若成功率增益不能覆盖 p95 延迟和 Token → 降级为 Workflow

本实验同类任务的当前数据优选：{best_architecture}
```
"""


def write_reports(report: dict, output_dir: Path = ARTIFACTS) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "architecture-comparison.json"
    md_path = output_dir / "architecture-comparison.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # 先按成功率降序，再按 p95 延迟和 Token 升序选择；安全硬约束由测试集先行过滤。
    ranked = sorted(
        report["summary"].items(),
        key=lambda item: (-item[1]["success_rate"], item[1]["p95_latency_ms"], item[1]["average_total_tokens"]),
    )
    best = ranked[0][0]
    lines = [
        "# 固定 Workflow、ReAct 与 Plan-and-Execute 实验报告", "",
        f"> 生成时间：{report['generated_at']}",
        f"> 模型：`{report['model']}`；每场景重复：`{report['repeats_per_scenario']}`；场景数：`{report['scenario_count']}`", "",
        "## 可比性约束", "",
        "- 三种架构使用同一模型实例与参数；",
        "- 使用同一工具 Schema、语料、测试问题、故障注入和评分器；",
        "- 每轮旋转执行顺序，降低服务时间漂移和缓存顺序偏差；",
        "- 差异只允许出现在控制策略。", "",
        "## 架构对比表", "",
        "| 架构 | 运行数 | 成功率 | 平均步骤 | 平均工具数 | 平均模型数 | 平均 Token | 平均延迟 ms | p95 ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, metric in report["summary"].items():
        lines.append(
            f"| {name} | {metric['runs']} | {metric['success_rate']:.1%} | {metric['average_steps']:.2f} | "
            f"{metric['average_tool_calls']:.2f} | {metric['average_model_calls']:.2f} | "
            f"{metric['average_total_tokens']:.1f} | {metric['average_latency_ms']:.1f} | {metric['p95_latency_ms']:.1f} |"
        )
    lines.extend(["", "## 分场景成功率", "", "| 架构 | normal | retrieval_failure | tool_failure | plan_invalidated |", "|---|---:|---:|---:|---:|"])
    for name, metric in report["summary"].items():
        values = metric["scenario_success"]
        lines.append(f"| {name} | {values['normal']:.1%} | {values['retrieval_failure']:.1%} | {values['tool_failure']:.1%} | {values['plan_invalidated']:.1%} |")
    lines.extend([
        "", "## 推荐决策树", "", decision_tree(best),
        "## 解释边界", "",
        f"当前数据排序优选 `{best}`，它只适用于本语料、本模型版本、本预算和受监管客服研究任务。",
        "不能把该结论外推为通用排名；换成开放网页研究、不可预测 UI 或不同延迟 SLO 后必须重跑。",
        "完整逐轨迹工具错误、Token 和 Trace 见同目录 JSON。", "",
    ])
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=ARTIFACTS)
    args = parser.parse_args()
    model = DeepSeekResearchModel()
    paths = write_reports(run_benchmark(model, repeats=args.repeats), args.output_dir)
    print("generated:", *(str(path) for path in paths))


if __name__ == "__main__":
    main()

