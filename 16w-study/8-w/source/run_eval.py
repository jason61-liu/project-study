"""第 8 周评测实验统一命令行入口。"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from agents import DegradedAgent, ReferenceAgent, TauReferenceReplayAgent
from benchmark_adapter import load_tau_subset
from graders import LLMJudge
from harness import EvaluationHarness
from models import EvalTask


ROOT = Path(__file__).parent


def load_local_tasks(path: Path) -> list[EvalTask]:
    return [EvalTask.model_validate(item) for item in json.loads(path.read_text(encoding="utf-8"))]


async def main_async() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("suite", choices=["local", "tau-adapter"])
    parser.add_argument("--tasks", type=Path, default=ROOT / "data" / "tasks.json")
    parser.add_argument("--tau-raw", type=Path, default=ROOT / "data" / "tau3-banking-v1.0.1" / "raw")
    parser.add_argument("--agent", choices=["baseline", "degraded"], default="baseline")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--llm-judge", action="store_true")
    args = parser.parse_args()

    judge = LLMJudge() if args.llm_judge else None
    if args.suite == "local":
        tasks = load_local_tasks(args.tasks)
        agent = ReferenceAgent() if args.agent == "baseline" else DegradedAgent()
    else:
        if args.agent != "baseline":
            parser.error("tau-adapter 只提供显式 Gold contract replay，不支持 degraded")
        tasks, gold_actions = load_tau_subset(args.tau_raw)
        agent = TauReferenceReplayAgent(gold_actions)

    _, report = await EvaluationHarness(
        args.output,
        concurrency=args.concurrency,
        llm_judge=judge,
    ).run(tasks, agent)
    print(report.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main_async())
