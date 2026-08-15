"""把固定的 τ³ banking_knowledge v1.0.1 子集适配到统一 Task Schema。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from models import (
    EnvironmentSpec,
    EvalTask,
    GraderKind,
    GraderSpec,
    SourceRef,
    SuccessCondition,
    TaskCategory,
    TaskInput,
)


REVISION = "v1.0.1"
SOURCE_URL = "https://github.com/sierra-research/tau2-bench/tree/v1.0.1"


def adapt_tau_task(raw: dict[str, Any]) -> EvalTask:
    """保留官方 Gold Actions 和 reward basis，但不把它们放进 Agent 可见环境。"""

    original_id = str(raw["id"])
    criteria = raw["evaluation_criteria"]
    actions = criteria.get("actions") or []
    communicate = criteria.get("communicate_info") or []
    conditions = [SuccessCondition(
        id="tau-actions",
        description="输出动作必须与官方 evaluation_criteria.actions 匹配",
    )]
    graders = [GraderSpec(
        id="tau-actions",
        kind=GraderKind.DETERMINISTIC,
        check="actions_match",
        config={"expected": actions, "reward_basis": criteria.get("reward_basis", [])},
    )]
    if communicate:
        conditions.append(SuccessCondition(
            id="tau-communicate",
            description="回答必须覆盖官方 communicate_info",
        ))
        graders.append(GraderSpec(
            id="tau-communicate",
            kind=GraderKind.DETERMINISTIC,
            check="communicate_info",
            config={"values": communicate},
        ))

    return EvalTask(
        id=f"tau3-banking-{original_id.replace('_', '-')}",
        version="1.0.1",
        category=TaskCategory.BENCHMARK,
        source=SourceRef(
            name="tau3-banking_knowledge/base-adapter-subset",
            revision=REVISION,
            url=SOURCE_URL,
            original_task_id=original_id,
        ),
        input=TaskInput(
            instruction=raw["user_scenario"]["instructions"],
            operation="tau-conversation",
            arguments={"user_tools": raw.get("user_tools") or []},
        ),
        environment=EnvironmentSpec(
            fixture_id=f"tau3-{original_id}",
            visible_context={
                "mode": "tau_conversation",
                "required_documents": raw.get("required_documents") or [],
                "reward_basis": criteria.get("reward_basis") or [],
            },
        ),
        success_conditions=conditions,
        graders=graders,
        tags=["tau3", "banking_knowledge", "base", "adapter-smoke"],
    )


def load_tau_subset(raw_dir: Path) -> tuple[list[EvalTask], dict[str, list[dict[str, Any]]]]:
    tasks: list[EvalTask] = []
    gold_actions: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(raw_dir.glob("task_*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        task = adapt_tau_task(raw)
        tasks.append(task)
        gold_actions[task.id] = raw["evaluation_criteria"].get("actions") or []
    if not tasks:
        raise FileNotFoundError(f"没有找到 τ³ 任务: {raw_dir}")
    return tasks, gold_actions
