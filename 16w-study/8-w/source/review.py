"""LLM Judge 人工复核队列与误判统计。

本模块不伪造“人工”标签。它要求操作员逐条输入结论，或提供已经由人填写的
review decisions JSON。CI 在 Judge 已运行时要求至少 20 条已完成复核。
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from enum import StrEnum
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from models import EvalTask, GraderKind, TrialRecord


class MisjudgmentType(StrEnum):
    NONE = "none"
    FALSE_POSITIVE = "false_positive"
    FALSE_NEGATIVE = "false_negative"
    RATIONALE_ERROR = "rationale_error"
    SEVERITY_ERROR = "severity_error"
    PROMPT_INJECTION = "prompt_injection"


class ReviewItem(BaseModel):
    task_id: str
    task_version: str
    category: str
    trial_trace_id: str
    instruction: str
    trusted_context: dict[str, Any]
    success_conditions: list[str]
    rubric: dict[str, Any]
    agent_status: str
    agent_answer: str
    citations: list[str]
    tool_calls: list[dict[str, Any]]
    judge_passed: bool
    judge_score: float
    judge_reason: str
    deterministic_reference_passed: bool


class HumanReview(BaseModel):
    task_id: str
    trial_trace_id: str
    reviewer: str
    human_passed: bool
    judge_passed: bool
    misjudgment_type: MisjudgmentType
    notes: str
    reviewed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


def build_review_queue(
    trials: list[TrialRecord],
    count: int = 20,
    tasks: list[EvalTask] | None = None,
) -> list[ReviewItem]:
    task_by_id = {task.id: task for task in tasks or []}
    candidates: list[ReviewItem] = []
    for trial in trials:
        judge = next(
            (grade for grade in trial.grades if grade.kind == GraderKind.LLM_RUBRIC and grade.passed is not None),
            None,
        )
        if judge is None:
            continue
        task = task_by_id.get(trial.task_id)
        rubric_spec = next(
            (grader for grader in task.graders if grader.kind == GraderKind.LLM_RUBRIC),
            None,
        ) if task else None
        deterministic = [
            grade.passed for grade in trial.grades
            if grade.kind == GraderKind.DETERMINISTIC and grade.hard_gate
        ]
        candidates.append(ReviewItem(
            task_id=trial.task_id,
            task_version=trial.task_version,
            category=trial.category.value,
            trial_trace_id=trial.trace_id,
            instruction=task.input.instruction if task else "",
            trusted_context=task.environment.visible_context if task else {},
            success_conditions=[item.description for item in task.success_conditions] if task else [],
            rubric=(rubric_spec.config.get("rubric", {}) if rubric_spec else {}),
            agent_status=trial.output.status.value,
            agent_answer=trial.output.answer,
            citations=trial.output.citations,
            tool_calls=[item.model_dump(mode="json") for item in trial.output.tool_calls],
            judge_passed=bool(judge.passed),
            judge_score=float(judge.score or 0),
            judge_reason=judge.reason,
            deterministic_reference_passed=all(deterministic),
        ))
    if len(candidates) < count:
        raise ValueError(f"可复核 Judge 结果只有 {len(candidates)} 条，少于要求的 {count} 条")

    # 优先抽取 Judge 与确定性硬事实分歧的样本，再补足其余样本。
    candidates.sort(
        key=lambda item: (item.judge_passed == item.deterministic_reference_passed, item.task_id),
    )
    return candidates[:count]


def classify_misjudgment(judge_passed: bool, human_passed: bool) -> MisjudgmentType:
    if judge_passed and not human_passed:
        return MisjudgmentType.FALSE_POSITIVE
    if not judge_passed and human_passed:
        return MisjudgmentType.FALSE_NEGATIVE
    return MisjudgmentType.NONE


def summarize_reviews(reviews: list[HumanReview]) -> dict[str, Any]:
    errors = [item for item in reviews if item.misjudgment_type != MisjudgmentType.NONE]
    by_type: dict[str, int] = {}
    for item in errors:
        by_type[item.misjudgment_type.value] = by_type.get(item.misjudgment_type.value, 0) + 1
    return {
        "review_count": len(reviews),
        "misjudgment_count": len(errors),
        "misjudgment_rate": len(errors) / len(reviews) if reviews else None,
        "by_type": by_type,
    }


def load_trials(path: Path) -> list[TrialRecord]:
    return [TrialRecord.model_validate(item) for item in json.loads(path.read_text(encoding="utf-8"))]


def load_tasks(path: Path) -> list[EvalTask]:
    return [EvalTask.model_validate(item) for item in json.loads(path.read_text(encoding="utf-8"))]


def interactive_review(queue: list[ReviewItem], reviewer: str) -> list[HumanReview]:
    reviews = []
    for index, item in enumerate(queue, start=1):
        print(f"\n[{index}/{len(queue)}] {item.task_id}")
        print(f"Task: {item.instruction}")
        print(f"Trusted context: {json.dumps(item.trusted_context, ensure_ascii=False)}")
        print(f"Success conditions: {json.dumps(item.success_conditions, ensure_ascii=False)}")
        print(f"Rubric: {json.dumps(item.rubric, ensure_ascii=False)}")
        print(f"Agent status: {item.agent_status}")
        print(f"Answer: {item.agent_answer}")
        print(f"Citations: {json.dumps(item.citations, ensure_ascii=False)}")
        print(f"Tool calls: {json.dumps(item.tool_calls, ensure_ascii=False)}")
        print(f"Judge: passed={item.judge_passed} score={item.judge_score:.2f}")
        print(f"Reason: {item.judge_reason}")
        while True:
            value = input("人工结论是否通过? [y/n]: ").strip().lower()
            if value in {"y", "n"}:
                break
        human_passed = value == "y"
        automatic_type = classify_misjudgment(item.judge_passed, human_passed)
        print("误判类型：")
        choices = list(MisjudgmentType)
        for choice_index, choice in enumerate(choices, start=1):
            marker = "（根据通过/失败结论自动推荐）" if choice == automatic_type else ""
            print(f"  {choice_index}. {choice.value}{marker}")
        while True:
            raw_type = input(f"请选择 [1-{len(choices)}]，直接回车采用推荐值: ").strip()
            if not raw_type:
                selected_type = automatic_type
                break
            if raw_type.isdigit() and 1 <= int(raw_type) <= len(choices):
                selected_type = choices[int(raw_type) - 1]
                break
        notes = input("复核说明（必须给出证据或理由）: ").strip()
        if not notes:
            raise ValueError("人工复核说明不能为空")
        reviews.append(HumanReview(
            task_id=item.task_id,
            trial_trace_id=item.trial_trace_id,
            reviewer=reviewer,
            human_passed=human_passed,
            judge_passed=item.judge_passed,
            misjudgment_type=selected_type,
            notes=notes,
        ))
    return reviews


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument(
        "--task-manifest",
        type=Path,
        help="默认为 trials.json 同目录的 task-manifest.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reviewer")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="只导出固定复核队列，不把机器检查冒充人工复核",
    )
    args = parser.parse_args()
    task_manifest = args.task_manifest or args.trials.with_name("task-manifest.json")
    queue = build_review_queue(load_trials(args.trials), args.count, load_tasks(task_manifest))
    if args.prepare_only:
        payload = {
            "created_at": datetime.now(UTC).isoformat(),
            "review_count": len(queue),
            "reviews": [item.model_dump(mode="json") for item in queue],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(args.output)
        return
    if not args.reviewer:
        parser.error("交互复核必须提供 --reviewer；仅导出队列请使用 --prepare-only")
    reviews = interactive_review(queue, args.reviewer)
    payload = {
        "summary": summarize_reviews(reviews),
        "reviews": [item.model_dump(mode="json") for item in reviews],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
