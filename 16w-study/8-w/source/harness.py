"""并发 Evaluation Harness：执行 Trial、保存 Trace、运行 Grader 并聚合结果。"""

from __future__ import annotations

import asyncio
from collections import Counter, defaultdict
from datetime import UTC, datetime
import json
from pathlib import Path
import statistics
import time
from typing import Iterable
import uuid

from agents import EvalAgent
from graders import LLMJudge, grade_deterministic, skipped_llm_grade
from models import (
    AgentOutput,
    AggregateReport,
    EvalTask,
    GraderKind,
    GraderResult,
    TrialStatus,
    TrialRecord,
)
from trace import TraceRecorder


class EvaluationHarness:
    """一个 Harness 实例对应一个输出目录，防止并发批次互相覆盖。"""

    def __init__(
        self,
        output_dir: Path,
        *,
        concurrency: int = 8,
        llm_judge: LLMJudge | None = None,
    ) -> None:
        if concurrency < 1:
            raise ValueError("concurrency 必须至少为 1")
        self.output_dir = output_dir
        self.concurrency = concurrency
        self.llm_judge = llm_judge

    async def run(self, tasks: list[EvalTask], agent: EvalAgent) -> tuple[list[TrialRecord], AggregateReport]:
        if not tasks:
            raise ValueError("任务集合不能为空")
        if self.output_dir.exists() and any(self.output_dir.iterdir()):
            raise FileExistsError(f"输出目录非空，拒绝污染新评测: {self.output_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "traces").mkdir()
        run_id = f"eval_{uuid.uuid4().hex[:16]}"
        semaphore = asyncio.Semaphore(self.concurrency)

        async def bounded(task: EvalTask) -> TrialRecord:
            async with semaphore:
                return await self._run_trial(run_id, task, agent)

        trials = await asyncio.gather(*(bounded(task) for task in tasks))
        report = aggregate(
            run_id, agent.version, trials,
            judge_requested=self.llm_judge is not None,
        )
        self._write_results(tasks, trials, report)
        return trials, report

    async def _run_trial(self, run_id: str, task: EvalTask, agent: EvalAgent) -> TrialRecord:
        trace = TraceRecorder()
        started_at = datetime.now(UTC)
        started_clock = time.perf_counter()
        try:
            with trace.span("trial", "agent-run", detail={"task_id": task.id}):
                output = await asyncio.wait_for(
                    agent.run(task.agent_view(), trace),
                    timeout=task.environment.timeout_ms / 1000,
                )
        except TimeoutError:
            # 单个 Trial 超时必须成为可评分结果，不能让 asyncio.gather 取消整批任务。
            output = AgentOutput(
                status=TrialStatus.FAILED,
                answer="Agent Trial 超时，未产生可验证结果。",
                safety_flags=["agent_timeout"],
            )
        except Exception as exc:
            output = AgentOutput(
                status=TrialStatus.FAILED,
                answer=f"Agent Trial 异常终止：{type(exc).__name__}",
                safety_flags=["agent_exception"],
            )

        grades: list[GraderResult] = []
        for spec in task.graders:
            if spec.kind == GraderKind.DETERMINISTIC:
                with trace.span("grader", spec.id, detail={"kind": spec.kind.value}):
                    grades.append(grade_deterministic(spec, output))
            elif self.llm_judge is None:
                grades.append(skipped_llm_grade(spec))
            else:
                try:
                    with trace.span("grader", spec.id, detail={"kind": spec.kind.value}):
                        grades.append(await self.llm_judge.grade(spec, task, output))
                except Exception as exc:
                    # Judge 故障与 Agent 失败是不同故障域。记录后继续保存 Trial，
                    # 后续 CI 会因复核数量不足而阻断，而不是丢失整批轨迹。
                    grades.append(GraderResult(
                        grader_id=spec.id,
                        grader_version=spec.version,
                        kind=spec.kind,
                        passed=None,
                        score=None,
                        hard_gate=spec.hard_gate,
                        reason=f"LLM Judge error: {type(exc).__name__}",
                        evidence={"error_type": type(exc).__name__},
                        status="error",
                    ))

        strict_success = all(
            grade.passed is True
            for grade in grades
            if grade.hard_gate and grade.status == "completed"
        )
        trace_path = self.output_dir / "traces" / f"{trace.trace_id}.json"
        trace_path.write_text(
            json.dumps([event.model_dump(mode="json") for event in trace.events], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        ended_at = datetime.now(UTC)
        return TrialRecord(
            run_id=run_id,
            task_id=task.id,
            task_version=task.version,
            category=task.category,
            agent_version=agent.version,
            trace_id=trace.trace_id,
            started_at=started_at,
            ended_at=ended_at,
            latency_ms=(time.perf_counter() - started_clock) * 1000,
            output=output,
            grades=grades,
            strict_success=strict_success,
            trace_path=str(trace_path),
        )

    def _write_results(
        self,
        tasks: list[EvalTask],
        trials: list[TrialRecord],
        report: AggregateReport,
    ) -> None:
        (self.output_dir / "task-manifest.json").write_text(
            json.dumps([task.model_dump(mode="json") for task in tasks], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (self.output_dir / "trials.json").write_text(
            json.dumps([trial.model_dump(mode="json") for trial in trials], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (self.output_dir / "summary.json").write_text(
            report.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )


def aggregate(
    run_id: str,
    agent_version: str,
    trials: Iterable[TrialRecord],
    *,
    judge_requested: bool = False,
) -> AggregateReport:
    items = list(trials)
    if not items:
        raise ValueError("没有 Trial 可聚合")

    categories: dict[str, list[TrialRecord]] = defaultdict(list)
    grader_values: dict[str, list[bool]] = defaultdict(list)
    for trial in items:
        categories[trial.category.value].append(trial)
        for grade in trial.grades:
            if grade.passed is not None:
                grader_values[grade.grader_id].append(grade.passed)

    input_tokens = [item.output.input_tokens for item in items]
    output_tokens = [item.output.output_tokens for item in items]
    return AggregateReport(
        run_id=run_id,
        agent_version=agent_version,
        task_count=len({item.task_id for item in items}),
        trial_count=len(items),
        strict_success_rate=sum(item.strict_success for item in items) / len(items),
        category_success_rate={
            name: sum(item.strict_success for item in group) / len(group)
            for name, group in sorted(categories.items())
        },
        grader_pass_rate={
            name: sum(values) / len(values) for name, values in sorted(grader_values.items())
        },
        status_counts=dict(Counter(item.output.status.value for item in items)),
        average_latency_ms=statistics.fmean(item.latency_ms for item in items),
        total_input_tokens=None if all(value is None for value in input_tokens) else sum(value or 0 for value in input_tokens),
        total_output_tokens=None if all(value is None for value in output_tokens) else sum(value or 0 for value in output_tokens),
        judge_requested=judge_requested,
        judge_expected=sum(
            grade.kind == GraderKind.LLM_RUBRIC for item in items for grade in item.grades
        ) if judge_requested else 0,
        judge_completed=sum(
            grade.kind == GraderKind.LLM_RUBRIC and grade.status == "completed"
            for item in items for grade in item.grades
        ),
        judge_errors=sum(
            grade.kind == GraderKind.LLM_RUBRIC and grade.status == "error"
            for item in items for grade in item.grades
        ),
        judge_input_tokens=sum(
            int(grade.evidence.get("input_tokens") or 0)
            for item in items for grade in item.grades
            if grade.kind == GraderKind.LLM_RUBRIC
        ),
        judge_output_tokens=sum(
            int(grade.evidence.get("output_tokens") or 0)
            for item in items for grade in item.grades
            if grade.kind == GraderKind.LLM_RUBRIC
        ),
    )
