from copy import deepcopy

import pytest

from ci_gate import evaluate_gate
from models import AggregateReport, GateConfig
from review import (
    HumanReview,
    MisjudgmentType,
    build_review_queue,
    classify_misjudgment,
    summarize_reviews,
)


def report(rate: float, *, judge_completed: int = 0) -> AggregateReport:
    return AggregateReport(
        run_id=f"run-{rate}", agent_version="agent", task_count=60, trial_count=60,
        strict_success_rate=rate,
        category_success_rate={
            "normal": rate, "boundary": rate, "failure": rate, "adversarial": rate,
        },
        grader_pass_rate={}, status_counts={"completed": 60}, average_latency_ms=1,
        total_input_tokens=None, total_output_tokens=None,
        judge_requested=judge_completed > 0, judge_expected=judge_completed,
        judge_completed=judge_completed, judge_errors=0,
    )


def config() -> GateConfig:
    return GateConfig(
        min_task_count=50,
        min_overall_success_rate=0.95,
        min_category_success_rate={"adversarial": 0.95, "failure": 0.95},
        max_success_drop=0.02,
    )


def test_gate_allows_healthy_baseline():
    result = evaluate_gate(report(1.0), report(0.99), config())
    assert result.passed is True


def test_gate_blocks_degraded_candidate():
    result = evaluate_gate(report(1.0), report(0.25), config())
    assert result.passed is False
    assert any("success_drop" in item for item in result.violations)
    assert any("category[adversarial]" in item for item in result.violations)


def test_gate_requires_twenty_human_reviews_when_judge_runs():
    result = evaluate_gate(
        report(1.0), report(1.0, judge_completed=20), config(),
        review_summary={"review_count": 19, "misjudgment_rate": 0},
    )
    assert result.passed is False
    assert "human_review_count 19 < 20" in result.violations


def test_review_misjudgment_taxonomy_and_summary():
    assert classify_misjudgment(True, False) == MisjudgmentType.FALSE_POSITIVE
    assert classify_misjudgment(False, True) == MisjudgmentType.FALSE_NEGATIVE
    reviews = [HumanReview(
        task_id=f"task-{i}", trial_trace_id=f"trace-{i}", reviewer="human",
        human_passed=i != 0, judge_passed=True,
        misjudgment_type=MisjudgmentType.FALSE_POSITIVE if i == 0 else MisjudgmentType.NONE,
        notes="checked against source",
    ) for i in range(20)]
    summary = summarize_reviews(reviews)
    assert summary["review_count"] == 20
    assert summary["misjudgment_rate"] == pytest.approx(0.05)


def test_review_queue_refuses_fewer_than_twenty_judgments():
    with pytest.raises(ValueError, match="少于要求"):
        build_review_queue([], 20)


def test_gate_blocks_judge_run_without_human_review():
    result = evaluate_gate(report(1.0), report(1.0, judge_completed=20), config())
    assert result.passed is False
    assert "human_review_count 0 < 20" in result.violations
