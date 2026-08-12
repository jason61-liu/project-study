"""结构化计划、恢复语义和 Manager 消融的确定性协议测试。"""

from __future__ import annotations

import json

import pytest

from models import ModelReply
from recoverable_agents import (
    ManagerSpecialistAblation, RecoverableSingleAgent, SimulatedProcessInterrupt,
)
from recovery_models import FailureClass, StructuredPlan, classify_failure
from run_recovery_ablation import load_tasks, run_ablation, write_report
from state_store import JsonPlanStore, StateConflict


class RecoveryTestModel:
    """只验证协议控制流；正式消融仍必须使用真实模型。"""

    model = "test-only-recovery-model"

    def complete_json(self, *, system: str, user: str, purpose: str) -> ModelReply:
        request = json.loads(user)
        evidence = request["evidence"]
        terms = "、".join(request["required_terms"])
        answer = f"结论覆盖：{terms}。依据结构化证据执行并记录恢复状态。"
        return ModelReply(
            data={"answer": answer, "citations": [item["id"] for item in evidence]},
            input_tokens=120, output_tokens=45, latency_ms=1.0, model=self.model,
        )


def task(identifier: str):
    return next(item for item in load_tasks() if item.id == identifier)


def test_plan_is_structured_and_tracks_each_item(tmp_path):
    agent = RecoverableSingleAgent(RecoveryTestModel(), JsonPlanStore(tmp_path))
    result = agent.run(task("t01-control-boundary"), 1)
    plan = agent.store.load(result.plan_id)
    assert [item.id for item in plan.items] == ["search", "read", "synthesize"]
    assert all(item.completion_condition for item in plan.items)
    assert [item.status for item in plan.items] == ["completed", "completed", "completed"]
    assert plan.version == plan.checkpoints


@pytest.mark.parametrize(
    ("error_type", "expected"),
    [
        ("tool_unavailable", FailureClass.RETRYABLE),
        ("plan_invalidated", FailureClass.REPLAN_REQUIRED),
        ("permission_denied", FailureClass.HUMAN_REQUIRED),
        ("invalid_arguments", FailureClass.UNRECOVERABLE),
    ],
)
def test_failure_classification_has_four_recovery_paths(error_type, expected):
    assert classify_failure(error_type) is expected


def test_process_interrupt_resumes_from_checkpoint_without_repeating_search(tmp_path):
    store = JsonPlanStore(tmp_path)
    first_process = RecoverableSingleAgent(RecoveryTestModel(), store)
    with pytest.raises(SimulatedProcessInterrupt) as interrupted:
        first_process.run(task("t16-process-resume"), 1)
    plan_id = str(interrupted.value)
    before = store.load(plan_id)
    search_records = {key for key in before.execution_records if ":search:" in key}

    second_process = RecoverableSingleAgent(RecoveryTestModel(), store)
    result = second_process.run(task("t16-process-resume"), 1, plan_id=plan_id)
    after = store.load(plan_id)
    assert result.success
    assert {key for key in after.execution_records if ":search:" in key} == search_records


def test_partial_success_is_preserved_instead_of_collapsed_to_failure(tmp_path):
    agent = RecoverableSingleAgent(RecoveryTestModel(), JsonPlanStore(tmp_path))
    result = agent.run(task("t14-partial"), 1)
    plan = agent.store.load(result.plan_id)
    assert plan.status == "partial"
    assert plan.item("read").status == "partial"
    assert plan.item("read").result["evidence"]
    assert result.failure_counts[FailureClass.RETRYABLE.value] == 1


def test_compare_and_swap_rejects_stale_state(tmp_path):
    store = JsonPlanStore(tmp_path)
    agent = RecoverableSingleAgent(RecoveryTestModel(), store)
    created = agent.create_plan(task("t01-control-boundary"), "conflict")
    writer_one = store.load(created.plan_id)
    writer_two = store.load(created.plan_id)
    writer_one.item("search").status = "in_progress"
    store.save(writer_one, expected_version=0)
    writer_two.item("search").status = "failed"
    with pytest.raises(StateConflict, match="actual 1"):
        store.save(writer_two, expected_version=0)


def test_duplicate_delegation_is_suppressed(tmp_path):
    store = JsonPlanStore(tmp_path)
    manager = ManagerSpecialistAblation(RecoveryTestModel(), store)
    result = manager.run(task("t01-control-boundary"), 1)
    before = store.load(result.plan_id)
    delegation_count = len(before.delegations)
    after = manager.invoke_duplicate_for_test(task("t01-control-boundary"), result.plan_id, "search")
    assert len(after.delegations) == delegation_count
    assert after.duplicate_delegations_suppressed == 1


def test_context_loss_rebuilds_from_checkpoint(tmp_path):
    agent = RecoverableSingleAgent(RecoveryTestModel(), JsonPlanStore(tmp_path))
    result = agent.run(task("t15-context-loss"), 1)
    assert result.success
    assert result.context_rebuilds == 1
    assert result.failure_counts[FailureClass.RETRYABLE.value] == 1


def test_plan_invalidation_increments_revision_and_finishes(tmp_path):
    agent = RecoverableSingleAgent(RecoveryTestModel(), JsonPlanStore(tmp_path))
    result = agent.run(task("t13-version-change"), 1)
    assert result.success
    assert result.plan_revision == 2
    assert result.failure_counts[FailureClass.REPLAN_REQUIRED.value] == 1


def test_ablation_has_twenty_tasks_three_repeats_and_equal_runs(tmp_path):
    report = run_ablation(RecoveryTestModel(), repeats=3, checkpoint_root=tmp_path / "states")
    assert report["task_count"] >= 20
    assert len(report["runs"]) == report["task_count"] * 3 * 2
    assert {value["runs"] for value in report["summary"].values()} == {report["task_count"] * 3}
    assert report["baseline"] == "recoverable_single_agent"
    assert report["ablation_only"] == "manager_specialist_ablation"


def test_ablation_report_records_three_repetitions(tmp_path):
    report = run_ablation(RecoveryTestModel(), repeats=3, checkpoint_root=tmp_path / "states")
    json_path, markdown_path = write_report(report, tmp_path / "report")
    persisted = json.loads(json_path.read_text(encoding="utf-8"))
    assert persisted["repeats_per_task"] == 3
    assert "Manager" in markdown_path.read_text(encoding="utf-8")


def test_ablation_rejects_fewer_than_three_repeats(tmp_path):
    with pytest.raises(ValueError, match="至少重复 3 次"):
        run_ablation(RecoveryTestModel(), repeats=2, checkpoint_root=tmp_path)
