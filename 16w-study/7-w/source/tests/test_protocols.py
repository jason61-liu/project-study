"""恢复、审批和幂等属于确定性协议，测试不依赖模型随机输出。"""

from __future__ import annotations

import pytest

from common import ArtifactLedger, ResearchTask, ResearchToolRuntime, TraceRecorder
from langgraph_workflow import LangGraphWorkflow
from native_baseline import NativeBaseline


@pytest.fixture
def task() -> ResearchTask:
    return ResearchTask(
        id="test-task",
        question="解释 Runtime、Checkpoint、幂等、审批、Subagent 与上下文",
        search_query="Runtime Checkpoint 幂等 审批 Subagent 上下文",
        required_terms=["Runtime", "Checkpoint", "幂等", "审批", "Subagent", "上下文"],
        required_sources=[
            "runtime-boundary", "durable-execution",
            "human-approval", "subagent-context",
        ],
    )


@pytest.mark.parametrize("implementation", [NativeBaseline, LangGraphWorkflow])
def test_start_stops_before_publish(tmp_path, task, implementation):
    workflow = implementation(tmp_path / implementation.architecture)
    pending = workflow.start(task, run_id=f"{implementation.architecture}-pending")
    assert pending.status == "waiting_approval"
    assert workflow.ledger.publication_count() == 0


@pytest.mark.parametrize("implementation", [NativeBaseline, LangGraphWorkflow])
def test_approval_resumes_and_publishes_once(tmp_path, task, implementation):
    workflow = implementation(tmp_path / implementation.architecture)
    pending = workflow.start(task, run_id=f"{implementation.architecture}-approve")
    completed = workflow.resume(
        pending.run_id, decision="approve", submission_id="approval-1",
    )
    assert completed.status == "completed"
    assert completed.success is True
    assert workflow.ledger.publication_count() == 1


@pytest.mark.parametrize("implementation", [NativeBaseline, LangGraphWorkflow])
def test_duplicate_submission_returns_cached_outcome(tmp_path, task, implementation):
    workflow = implementation(tmp_path / implementation.architecture)
    pending = workflow.start(task, run_id=f"{implementation.architecture}-duplicate")
    first = workflow.resume(pending.run_id, decision="approve", submission_id="same-request")
    duplicate = workflow.resume(pending.run_id, decision="approve", submission_id="same-request")
    assert first.status == duplicate.status == "completed"
    assert duplicate.duplicate_submissions == 1
    assert workflow.ledger.publication_count() == 1


@pytest.mark.parametrize("implementation", [NativeBaseline, LangGraphWorkflow])
def test_rejection_terminates_without_publish(tmp_path, task, implementation):
    workflow = implementation(tmp_path / implementation.architecture)
    pending = workflow.start(task, run_id=f"{implementation.architecture}-reject")
    rejected = workflow.resume(
        pending.run_id, decision="reject", submission_id="reject-request",
    )
    assert rejected.status == "rejected"
    assert workflow.ledger.publication_count() == 0


def test_langgraph_state_survives_new_process_object(tmp_path, task):
    root = tmp_path / "durable"
    first_process = LangGraphWorkflow(root)
    pending = first_process.start(task, run_id="durable-thread")
    assert pending.status == "waiting_approval"
    first_process.connection.close()

    second_process = LangGraphWorkflow(root)
    completed = second_process.resume(
        "durable-thread", decision="approve", submission_id="resume-after-restart",
    )
    assert completed.status == "completed"
    assert second_process.ledger.publication_count() == 1


@pytest.mark.parametrize("implementation", [NativeBaseline, LangGraphWorkflow])
def test_transient_tool_exception_is_retried_once(tmp_path, task, implementation):
    workflow = implementation(
        tmp_path / implementation.architecture, fault="tool_exception",
    )
    pending = workflow.start(task, run_id=f"{implementation.architecture}-tool-error")
    assert pending.status == "waiting_approval"
    read_events = [event for event in pending.trace if event.name == "read_document"]
    assert any(event.status == "error" for event in read_events)
    assert any(event.status == "success" for event in read_events)


def test_publish_tool_requires_approval_even_if_model_requests_it(tmp_path):
    ledger = ArtifactLedger(tmp_path / "ledger.sqlite")
    trace = TraceRecorder("test")
    runtime = ResearchToolRuntime(ledger, trace)
    result = runtime.call("publish_report", {
        "draft_id": "draft-x", "approved": False,
        "idempotency_key": "publish-x",
    })
    assert result.status == "error"
    assert result.error_type == "approval_required"
    assert ledger.publication_count() == 0


def test_unknown_tool_is_structured_error(tmp_path):
    runtime = ResearchToolRuntime(
        ArtifactLedger(tmp_path / "ledger.sqlite"), TraceRecorder("test"),
    )
    result = runtime.call("delete_everything", {})
    assert result.status == "error"
    assert result.error_type == "unknown_tool"

