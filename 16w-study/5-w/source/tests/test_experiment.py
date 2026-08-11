"""测试控制流和故障恢复；TestResearchModel 不得生成正式实验报告。"""

from __future__ import annotations

import json

import pytest

from architectures import FixedWorkflow, PlanAndExecute, ReAct
from models import ModelReply
from run_experiment import load_scenarios, run_benchmark, write_reports
from tools import ResearchToolRuntime


class TestResearchModel:
    """确定性协议测试替身，只验证编排，不代表模型质量或真实 Token。"""

    __test__ = False
    model = "test-only-not-for-benchmark"

    def complete_json(self, *, system: str, user: str, purpose: str) -> ModelReply:
        if purpose in {"plan", "replan"}:
            data = {"queries": ["workflow runtime latency recovery planning trace safety"], "read_limit": 6}
        elif purpose == "react_decision":
            state = json.loads(user)
            observations = state["observations"]
            read_ids = state["read_evidence_ids"]
            if not observations:
                data = {"type": "tool", "tool_name": "search_documents",
                        "arguments": {"query": state["question"], "max_results": 6}}
            elif observations[-1]["error_type"] == "retriever_unavailable":
                data = {"type": "tool", "tool_name": "search_documents",
                        "arguments": {"query": state["question"], "max_results": 6}}
            else:
                order = ["arch-boundary", "safety-runtime", "latency-cost", "observability", "planning"]
                missing = next((doc_id for doc_id in order if doc_id not in read_ids), None)
                if missing:
                    version = observations[-1]["corpus_version"]
                    data = {"type": "tool", "tool_name": "read_document",
                            "arguments": {"doc_id": missing, "expected_version": version}}
                else:
                    data = {"type": "final", "answer": _answer(), "citations": order}
        else:
            data = {"answer": _answer(), "citations": [
                "arch-boundary", "safety-runtime", "latency-cost", "observability", "planning"
            ]}
        return ModelReply(data=data, input_tokens=100, output_tokens=40, latency_ms=2.0, model=self.model)


def _answer() -> str:
    return (
        "推荐固定 Workflow，并由 Runtime 强制权限。检索失败应重试，工具部分失败要记录 Trace 并降级；"
        "依据证据控制延迟与恢复。计划失效时重规划，可采用 Rolling-Horizon。"
    )


def scenario(identifier: str):
    return next(item for item in load_scenarios() if item.id == identifier)


def test_search_returns_versioned_documents():
    runtime = ResearchToolRuntime()
    result = runtime.call("search_documents", {"query": "workflow runtime safety", "max_results": 4})
    assert result.status == "success"
    assert result.corpus_version == 1
    assert result.data["hits"]


def test_retrieval_failure_is_retryable_and_then_recovers():
    runtime = ResearchToolRuntime(fault="retrieval_failure")
    first = runtime.call("search_documents", {"query": "workflow", "max_results": 4})
    second = runtime.call("search_documents", {"query": "workflow", "max_results": 4})
    assert first.error_type == "retriever_unavailable" and first.retryable
    assert second.status == "success"


def test_tool_failure_is_structured_and_transient():
    runtime = ResearchToolRuntime(fault="tool_failure")
    first = runtime.call("read_document", {"doc_id": "arch-boundary", "expected_version": 1})
    second = runtime.call("read_document", {"doc_id": "arch-boundary", "expected_version": 1})
    assert first.error_type == "tool_unavailable" and first.retryable
    assert second.status == "success"


def test_plan_invalidation_exposes_new_version():
    runtime = ResearchToolRuntime(fault="plan_invalidated")
    stale = runtime.call("read_document", {"doc_id": "planning", "expected_version": 1})
    fresh = runtime.call("read_document", {"doc_id": "planning", "expected_version": stale.corpus_version})
    assert stale.error_type == "plan_invalidated"
    assert fresh.status == "success" and fresh.corpus_version == 2


@pytest.mark.parametrize("architecture", [FixedWorkflow, ReAct, PlanAndExecute])
def test_all_architectures_complete_normal_scenario(architecture):
    result = architecture(TestResearchModel()).run(scenario("normal"), 1)
    assert result.success
    assert result.tool_calls > 0
    assert result.input_tokens > 0
    assert result.trace


def test_react_recovers_from_retrieval_failure():
    result = ReAct(TestResearchModel()).run(scenario("retrieval_failure"), 1)
    assert result.success
    assert "retriever_unavailable" in result.failure_types


def test_plan_and_execute_replans_when_corpus_changes():
    result = PlanAndExecute(TestResearchModel()).run(scenario("plan_invalidated"), 1)
    assert result.success
    assert "plan_invalidated" in result.failure_types
    assert any(event.name == "replan" for event in result.trace)


def test_repeats_below_three_are_rejected():
    with pytest.raises(ValueError, match="至少运行 3 次"):
        run_benchmark(TestResearchModel(), repeats=2)


def test_full_benchmark_has_equal_runs_and_shared_model():
    report = run_benchmark(TestResearchModel(), repeats=3)
    assert report["comparability"]["same_model"]
    assert len(report["runs"]) == 3 * 4 * 3
    assert {metrics["runs"] for metrics in report["summary"].values()} == {12}


def test_report_contains_comparison_table_and_decision_tree(tmp_path):
    report = run_benchmark(TestResearchModel(), repeats=3)
    json_path, md_path = write_reports(report, tmp_path)
    markdown = md_path.read_text(encoding="utf-8")
    assert json_path.exists()
    assert "架构对比表" in markdown
    assert "推荐决策树" in markdown
    assert "Fixed Workflow" in markdown

