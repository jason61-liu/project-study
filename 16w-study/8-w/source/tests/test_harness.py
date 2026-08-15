import asyncio
import json
from pathlib import Path

from agents import DegradedAgent, ReferenceAgent
from generate_tasks import generate
from harness import EvaluationHarness
from models import AgentOutput, AgentTaskView, TrialStatus
from trace import TraceRecorder


async def test_harness_persists_one_trace_per_trial(tmp_path):
    tasks = generate()[:8]
    trials, report = await EvaluationHarness(
        tmp_path / "run", concurrency=4,
    ).run(tasks, ReferenceAgent())
    assert report.trial_count == 8
    assert len(list((tmp_path / "run" / "traces").glob("*.json"))) == 8
    for trial in trials:
        events = json.loads(Path(trial.trace_path).read_text(encoding="utf-8"))
        assert any(event["kind"] == "agent" for event in events)
        assert any(event["kind"] == "grader" for event in events)


async def test_baseline_passes_all_required_slices(tmp_path):
    _, report = await EvaluationHarness(
        tmp_path / "baseline", concurrency=8,
    ).run(generate(), ReferenceAgent())
    assert report.task_count == 60
    assert report.strict_success_rate == 1.0
    assert set(report.category_success_rate.values()) == {1.0}


async def test_degraded_prompt_fails_security_and_failure_slices(tmp_path):
    _, report = await EvaluationHarness(
        tmp_path / "degraded", concurrency=8,
    ).run(generate(), DegradedAgent())
    assert report.strict_success_rate < 0.5
    assert report.category_success_rate["adversarial"] == 0
    assert report.category_success_rate["failure"] == 0


class ConcurrencyProbeAgent:
    version = "concurrency-probe@1"

    def __init__(self) -> None:
        self.active = 0
        self.maximum = 0

    async def run(self, task: AgentTaskView, trace: TraceRecorder) -> AgentOutput:
        self.active += 1
        self.maximum = max(self.maximum, self.active)
        await asyncio.sleep(0.02)
        self.active -= 1
        return AgentOutput(
            status=TrialStatus.COMPLETED,
            answer=f"{task.environment.visible_context['records'][0]['value']}",
            citations=[task.environment.visible_context["records"][0]["source_id"]],
        )


async def test_harness_runs_trials_concurrently(tmp_path):
    agent = ConcurrencyProbeAgent()
    await EvaluationHarness(tmp_path / "parallel", concurrency=4).run(generate()[:8], agent)
    assert agent.maximum == 4


async def test_harness_rejects_nonempty_output_directory(tmp_path):
    output = tmp_path / "dirty"
    output.mkdir()
    (output / "old.json").write_text("{}", encoding="utf-8")
    try:
        await EvaluationHarness(output).run(generate()[:1], ReferenceAgent())
    except FileExistsError as exc:
        assert "拒绝污染" in str(exc)
    else:
        raise AssertionError("应拒绝复用非空输出目录")


class FailingAgent:
    version = "failing-agent@1"

    async def run(self, task: AgentTaskView, trace: TraceRecorder) -> AgentOutput:
        raise RuntimeError("boom")


async def test_one_agent_exception_becomes_a_failed_trial(tmp_path):
    trials, report = await EvaluationHarness(tmp_path / "failure").run(
        generate()[:2], FailingAgent(),
    )
    assert len(trials) == 2
    assert report.strict_success_rate == 0
    assert all(trial.output.status == TrialStatus.FAILED for trial in trials)
    assert all("agent_exception" in trial.output.safety_flags for trial in trials)
