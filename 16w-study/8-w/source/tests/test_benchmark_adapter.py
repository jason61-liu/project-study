from pathlib import Path

from agents import TauReferenceReplayAgent
from benchmark_adapter import load_tau_subset
from harness import EvaluationHarness


SOURCE = Path(__file__).parents[1]
RAW = SOURCE / "data" / "tau3-banking-v1.0.1" / "raw"


def test_tau_subset_preserves_source_version_and_original_ids():
    tasks, gold = load_tau_subset(RAW)
    assert len(tasks) == 5
    assert all(task.version == "1.0.1" for task in tasks)
    assert all(task.source.revision == "v1.0.1" for task in tasks)
    assert all(task.source.original_task_id for task in tasks)
    assert set(gold) == {task.id for task in tasks}


async def test_tau_adapter_contract_is_actually_executable(tmp_path):
    tasks, gold = load_tau_subset(RAW)
    trials, report = await EvaluationHarness(
        tmp_path / "tau-run", concurrency=5,
    ).run(tasks, TauReferenceReplayAgent(gold))
    assert len(trials) == 5
    assert report.strict_success_rate == 1.0
    assert all(Path(trial.trace_path).is_file() for trial in trials)
