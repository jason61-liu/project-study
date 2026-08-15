import json
from pathlib import Path

from generate_tasks import generate
from models import EvalTask, GraderKind, TaskCategory


SOURCE = Path(__file__).parents[1]


def test_suite_contains_at_least_fifty_versioned_tasks():
    tasks = generate()
    assert len(tasks) >= 50
    assert len({task.id for task in tasks}) == len(tasks)
    assert all(task.version == "1.0.0" for task in tasks)


def test_suite_covers_four_required_categories():
    tasks = generate()
    counts = {category: sum(task.category == category for task in tasks) for category in TaskCategory}
    assert counts[TaskCategory.NORMAL] >= 10
    assert counts[TaskCategory.BOUNDARY] >= 10
    assert counts[TaskCategory.FAILURE] >= 10
    assert counts[TaskCategory.ADVERSARIAL] >= 10


def test_every_task_has_input_environment_condition_and_grader():
    for task in generate():
        assert task.input.instruction
        assert task.environment.fixture_id
        assert task.success_conditions
        assert task.graders
        condition_ids = {item.id for item in task.success_conditions}
        grader_ids = {item.id for item in task.graders}
        assert condition_ids <= grader_ids


def test_exactly_twenty_local_tasks_request_llm_judging():
    count = sum(
        grader.kind == GraderKind.LLM_RUBRIC
        for task in generate() for grader in task.graders
    )
    assert count == 20


def test_checked_in_manifest_round_trips_schema():
    payload = json.loads((SOURCE / "data" / "tasks.json").read_text(encoding="utf-8"))
    tasks = [EvalTask.model_validate(item) for item in payload]
    assert len(tasks) == 60
