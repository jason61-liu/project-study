"""不调用网络也能验证 SDK/Harness 采用了真实框架入口。"""

from pathlib import Path

import pytest

from run_comparison import run_all


SOURCE = Path(__file__).parents[1]


def test_agents_sdk_uses_native_primitives():
    text = (SOURCE / "agents_sdk_workflow.py").read_text(encoding="utf-8")
    for symbol in ["Runner.run", "@function_tool", "handoff(", "@input_guardrail", "@output_guardrail", "needs_approval=True"]:
        assert symbol in text
    assert "RunState.from_json" in text
    assert 'metadata.get("trace", [])' in text


def test_deep_agents_uses_one_bounded_subagent():
    text = (SOURCE / "deep_agents_workflow.py").read_text(encoding="utf-8")
    assert "create_deep_agent(" in text
    assert '"name": "evidence-researcher"' in text
    assert "PARENT_ONLY_7W" in text


@pytest.mark.asyncio
async def test_experiment_refuses_reused_run_directory(tmp_path):
    (tmp_path / "runs").mkdir()
    with pytest.raises(FileExistsError, match="避免 Checkpoint/幂等账本污染"):
        await run_all(tmp_path)
