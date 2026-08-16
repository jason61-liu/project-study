from __future__ import annotations

import asyncio
import json

import pytest

from dashboard import build_dashboard
from release_drill import run_release_drill, verify_trace_version_combinations
from runtime import AgentRuntime
from state_store import StateStore
from telemetry import Telemetry, UnsafeTelemetry
from versioning import CANDIDATE_VERSIONS


def test_otel_exports_dashboard_and_complete_version_trace_without_content(tmp_path):
    async def scenario():
        output = tmp_path / "otel"
        telemetry = Telemetry(output)
        runtime = AgentRuntime(StateStore(tmp_path / "state.db"), telemetry)
        runtime.submit(
            message_id="safe-telemetry",
            tenant_id="tenant-a",
            input_data={"text": "alice@example.com sk_live_abcdefghijk"},
            versions=CANDIDATE_VERSIONS,
        )
        await runtime.run_until_idle()
        release = run_release_drill(telemetry, output / "release.json")
        telemetry.shutdown()

        trace_check = verify_trace_version_combinations(output / "traces.jsonl", release)
        dashboard = build_dashboard(output, output)
        assert trace_check["passed"] is True
        assert release["rollback_verified"] is True
        assert dashboard["export_counts"]["traces"] > 0
        assert dashboard["export_counts"]["metric_batches"] > 0
        assert dashboard["export_counts"]["logs"] > 0
        assert dashboard["groups"][0]["versions"] == CANDIDATE_VERSIONS.as_dict()

        exported = "\n".join(
            (output / name).read_text(encoding="utf-8")
            for name in ("traces.jsonl", "metrics.jsonl", "logs.jsonl")
        )
        assert "alice@example.com" not in exported
        assert "sk_live_abcdefghijk" not in exported
        assert "gen_ai.input.messages" not in exported
        assert "gen_ai.tool.call.result" not in exported

    asyncio.run(scenario())


def test_telemetry_rejects_content_bearing_attributes(tmp_path):
    telemetry = Telemetry(tmp_path)
    with pytest.raises(UnsafeTelemetry):
        with telemetry.span(
            "unsafe",
            attributes={"gen_ai.input.messages": "user private content"},
        ):
            pass
    telemetry.shutdown()


def test_release_drill_uses_week8_gate_and_rolls_back(tmp_path):
    telemetry = Telemetry(tmp_path / "otel")
    report = run_release_drill(telemetry, tmp_path / "release.json")
    telemetry.shutdown()
    assert report["actual_sequence"] == ["shadow", "canary", "rollback"]
    assert report["phases"][0]["gate"]["passed"] is True
    assert report["phases"][1]["gate"]["passed"] is False
    assert report["phases"][1]["action"] == "rollback"
    assert report["rollback_verified"] is True

