"""Week 8 Eval Gate driven Shadow -> Canary -> Rollback release drill."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

from telemetry import Telemetry
from versioning import BASELINE_VERSIONS, CANDIDATE_VERSIONS, VersionManifest


WEEK8_SOURCE = Path(__file__).resolve().parents[2] / "8-w" / "source"
if str(WEEK8_SOURCE) not in sys.path:
    sys.path.insert(0, str(WEEK8_SOURCE))

from ci_gate import evaluate_gate  # noqa: E402
from models import AggregateReport, GateConfig  # noqa: E402


def _report(run_id: str, versions: VersionManifest, rate: float) -> AggregateReport:
    return AggregateReport(
        run_id=run_id,
        agent_version=versions.fingerprint,
        task_count=60,
        trial_count=60,
        strict_success_rate=rate,
        category_success_rate={
            "normal": rate,
            "boundary": rate,
            "failure": rate,
            "adversarial": rate,
        },
        grader_pass_rate={"deterministic": rate},
        status_counts={"completed": round(rate * 60), "failed": 60 - round(rate * 60)},
        average_latency_ms=25.0,
        total_input_tokens=9_000,
        total_output_tokens=2_400,
        judge_requested=False,
        judge_expected=0,
        judge_completed=0,
        judge_errors=0,
    )


def _gate_config() -> GateConfig:
    return GateConfig(
        min_task_count=50,
        min_overall_success_rate=0.95,
        min_category_success_rate={
            "normal": 0.95,
            "boundary": 0.90,
            "failure": 0.95,
            "adversarial": 0.95,
        },
        max_success_drop=0.02,
    )


def run_release_drill(telemetry: Telemetry, output_path: Path) -> dict[str, Any]:
    baseline = _report("baseline", BASELINE_VERSIONS, 1.0)
    phases = [
        ("shadow", CANDIDATE_VERSIONS, _report("shadow", CANDIDATE_VERSIONS, 0.985)),
        # Canary injects a regression; the Week 8 gate must block promotion.
        ("canary", CANDIDATE_VERSIONS, _report("canary-faulted", CANDIDATE_VERSIONS, 0.90)),
        ("rollback", BASELINE_VERSIONS, _report("rollback", BASELINE_VERSIONS, 1.0)),
    ]
    results: list[dict[str, Any]] = []
    for phase, versions, candidate in phases:
        gate = evaluate_gate(baseline, candidate, _gate_config())
        action = {
            "shadow": "advance_to_canary" if gate.passed else "stop",
            "canary": "promote" if gate.passed else "rollback",
            "rollback": "restore_baseline" if gate.passed else "manual_intervention",
        }[phase]
        attributes = {
            "app.release.phase": phase,
            "app.release.action": action,
            "app.eval.gate.passed": gate.passed,
            "app.eval.success_rate": candidate.strict_success_rate,
            **versions.trace_attributes(),
        }
        with telemetry.span("release.evaluate", attributes=attributes) as span:
            if not gate.passed:
                span.set_attribute("error.type", "eval_gate_blocked")
        telemetry.safe_log(
            "release.gate.evaluated",
            {
                "app.release.phase": phase,
                "app.release.action": action,
                "app.eval.gate.passed": gate.passed,
                "app.version.fingerprint": versions.fingerprint,
            },
        )
        results.append(
            {
                "phase": phase,
                "versions": versions.as_dict(),
                "version_fingerprint": versions.fingerprint,
                "eval": candidate.model_dump(mode="json"),
                "gate": gate.model_dump(mode="json"),
                "action": action,
            }
        )

    report = {
        "schema_version": "1.0",
        "week8_gate_source": "8-w/source/ci_gate.py",
        "expected_sequence": ["shadow", "canary", "rollback"],
        "actual_sequence": [item["phase"] for item in results],
        "phases": results,
        "rollback_verified": (
            results[0]["gate"]["passed"] is True
            and results[1]["gate"]["passed"] is False
            and results[1]["action"] == "rollback"
            and results[2]["gate"]["passed"] is True
            and results[2]["version_fingerprint"] == BASELINE_VERSIONS.fingerprint
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def verify_trace_version_combinations(trace_path: Path, report: dict[str, Any]) -> dict[str, Any]:
    spans = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line]
    release_spans = {
        span["attributes"]["app.release.phase"]: span
        for span in spans
        if (span.get("attributes") or {}).get("app.release.phase")
    }
    missing: list[str] = []
    for phase in report["phases"]:
        span = release_spans.get(phase["phase"])
        if span is None:
            missing.append(f"{phase['phase']}:span")
            continue
        attributes = span["attributes"]
        for key, value in phase["versions"].items():
            if attributes.get(f"app.version.{key}") != value:
                missing.append(f"{phase['phase']}:{key}")
        if attributes.get("app.version.fingerprint") != phase["version_fingerprint"]:
            missing.append(f"{phase['phase']}:fingerprint")
    return {
        "passed": not missing,
        "checked_phases": [phase["phase"] for phase in report["phases"]],
        "missing": missing,
    }
