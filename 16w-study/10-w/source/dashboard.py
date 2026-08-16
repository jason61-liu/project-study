"""Build a static dashboard from exported OpenTelemetry spans."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
import statistics
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _duration_ms(span: dict[str, Any]) -> float:
    start = datetime.fromisoformat(span["start_time"].replace("Z", "+00:00"))
    end = datetime.fromisoformat(span["end_time"].replace("Z", "+00:00"))
    return (end - start).total_seconds() * 1000


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, int((len(ordered) - 1) * q))
    return ordered[index]


def build_dashboard(telemetry_dir: Path, output_dir: Path) -> dict[str, Any]:
    spans = _read_jsonl(telemetry_dir / "traces.jsonl")
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for span in spans:
        attrs = span.get("attributes") or {}
        if attrs.get("app.operation.summary") is True:
            groups[(str(attrs["tenant.id"]), str(attrs["app.version.fingerprint"]))].append(span)

    rows: list[dict[str, Any]] = []
    for (tenant, fingerprint), items in sorted(groups.items()):
        attrs = items[0]["attributes"]
        durations = [_duration_ms(item) for item in items]
        successes = sum((item.get("attributes") or {}).get("app.outcome") == "success" for item in items)
        logical_tasks: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in items:
            logical_tasks[str(item["attributes"]["task.id"])].append(item)
        successful_tasks = sum(
            any(attempt["attributes"].get("app.outcome") == "success" for attempt in attempts)
            for attempts in logical_tasks.values()
        )
        versions = {
            key.removeprefix("app.version."): value
            for key, value in attrs.items()
            if key.startswith("app.version.") and key != "app.version.fingerprint"
        }
        rows.append(
            {
                "tenant_id": tenant,
                "version_fingerprint": fingerprint,
                "versions": versions,
                "attempts": len(items),
                "logical_tasks": len(logical_tasks),
                "attempt_success_rate": successes / len(items),
                "attempt_error_rate": 1 - successes / len(items),
                "task_success_rate": successful_tasks / len(logical_tasks),
                "latency_ms": {
                    "avg": round(statistics.fmean(durations), 3),
                    "p95": round(_percentile(durations, 0.95), 3),
                },
                "tokens": sum(int((item["attributes"]).get("gen_ai.usage.input_tokens", 0)) + int((item["attributes"]).get("gen_ai.usage.output_tokens", 0)) for item in items),
                "cost_microusd": sum(int((item["attributes"]).get("app.cost.microusd", 0)) for item in items),
            }
        )

    dashboard = {
        "schema_version": "1.0",
        "source": "OpenTelemetry JSONL exports",
        "export_counts": {
            "traces": len(spans),
            "metric_batches": len(_read_jsonl(telemetry_dir / "metrics.jsonl")),
            "logs": len(_read_jsonl(telemetry_dir / "logs.jsonl")),
        },
        "groups": rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dashboard.json").write_text(json.dumps(dashboard, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# 第 10 周 Agent Runtime Dashboard",
        "",
        "> 数据源：OpenTelemetry JSONL Trace/Metric/Log 导出。",
        "",
        "| Tenant | Version Fingerprint | Tasks | Attempts | Task Success | Attempt Error | Avg ms | p95 ms | Tokens | Cost microUSD |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['tenant_id']} | `{row['version_fingerprint']}` | {row['logical_tasks']} | "
            f"{row['attempts']} | {row['task_success_rate']:.1%} | {row['attempt_error_rate']:.1%} | "
            f"{row['latency_ms']['avg']:.3f} | {row['latency_ms']['p95']:.3f} | "
            f"{row['tokens']} | {row['cost_microusd']} |"
        )
    lines.extend(["", "完整版本组合见 `dashboard.json` 的 `groups[].versions`。", ""])
    (output_dir / "dashboard.md").write_text("\n".join(lines), encoding="utf-8")
    return dashboard
