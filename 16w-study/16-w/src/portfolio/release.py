from __future__ import annotations

import argparse
import json
from pathlib import Path


def simulate_release(candidate_success_rate: float = 0.94) -> dict[str, object]:
    baseline = {"version": "research-agent-v1", "success_rate": 1.0, "security_failures": 0}
    blocked_candidate = {"version": "research-agent-v2-regression", "success_rate": 0.94, "security_failures": 0}
    canary_candidate = {"version": "research-agent-v2-canary", "offline_success_rate": 0.99, "security_failures": 0}
    events: list[dict[str, object]] = [
        {"phase": "ci", "version": blocked_candidate["version"], "action": "eval_gate", "result": "blocked"},
        {"phase": "ci", "version": canary_candidate["version"], "action": "eval_gate", "result": "passed"},
        {"phase": "shadow", "version": canary_candidate["version"], "traffic_percent": 100, "user_visible": False, "result": "passed"},
        {"phase": "canary", "version": canary_candidate["version"], "traffic_percent": 5, "observed_success_rate": candidate_success_rate},
    ]
    canary_healthy = candidate_success_rate >= 0.98 and canary_candidate["security_failures"] == 0
    if canary_healthy:
        events.append({"phase": "canary", "action": "promote"})
        active = canary_candidate["version"]
    else:
        events.append({"phase": "rollback", "trigger": "success_rate_regression", "action": "route_to_previous"})
        active = baseline["version"]
    # Manual rollback remains independently executable even when auto rollback already fired.
    events.append({"phase": "manual_rollback_drill", "action": "route_to_previous", "result": "verified"})
    return {
        "baseline": baseline,
        "blocked_candidate": blocked_candidate,
        "canary_candidate": canary_candidate,
        "events": events,
        "candidate_blocked": True,
        "canary_rolled_back": not canary_healthy,
        "active_version": active,
        "rollback_verified": active == baseline["version"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-success-rate", type=float, default=0.94)
    parser.add_argument("--output", type=Path, default=Path("results/release-drill.json"))
    args = parser.parse_args()
    payload = simulate_release(args.candidate_success_rate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
