from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from .eval_harness import run_suite


def main() -> None:
    parser = argparse.ArgumentParser(description="Block a release when the deterministic eval regresses")
    parser.add_argument("--baseline", type=Path, default=Path("evals/baseline.json"))
    parser.add_argument("--max-success-rate-drop", type=float, default=0.02)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text())
    with tempfile.TemporaryDirectory() as directory:
        candidate = run_suite(Path(directory) / "candidate.json")
    floor = max(0.98, float(baseline["success_rate"]) - args.max_success_rate_drop)
    failures = []
    if float(candidate["success_rate"]) < floor:
        failures.append(f"success_rate {candidate['success_rate']} < {floor}")
    if int(candidate["security_scenarios_defined"]) < 15:
        failures.append("security suite contains fewer than 15 scenarios")
    if int(candidate["fault_scenarios_defined"]) < 10:
        failures.append("fault suite contains fewer than 10 scenarios")
    if failures:
        raise SystemExit("EVAL GATE FAILED: " + "; ".join(failures))
    print(f"EVAL GATE PASSED: {candidate['passed']}/{candidate['tasks']} tasks")


if __name__ == "__main__":
    main()

