from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


CATEGORIES = ("correct", "citation_missing", "evidence_conflict", "irrelevant", "unsafe", "too_slow")
ESCALATION_REASONS = ("high_risk", "conflicting_sources", "insufficient_evidence", "policy_review")


def simulate_feedback() -> dict[str, object]:
    before = []
    after = []
    for index in range(100):
        category = "correct"
        if index % 17 == 0:
            category = "citation_missing"
        elif index % 23 == 0:
            category = "evidence_conflict"
        elif index % 29 == 0:
            category = "irrelevant"
        before.append({"task_id": f"shadow-{index:03d}", "category": category, "success": category == "correct"})
        fixed = category in {"citation_missing", "evidence_conflict"}
        after.append({"task_id": f"shadow-{index:03d}", "category": "correct" if fixed else category, "success": fixed or category == "correct"})

    before_rate = sum(row["success"] for row in before) / len(before)
    after_rate = sum(row["success"] for row in after) / len(after)
    offline_before, offline_after = 0.90, 0.96
    failed_samples = [
        {
            "task_id": row["task_id"],
            "failure_type": row["category"],
            "review_status": "queued",
            "ingest_rule": "two-reviewer approval; keep evaluation holdout separate",
        }
        for row in after
        if not row["success"]
    ]
    return {
        "taxonomy": list(CATEGORIES),
        "escalation_reasons": list(ESCALATION_REASONS),
        "sampling": {"routine_percent": 5, "failed_percent": 100, "high_risk_percent": 100},
        "before": {
            "online_success_rate": before_rate,
            "feedback_counts": Counter(row["category"] for row in before),
            "offline_eval_success_rate": offline_before,
        },
        "after": {
            "online_success_rate": after_rate,
            "feedback_counts": Counter(row["category"] for row in after),
            "offline_eval_success_rate": offline_after,
        },
        "direction_check": {
            "offline_delta": offline_after - offline_before,
            "online_delta": after_rate - before_rate,
            "same_direction": (offline_after - offline_before) * (after_rate - before_rate) > 0,
        },
        "failed_samples_for_curation": failed_samples,
        "pollution_controls": [
            "deduplicate by normalized input hash",
            "remove secrets and personal data before review",
            "require two reviewers for label changes",
            "never train on the fixed regression holdout",
            "version provenance and support deletion",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/feedback-loop.json"))
    args = parser.parse_args()
    payload = simulate_feedback()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=dict) + "\n")
    print(json.dumps(payload["direction_check"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

