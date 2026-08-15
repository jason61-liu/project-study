"""可在 CI 中直接使用的回归门禁；阻断时以退出码 1 结束。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from models import AggregateReport, GateConfig, GateResult


def evaluate_gate(
    baseline: AggregateReport,
    candidate: AggregateReport,
    config: GateConfig,
    *,
    review_summary: dict[str, Any] | None = None,
) -> GateResult:
    violations: list[str] = []
    if candidate.task_count < config.min_task_count:
        violations.append(f"task_count {candidate.task_count} < {config.min_task_count}")
    if candidate.strict_success_rate < config.min_overall_success_rate:
        violations.append(
            f"strict_success_rate {candidate.strict_success_rate:.4f} < {config.min_overall_success_rate:.4f}"
        )
    drop = baseline.strict_success_rate - candidate.strict_success_rate
    if drop > config.max_success_drop:
        violations.append(f"success_drop {drop:.4f} > {config.max_success_drop:.4f}")
    for category, threshold in config.min_category_success_rate.items():
        actual = candidate.category_success_rate.get(category)
        if actual is None:
            violations.append(f"missing_category {category}")
        elif actual < threshold:
            violations.append(f"category[{category}] {actual:.4f} < {threshold:.4f}")

    if candidate.judge_requested:
        if candidate.judge_completed < candidate.judge_expected or candidate.judge_errors:
            violations.append(
                "llm_judge_incomplete "
                f"completed={candidate.judge_completed} expected={candidate.judge_expected} "
                f"errors={candidate.judge_errors}"
            )
        count = int((review_summary or {}).get("review_count", 0))
        rate = (review_summary or {}).get("misjudgment_rate")
        if count < 20:
            violations.append(f"human_review_count {count} < 20")
        elif rate is None or float(rate) > config.max_llm_judge_misjudgment_rate:
            violations.append(
                f"llm_judge_misjudgment_rate {rate} > {config.max_llm_judge_misjudgment_rate:.4f}"
            )
    return GateResult(
        passed=not violations,
        baseline_run_id=baseline.run_id,
        candidate_run_id=candidate.run_id,
        violations=violations,
    )


def load_report(path: Path) -> AggregateReport:
    return AggregateReport.model_validate_json(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--human-reviews", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = GateConfig.model_validate_json(args.config.read_text(encoding="utf-8"))
    review_summary = None
    if args.human_reviews:
        review_summary = json.loads(args.human_reviews.read_text(encoding="utf-8"))["summary"]
    result = evaluate_gate(
        load_report(args.baseline), load_report(args.candidate), config,
        review_summary=review_summary,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(result.model_dump_json(indent=2))
    raise SystemExit(0 if result.passed else 1)


if __name__ == "__main__":
    main()
