"""Execute attack cases and persist JSON plus Markdown evidence reports."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

from attack_cases import live_e2b_status, run_suite


ROOT = Path(__file__).resolve().parent


def main() -> None:
    results = run_suite()
    live = live_e2b_status()
    summary = {
        "total": len(results),
        "passed": sum(item.status == "PASS" for item in results),
        "failed": sum(item.status == "FAIL" for item in results),
        "errors": sum(item.status == "ERROR" for item in results),
        "live_e2b": live,
        "results": [asdict(item) for item in results],
    }
    output = ROOT / "artifacts"
    output.mkdir(parents=True, exist_ok=True)
    (output / "attack-results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# 第 9 周攻击测试结果",
        "",
        f"> 自动攻击用例：{summary['total']}；PASS：{summary['passed']}；FAIL：{summary['failed']}；ERROR：{summary['errors']}。",
        "",
        "| ID | 类别 | 攻击 | 预期防护 | 实际结果 | 残余风险 |",
        "|---|---|---|---|---|---|",
    ]
    for item in results:
        evidence = json.dumps(item.evidence, ensure_ascii=False, separators=(",", ":"))
        actual = f"{item.status}：{item.actual_result}；证据 `{evidence}`"
        cells = [item.id, item.category, item.attack, item.expected_defense, actual, item.residual_risk]
        lines.append("| " + " | ".join(cell.replace("|", "\\|").replace("\n", " ") for cell in cells) + " |")
    lines.extend([
        "",
        "## 托管 E2B Live Smoke",
        "",
        f"- 状态：`{live['status']}`",
        f"- 实际结果：{live['actual_result']}",
        f"- 残余风险：{live['residual_risk']}",
        "",
        "说明：自动用例验证本项目的确定性边界；托管 E2B 项只在 Host 已配置 `E2B_API_KEY` 时执行，未配置时明确记录为 SKIPPED。",
    ])
    (output / "attack-results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("total", "passed", "failed", "errors")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
