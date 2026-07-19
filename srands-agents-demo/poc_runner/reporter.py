"""POC 报告生成器 — 为每个测试用例输出结构化 JSON 证据"""

import json
import os
import time
from pathlib import Path
from typing import Any


class PocReporter:
    """单个测试用例的报告记录器

    用法：
        reporter = PocReporter("TC-SES-01", "对话历史隔离")
        reporter.add_event("sse", {"type": "agent.tool_use", ...})
        reporter.add_hook_event("BeforeToolCallEvent", {...})
        reporter.add_assertion("agent_a.messages 不含 AgentB 输入", True, True)
        reporter.finalize("PASS")  # → 写入 reports/TC-SES-01.json
    """

    def __init__(self, test_id: str, name: str, output_dir: str | None = None):
        self.test_id = test_id
        self.name = name
        self.output_dir = Path(output_dir or os.path.join(os.getcwd(), "reports"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.status: str = "PENDING"
        self.events: list[dict[str, Any]] = []
        self.hooks: list[dict[str, Any]] = []
        self.assertions: list[dict[str, Any]] = []
        self.blocker: str | None = None
        self.start_time = time.time()

    def add_event(self, event: dict[str, Any]) -> None:
        """记录 SSE 事件"""
        self.events.append(event)

    def add_events(self, events: list[dict[str, Any]]) -> None:
        """批量记录 SSE 事件"""
        self.events.extend(events)

    def add_hook_event(self, hook_type: str, payload: dict[str, Any]) -> None:
        """记录 Hook 事件"""
        self.hooks.append({
            "hook": hook_type,
            "timestamp": time.time(),
            "payload": payload,
        })

    def add_assertion(
        self, assertion: str, expected: Any, actual: Any, match: bool | None = None
    ) -> None:
        """记录断言

        Args:
            assertion: 断言描述
            expected: 预期值
            actual: 实际值
            match: 是否匹配（None 表示自动计算）
        """
        if match is None:
            match = (expected == actual)
        self.assertions.append({
            "assertion": assertion,
            "expected": str(expected)[:500],
            "actual": str(actual)[:500],
            "match": match,
        })

    def set_blocker(self, reason: str) -> None:
        """设置阻塞原因（仅 BLOCKED 状态时使用）"""
        self.blocker = reason

    def finalize(self, status: str = "PASS") -> str:
        """完成报告并写入 JSON 文件

        Args:
            status: "PASS" | "FAIL" | "BLOCKED"

        Returns:
            输出文件路径
        """
        self.status = status
        elapsed = time.time() - self.start_time

        report = {
            "id": self.test_id,
            "name": self.name,
            "status": self.status,
            "elapsed_seconds": round(elapsed, 3),
            "events_count": len(self.events),
            "events": self.events,
            "hooks_count": len(self.hooks),
            "hooks": self.hooks,
            "assertions_count": len(self.assertions),
            "assertions": self.assertions,
            "blocker": self.blocker,
        }

        filepath = self.output_dir / f"{self.test_id.lower()}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)

        return str(filepath)


class PocReportSuite:
    """测试套件级别的报告汇总"""

    def __init__(self, output_dir: str | None = None):
        self.output_dir = Path(output_dir or os.path.join(os.getcwd(), "reports"))
        self.reporters: dict[str, PocReporter] = {}

    def create(self, test_id: str, name: str) -> PocReporter:
        """创建或获取一个测试报告器"""
        if test_id not in self.reporters:
            self.reporters[test_id] = PocReporter(
                test_id, name, str(self.output_dir)
            )
        return self.reporters[test_id]

    def generate_summary(self) -> str:
        """生成汇总报告 — 扫描磁盘上所有 JSON 报告文件合并汇总"""
        results = {}

        # 扫描 reports/ 目录下所有 tc-*.json 文件
        if self.output_dir.exists():
            for f in sorted(self.output_dir.glob("tc-*.json")):
                try:
                    with open(f, "r") as fh:
                        data = json.load(fh)
                    tid = data.get("id", f.stem.upper())
                    results[tid] = {
                        "name": data.get("name", ""),
                        "status": data.get("status", "UNKNOWN"),
                        "assertions": data.get("assertions_count", 0),
                    }
                except (json.JSONDecodeError, KeyError):
                    continue

        # 合并内存中的 reporters（可能包含尚未写入磁盘的）
        for rid, r in self.reporters.items():
            if rid not in results:
                results[rid] = {"name": r.name, "status": r.status, "assertions": len(r.assertions)}

        total = len(results)
        passed = sum(1 for v in results.values() if v["status"] == "PASS")
        failed = sum(1 for v in results.values() if v["status"] == "FAIL")
        blocked = sum(1 for v in results.values() if v["status"] == "BLOCKED")

        summary = {
            "total": total,
            "passed": passed,
            "failed": failed,
            "blocked": blocked,
            "results": dict(sorted(results.items())),
        }

        filepath = self.output_dir / "_summary.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        return str(filepath)
