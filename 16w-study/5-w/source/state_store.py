"""带乐观并发控制的 JSON Checkpoint Store。"""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

from recovery_models import StructuredPlan


class StateConflict(RuntimeError):
    """调用方基于旧版本写入，必须重新加载而不能覆盖新状态。"""


class JsonPlanStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, plan_id: str) -> Path:
        return self.root / f"{plan_id}.json"

    def create(self, plan: StructuredPlan) -> None:
        path = self.path_for(plan.plan_id)
        if path.exists():
            raise StateConflict(f"plan already exists: {plan.plan_id}")
        self._atomic_write(path, plan.to_dict())

    def load(self, plan_id: str) -> StructuredPlan:
        return StructuredPlan.from_dict(
            json.loads(self.path_for(plan_id).read_text(encoding="utf-8"))
        )

    def save(self, plan: StructuredPlan, *, expected_version: int) -> StructuredPlan:
        """Compare-And-Swap：只有磁盘版本等于 expected_version 才提交。"""

        current = self.load(plan.plan_id)
        if current.version != expected_version:
            raise StateConflict(
                f"expected version {expected_version}, actual {current.version}"
            )
        plan.version = expected_version + 1
        plan.checkpoints += 1
        self._atomic_write(self.path_for(plan.plan_id), plan.to_dict())
        return plan

    @staticmethod
    def _atomic_write(path: Path, value: dict) -> None:
        """同目录临时文件 + replace，避免进程中断留下半个 JSON。"""

        descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(value, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
