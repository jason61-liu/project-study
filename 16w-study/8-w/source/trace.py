"""轻量 Trace Recorder；每个 Trial 独占实例，因此并发运行时无需共享锁。"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime
import time
from typing import Any, Iterator
import uuid

from models import TraceEvent


class TraceRecorder:
    def __init__(self) -> None:
        self.trace_id = f"trace_{uuid.uuid4().hex}"
        self.events: list[TraceEvent] = []

    @contextmanager
    def span(
        self,
        kind: str,
        name: str,
        *,
        parent_span_id: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """记录开始、结束、异常和延迟；调用者可在 yield 后补充 detail。"""

        started_clock = time.perf_counter()
        started_at = datetime.now(UTC)
        span_id = f"span_{uuid.uuid4().hex[:16]}"
        mutable_detail = dict(detail or {})
        status = "success"
        try:
            yield mutable_detail
        except Exception as exc:
            status = "error"
            mutable_detail["error_type"] = type(exc).__name__
            raise
        finally:
            ended_at = datetime.now(UTC)
            self.events.append(TraceEvent(
                trace_id=self.trace_id,
                span_id=span_id,
                parent_span_id=parent_span_id,
                kind=kind,
                name=name,
                status=status,
                started_at=started_at,
                ended_at=ended_at,
                latency_ms=(time.perf_counter() - started_clock) * 1000,
                detail=mutable_detail,
            ))
