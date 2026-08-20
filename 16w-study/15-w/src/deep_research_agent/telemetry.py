from __future__ import annotations

import contextlib
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator


@dataclass
class Span:
    trace_id: str
    span_id: str
    name: str
    started_at: float
    attributes: dict[str, Any] = field(default_factory=dict)


class Tracer:
    """Small OTel-shaped adapter; records metadata, never prompts or source bodies."""

    def __init__(self, sink: Callable[[dict[str, Any]], None]) -> None:
        self._sink = sink

    @contextlib.contextmanager
    def span(self, trace_id: str, name: str, **attributes: Any) -> Iterator[Span]:
        span = Span(trace_id, uuid.uuid4().hex[:16], name, time.time(), attributes)
        status = "ok"
        error_type: str | None = None
        try:
            yield span
        except Exception as exc:
            status = "error"
            error_type = type(exc).__name__
            raise
        finally:
            self._sink(
                {
                    "trace_id": trace_id,
                    "span_id": span.span_id,
                    "name": name,
                    "started_at": span.started_at,
                    "duration_ms": round((time.time() - span.started_at) * 1000, 3),
                    "status": status,
                    "error_type": error_type,
                    "attributes": attributes,
                }
            )
