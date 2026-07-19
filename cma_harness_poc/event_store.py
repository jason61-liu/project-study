# cma_harness_poc/event_store.py — Append-only CMA event store
from __future__ import annotations
import time
from typing import Dict, List, Optional

from cma_harness_poc.models import CmaEvent


class CmaEventStore:
    """In-memory append-only event store per session."""

    def __init__(self):
        self._events: Dict[str, List[CmaEvent]] = {}
        self._seq: Dict[str, int] = {}

    def append(self, session_id: str, event: CmaEvent) -> CmaEvent:
        seq = self._seq.get(session_id, 0) + 1
        self._seq[session_id] = seq
        # 如果 event.id 已经设了（如 tool_call_id），就保留；否则自动生成
        if not event.id:
            event.id = f"evt_{session_id}_{seq}"
        event.timestamp = time.time()
        self._events.setdefault(session_id, []).append(event)
        return event

    def get_events(self, session_id: str,
                   since_id: Optional[str] = None,
                   limit: Optional[int] = None) -> List[CmaEvent]:
        events = self._events.get(session_id, [])
        if since_id:
            start = next(
                (i for i, e in enumerate(events) if e.id == since_id),
                len(events),
            )
            events = events[start + 1:]
        if limit is not None:
            events = events[:limit]
        return events

    def delete_session(self, session_id: str) -> None:
        self._events.pop(session_id, None)
        self._seq.pop(session_id, None)
