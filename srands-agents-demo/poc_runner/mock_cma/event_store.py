"""Mock CMA EventStore — 内存 append-only 事件存储"""

import threading
from collections import defaultdict
from typing import Any


class CmaEventStore:
    """内存 append-only EventStore，模拟 CMA EventStore 行为"""

    def __init__(self):
        self._streams: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._lock = threading.Lock()

    def create_stream(self, session_id: str) -> None:
        with self._lock:
            if session_id not in self._streams:
                self._streams[session_id] = []

    def append(self, session_id: str, event: dict[str, Any]) -> int:
        """追加事件，返回 seq 编号（从 1 开始）"""
        with self._lock:
            if session_id not in self._streams:
                self._streams[session_id] = []
            seq = len(self._streams[session_id]) + 1
            event["seq"] = seq
            self._streams[session_id].append(event)
            return seq

    def get_events(self, session_id: str) -> list[dict[str, Any]]:
        """获取 session 的所有事件（按 seq 排序）"""
        with self._lock:
            return list(self._streams.get(session_id, []))

    def get_event_count(self, session_id: str) -> int:
        with self._lock:
            return len(self._streams.get(session_id, []))

    def clear(self) -> None:
        with self._lock:
            self._streams.clear()
