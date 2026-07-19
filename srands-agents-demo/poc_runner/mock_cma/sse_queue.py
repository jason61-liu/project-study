"""Mock CMA SSE Queue — 模拟 CMA 的 SSE 事件推送队列"""

import asyncio
from typing import Any


class SSEClient:
    """内存 SSE 事件队列，收集所有推送的 CMA 格式事件"""

    def __init__(self):
        self._queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._events: list[dict[str, Any]] = []  # 持久记录所有已推送事件

    async def push(self, event: dict[str, Any]) -> None:
        """推送一个 CMA 事件到 SSE 队列"""
        self._events.append(event)
        await self._queue.put(event)

    def push_sync(self, event: dict[str, Any]) -> None:
        """同步推送事件（用于 hook 回调中）"""
        self._events.append(event)
        # 如果事件循环在运行，用 call_soon_threadsafe
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                loop.call_soon_threadsafe(self._queue.put_nowait, event)
        except RuntimeError:
            # 没有运行中的事件循环，直接加入
            pass

    def get_events(self) -> list[dict[str, Any]]:
        """获取所有已推送的事件"""
        return list(self._events)

    def get_events_by_type(self, event_type: str) -> list[dict[str, Any]]:
        """按类型过滤事件"""
        return [e for e in self._events if e.get("type") == event_type]

    def clear(self) -> None:
        """清空事件历史"""
        self._events.clear()
        # 清空队列
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
