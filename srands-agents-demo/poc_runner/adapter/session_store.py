"""CMA EventStore Session Repository — 用 CMA EventStore 替换 Strands Session 存储后端"""

from typing import Any

from strands.session.session_repository import SessionRepository
from strands.types.session import Session, SessionAgent, SessionMessage

from poc_runner.mock_cma.event_store import CmaEventStore


class CmaEventStoreSessionRepository(SessionRepository):
    """用 CMA EventStore 实现 SessionRepository 接口

    核心方法：create_message / list_messages。
    其他方法用最小实现支持 SessionManager 的生命周期调用。
    """

    def __init__(self, event_store: CmaEventStore):
        self.event_store = event_store
        self._sessions: dict[str, Session] = {}
        self._agents: dict[str, dict[str, SessionAgent]] = {}

    # ---- Session CRUD ----

    async def create_session(self, session: Session) -> None:
        self._sessions[session.session_id] = session
        self.event_store.create_stream(session.session_id)
        self._agents[session.session_id] = {}

    async def read_session(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    # ---- Agent CRUD ----

    async def create_agent(self, session_id: str, session_agent: SessionAgent) -> None:
        if session_id not in self._agents:
            self._agents[session_id] = {}
        self._agents[session_id][session_agent.agent_id] = session_agent

    async def read_agent(self, session_id: str, agent_id: str) -> SessionAgent | None:
        return self._agents.get(session_id, {}).get(agent_id)

    async def update_agent(self, session_id: str, session_agent: SessionAgent) -> None:
        if session_id in self._agents:
            self._agents[session_id][session_agent.agent_id] = session_agent

    # ---- Message CRUD（核心） ----

    async def create_message(
        self, session_id: str, agent_id: str, message: Any
    ) -> None:
        """将消息追加到 EventStore"""
        event = _message_to_cma_event(message)
        self.event_store.append(session_id, event)

    async def read_message(
        self, session_id: str, agent_id: str, message_id: str
    ) -> SessionMessage | None:
        events = self.event_store.get_events(session_id)
        for e in events:
            if e.get("message_id") == message_id:
                return _cma_event_to_session_message(e)
        return None

    async def update_message(
        self, session_id: str, agent_id: str, message: Any
    ) -> None:
        # 更新语义：追加一个更新事件
        await self.create_message(session_id, agent_id, message)

    async def list_messages(
        self,
        session_id: str,
        agent_id: str,
        limit: int | None = None,
        offset: int | None = None,
    ) -> list[SessionMessage]:
        """从 EventStore 列出消息"""
        events = self.event_store.get_events(session_id)
        messages = [_cma_event_to_session_message(e) for e in events]

        if offset:
            messages = messages[offset:]
        if limit:
            messages = messages[:limit]

        return messages

    # ---- MultiAgent CRUD (stub) ----

    async def create_multi_agent(self, session_id: str, multi_agent: Any) -> None:
        pass

    async def read_multi_agent(self, session_id: str, multi_agent_id: str) -> Any:
        return None

    async def update_multi_agent(self, session_id: str, multi_agent: Any) -> None:
        pass


def _message_to_cma_event(message: Any) -> dict[str, Any]:
    """将 SessionMessage 或 Message 转换为 CMA 事件格式"""
    # SessionMessage 包装
    if hasattr(message, 'message'):
        msg = message.message
        msg_id = message.message_id
    elif isinstance(message, dict):
        msg = message.get("message", message)
        msg_id = message.get("message_id", "")
    else:
        msg = message
        msg_id = ""

    role = msg.get("role", "unknown") if isinstance(msg, dict) else getattr(msg, "role", "unknown")
    content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")

    return {
        "type": f"{role}.message",
        "role": role,
        "content": content,
        "message_id": msg_id,
    }


def _cma_event_to_session_message(event: dict[str, Any]) -> SessionMessage:
    """将 CMA 事件转换回 SessionMessage"""
    from strands.types.session import SessionMessage

    msg = {
        "role": event.get("role", ""),
        "content": event.get("content", ""),
    }
    return SessionMessage(
        message=msg,
        message_id=event.get("message_id", str(event.get("seq", ""))),
    )
