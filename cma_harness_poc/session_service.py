# cma_harness_poc/session_service.py — Session state machine + event orchestration
from __future__ import annotations
import time
import uuid
from typing import Dict, Optional

from cma_harness_poc.models import SessionState, SessionRecord, CmaEvent, CompactContext
from cma_harness_poc.event_store import CmaEventStore


class CmaSessionService:
    """
    Manages session lifecycle and state transitions.
    Event storage delegated to CmaEventStore.
    """

    def __init__(self, event_store: CmaEventStore):
        self._event_store = event_store
        self._sessions: Dict[str, SessionRecord] = {}

    def create_session(self, agent_id: str, agent_version: int,
                       environment_id: str = "env_default") -> SessionRecord:
        session_id = f"sesn_{uuid.uuid4().hex[:12]}"
        now = time.time()
        record = SessionRecord(
            id=session_id,
            status=SessionState.IDLE,
            agent_id=agent_id,
            agent_version=agent_version,
            environment_id=environment_id,
            created_at=now,
            updated_at=now,
        )
        self._sessions[session_id] = record
        return record

    def get_session(self, session_id: str) -> Optional[SessionRecord]:
        return self._sessions.get(session_id)

    def _set_state(self, session_id: str, state: SessionState) -> None:
        record = self._sessions.get(session_id)
        if record:
            record.status = state
            record.updated_at = time.time()

    def on_event_appended(self, session_id: str, event: CmaEvent) -> None:
        """React to an event being appended — manage state transitions."""
        event_type = event.type
        if event_type == "user.message":
            self._set_state(session_id, SessionState.RUNNING)
        elif event_type == "session.status_idle":
            self._set_state(session_id, SessionState.IDLE)
        elif event_type == "session.status_terminated":
            self._set_state(session_id, SessionState.TERMINATED)
        elif event_type == "session.error":
            is_fatal = (event.error or {}).get("fatal", False)
            if is_fatal:
                self._set_state(session_id, SessionState.TERMINATED)
            else:
                self._set_state(session_id, SessionState.RESCHEDULING)

    def append_event_and_update_state(self, session_id: str,
                                       event: CmaEvent) -> CmaEvent:
        stored = self._event_store.append(session_id, event)
        self.on_event_appended(session_id, stored)
        return stored

    def update_compact_context(
        self, session_id: str, compact_ctx: CompactContext,
    ) -> None:
        record = self._sessions.get(session_id)
        if record:
            record.compact_context = compact_ctx

    def get_compact_context(self, session_id: str) -> Optional[CompactContext]:
        record = self._sessions.get(session_id)
        return record.compact_context if record else None

    def update_session_state(
        self, session_id: str, state: SessionState,
    ) -> None:
        self._set_state(session_id, state)
