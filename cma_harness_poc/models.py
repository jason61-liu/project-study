# cma_harness_poc/models.py — Core data models
from __future__ import annotations
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


@dataclass
class CompactContext:
    """压缩状态，跨轮持久化。Hermes 压缩后提取 summary 到此。"""
    compacted_up_to: int = 0
    summary: str = ""


class SessionState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    RESCHEDULING = "rescheduling"
    TERMINATED = "terminated"


@dataclass
class CmaEvent:
    """A single CMA event — append-only, immutable after storage."""
    session_id: str
    type: str
    id: str = ""
    timestamp: float = 0.0
    content: Optional[List[Dict[str, Any]]] = None
    name: Optional[str] = None          # agent.tool_use
    input: Optional[Dict[str, Any]] = None  # agent.tool_use
    tool_use_id: Optional[str] = None   # agent.tool_result
    is_error: Optional[bool] = None     # agent.tool_result
    stop_reason: Optional[Dict[str, Any]] = None  # session.status_idle
    error: Optional[Dict[str, Any]] = None  # session.error

    def to_sse_dict(self) -> Dict[str, Any]:
        """Serialize to CMA SSE JSON format."""
        d = {
            "type": self.type,
            "id": self.id,
            "processed_at": self.timestamp,
        }
        if self.content is not None:
            d["content"] = self.content
        if self.name is not None:
            d["name"] = self.name
        if self.input is not None:
            d["input"] = self.input
        if self.tool_use_id is not None:
            d["tool_use_id"] = self.tool_use_id
        if self.is_error is not None:
            d["is_error"] = self.is_error
        if self.stop_reason is not None:
            d["stop_reason"] = self.stop_reason
        if self.error is not None:
            d["error"] = self.error
        return d


@dataclass
class AgentConfig:
    """Agent model stored in AgentStore."""
    id: str
    version: int = 1
    name: str = ""
    model: str = ""
    system: str = ""
    tools: List[Dict[str, Any]] = field(default_factory=list)
    skills: List[str] = field(default_factory=list)
    mcp_servers: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)
    created_at: float = 0.0


@dataclass
class SessionRecord:
    """Session record returned by REST API."""
    id: str
    status: SessionState
    agent_id: str
    agent_version: int
    environment_id: str
    created_at: float
    updated_at: float
    usage: Dict[str, int] = field(default_factory=lambda: {
        "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
    })
    compact_context: Optional[CompactContext] = None
