# cma_harness_poc/agent_store.py — In-memory Agent CRUD
from __future__ import annotations
import time
import uuid
from typing import Dict, List, Optional, Any

from cma_harness_poc.models import AgentConfig


class AgentStore:
    """In-memory agent storage. No versioning in POC."""

    def __init__(self):
        self._agents: Dict[str, AgentConfig] = {}

    def create(self, name: str, model: str, system: str = "",
               tools: Optional[List[Dict[str, Any]]] = None,
               skills: Optional[List[str]] = None,
               mcp_servers: Optional[List[Dict[str, Any]]] = None,
               metadata: Optional[Dict[str, str]] = None) -> AgentConfig:
        agent_id = f"agent_{uuid.uuid4().hex[:12]}"
        now = time.time()
        config = AgentConfig(
            id=agent_id,
            version=1,
            name=name,
            model=model,
            system=system,
            tools=tools or [],
            skills=skills or [],
            mcp_servers=mcp_servers or [],
            metadata=metadata or {},
            created_at=now,
        )
        self._agents[agent_id] = config
        return config

    def get(self, agent_id: str) -> Optional[AgentConfig]:
        return self._agents.get(agent_id)

    def list(self) -> List[AgentConfig]:
        return list(self._agents.values())

    def to_dict(self, config: AgentConfig) -> dict:
        return {
            "id": config.id,
            "version": config.version,
            "name": config.name,
            "model": {"id": config.model},
            "system": config.system,
            "tools": config.tools,
            "skills": [{"type": "skill", "name": s} for s in config.skills],
            "mcp_servers": config.mcp_servers,
            "metadata": config.metadata,
            "created_at": config.created_at,
        }
