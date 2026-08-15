"""Tenant-scoped RAG, memory and cache stores used by the Week 9 lab.

The in-memory backend is intentionally small.  The important property is that
tenant selection happens before lookup or ranking, and that deletion fans out
to every derived store instead of deleting only the primary row.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


JSON = dict[str, Any]


class ResourceNotFound(Exception):
    """Return one indistinguishable result for missing and foreign resources."""


@dataclass
class TenantDataStore:
    rag: dict[str, dict[str, JSON]] = field(default_factory=dict)
    memories: dict[str, dict[str, JSON]] = field(default_factory=dict)
    cache: dict[str, dict[str, JSON]] = field(default_factory=dict)
    deletion_ledger: list[JSON] = field(default_factory=list)

    @classmethod
    def sample(cls) -> "TenantDataStore":
        return cls(
            rag={
                "tenant-a": {
                    "rag-a-1": {
                        "id": "rag-a-1",
                        "text": "Atlas release is approved.",
                        "classification": "internal",
                    }
                },
                "tenant-b": {
                    "rag-b-1": {
                        "id": "rag-b-1",
                        "text": "COBALT-SECRET belongs only to tenant-b.",
                        "classification": "confidential",
                    }
                },
            },
            memories={
                "tenant-a": {
                    "mem-a-1": {"id": "mem-a-1", "owner_id": "alice", "text": "Alice prefers concise reports."}
                },
                "tenant-b": {
                    "mem-b-1": {"id": "mem-b-1", "owner_id": "bob", "text": "Bob project code is NEBULA."}
                },
            },
            cache={
                "tenant-a": {"answer:atlas": {"value": "approved", "owner_id": "alice"}},
                "tenant-b": {"answer:cobalt": {"value": "COBALT-SECRET", "owner_id": "bob"}},
            },
        )

    def search_rag(self, tenant_id: str, query: str, *, limit: int = 5) -> list[JSON]:
        """Select the tenant partition before matching or ranking."""

        needle = query.casefold()
        rows = self.rag.get(tenant_id, {}).values()
        return [deepcopy(row) for row in rows if needle in row["text"].casefold()][:limit]

    def read_memory(self, tenant_id: str, memory_id: str, user_id: str) -> JSON:
        row = self.memories.get(tenant_id, {}).get(memory_id)
        if row is None or row["owner_id"] != user_id:
            raise ResourceNotFound("memory not found")
        return deepcopy(row)

    def write_memory(self, tenant_id: str, user_id: str, memory_id: str, text: str) -> JSON:
        row = {"id": memory_id, "owner_id": user_id, "text": text}
        self.memories.setdefault(tenant_id, {})[memory_id] = row
        return deepcopy(row)

    def get_cache(self, tenant_id: str, key: str, user_id: str) -> JSON:
        row = self.cache.get(tenant_id, {}).get(key)
        if row is None or row["owner_id"] != user_id:
            raise ResourceNotFound("cache entry not found")
        return deepcopy(row)

    def export_tenant(self, tenant_id: str) -> JSON:
        """Export only the selected tenant; caller authorization happens upstream."""

        return {
            "tenant_id": tenant_id,
            "rag": deepcopy(list(self.rag.get(tenant_id, {}).values())),
            "memories": deepcopy(list(self.memories.get(tenant_id, {}).values())),
            "cache": deepcopy(self.cache.get(tenant_id, {})),
        }

    def delete_tenant(self, tenant_id: str, *, request_id: str) -> JSON:
        """Delete primary and derived data, then record a content-free tombstone."""

        counts = {
            "rag": len(self.rag.pop(tenant_id, {})),
            "memories": len(self.memories.pop(tenant_id, {})),
            "cache": len(self.cache.pop(tenant_id, {})),
        }
        tombstone = {
            "tenant_id": tenant_id,
            "request_id": request_id,
            "deleted_at": datetime.now(UTC).isoformat(),
            "counts": counts,
        }
        self.deletion_ledger.append(tombstone)
        return deepcopy(tombstone)

    def has_tenant_data(self, tenant_id: str) -> bool:
        return any(tenant_id in store and bool(store[tenant_id]) for store in (self.rag, self.memories, self.cache))
