"""Version manifest carried by every task, span and release decision."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any


@dataclass(frozen=True)
class VersionManifest:
    prompt: str
    model: str
    tool_schema: str
    mcp_server: str
    memory_policy: str
    eval_set: str
    runtime: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        canonical = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def trace_attributes(self) -> dict[str, Any]:
        values: dict[str, Any] = {
            f"app.version.{name}": value for name, value in self.as_dict().items()
        }
        values["app.version.fingerprint"] = self.fingerprint
        return values


BASELINE_VERSIONS = VersionManifest(
    prompt="support-agent@1.4.0",
    model="model-stable@2026-08-01",
    tool_schema="tools@3.2.0",
    mcp_server="business-mcp@2.1.0",
    memory_policy="tenant-memory@1.3.0",
    eval_set="week8-regression@1.0.0",
    runtime="week10-runtime@1.0.0",
)

CANDIDATE_VERSIONS = VersionManifest(
    prompt="support-agent@1.5.0",
    model="model-candidate@2026-08-15",
    tool_schema="tools@3.3.0",
    mcp_server="business-mcp@2.1.0",
    memory_policy="tenant-memory@1.4.0",
    eval_set="week8-regression@1.0.0",
    runtime="week10-runtime@1.1.0",
)

