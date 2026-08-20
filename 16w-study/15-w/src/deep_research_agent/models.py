from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


def _reject_unknown(value: dict[str, Any], allowed: set[str], name: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"unknown {name} fields: {', '.join(sorted(unknown))}")


class RunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    COMPLETED = "completed"
    REFUSED = "refused"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ErrorType(StrEnum):
    INVALID_INPUT = "invalid_input"
    UNAUTHORIZED = "unauthorized"
    BUDGET_EXCEEDED = "budget_exceeded"
    NO_EVIDENCE = "no_evidence"
    APPROVAL_REQUIRED = "approval_required"
    CONFLICT = "idempotency_conflict"
    INTERNAL = "internal"


@dataclass(frozen=True)
class Identity:
    tenant_id: str
    user_id: str
    roles: tuple[str, ...] = ("researcher",)
    scopes: tuple[str, ...] = ("research:run",)


@dataclass(frozen=True)
class Source:
    source_id: str
    title: str
    url: str
    content: str
    claims: dict[str, str] = field(default_factory=dict)
    trust: str = "external"

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Source":
        _reject_unknown(value, {"source_id", "title", "url", "content", "claims", "trust"}, "source")
        return cls(
            source_id=str(value["source_id"]),
            title=str(value["title"]),
            url=str(value["url"]),
            content=str(value["content"]),
            claims={str(k): str(v) for k, v in value.get("claims", {}).items()},
            trust=str(value.get("trust", "external")),
        )


@dataclass(frozen=True)
class Budget:
    max_steps: int = 12
    max_sources: int = 20
    max_chars: int = 120_000
    max_cost_usd: float = 1.0


@dataclass(frozen=True)
class RunRequest:
    question: str
    identity: Identity
    sources: tuple[Source, ...]
    budget: Budget = field(default_factory=Budget)
    require_approval: bool = False
    idempotency_key: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RunRequest":
        _reject_unknown(
            value,
            {"question", "identity", "sources", "budget", "require_approval", "idempotency_key"},
            "request",
        )
        identity = value.get("identity", {})
        budget = value.get("budget", {})
        _reject_unknown(identity, {"tenant_id", "user_id", "roles", "scopes"}, "identity")
        return cls(
            question=str(value["question"]),
            identity=Identity(
                tenant_id=str(identity["tenant_id"]),
                user_id=str(identity["user_id"]),
                roles=tuple(identity.get("roles", ["researcher"])),
                scopes=tuple(identity.get("scopes", ["research:run"])),
            ),
            sources=tuple(Source.from_dict(item) for item in value.get("sources", [])),
            budget=Budget(**budget),
            require_approval=bool(value.get("require_approval", False)),
            idempotency_key=value.get("idempotency_key"),
        )


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    source_id: str
    title: str
    url: str
    excerpt: str
    score: float
    claims: dict[str, str]


@dataclass
class RunState:
    run_id: str
    tenant_id: str
    user_id: str
    question: str
    status: RunStatus = RunStatus.PENDING
    current_step: str = "created"
    version: int = 0
    subqueries: list[str] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    report: str | None = None
    refusal_reason: str | None = None
    cost_usd: float = 0.0
    steps_used: int = 0
    max_steps: int = 12
    max_cost_usd: float = 1.0
    require_approval: bool = False

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["status"] = self.status.value
        return result

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RunState":
        value = dict(value)
        value["status"] = RunStatus(value["status"])
        return cls(**value)
