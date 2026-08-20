from __future__ import annotations

import re
from urllib.parse import urlparse

from .models import RunRequest


class PolicyViolation(ValueError):
    pass


INJECTION_PATTERNS = (
    re.compile(r"ignore (all|any|the|previous) .*instructions?", re.I),
    re.compile(r"system prompt", re.I),
    re.compile(r"reveal .*?(secret|token|credential)", re.I),
    re.compile(r"执行.{0,8}(rm\s+-rf|删库|转账)", re.I),
)
SECRET_PATTERN = re.compile(
    r"(?:sk-[A-Za-z0-9_-]{16,}|AKIA[0-9A-Z]{16}|Bearer\s+[A-Za-z0-9._-]{12,})"
)


def validate_request(request: RunRequest) -> None:
    if not request.question.strip() or len(request.question) > 4_000:
        raise PolicyViolation("question must contain 1..4000 characters")
    if not request.identity.tenant_id or not request.identity.user_id:
        raise PolicyViolation("tenant_id and user_id are required")
    if "research:run" not in request.identity.scopes:
        raise PolicyViolation("missing research:run scope")
    required_steps = 7 if request.require_approval else 5
    if request.budget.max_steps < required_steps or request.budget.max_sources < 1:
        raise PolicyViolation("budget is too small to run safely")
    if request.budget.max_chars < 1 or request.budget.max_cost_usd <= 0:
        raise PolicyViolation("character and cost budgets must be positive")
    if len(request.sources) > 100:
        raise PolicyViolation("too many sources")
    if request.idempotency_key is not None and not (1 <= len(request.idempotency_key) <= 200):
        raise PolicyViolation("idempotency_key must contain 1..200 characters")
    total_chars = sum(len(source.content) for source in request.sources)
    if total_chars > request.budget.max_chars:
        raise PolicyViolation("source content exceeds character budget")
    for source in request.sources:
        if source.trust not in {"external", "internal", "authoritative"}:
            raise PolicyViolation(f"invalid source trust: {source.source_id}")
        parsed = urlparse(source.url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise PolicyViolation(f"unsafe source URL: {source.source_id}")


def source_taints(content: str) -> tuple[str, ...]:
    return tuple("prompt_injection_suspected" for pattern in INJECTION_PATTERNS if pattern.search(content))


def sanitize_excerpt(content: str, limit: int = 480) -> str:
    clean = SECRET_PATTERN.sub("[REDACTED]", content)
    clean = " ".join(clean.split())
    return clean[:limit]


def validate_output(report: str, evidence_ids: set[str]) -> None:
    if SECRET_PATTERN.search(report):
        raise PolicyViolation("output contains a credential-like value")
    cited = set(re.findall(r"\[(S\d+)\]", report))
    if not cited or not cited.issubset(evidence_ids):
        raise PolicyViolation("report contains missing or invalid citations")
