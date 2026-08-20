from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass
from typing import Any


class AuthorizationError(PermissionError):
    pass


@dataclass(frozen=True)
class TokenClaims:
    issuer: str
    audience: str
    tenant_id: str
    user_id: str
    scopes: tuple[str, ...]
    expires_at: float
    token_id: str


class TokenVerifier:
    """Deterministic post-IdP checks. Raw tokens never cross this boundary."""

    def __init__(self, issuer: str, audience: str) -> None:
        self.issuer = issuer
        self.audience = audience
        self._revoked: set[str] = set()

    def revoke(self, token_id: str) -> None:
        self._revoked.add(token_id)

    def verify(self, claims: TokenClaims, required_scope: str, now: float | None = None) -> dict[str, Any]:
        current = time.time() if now is None else now
        if claims.issuer != self.issuer or claims.audience != self.audience:
            raise AuthorizationError("issuer or audience mismatch")
        if claims.expires_at <= current or claims.token_id in self._revoked:
            raise AuthorizationError("token expired or revoked")
        if required_scope not in claims.scopes:
            raise AuthorizationError("required scope missing")
        return {
            "tenant_id": claims.tenant_id,
            "user_id": claims.user_id,
            "scopes": claims.scopes,
            "credential_ref": hashlib.sha256(claims.token_id.encode()).hexdigest()[:12],
        }


class TenantMemory:
    """A provenance-aware memory/cache reference baseline with tenant deletion."""

    def __init__(self) -> None:
        self._facts: dict[tuple[str, str], dict[str, Any]] = {}
        self._cache: dict[tuple[str, str], Any] = {}

    def put_fact(self, tenant_id: str, fact_id: str, value: str, source_id: str, approved: bool) -> None:
        if not approved:
            raise AuthorizationError("unapproved facts cannot enter long-term memory")
        self._facts[(tenant_id, fact_id)] = {
            "value": value,
            "source_id": source_id,
            "version": self._facts.get((tenant_id, fact_id), {}).get("version", 0) + 1,
        }

    def get_fact(self, tenant_id: str, fact_id: str) -> dict[str, Any] | None:
        value = self._facts.get((tenant_id, fact_id))
        return dict(value) if value else None

    def put_cache(self, tenant_id: str, key: str, value: Any) -> None:
        self._cache[(tenant_id, key)] = value

    def get_cache(self, tenant_id: str, key: str) -> Any:
        return self._cache.get((tenant_id, key))

    def export(self, tenant_id: str) -> dict[str, Any]:
        return {
            "facts": {key: value for (tenant, key), value in self._facts.items() if tenant == tenant_id},
            "cache_keys": [key for tenant, key in self._cache if tenant == tenant_id],
        }

    def delete_tenant(self, tenant_id: str) -> None:
        self._facts = {key: value for key, value in self._facts.items() if key[0] != tenant_id}
        self._cache = {key: value for key, value in self._cache.items() if key[0] != tenant_id}


SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9]{16,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"Bearer\s+[A-Za-z0-9._-]{12,}"),
    re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}"),
)


def scan_public_artifact(text: str) -> list[str]:
    return [pattern.pattern for pattern in SECRET_PATTERNS if pattern.search(text)]
