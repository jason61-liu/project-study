"""Deterministic security gateway for an agent's tools and tenant data.

The model may propose a tool name and business arguments.  It never receives
credentials and it cannot decide authorization, approval, tenant identity or
sandbox policy.  Those decisions are made here before any side effect.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
import hashlib
import hmac
import json
import re
import time
from typing import Any, Callable, Mapping
from uuid import uuid4

from jsonschema import Draft202012Validator

from sandbox import E2BSandboxExecutor, SandboxPolicyError, UnavailableSandboxExecutor
from tenant_store import ResourceNotFound, TenantDataStore


JSON = dict[str, Any]


class SecurityError(Exception):
    def __init__(self, code: str, message: str, *, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = dict(details or {})


@dataclass(frozen=True)
class Principal:
    subject_id: str
    actor_id: str
    tenant_id: str
    roles: frozenset[str]
    scopes: frozenset[str]
    token_id: str
    attributes: Mapping[str, Any] = field(default_factory=dict)


class AccessTokenService:
    """Small HS256 resource-server simulator with jti revocation.

    This is a deterministic lab substitute for OIDC/JWKS or introspection, not
    a production authorization server.
    """

    def __init__(self, secret: bytes, *, issuer: str = "week9-idp", audience: str = "week9-tools") -> None:
        if len(secret) < 32:
            raise ValueError("token secret must be at least 32 bytes")
        self._secret = secret
        self.issuer = issuer
        self.audience = audience
        self._revoked: set[str] = set()

    def issue(
        self,
        *,
        subject_id: str,
        actor_id: str,
        tenant_id: str,
        roles: set[str],
        scopes: set[str],
        attributes: Mapping[str, Any] | None = None,
        token_id: str | None = None,
        expires_in_s: int = 300,
    ) -> str:
        now = int(time.time())
        payload = {
            "iss": self.issuer,
            "aud": self.audience,
            "sub": subject_id,
            "act": actor_id,
            "tenant_id": tenant_id,
            "roles": sorted(roles),
            "scope": " ".join(sorted(scopes)),
            "attrs": dict(attributes or {}),
            "iat": now,
            "exp": now + expires_in_s,
            "jti": token_id or uuid4().hex,
        }
        header = {"alg": "HS256", "typ": "JWT"}
        signing_input = f"{_b64json(header)}.{_b64json(payload)}"
        signature = _b64(hmac.new(self._secret, signing_input.encode(), hashlib.sha256).digest())
        return f"{signing_input}.{signature}"

    def revoke(self, token_id: str) -> None:
        self._revoked.add(token_id)

    def verify(self, raw_token: str | None) -> Principal:
        if not raw_token:
            raise SecurityError("AUTH_MISSING", "missing access token")
        try:
            header_part, payload_part, signature = raw_token.split(".")
            signing_input = f"{header_part}.{payload_part}"
            expected = _b64(hmac.new(self._secret, signing_input.encode(), hashlib.sha256).digest())
            if not hmac.compare_digest(signature, expected):
                raise ValueError("bad signature")
            header = _decode_json(header_part)
            payload = _decode_json(payload_part)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            raise SecurityError("AUTH_INVALID", "invalid access token") from exc
        if header != {"alg": "HS256", "typ": "JWT"}:
            raise SecurityError("AUTH_INVALID", "untrusted token algorithm")
        if payload.get("iss") != self.issuer or payload.get("aud") != self.audience:
            raise SecurityError("AUTH_INVALID", "invalid issuer or audience")
        if not all(payload.get(name) for name in ("sub", "act", "tenant_id", "jti")):
            raise SecurityError("AUTH_INVALID", "delegation claims are incomplete")
        if not isinstance(payload.get("exp"), int) or payload["exp"] <= int(time.time()):
            raise SecurityError("TOKEN_EXPIRED", "access token expired")
        if payload["jti"] in self._revoked:
            raise SecurityError("TOKEN_REVOKED", "access token revoked")
        return Principal(
            subject_id=str(payload["sub"]),
            actor_id=str(payload["act"]),
            tenant_id=str(payload["tenant_id"]),
            roles=frozenset(str(item) for item in payload.get("roles", [])),
            scopes=frozenset(str(payload.get("scope", "")).split()),
            token_id=str(payload["jti"]),
            attributes=dict(payload.get("attrs", {})),
        )


def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode()


def _b64json(value: JSON) -> str:
    return _b64(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def _decode_json(value: str) -> JSON:
    result = json.loads(base64.urlsafe_b64decode(value + "=" * (-len(value) % 4)))
    if not isinstance(result, dict):
        raise ValueError("JWT part is not an object")
    return result


class DataSanitizer:
    """Redact credentials and PII before logs or model observations."""

    SENSITIVE_KEYS = re.compile(r"(?i)(authorization|access.?token|api.?key|secret|password|cookie)")
    CREDENTIALS = re.compile(
        r"(?i)(bearer\s+[A-Za-z0-9._~-]+"
        r"|\beyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{8,}\b"
        r"|\b(?:sk|e2b)_[A-Za-z0-9_-]{8,}\b)"
    )
    EMAIL = re.compile(r"\b([A-Za-z0-9._%+-])[^@\s]*@([A-Za-z0-9.-]+\.[A-Za-z]{2,})\b")
    PHONE = re.compile(r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)")
    INJECTION = re.compile(r"(?i)(ignore (?:all |the )?(?:previous|system)|system prompt|exfiltrate|developer message)")

    @classmethod
    def contains_credential(cls, value: Any, *, key: str = "") -> bool:
        if key and cls.SENSITIVE_KEYS.search(key):
            return True
        if isinstance(value, str):
            return bool(cls.CREDENTIALS.search(value))
        if isinstance(value, Mapping):
            return any(cls.contains_credential(item, key=str(name)) for name, item in value.items())
        if isinstance(value, (list, tuple)):
            return any(cls.contains_credential(item) for item in value)
        return False

    @classmethod
    def redact(cls, value: Any, *, key: str = "") -> Any:
        if key and cls.SENSITIVE_KEYS.search(key):
            return "[REDACTED_CREDENTIAL]"
        if isinstance(value, str):
            value = cls.CREDENTIALS.sub("[REDACTED_CREDENTIAL]", value)
            value = cls.EMAIL.sub(lambda m: f"{m.group(1)}***@{m.group(2)}", value)
            value = cls.PHONE.sub(lambda m: f"***{m.group(0)[-4:]}", value)
            return value
        if isinstance(value, Mapping):
            return {str(name): cls.redact(item, key=str(name)) for name, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls.redact(item) for item in value]
        return value

    @classmethod
    def model_context(cls, messages: list[JSON]) -> list[JSON]:
        if cls.contains_credential(messages):
            raise SecurityError("CREDENTIAL_IN_CONTEXT", "raw credentials cannot enter model context")
        return cls.redact(messages)

    @classmethod
    def user_input(cls, content: str) -> JSON:
        """Preserve user text as data and attach risk signals, never authority."""

        flags = ["untrusted_user_input"]
        if cls.INJECTION.search(content):
            flags.append("instruction_like_content")
        return {"source": "user", "trust": "untrusted", "flags": flags, "content": cls.redact(content)}

    @classmethod
    def observation(cls, tool_name: str, result: Any) -> JSON:
        text = json.dumps(result, ensure_ascii=False, default=str)
        flags = ["untrusted_tool_result"]
        if cls.INJECTION.search(text):
            flags.append("instruction_like_content")
        return {
            "source": f"tool:{tool_name}",
            "trust": "untrusted",
            "flags": flags,
            "data": cls.redact(result),
        }


@dataclass
class AuditSink:
    events: list[JSON] = field(default_factory=list)

    def record(self, event: JSON) -> None:
        self.events.append(DataSanitizer.redact(event))


@dataclass
class ApprovalTicket:
    approval_id: str
    action_hash: str
    tenant_id: str
    requester_id: str
    expires_at: datetime
    approved_by: str | None = None
    consumed: bool = False


class ApprovalStore:
    def __init__(self) -> None:
        self._tickets: dict[str, ApprovalTicket] = {}

    def request(self, principal: Principal, action_hash: str, *, ttl_s: int = 300) -> ApprovalTicket:
        ticket = ApprovalTicket(
            approval_id=uuid4().hex,
            action_hash=action_hash,
            tenant_id=principal.tenant_id,
            requester_id=principal.subject_id,
            expires_at=datetime.now(UTC) + timedelta(seconds=ttl_s),
        )
        self._tickets[ticket.approval_id] = ticket
        return ticket

    def approve(self, approval_id: str, approver: Principal) -> None:
        ticket = self._tickets.get(approval_id)
        if ticket is None:
            raise SecurityError("APPROVAL_INVALID", "approval does not exist")
        if "security_approver" not in approver.roles or approver.tenant_id != ticket.tenant_id:
            raise SecurityError("APPROVAL_FORBIDDEN", "approver is not authorized")
        if approver.subject_id == ticket.requester_id:
            raise SecurityError("APPROVAL_SELF_REVIEW", "requester cannot approve the same action")
        ticket.approved_by = approver.subject_id

    def consume(self, approval_id: str | None, action_hash: str, principal: Principal) -> None:
        ticket = self._tickets.get(approval_id or "")
        if ticket is None or ticket.approved_by is None:
            raise SecurityError("APPROVAL_REQUIRED", "approved high-risk action ticket is required")
        if ticket.consumed or ticket.expires_at <= datetime.now(UTC):
            raise SecurityError("APPROVAL_EXPIRED", "approval is expired or already consumed")
        if ticket.tenant_id != principal.tenant_id or ticket.requester_id != principal.subject_id:
            raise SecurityError("APPROVAL_MISMATCH", "approval principal does not match")
        if not hmac.compare_digest(ticket.action_hash, action_hash):
            raise SecurityError("APPROVAL_MISMATCH", "approved action was modified")
        ticket.consumed = True


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: JSON
    handler: Callable[[Principal, JSON], Any]
    roles: frozenset[str]
    scope: str
    high_risk: bool = False

    def model_definition(self) -> JSON:
        return {"name": self.name, "description": self.description, "inputSchema": self.input_schema}


class PolicyEngine:
    def authorize(self, principal: Principal, spec: ToolSpec, arguments: JSON) -> None:
        if spec.scope not in principal.scopes:
            raise SecurityError("INSUFFICIENT_SCOPE", "required OAuth scope is missing", details={"scope": spec.scope})
        if not principal.roles.intersection(spec.roles):
            raise SecurityError("RBAC_DENIED", "role is not allowed to use this tool")
        resource_tenant = str(arguments.get("tenant_id", principal.tenant_id))
        if resource_tenant != principal.tenant_id:
            raise SecurityError("ABAC_DENIED", "resource tenant does not match principal tenant")
        requested_region = arguments.get("region")
        allowed_regions = set(principal.attributes.get("regions", []))
        if requested_region and requested_region not in allowed_regions:
            raise SecurityError("ABAC_DENIED", "requested region is outside principal attributes")


class ToolGateway:
    def __init__(
        self,
        token_service: AccessTokenService,
        store: TenantDataStore | None = None,
        *,
        sandbox: Any | None = None,
        approvals: ApprovalStore | None = None,
        audit: AuditSink | None = None,
    ) -> None:
        self.tokens = token_service
        self.store = store or TenantDataStore.sample()
        self.sandbox = sandbox or UnavailableSandboxExecutor()
        self.approvals = approvals or ApprovalStore()
        self.audit = audit or AuditSink()
        self.policy = PolicyEngine()
        self.tools = self._build_tools()

    def model_tool_definitions(self) -> list[JSON]:
        definitions = [spec.model_definition() for spec in self.tools.values()]
        if DataSanitizer.contains_credential(definitions):
            raise AssertionError("tool schema exposed a credential field")
        return definitions

    def action_hash(self, tool_name: str, arguments: JSON, principal: Principal) -> str:
        clean = {name: value for name, value in arguments.items() if name != "approval_id"}
        value = {
            "tool": tool_name,
            "arguments": clean,
            "tenant_id": principal.tenant_id,
            "subject_id": principal.subject_id,
            "actor_id": principal.actor_id,
        }
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def request_approval(self, raw_token: str, tool_name: str, arguments: JSON) -> ApprovalTicket:
        principal = self.tokens.verify(raw_token)
        spec = self.tools.get(tool_name)
        if spec is None or not spec.high_risk:
            raise SecurityError("APPROVAL_NOT_APPLICABLE", "tool is unknown or not high risk")
        self._validate(spec, arguments, allow_missing_approval=True)
        self.policy.authorize(principal, spec, arguments)
        return self.approvals.request(principal, self.action_hash(tool_name, arguments, principal))

    def invoke(self, tool_name: str, arguments: JSON, *, raw_token: str | None) -> JSON:
        trace_id = uuid4().hex
        started = time.monotonic()
        principal: Principal | None = None
        try:
            spec = self.tools.get(tool_name)
            if spec is None:
                raise SecurityError("TOOL_NOT_ALLOWLISTED", "tool is not registered")
            if DataSanitizer.contains_credential(arguments):
                raise SecurityError("CREDENTIAL_ARGUMENT", "credentials are host-only and forbidden in model arguments")
            principal = self.tokens.verify(raw_token)
            self._validate(spec, arguments)
            self.policy.authorize(principal, spec, arguments)
            if spec.high_risk:
                self.approvals.consume(
                    str(arguments.get("approval_id", "")),
                    self.action_hash(tool_name, arguments, principal),
                    principal,
                )
            data = spec.handler(principal, arguments)
            observation = DataSanitizer.observation(tool_name, data)
            result = {"ok": True, "data": data, "observation": observation, "error": None}
            status = "allowed"
        except SecurityError as exc:
            result = {"ok": False, "data": None, "observation": None, "error": {"code": exc.code, "message": exc.message, "details": exc.details}}
            status = exc.code
        except (ResourceNotFound, SandboxPolicyError) as exc:
            code = "RESOURCE_NOT_FOUND" if isinstance(exc, ResourceNotFound) else "SANDBOX_POLICY_DENIED"
            result = {"ok": False, "data": None, "observation": None, "error": {"code": code, "message": str(exc), "details": {}}}
            status = code
        except Exception:
            result = {"ok": False, "data": None, "observation": None, "error": {"code": "TOOL_FAILURE", "message": "tool failed", "details": {}}}
            status = "TOOL_FAILURE"

        self.audit.record(
            {
                "trace_id": trace_id,
                "timestamp": datetime.now(UTC).isoformat(),
                "tool": tool_name,
                "status": status,
                "tenant_id": principal.tenant_id if principal else None,
                "subject_id": principal.subject_id if principal else None,
                "actor_id": principal.actor_id if principal else None,
                "token_id": principal.token_id if principal else None,
                "duration_ms": round((time.monotonic() - started) * 1000, 3),
                "arguments": arguments,
            }
        )
        result["meta"] = {"trace_id": trace_id, "status": status}
        return result

    @staticmethod
    def _validate(spec: ToolSpec, arguments: JSON, *, allow_missing_approval: bool = False) -> None:
        candidate = dict(arguments)
        if allow_missing_approval and spec.high_risk and "approval_id" not in candidate:
            candidate["approval_id"] = "pending-approval-placeholder"
        errors = sorted(Draft202012Validator(spec.input_schema).iter_errors(candidate), key=lambda item: list(item.path))
        if errors:
            raise SecurityError(
                "INVALID_ARGUMENTS",
                "arguments failed JSON Schema",
                details={"errors": [{"path": list(error.path), "message": error.message} for error in errors]},
            )

    def _build_tools(self) -> dict[str, ToolSpec]:
        closed = {"type": "object", "additionalProperties": False}
        tenant = {"tenant_id": {"type": "string", "pattern": "^tenant-[a-z0-9-]+$"}}
        approval = {"approval_id": {"type": "string", "minLength": 8, "maxLength": 128}}

        def rag(principal: Principal, args: JSON) -> Any:
            return {"items": self.store.search_rag(principal.tenant_id, args["query"], limit=args.get("limit", 5))}

        def memory_read(principal: Principal, args: JSON) -> Any:
            return self.store.read_memory(principal.tenant_id, args["memory_id"], principal.subject_id)

        def memory_write(principal: Principal, args: JSON) -> Any:
            return self.store.write_memory(principal.tenant_id, principal.subject_id, args["memory_id"], args["text"])

        def cache_get(principal: Principal, args: JSON) -> Any:
            return self.store.get_cache(principal.tenant_id, args["key"], principal.subject_id)

        def export(principal: Principal, _args: JSON) -> Any:
            return self.store.export_tenant(principal.tenant_id)

        def delete(principal: Principal, args: JSON) -> Any:
            return self.store.delete_tenant(principal.tenant_id, request_id=args["request_id"])

        def run_shell(_principal: Principal, args: JSON) -> Any:
            result = self.sandbox.run_shell(args["command"])
            return result.__dict__

        def run_code(_principal: Principal, args: JSON) -> Any:
            result = self.sandbox.run_code(args["code"])
            return result.__dict__

        specs = [
            ToolSpec("rag_search", "Search only the verified tenant RAG partition.", {**closed, "properties": {**tenant, "query": {"type": "string", "minLength": 1, "maxLength": 500}, "limit": {"type": "integer", "minimum": 1, "maximum": 20}}, "required": ["tenant_id", "query"]}, rag, frozenset({"viewer", "editor", "admin"}), "rag.read"),
            ToolSpec("memory_read", "Read one user-owned memory in the verified tenant.", {**closed, "properties": {**tenant, "memory_id": {"type": "string", "pattern": "^mem-[a-z0-9-]+$"}}, "required": ["tenant_id", "memory_id"]}, memory_read, frozenset({"viewer", "editor", "admin"}), "memory.read"),
            ToolSpec("memory_write", "Write one user-owned memory in the verified tenant.", {**closed, "properties": {**tenant, "memory_id": {"type": "string", "pattern": "^mem-[a-z0-9-]+$"}, "text": {"type": "string", "minLength": 1, "maxLength": 2000}}, "required": ["tenant_id", "memory_id", "text"]}, memory_write, frozenset({"editor", "admin"}), "memory.write"),
            ToolSpec("cache_get", "Read a user-owned cache entry in the verified tenant.", {**closed, "properties": {**tenant, "key": {"type": "string", "minLength": 1, "maxLength": 200}}, "required": ["tenant_id", "key"]}, cache_get, frozenset({"viewer", "editor", "admin"}), "cache.read"),
            ToolSpec("tenant_export", "Export all data for the verified tenant after approval.", {**closed, "properties": {**tenant, **approval}, "required": ["tenant_id", "approval_id"]}, export, frozenset({"admin"}), "tenant.export", True),
            ToolSpec("tenant_delete", "Delete RAG, memory and cache data after approval.", {**closed, "properties": {**tenant, **approval, "request_id": {"type": "string", "pattern": "^delete-[a-z0-9-]{8,64}$"}}, "required": ["tenant_id", "request_id", "approval_id"]}, delete, frozenset({"admin"}), "tenant.delete", True),
            ToolSpec("run_shell", "Run one allowlisted argv in a disposable network-denied E2B sandbox.", {**closed, "properties": {**tenant, **approval, "command": {"type": "string", "minLength": 1, "maxLength": 8000}}, "required": ["tenant_id", "command", "approval_id"]}, run_shell, frozenset({"operator", "admin"}), "sandbox.execute", True),
            ToolSpec("run_code", "Run Python in a disposable network-denied E2B sandbox.", {**closed, "properties": {**tenant, **approval, "code": {"type": "string", "minLength": 1, "maxLength": 32768}}, "required": ["tenant_id", "code", "approval_id"]}, run_code, frozenset({"operator", "admin"}), "sandbox.execute", True),
        ]
        return {spec.name: spec for spec in specs}


def live_gateway_from_environment(token_service: AccessTokenService, store: TenantDataStore | None = None) -> ToolGateway:
    """Use managed E2B only when the host has a key; never fall back to host exec."""

    import os

    sandbox = E2BSandboxExecutor() if os.getenv("E2B_API_KEY") else UnavailableSandboxExecutor()
    return ToolGateway(token_service, store, sandbox=sandbox)
