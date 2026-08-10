"""第三周编码实验：带契约、授权、超时和幂等控制的最小工具运行时。

模块按“定义契约 → 验证身份 → 校验参数 → 执行 → 校验输出 → 记录结果”的顺序
实现工具调用。模型只产生工具名和业务参数；Token 获取、授权、线程调度、幂等记录
与日志均由确定性的 Runtime 完成，不能委托给概率性模型判断。

代码中的 HS256 TokenService 仅用于离线学习和测试。它模拟 OAuth Resource Server
需要检查的核心 Claims，但不负责浏览器授权流程，也不能替代生产授权服务器。
"""

from __future__ import annotations

import ast
import base64
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass, field
import hashlib
import hmac
import json
import logging
import operator
import time
from typing import Any, Callable, Mapping
from uuid import uuid4

from jsonschema import Draft202012Validator


JSON = dict[str, Any]
LOGGER = logging.getLogger("week3.tools")


@dataclass(frozen=True)
class AuthorizationContext:
    """验证 Token 后的最小授权上下文；不保存、也不透传原始 Token。"""

    # subject：本次调用代表的最终用户。
    user_id: str
    # 数据隔离边界；即使两个租户存在相同资源 ID，也不能交叉读取。
    tenant_id: str
    # Token 验证完成后的不可变 Scope 集合，后续工具只能继续缩权。
    scopes: frozenset[str]
    # 仅保留 jti 供审计/撤销关联，不保留可重放的原始 JWT。
    token_id: str


class ToolFailure(Exception):
    """可安全转换为结构化工具错误的预期失败。"""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        status: str = "business_failure",
        retryable: bool = False,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        """保存稳定错误码、用户消息、重试建议和机器可读细节。"""

        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status
        self.retryable = retryable
        self.details = dict(details or {})


class TokenService:
    """用于本地实验的 HS256 JWT 签发与验证器。

    生产环境应接入真正的 OAuth Authorization Server/JWKS 或 introspection；
    这里保留 issuer、audience、exp、jti、Scope 和撤销语义，便于确定性测试。
    """

    def __init__(self, secret: bytes, *, issuer: str = "week3-auth", audience: str = "tool-runtime") -> None:
        """初始化签发/验证配置和内存撤销集合。

        32 字节下限避免教学代码意外使用过短 HMAC 密钥。真实系统更推荐授权服务器
        用非对称私钥签名，Resource Server 只持有可轮换的 JWKS 公钥。
        """

        if len(secret) < 32:
            raise ValueError("HMAC secret 至少 32 字节")
        self.secret = secret
        self.issuer = issuer
        self.audience = audience
        self._revoked: set[str] = set()

    def issue(
        self,
        *,
        user_id: str,
        tenant_id: str,
        scopes: set[str],
        expires_in_s: int = 300,
        token_id: str | None = None,
        now: int | None = None,
    ) -> str:
        """签发包含用户、租户、Scope、过期时间和唯一 jti 的实验 JWT。"""

        issued_at = int(time.time()) if now is None else now
        # scope 使用 OAuth 常见的空格分隔形式；排序保证相同 Scope 集合生成稳定载荷。
        payload = {
            "iss": self.issuer,
            "aud": self.audience,
            "sub": user_id,
            "tenant_id": tenant_id,
            "scope": " ".join(sorted(scopes)),
            "iat": issued_at,
            "exp": issued_at + expires_in_s,
            "jti": token_id or uuid4().hex,
            "token_type": "user",
        }
        header = {"alg": "HS256", "typ": "JWT"}
        # JWT 签名覆盖 header 和 payload 的 Base64URL 文本。Payload 可被解码查看，
        # 保密依赖 TLS 和安全存储；签名只提供完整性与签发者真实性。
        signing_input = f"{_b64json(header)}.{_b64json(payload)}"
        signature = _b64(hmac.new(self.secret, signing_input.encode(), hashlib.sha256).digest())
        return f"{signing_input}.{signature}"

    def revoke(self, token_id: str) -> None:
        """按 jti 撤销一枚 Token；模拟 introspection/denylist 的状态检查。"""

        self._revoked.add(token_id)

    def verify(self, token: str | None, required_scopes: frozenset[str]) -> AuthorizationContext:
        """验证 Token 并返回脱敏上下文；任一失败均产生稳定结构化错误。

        验证顺序先确认格式和签名，再信任 Claims；随后检查 issuer/audience、用户类型、
        有效期、撤销状态和 Scope。调用方永远不能仅仅 decode JWT 后相信其中字段。
        """

        if not token:
            raise ToolFailure("AUTH_MISSING", "缺少用户 Access Token")
        try:
            header_part, payload_part, signature = token.split(".")
            signing_input = f"{header_part}.{payload_part}"
            expected = _b64(hmac.new(self.secret, signing_input.encode(), hashlib.sha256).digest())
            # compare_digest 使用常量时间比较，避免普通字符串比较暴露签名匹配前缀。
            if not hmac.compare_digest(signature, expected):
                raise ValueError("signature mismatch")
            header = _decode_json(header_part)
            payload = _decode_json(payload_part)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            raise ToolFailure("AUTH_INVALID", "Access Token 格式或签名无效") from exc

        # 算法必须由服务端白名单决定，不能接受 Token 自己指定的任意 alg。
        if header != {"alg": "HS256", "typ": "JWT"}:
            raise ToolFailure("AUTH_INVALID", "Access Token 算法不受信任")
        if payload.get("iss") != self.issuer or payload.get("aud") != self.audience:
            raise ToolFailure("AUTH_INVALID", "Access Token issuer 或 audience 无效")
        if payload.get("token_type") != "user" or not payload.get("sub") or not payload.get("tenant_id"):
            raise ToolFailure("AUTH_INVALID", "必须使用包含用户和租户的用户令牌")
        if not isinstance(payload.get("exp"), int) or payload["exp"] <= int(time.time()):
            raise ToolFailure("TOKEN_EXPIRED", "Access Token 已过期")
        token_id = payload.get("jti")
        if not isinstance(token_id, str):
            raise ToolFailure("AUTH_INVALID", "Access Token 缺少 jti")
        if token_id in self._revoked:
            raise ToolFailure("TOKEN_REVOKED", "Access Token 已撤销")

        # 有效权限是 Token Scope 与工具所需 Scope 的包含关系；模型描述或参数不能
        # 增加 Scope。这里只做粗粒度授权，资源归属稍后仍由 handler 检查。
        scopes = frozenset(str(payload.get("scope", "")).split())
        missing = sorted(required_scopes - scopes)
        if missing:
            raise ToolFailure(
                "INSUFFICIENT_SCOPE",
                "Access Token 权限不足",
                details={"required_scopes": sorted(required_scopes), "missing_scopes": missing},
            )
        return AuthorizationContext(payload["sub"], payload["tenant_id"], scopes, token_id)


def _b64(value: bytes) -> str:
    """生成 JWT 使用的无填充 Base64URL 文本。"""

    return base64.urlsafe_b64encode(value).rstrip(b"=").decode()


def _b64json(value: JSON) -> str:
    """用稳定字段顺序压缩 JSON 后执行 Base64URL 编码。"""

    return _b64(json.dumps(value, separators=(",", ":"), sort_keys=True).encode())


def _decode_json(value: str) -> JSON:
    """恢复 Base64URL 填充并解码 JSON 对象，拒绝数组或标量。"""

    padded = value + "=" * (-len(value) % 4)
    decoded = json.loads(base64.urlsafe_b64decode(padded))
    if not isinstance(decoded, dict):
        raise ValueError("JWT section is not an object")
    return decoded


@dataclass(frozen=True)
class ToolSpec:
    """一个工具的完整执行契约。

    input/output Schema 同时约束模型生成和实际 Runtime 数据；timeout_s 是每次调用
    的硬等待预算；required_scopes 与 side_effect 控制授权、确认和幂等路径。
    """

    name: str
    description: str
    input_schema: JSON
    output_schema: JSON
    handler: Callable[..., JSON]
    timeout_s: float = 1.0
    required_scopes: frozenset[str] = frozenset()
    side_effect: bool = False

    def model_definition(self) -> JSON:
        """只向模型公开业务参数；Access Token 永远不进入工具 Schema。"""

        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
            "outputSchema": self.output_schema,
        }


@dataclass
class InMemoryStore:
    """用于教学测试的 tenant-aware 数据库和幂等结果仓库。

    字典不是生产持久化方案，但它清晰展示三类状态必须共同提交：业务草稿、资源数据
    和幂等记录。生产环境应通过数据库唯一约束/事务避免两个进程同时执行同一意图。
    """

    documents: dict[str, dict[str, JSON]] = field(default_factory=dict)
    drafts: dict[str, JSON] = field(default_factory=dict)
    idempotency: dict[tuple[str, str, str, str], tuple[str, JSON]] = field(default_factory=dict)

    @classmethod
    def sample(cls) -> "InMemoryStore":
        """构造两个租户的样本，便于测试相同 Runtime 中的数据隔离。"""

        return cls(
            documents={
                "tenant-a": {
                    "doc-1": {"id": "doc-1", "title": "MCP 入门", "content": "MCP 连接 Agent 与工具和数据。"},
                    "doc-2": {"id": "doc-2", "title": "OAuth 笔记", "content": "Access Token 用于资源访问授权。"},
                },
                "tenant-b": {
                    "doc-3": {"id": "doc-3", "title": "隔离文档", "content": "tenant-b only"},
                },
            }
        )


class ToolRuntime:
    """统一执行 Schema、授权、确认、超时、输出校验、日志和幂等控制。"""

    def __init__(self, token_service: TokenService, store: InMemoryStore | None = None) -> None:
        """注入 Token 验证器和状态仓库，并创建有界工作线程池。"""

        self.token_service = token_service
        self.store = store or InMemoryStore.sample()
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="week3-tool")
        self.tools = _build_tools(self.store)

    def close(self) -> None:
        """停止接受新任务并取消尚未开始的 Future；运行中的线程不能被强杀。"""

        self._executor.shutdown(wait=False, cancel_futures=True)

    def invoke(self, name: str, arguments: JSON, *, token: str | None) -> JSON:
        """面向普通 Host 的入口：先验证 Token，再进入统一执行管线。

        未知工具在读取 Token 前直接返回 UNKNOWN_TOOL；已注册工具则按照各自声明的
        Scope 验证。原始 Token 只存在于这个短调用栈中，不写入 AuthorizationContext。
        """

        spec = self.tools.get(name)
        if spec is None:
            return self._error(name, "UNKNOWN_TOOL", f"未知工具：{name}")
        try:
            auth = self.token_service.verify(token, spec.required_scopes)
        except ToolFailure as exc:
            return self._failure(name, exc)
        return self.invoke_authorized(name, arguments, auth=auth)

    def invoke_authorized(self, name: str, arguments: JSON, *, auth: AuthorizationContext) -> JSON:
        """供已验证 Token 的 Host/MCP 适配层调用，只接收脱敏授权上下文。"""

        # 每一次物理调用都有新的 trace_id。幂等重放仍是一次新调用，因此会获得新
        # trace_id，但业务 data 保持首次提交结果。
        trace_id = uuid4().hex
        started = time.monotonic()
        spec = self.tools.get(name)
        if spec is None:
            return self._error(name, "UNKNOWN_TOOL", f"未知工具：{name}", trace_id=trace_id)

        # MCP Host 可能直接传入已验证上下文，所以这里必须再次执行 Scope 检查，
        # 不能假设 invoke() 是唯一入口。
        missing_scopes = sorted(spec.required_scopes - auth.scopes)
        if missing_scopes:
            return self._error(
                name,
                "INSUFFICIENT_SCOPE",
                "授权上下文权限不足",
                details={"missing_scopes": missing_scopes},
                trace_id=trace_id,
                auth=auth,
            )
        try:
            # 使用完整 Draft 2020-12 校验器而不是手写字段判断；收集全部错误能让
            # Agent 一次修正多个参数，path 则用于定位嵌套字段。
            errors = sorted(Draft202012Validator(spec.input_schema).iter_errors(arguments), key=lambda e: list(e.path))
            if errors:
                details = [{"path": _json_path(error.path), "message": error.message} for error in errors]
                raise ToolFailure("INVALID_ARGUMENTS", "工具参数不满足 JSON Schema", details={"errors": details})
            # Runtime 强制确认而不是依赖工具描述中的自然语言。dry-run 无副作用，
            # 所以允许 confirmed=false；真实写入必须显式确认。
            if spec.side_effect and not arguments["dry_run"] and not arguments["confirmed"]:
                raise ToolFailure("CONFIRMATION_REQUIRED", "真实写入前必须设置 confirmed=true")

            replay = self._idempotent_replay(spec, arguments, auth)
            if replay is not None:
                replay["meta"] = {**replay["meta"], "trace_id": trace_id, "idempotent_replay": True}
                self._log(name, trace_id, "success", started, auth)
                return replay

            # handler 收到的是 AuthorizationContext，不是 Token。这样业务代码即使
            # 被日志或异常捕获，也没有可泄露、可重放的凭证。
            future = self._executor.submit(spec.handler, auth, **arguments)
            try:
                data = future.result(timeout=spec.timeout_s)
            except FutureTimeout as exc:
                # Future.cancel 只能取消尚未运行的任务。若线程已经开始，状态可能未知；
                # 因此带副作用工具必须依赖幂等键对冲重复提交风险。
                future.cancel()
                raise ToolFailure(
                    "TOOL_TIMEOUT",
                    f"工具超过 {spec.timeout_s:.3f}s Deadline",
                    status="system_failure",
                    retryable=not spec.side_effect,
                    details={"execution_state": "unknown" if spec.side_effect else "not_completed"},
                ) from exc

            # 输入合法不代表工具实现一定正确；输出同样是不可信边界，必须在交给模型
            # 作为 observation 前验证，防止错误类型污染后续循环。
            output_errors = list(Draft202012Validator(spec.output_schema).iter_errors(data))
            if output_errors:
                raise ToolFailure("INVALID_TOOL_OUTPUT", "工具返回值不满足输出 Schema", status="system_failure")
            result = self._success(name, data, trace_id, auth, started)
            self._remember_idempotency(spec, arguments, auth, result)
            return result
        except ToolFailure as exc:
            result = self._failure(name, exc, trace_id=trace_id, auth=auth, started=started)
            return result
        except Exception:
            LOGGER.exception("tool_unhandled trace_id=%s tool=%s", trace_id, name)
            return self._error(
                name,
                "TOOL_INTERNAL_ERROR",
                "工具执行发生未处理异常",
                status="system_failure",
                retryable=False,
                trace_id=trace_id,
                auth=auth,
                started=started,
            )

    def model_tool_definitions(self, *, read_only: bool = False) -> list[JSON]:
        """导出模型可见契约，可选只返回无副作用工具。"""

        specs = (spec for spec in self.tools.values() if not read_only or not spec.side_effect)
        return [spec.model_definition() for spec in specs]

    def _idempotent_replay(self, spec: ToolSpec, arguments: JSON, auth: AuthorizationContext) -> JSON | None:
        """查找同一业务意图的已提交结果，并检测“同键不同参数”。"""

        if not spec.side_effect or arguments["dry_run"]:
            return None
        # 作用域包含租户、用户和工具名，避免不同调用主体或不同动作共享同一键。
        key = (auth.tenant_id, auth.user_id, spec.name, arguments["idempotency_key"])
        fingerprint = _business_fingerprint(arguments)
        if key not in self.store.idempotency:
            return None
        previous_fingerprint, previous_result = self.store.idempotency[key]
        if previous_fingerprint != fingerprint:
            raise ToolFailure("IDEMPOTENCY_CONFLICT", "同一幂等键对应了不同业务参数")
        return json.loads(json.dumps(previous_result))

    def _remember_idempotency(self, spec: ToolSpec, arguments: JSON, auth: AuthorizationContext, result: JSON) -> None:
        """只缓存真实提交的成功结果；dry-run 不消耗业务幂等键。"""

        if spec.side_effect and not arguments["dry_run"]:
            key = (auth.tenant_id, auth.user_id, spec.name, arguments["idempotency_key"])
            self.store.idempotency[key] = (_business_fingerprint(arguments), json.loads(json.dumps(result)))

    def _success(self, name: str, data: JSON, trace_id: str, auth: AuthorizationContext, started: float) -> JSON:
        """构造统一成功信封并在返回前落一条完成日志。"""

        self._log(name, trace_id, "success", started, auth)
        return {
            "ok": True,
            "status": "success",
            "data": data,
            "error": None,
            "meta": self._meta(trace_id, auth, started, idempotent_replay=False),
        }

    def _failure(
        self,
        name: str,
        failure: ToolFailure,
        *,
        trace_id: str | None = None,
        auth: AuthorizationContext | None = None,
        started: float | None = None,
    ) -> JSON:
        """把 ToolFailure 的领域字段转交给统一错误构造器。"""

        return self._error(
            name,
            failure.code,
            failure.message,
            status=failure.status,
            retryable=failure.retryable,
            details=failure.details,
            trace_id=trace_id,
            auth=auth,
            started=started,
        )

    def _error(
        self,
        name: str,
        code: str,
        message: str,
        *,
        status: str = "business_failure",
        retryable: bool = False,
        details: Mapping[str, Any] | None = None,
        trace_id: str | None = None,
        auth: AuthorizationContext | None = None,
        started: float | None = None,
    ) -> JSON:
        """构造不会泄露异常栈和 Token 的稳定失败信封。"""

        trace_id = trace_id or uuid4().hex
        started = time.monotonic() if started is None else started
        self._log(name, trace_id, code, started, auth)
        return {
            "ok": False,
            "status": status,
            "data": None,
            "error": {"code": code, "message": message, "retryable": retryable, "details": dict(details or {})},
            "meta": self._meta(trace_id, auth, started),
        }

    @staticmethod
    def _meta(trace_id: str, auth: AuthorizationContext | None, started: float, **extra: Any) -> JSON:
        """生成结果元数据；未认证失败会用 null 表示 tenant/user。"""

        return {
            "trace_id": trace_id,
            "duration_ms": round((time.monotonic() - started) * 1000, 3),
            "tenant_id": auth.tenant_id if auth else None,
            "user_id": auth.user_id if auth else None,
            **extra,
        }

    @staticmethod
    def _log(name: str, trace_id: str, status: str, started: float, auth: AuthorizationContext | None) -> None:
        # 日志只记录脱敏身份和 Trace；绝不记录 token、草稿正文等敏感数据。
        LOGGER.info(
            "tool_call tool=%s trace_id=%s status=%s duration_ms=%.3f tenant_id=%s user_id=%s",
            name,
            trace_id,
            status,
            (time.monotonic() - started) * 1000,
            auth.tenant_id if auth else "-",
            auth.user_id if auth else "-",
        )


OBJECT = {"type": "object", "additionalProperties": True}


def _build_tools(store: InMemoryStore) -> dict[str, ToolSpec]:
    """定义五个 handler，并把它们与 Schema、Scope 和副作用属性绑定。

    handler 只实现领域行为；所有横切能力由 ToolRuntime 统一处理。这样新增工具时
    不容易遗漏授权、超时或日志，同时每个 ToolSpec 又能声明自己的最小权限。
    """

    def search(auth: AuthorizationContext, query: str, limit: int = 5) -> JSON:
        """只搜索授权租户；返回摘要而不是正文，降低无意数据暴露。"""

        query_lower = query.lower()
        # 先按 tenant_id 选分区，再进行内容匹配。绝不能先全局搜索后只在结果中
        # 标记 tenant，否则排序、计数或错误信息仍可能形成跨租户侧信道。
        matches = [
            {"id": doc["id"], "title": doc["title"]}
            for doc in store.documents.get(auth.tenant_id, {}).values()
            if query_lower in f"{doc['title']} {doc['content']}".lower()
        ]
        return {"items": matches[:limit], "count": min(len(matches), limit)}

    def read(auth: AuthorizationContext, document_id: str) -> JSON:
        """读取 tenant 分区中的文档，不区分“不存在”和“无权限”。"""

        document = store.documents.get(auth.tenant_id, {}).get(document_id)
        if document is None:
            # 统一返回 NOT_FOUND，避免攻击者根据 403/404 差异枚举其他租户资源。
            raise ToolFailure("DOCUMENT_NOT_FOUND", "文档不存在或不属于当前租户")
        return dict(document)

    def calculate(_auth: AuthorizationContext, expression: str) -> JSON:
        """调用 AST 白名单计算器；不使用 Python eval。"""

        return {"expression": expression, "value": _safe_calculate(expression)}

    def save(
        auth: AuthorizationContext,
        title: str,
        content: str,
        idempotency_key: str,
        dry_run: bool,
        confirmed: bool,
    ) -> JSON:
        """预演或保存草稿；确认与幂等由进入 handler 前的 Runtime 保证。"""

        if dry_run:
            # 预演只返回足以让用户确认的信息，不把完整正文再复制到 observation。
            return {"mode": "dry_run", "would_save": {"title": title, "content_length": len(content)}}
        # 到达这里意味着 Schema 合法、Scope 足够、confirmed=true 且不存在可重放
        # 的幂等结果。生产数据库应把草稿写入和幂等记录放入同一个事务。
        draft_id = f"draft-{uuid4().hex[:12]}"
        store.drafts[draft_id] = {
            "draft_id": draft_id,
            "tenant_id": auth.tenant_id,
            "user_id": auth.user_id,
            "title": title,
            "content": content,
            "state": "saved",
        }
        return {"mode": "committed", "draft_id": draft_id, "state": "saved"}

    def status(auth: AuthorizationContext, draft_id: str) -> JSON:
        """按 tenant + user 查询状态，并故意省略草稿正文。"""

        draft = store.drafts.get(draft_id)
        if draft is None or draft["tenant_id"] != auth.tenant_id or draft["user_id"] != auth.user_id:
            raise ToolFailure("DRAFT_NOT_FOUND", "草稿不存在或无权查看")
        return {"draft_id": draft_id, "state": draft["state"], "title": draft["title"]}

    # 所有输入对象关闭额外字段，模型把 document_id 拼成 documentId 时会立即失败，
    # 而不是静默忽略并执行一个与预期不同的调用。
    closed = {"type": "object", "additionalProperties": False}
    return {
        "search_documents": ToolSpec(
            "search_documents",
            "按关键词搜索当前租户文档；只读，不返回正文。",
            {**closed, "properties": {"query": {"type": "string", "minLength": 1}, "limit": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5}}, "required": ["query"]},
            OBJECT,
            search,
            required_scopes=frozenset({"documents.read"}),
        ),
        "read_document": ToolSpec(
            "read_document",
            "读取当前租户中的指定文档；只读。",
            {**closed, "properties": {"document_id": {"type": "string", "pattern": "^doc-[0-9]+$"}}, "required": ["document_id"]},
            OBJECT,
            read,
            required_scopes=frozenset({"documents.read"}),
        ),
        "calculate": ToolSpec(
            "calculate",
            "计算仅包含数字、括号和 + - * / 的算术表达式；只读。",
            {**closed, "properties": {"expression": {"type": "string", "minLength": 1, "maxLength": 100}}, "required": ["expression"]},
            OBJECT,
            calculate,
            required_scopes=frozenset({"calculate.use"}),
        ),
        "save_draft": ToolSpec(
            "save_draft",
            "保存用户草稿。先 dry_run；真实写入要求 confirmed=true、稳定幂等键及 drafts.write。",
            {
                **closed,
                "properties": {
                    "title": {"type": "string", "minLength": 1, "maxLength": 100},
                    "content": {"type": "string", "minLength": 1, "maxLength": 5000},
                    "idempotency_key": {"type": "string", "pattern": "^[A-Za-z0-9_-]{16,128}$"},
                    "dry_run": {"type": "boolean"},
                    "confirmed": {"type": "boolean"},
                },
                "required": ["title", "content", "idempotency_key", "dry_run", "confirmed"],
            },
            OBJECT,
            save,
            required_scopes=frozenset({"drafts.write"}),
            side_effect=True,
        ),
        "get_draft_status": ToolSpec(
            "get_draft_status",
            "查询当前用户草稿状态；只读，不返回正文。",
            {**closed, "properties": {"draft_id": {"type": "string", "pattern": "^draft-[a-f0-9]{12}$"}}, "required": ["draft_id"]},
            OBJECT,
            status,
            required_scopes=frozenset({"drafts.read"}),
        ),
    }


def _json_path(path: Any) -> str:
    """把 jsonschema 的 deque 路径转换成适合错误详情展示的 JSONPath。"""

    return "$" + "".join(f"[{item}]" if isinstance(item, int) else f".{item}" for item in path)


def _business_fingerprint(arguments: JSON) -> str:
    """对真正决定副作用的参数生成稳定摘要。

    dry_run/confirmed 是执行控制，idempotency_key 是意图标识；三者变化都不应把
    同一标题和正文变成不同业务动作，因此从指纹中排除。
    """

    business = {key: value for key, value in arguments.items() if key not in {"dry_run", "confirmed", "idempotency_key"}}
    return hashlib.sha256(json.dumps(business, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


def _safe_calculate(expression: str) -> int | float:
    """用 AST 白名单计算，禁止名称、属性、调用和 Python 任意代码执行。"""

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ToolFailure("INVALID_EXPRESSION", "算术表达式语法错误") from exc

    def visit(node: ast.AST) -> int | float:
        """递归解释允许节点；没有显式分支的 AST 类型一律拒绝。"""

        if isinstance(node, ast.Expression):
            return visit(node.body)
        # 使用 type(...) 而不是 isinstance(..., int)，避免把 bool 当作数字接受。
        if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            try:
                return _OPS[type(node.op)](visit(node.left), visit(node.right))
            except ZeroDivisionError as exc:
                raise ToolFailure("DIVISION_BY_ZERO", "除数不能为零") from exc
        # Name、Call、Attribute、Subscript、幂运算等都会到达这里，因而表达式无法
        # 读取变量、导入模块、调用函数或构造超大指数运算。
        raise ToolFailure("UNSAFE_EXPRESSION", "表达式包含不允许的语法")

    value = visit(tree)
    # 即使语法安全，也限制结果规模，避免后续序列化或业务层处理异常巨大数值。
    if abs(value) > 1e15:
        raise ToolFailure("RESULT_OUT_OF_RANGE", "计算结果超出允许范围")
    return value
