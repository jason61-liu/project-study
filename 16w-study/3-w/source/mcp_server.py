"""只读 MCP Server：暴露 2 个 Tools 和 1 个 Resource。

本模块是协议适配层，不重复实现搜索和读取业务。所有实际调用仍进入 ToolRuntime，
以复用 Schema、Scope、Deadline、结构化错误和日志。Server 的核心安全约束是：
原始 Token 在 Host 侧验证后即被丢弃，只把 tenant/user 组成的脱敏上下文带入 MCP。
"""

from __future__ import annotations

import json
import os
from typing import Any

import anyio
from mcp.server.fastmcp import Context, FastMCP
from mcp.types import ToolAnnotations

from tool_runtime import AuthorizationContext, InMemoryStore, TokenService, ToolFailure, ToolRuntime


READ_ONLY_SCOPES = frozenset({"documents.read", "drafts.read"})


def build_mcp_server(runtime: ToolRuntime, host_auth: AuthorizationContext) -> FastMCP:
    """建立仅暴露只读能力的 Server。

    ``host_auth`` 必须由 Host 在 MCP 连接外验证用户 Token 后构造。每次调用只透传
    tenant_id/user_id，MCP 消息、工具 Schema、结果和日志中都没有原始 Token。
    stdio Server 与启动它的 Host 构成受信边界；远程 HTTP 部署应改用 MCP OAuth。
    """

    # MCP Server 的 instructions 是面向客户端/模型的能力说明，不是安全策略。
    # 真正的只读边界来自“只注册只读函数”和 Runtime 的授权检查。
    server = FastMCP(
        "week3-readonly",
        instructions="只读文档服务；所有结果按 tenant_id/user_id 隔离。",
        log_level="WARNING",
    )

    @server.tool(
        name="search_documents",
        description="搜索当前授权租户的文档标题；只读。wait_ms 仅用于演示取消。",
        annotations=ToolAnnotations(
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=False,
        ),
        structured_output=True,
    )
    async def search_documents(query: str, limit: int = 5, wait_ms: int = 0, ctx: Context | None = None) -> dict[str, Any]:
        """搜索当前租户的文档标题，并支持教学用途的可取消等待。"""

        # wait_ms 属于协议测试参数，限制上界避免客户端用一个请求长期占用任务。
        if wait_ms < 0 or wait_ms > 2_000:
            return _adapter_error("INVALID_ARGUMENTS", "wait_ms 必须在 0..2000")
        # 使用可取消的异步等待，客户端取消请求时不会继续占用 Server 任务。
        await anyio.sleep(wait_ms / 1_000)
        # 从 MCP Request Context 读取脱敏身份并与 Host 会话绑定值比较。通过后只把
        # AuthorizationContext 交给 Runtime；查询参数里从来没有 token。
        auth = _authorization_from_context(ctx, host_auth)
        return runtime.invoke_authorized("search_documents", {"query": query, "limit": limit}, auth=auth)

    @server.tool(
        name="read_document",
        description="读取当前授权租户中的指定文档；只读。",
        annotations=ToolAnnotations(
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=False,
        ),
        structured_output=True,
    )
    async def read_document(document_id: str, ctx: Context | None = None) -> dict[str, Any]:
        """读取单个文档；tenant 隔离和 ID Schema 均由 Runtime 强制执行。"""

        auth = _authorization_from_context(ctx, host_auth)
        return runtime.invoke_authorized("read_document", {"document_id": document_id}, auth=auth)

    @server.resource(
        "catalog://documents",
        name="document_catalog",
        description="当前授权租户可见的文档目录，不包含正文和 Token。",
        mime_type="application/json",
    )
    async def document_catalog() -> str:
        """返回当前 Host 授权租户的静态目录 Resource，不泄露文档正文。"""

        # Resource 不接受模型参数，因此不能由调用方在 URI 中切换 tenant。
        # tenant_id 固定来自已经验证的 host_auth。
        documents = runtime.store.documents.get(host_auth.tenant_id, {})
        payload = {
            "tenant_id": host_auth.tenant_id,
            "user_id": host_auth.user_id,
            "items": [{"id": doc["id"], "title": doc["title"]} for doc in documents.values()],
        }
        return json.dumps(payload, ensure_ascii=False)

    return server


def _authorization_from_context(ctx: Context | None, fallback: AuthorizationContext) -> AuthorizationContext:
    """读取 Client request `_meta` 中的脱敏身份，并阻止其越过 Host 会话边界。"""

    # FastMCP 把 tools/call 的 `_meta` 解析到 request_context.meta。model_dump()
    # 同时保留标准字段和 tenant_id/user_id 这类扩展字段。
    raw: dict[str, Any] = {}
    if ctx is not None and ctx.request_context.meta is not None:
        raw = ctx.request_context.meta.model_dump()
    tenant_id = raw.get("tenant_id", fallback.tenant_id)
    user_id = raw.get("user_id", fallback.user_id)
    # `_meta` 由客户端发送，所以不能仅凭它提升身份。它只能与 Host 已验证的会话
    # 身份一致；任何 tenant/user 变化都按越权处理。
    if tenant_id != fallback.tenant_id or user_id != fallback.user_id:
        raise ToolFailure("AUTH_CONTEXT_MISMATCH", "MCP 请求身份与已验证 Host 会话不一致")
    return AuthorizationContext(user_id, tenant_id, fallback.scopes, fallback.token_id)


def _adapter_error(code: str, message: str) -> dict[str, Any]:
    """把 MCP 适配层自己的参数错误转换成与 Runtime 一致的结果信封。"""

    return {
        "ok": False,
        "status": "business_failure",
        "data": None,
        "error": {"code": code, "message": message, "retryable": False, "details": {}},
        "meta": {},
    }


def _server_from_environment() -> FastMCP:
    """stdio 启动入口：身份由已完成 OAuth 验证的 Host 通过进程配置注入。"""

    # 只有启动 stdio 子进程的可信 Host 能控制这些值。若 Server 暴露到网络，环境
    # 变量无法代表每个请求的用户，必须换成 MCP OAuth TokenVerifier。
    tenant_id = os.environ.get("MCP_TENANT_ID", "tenant-a")
    user_id = os.environ.get("MCP_USER_ID", "user-demo")
    auth = AuthorizationContext(user_id, tenant_id, READ_ONLY_SCOPES, "host-verified")
    runtime = ToolRuntime(TokenService(b"week3-demo-secret-must-be-at-least-32-bytes"), InMemoryStore.sample())
    return build_mcp_server(runtime, auth)


if __name__ == "__main__":
    # stdio Transport 要求 stdout 只输出协议消息；普通日志应写 stderr。
    _server_from_environment().run(transport="stdio")
