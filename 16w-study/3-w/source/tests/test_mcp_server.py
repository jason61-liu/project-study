"""通过真实 MCP ClientSession 验证发现、调用、Resource、错误和取消。"""

from __future__ import annotations

import asyncio
import json

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from mcp_server import READ_ONLY_SCOPES, build_mcp_server
from tool_runtime import AuthorizationContext, InMemoryStore, TokenService, ToolRuntime


@pytest.fixture
def mcp_services():
    """创建绑定 tenant-a/user-1 的只读 Server 和独立 Runtime。"""

    runtime = ToolRuntime(TokenService(b"mcp-test-secret-that-is-at-least-32-bytes"), InMemoryStore.sample())
    auth = AuthorizationContext("user-1", "tenant-a", READ_ONLY_SCOPES, "verified-token-id")
    server = build_mcp_server(runtime, auth)
    yield runtime, server
    runtime.close()


@pytest.mark.anyio
async def test_client_discovers_exactly_two_readonly_tools_and_one_resource(mcp_services) -> None:
    """发现结果必须从结构上证明 Server 只读，而不是只依赖描述文字。"""

    _runtime, server = mcp_services
    async with create_connected_server_and_client_session(server) as session:
        tools = await session.list_tools()
        resources = await session.list_resources()

    assert {tool.name for tool in tools.tools} == {"search_documents", "read_document"}
    assert {str(resource.uri) for resource in resources.resources} == {"catalog://documents"}
    assert "save_draft" not in {tool.name for tool in tools.tools}
    assert all(tool.annotations.readOnlyHint is True for tool in tools.tools)
    assert all(tool.annotations.destructiveHint is False for tool in tools.tools)
    assert "token" not in json.dumps([tool.inputSchema for tool in tools.tools]).lower()


@pytest.mark.anyio
async def test_client_calls_tools_with_sanitized_authorization_context(mcp_services) -> None:
    """验证 `_meta` 身份能到达 Runtime，同时 Schema 和结果中没有 Token。"""

    _runtime, server = mcp_services
    meta = {"tenant_id": "tenant-a", "user_id": "user-1"}
    async with create_connected_server_and_client_session(server) as session:
        result = await session.call_tool("search_documents", {"query": "MCP", "limit": 5}, meta=meta)

    assert result.isError is False
    assert result.structuredContent["data"]["items"][0]["id"] == "doc-1"
    assert result.structuredContent["meta"]["tenant_id"] == "tenant-a"
    assert "token" not in json.dumps(result.structuredContent).lower()


@pytest.mark.anyio
async def test_client_reads_tenant_filtered_resource(mcp_services) -> None:
    """Resource 目录只能列出 Host 会话授权租户的 doc-1/doc-2。"""

    _runtime, server = mcp_services
    async with create_connected_server_and_client_session(server) as session:
        result = await session.read_resource("catalog://documents")

    payload = json.loads(result.contents[0].text)
    assert payload["tenant_id"] == "tenant-a"
    assert {item["id"] for item in payload["items"]} == {"doc-1", "doc-2"}
    assert "doc-3" not in result.contents[0].text
    assert "token" not in result.contents[0].text.lower()


@pytest.mark.anyio
async def test_client_receives_schema_and_unknown_tool_errors(mcp_services) -> None:
    """分别验证已注册 Tool 的参数错误和未注册写 Tool 的协议错误。"""

    _runtime, server = mcp_services
    async with create_connected_server_and_client_session(server) as session:
        invalid = await session.call_tool("read_document", {"document_id": 123})
        unknown = await session.call_tool("save_draft", {})

    assert invalid.isError is True
    assert unknown.isError is True


@pytest.mark.anyio
async def test_mcp_request_can_be_cancelled_and_session_remains_usable(mcp_services) -> None:
    """取消慢请求后再读取文档，证明取消单个 request 不会摧毁 MCP Session。"""

    _runtime, server = mcp_services
    async with create_connected_server_and_client_session(server) as session:
        # asyncio.wait_for 取消 Client 的等待；MCP SDK 将请求取消传播给 Server，
        # Server 中的 anyio.sleep 是协作式取消点，不会继续占用整个等待时间。
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                session.call_tool("search_documents", {"query": "MCP", "wait_ms": 1_000}),
                timeout=0.03,
            )
        recovered = await session.call_tool("read_document", {"document_id": "doc-1"})

    assert recovered.isError is False
    assert recovered.structuredContent["data"]["id"] == "doc-1"


@pytest.mark.anyio
async def test_mcp_rejects_context_escalation(mcp_services) -> None:
    """客户端不能通过伪造 `_meta.tenant_id` 把 tenant-a 会话升级为 tenant-b。"""

    _runtime, server = mcp_services
    async with create_connected_server_and_client_session(server) as session:
        result = await session.call_tool(
            "search_documents",
            {"query": "隔离"},
            meta={"tenant_id": "tenant-b", "user_id": "user-1"},
        )

    assert result.isError is True
