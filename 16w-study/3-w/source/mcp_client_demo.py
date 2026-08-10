"""使用官方 Python MCP Client 通过 stdio 完成发现、调用与 Resource 读取。

该脚本不是直接调用 Server 内部函数，而是启动独立子进程，通过 stdio 交换真正的
MCP 消息。它因此可以同时验证进程启动、initialize、能力发现、tools/call 和
resources/read 的协议链路。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys

from mcp import ClientSession, StdioServerParameters, stdio_client


async def main() -> None:
    """启动只读 Server，建立 ClientSession，并执行最小验收轨迹。"""

    server_file = Path(__file__).with_name("mcp_server.py")
    # stdio 模式由 Client/Host 启动 Server 子进程。环境变量是本地教学场景下的
    # 已验证会话上下文，不包含原始 Access Token；远程部署应使用 MCP OAuth。
    parameters = StdioServerParameters(
        command=sys.executable,
        args=[str(server_file)],
        env={"MCP_TENANT_ID": "tenant-a", "MCP_USER_ID": "user-demo"},
    )
    # 第一层上下文管理器管理子进程和字节流，第二层管理 MCP 会话。initialize()
    # 完成协议协商后才能执行 list/call/read 等操作。
    async with stdio_client(parameters) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            # 发现结果来自 Server 注册表，不依赖模型猜测。只读 Server 应只出现
            # 两个 Tool，并且 save_draft 永远不会出现在这里。
            tools = await session.list_tools()
            resources = await session.list_resources()
            print("tools:", [tool.name for tool in tools.tools])
            print("resources:", [str(resource.uri) for resource in resources.resources])

            # Client 只透传脱敏身份；Server 会与连接创建时绑定的 Host 身份比较。
            # Token 既不属于模型参数，也不属于这条 tools/call 消息。
            meta = {"tenant_id": "tenant-a", "user_id": "user-demo"}
            result = await session.call_tool("search_documents", {"query": "MCP"}, meta=meta)
            print("search:", json.dumps(result.structuredContent, ensure_ascii=False, indent=2))

            # Resource 与 Tool 使用相同租户边界，但 resources/read 当前没有业务参数，
            # 因而直接使用创建 Server 时绑定的 AuthorizationContext 过滤目录。
            resource = await session.read_resource("catalog://documents")
            print("catalog:", resource.contents[0].text)


if __name__ == "__main__":
    asyncio.run(main())
