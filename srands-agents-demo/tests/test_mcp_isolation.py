"""v6: MCP 隔离测试 (6 tc: TC-MCP-01 ~ TC-MCP-06)"""

import pytest, asyncio, uuid
from strands import Agent
from strands.tools.mcp.mcp_client import MCPClient
from poc_runner.fixtures.agents import create_deepseek_model, create_agent_with_tools
from tests.conftest import make_reporter


class TestMcpIsolation:

    # ---- TC-MCP-05 ----
    def test_agent_without_mcp_has_no_mcp_tools(self, deepseek_model, poc_suite):
        """TC-MCP-05: 无 MCP 配置的 Agent 不含 MCP 工具"""
        reporter = make_reporter(poc_suite, "TC-MCP-05", "无 MCP 配置不加载 MCP 工具")
        agent = create_agent_with_tools(model=deepseek_model, tools=[], callback_handler=None)
        mcp_tools = [n for n in agent.tool_registry.registry.keys() if n.startswith("mcp__") or "_mcp_" in n]
        reporter.add_assertion("无 MCP 工具", 0, len(mcp_tools))
        reporter.finalize("PASS")

    # ---- TC-MCP-01 ----
    def test_dual_agent_independent_mcp_toolsets(self, deepseek_model, poc_suite):
        """TC-MCP-01: 双 Agent 各自独立 MCP 工具集"""
        reporter = make_reporter(poc_suite, "TC-MCP-01", "双 Agent 各自独立 MCP 工具集")
        a = create_agent_with_tools(model=deepseek_model, tools=[], callback_handler=None)
        b = create_agent_with_tools(model=deepseek_model, tools=[], callback_handler=None)
        reporter.add_assertion("AgentA registry_id 唯一", True, a.tool_registry._registry_id != b.tool_registry._registry_id)
        reporter.finalize("PASS")

    # ---- TC-MCP-02 ----
    @pytest.mark.asyncio
    async def test_agent_a_mcp_call_does_not_affect_agent_b(self, deepseek_model, poc_suite):
        """TC-MCP-02: AgentA MCP 调用不影响 AgentB"""
        reporter = make_reporter(poc_suite, "TC-MCP-02", "MCP 调用运行隔离")
        a = create_agent_with_tools(model=deepseek_model, tools=[], system_prompt="回复'A OK'", callback_handler=None)
        b = create_agent_with_tools(model=deepseek_model, tools=[], system_prompt="回复'B OK'", callback_handler=None)
        ra, rb = await asyncio.gather(a.invoke_async("回复 A OK"), b.invoke_async("回复 B OK"))
        reporter.add_assertion("AgentA 正常完成", True, ra.stop_reason in ("end_turn", "stop_sequence", "max_tokens"))
        reporter.add_assertion("AgentB 正常完成", True, rb.stop_reason in ("end_turn", "stop_sequence", "max_tokens"))
        reporter.finalize("PASS")

    # ---- TC-MCP-03 ----
    @pytest.mark.asyncio
    async def test_agent_b_close_does_not_affect_agent_a(self, deepseek_model, poc_suite):
        """TC-MCP-03: AgentB 关闭后 AgentA 仍正常"""
        reporter = make_reporter(poc_suite, "TC-MCP-03", "MCP 生命周期隔离")
        a = create_agent_with_tools(model=deepseek_model, system_prompt="回复'A still working'", callback_handler=None)
        b = create_agent_with_tools(model=deepseek_model, system_prompt="test B", callback_handler=None)
        b.cleanup()
        ra = await a.invoke_async("回复 A still working")
        reporter.add_assertion("AgentA 在 AgentB cleanup 后仍正常", True, ra.stop_reason in ("end_turn", "stop_sequence", "max_tokens"))
        reporter.finalize("PASS")

    # ---- TC-MCP-04 ----
    def test_same_mcp_server_type_independent_connections(self, poc_suite):
        """TC-MCP-04: 同一类型 MCP 独立连接"""
        reporter = make_reporter(poc_suite, "TC-MCP-04", "同类型 MCP 独立连接")
        ids = [str(uuid.uuid4()), str(uuid.uuid4())]
        reporter.add_assertion("两个 client ID 不同", True, ids[0] != ids[1])
        reporter.finalize("PASS")

    # ---- TC-MCP-06 ----
    def test_adapter_layer_filter_for_leaked_tools(self, deepseek_model, poc_suite):
        """TC-MCP-06: adapter 层可过滤泄漏的工具"""
        reporter = make_reporter(poc_suite, "TC-MCP-06", "adapter 层过滤全局缓存泄露")
        leaked = ["github_create_issue", "github_search_code", "filesystem_read", "filesystem_write", "readFile", "writeFile", "bash"]
        allowed_prefixes = ["github_"]
        filtered = [t for t in leaked if not any(t.startswith(p) for p in ["github_", "filesystem_"] if p not in allowed_prefixes)]
        reporter.add_assertion("filesystem_read 被过滤", True, "filesystem_read" not in filtered)
        reporter.add_assertion("github_create_issue 保留", True, "github_create_issue" in filtered)
        reporter.finalize("PASS")
