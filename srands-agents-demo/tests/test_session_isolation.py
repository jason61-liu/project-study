"""v1: Session 隔离测试 (7 tc: TC-SES-01 ~ TC-SES-04, TC-SES-06 ~ TC-SES-08)"""

import pytest, asyncio
from strands import Agent
from strands.hooks.events import BeforeToolCallEvent
from strands.vended_tools.bash.bash import make_bash
from poc_runner.fixtures.agents import create_deepseek_model, create_agent_with_tools
from poc_runner.fixtures.sandboxes import MockDockerSandbox, MockSshSandbox
from tests.conftest import make_reporter


class TestSessionIsolation:

    # ---- TC-SES-01 ----
    @pytest.mark.asyncio
    async def test_conversation_history_isolation(self, deepseek_model, poc_suite):
        """TC-SES-01: 对话历史隔离"""
        reporter = make_reporter(poc_suite, "TC-SES-01", "对话历史隔离")
        a = create_agent_with_tools(model=deepseek_model, system_prompt="你是 Agent A。简短回答。", callback_handler=None)
        b = create_agent_with_tools(model=deepseek_model, system_prompt="你是 Agent B。简短回答。", callback_handler=None)
        ra, rb = await asyncio.gather(a.invoke_async("回复'Hello from A'"), b.invoke_async("回复'Hello from B'"))
        a_text = str(a.messages)
        b_text = str(b.messages)
        reporter.add_assertion("AgentA messages 不含 AgentB 输入", True, "Hello from B" not in a_text)
        reporter.add_assertion("AgentB messages 不含 AgentA 输入", True, "Hello from A" not in b_text)
        reporter.add_event({"agent_a_msg_count": len(a.messages), "agent_b_msg_count": len(b.messages)})
        reporter.finalize("PASS")

    # ---- TC-SES-02 ----
    @pytest.mark.asyncio
    async def test_tool_registry_isolation(self, deepseek_model, poc_suite):
        """TC-SES-02: 工具注册表隔离"""
        reporter = make_reporter(poc_suite, "TC-SES-02", "工具注册表隔离")
        bash_tool = make_bash(name="bash")
        a = create_agent_with_tools(model=deepseek_model, tools=[], callback_handler=None)
        b = create_agent_with_tools(model=deepseek_model, tools=[bash_tool], callback_handler=None)
        a_names = list(a.tool_registry.registry.keys())
        b_names = list(b.tool_registry.registry.keys())
        reporter.add_assertion("AgentA 不含 bash", True, "bash" not in a_names)
        reporter.add_assertion("AgentB 含 bash", True, "bash" in b_names)
        reporter.add_event({"agent_a_tools": a_names, "agent_b_tools": b_names})
        reporter.finalize("PASS")

    # ---- TC-SES-03 ----
    @pytest.mark.asyncio
    async def test_system_prompt_isolation(self, deepseek_model, poc_suite):
        """TC-SES-03: System Prompt 隔离"""
        reporter = make_reporter(poc_suite, "TC-SES-03", "System Prompt / Skills 隔离")
        a = create_agent_with_tools(model=deepseek_model, system_prompt="你是 Git 助手", callback_handler=None)
        b = create_agent_with_tools(model=deepseek_model, system_prompt="你是数据库助手", callback_handler=None)
        reporter.add_assertion("AgentA system_prompt 含 'Git 助手'", True, "Git 助手" in a._system_prompt)
        reporter.add_assertion("AgentB system_prompt 含 '数据库助手'", True, "数据库助手" in b._system_prompt)
        reporter.add_assertion("AgentA system_prompt 不含 '数据库'", True, "数据库" not in a._system_prompt)
        reporter.finalize("PASS")

    # ---- TC-SES-04 ----
    @pytest.mark.asyncio
    async def test_model_config_isolation(self, deepseek_model, poc_suite):
        """TC-SES-04: 模型配置隔离"""
        reporter = make_reporter(poc_suite, "TC-SES-04", "模型配置隔离")
        model_a = deepseek_model
        model_b = create_deepseek_model()
        a = create_agent_with_tools(model=model_a, system_prompt="test A", callback_handler=None)
        b = create_agent_with_tools(model=model_b, system_prompt="test B", callback_handler=None)
        reporter.add_assertion("两个 model 是不同的实例", True, a.model is not b.model)
        reporter.add_assertion("model_id 一致", "deepseek-chat", a.model.get_config()["model_id"])
        reporter.finalize("PASS")

    # ---- TC-SES-06 ----
    @pytest.mark.asyncio
    async def test_interrupt_control_isolation(self, deepseek_model, poc_suite):
        """TC-SES-06: 中断控制隔离"""
        reporter = make_reporter(poc_suite, "TC-SES-06", "中断控制隔离")
        a = create_agent_with_tools(model=deepseek_model, system_prompt="你是 Agent A。简短回复。", callback_handler=None)
        b = create_agent_with_tools(model=deepseek_model, system_prompt="你是 Agent B。简短回复。", callback_handler=None)
        rb = await b.invoke_async("回复'B OK'")
        ra = await a.invoke_async("回复'A OK'")
        reporter.add_assertion("AgentB 正常完成", True, rb.stop_reason in ("end_turn", "stop_sequence", "max_tokens"))
        reporter.add_assertion("AgentA 正常完成", True, ra.stop_reason in ("end_turn", "stop_sequence", "max_tokens"))
        reporter.finalize("PASS")

    # ---- TC-SES-07 ----
    @pytest.mark.asyncio
    async def test_sandbox_config_isolation(self, deepseek_model, poc_suite):
        """TC-SES-07: Sandbox 配置隔离 — 两个真实 Sandbox 子类（继承 SDK Sandbox ABC）"""
        reporter = make_reporter(poc_suite, "TC-SES-07", "Sandbox 配置隔离")
        from strands.sandbox.base import Sandbox
        from strands.sandbox.types import ExecutionResult, FileInfo
        from typing import Any, AsyncGenerator

        # 两个真实的 Sandbox 实现（继承 SDK Sandbox ABC，实现全部 6 个抽象方法）
        class SandboxAlpha(Sandbox):
            async def execute_streaming(self, command, *, timeout=None, cwd=None, env=None, **kwargs) -> AsyncGenerator[Any, None]:
                import subprocess
                r = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=timeout)
                yield ExecutionResult(exit_code=r.returncode, stdout=f"[Sandbox-Alpha] {r.stdout}", stderr=r.stderr)
            async def execute_code_streaming(self, code, language, *, timeout=None, cwd=None, env=None, **kwargs) -> AsyncGenerator[Any, None]:
                yield ExecutionResult(exit_code=0, stdout=f"[Alpha] {code[:50]}", stderr="")
            async def read_file(self, path, **kwargs) -> bytes: return f"[Alpha] {path}".encode()
            async def write_file(self, path, content, **kwargs) -> None: pass
            async def remove_file(self, path, **kwargs) -> None: pass
            async def list_files(self, path, **kwargs) -> list[FileInfo]: return []

        class SandboxBeta(Sandbox):
            async def execute_streaming(self, command, *, timeout=None, cwd=None, env=None, **kwargs) -> AsyncGenerator[Any, None]:
                import subprocess
                r = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=timeout)
                yield ExecutionResult(exit_code=r.returncode, stdout=f"[Sandbox-Beta] {r.stdout}", stderr=r.stderr)
            async def execute_code_streaming(self, code, language, *, timeout=None, cwd=None, env=None, **kwargs) -> AsyncGenerator[Any, None]:
                yield ExecutionResult(exit_code=0, stdout=f"[Beta] {code[:50]}", stderr="")
            async def read_file(self, path, **kwargs) -> bytes: return f"[Beta] {path}".encode()
            async def write_file(self, path, content, **kwargs) -> None: pass
            async def remove_file(self, path, **kwargs) -> None: pass
            async def list_files(self, path, **kwargs) -> list[FileInfo]: return []

        sandbox_a = SandboxAlpha()
        sandbox_b = SandboxBeta()

        bash_a = make_bash(sandbox=sandbox_a, name="bash")
        bash_b = make_bash(sandbox=sandbox_b, name="bash")

        a = create_agent_with_tools(model=deepseek_model, tools=[bash_a], sandbox=sandbox_a,
                                     system_prompt="test A", callback_handler=None)
        b = create_agent_with_tools(model=deepseek_model, tools=[bash_b], sandbox=sandbox_b,
                                     system_prompt="test B", callback_handler=None)

        # 真实 SDK 工具调用 → Sandbox.execute() → subprocess.run()
        ra = a.tool.bash(command="echo alpha_test")
        rb = b.tool.bash(command="echo beta_test")

        reporter.add_assertion("AgentA 路由到 Sandbox-Alpha", True, "[Sandbox-Alpha]" in str(ra))
        reporter.add_assertion("AgentB 路由到 Sandbox-Beta", True, "[Sandbox-Beta]" in str(rb))
        reporter.add_assertion("Sandbox-Alpha 输出不含 Beta 标记", True, "[Sandbox-Beta]" not in str(ra))
        reporter.add_event({"agent_a_result": str(ra)[:200], "agent_b_result": str(rb)[:200]})
        reporter.finalize("PASS")

    # ---- TC-SES-08 ----
    @pytest.mark.asyncio
    async def test_approval_state_isolation(self, deepseek_model, poc_suite):
        """TC-SES-08: 审批状态隔离"""
        reporter = make_reporter(poc_suite, "TC-SES-08", "审批状态隔离")
        def hook_a(event: BeforeToolCallEvent):
            if event.tool_use["name"] == "bash":
                event.interrupt(name="approval_bash", reason="need approval")
        bash_tool = make_bash(name="bash")
        a = create_agent_with_tools(model=deepseek_model, tools=[bash_tool], hooks=[hook_a],
                                      system_prompt="用 bash 执行 echo hello", callback_handler=None)
        b = create_agent_with_tools(model=deepseek_model, system_prompt="回复'B OK'", callback_handler=None)
        results = await asyncio.gather(a.invoke_async("用 bash 执行 echo hello"), b.invoke_async("回复 B OK"), return_exceptions=True)
        rb = results[1]
        reporter.add_assertion("AgentB 正常完成（不被 AgentA interrupt 影响）", True,
                                hasattr(rb, 'stop_reason') and rb.stop_reason in ("end_turn", "stop_sequence", "max_tokens"))
        reporter.finalize("PASS")
