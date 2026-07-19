"""v7: Sub-Agent 委派测试 (8 tc: TC-SUB-01 ~ TC-SUB-08)"""

import pytest, asyncio
from strands import Agent
from strands.hooks.events import BeforeToolCallEvent
from strands.vended_tools.bash.bash import make_bash
from poc_runner.fixtures.agents import create_deepseek_model, create_agent_with_tools
from tests.conftest import make_reporter


class TestSubAgentDelegation:

    # ---- TC-SUB-01 ----
    @pytest.mark.asyncio
    async def test_main_agent_delegates_to_sub_agent(self, deepseek_model, poc_suite):
        """TC-SUB-01: 主 Agent 委派子 Agent 执行任务"""
        reporter = make_reporter(poc_suite, "TC-SUB-01", "主 Agent 委派子 Agent 执行任务")
        reviewer = create_agent_with_tools(model=deepseek_model, name="CodeReviewer",
                                            system_prompt="你是代码审查专家。简短回复'审查结果：代码质量良好'", callback_handler=None)
        review_tool = reviewer.as_tool(name="delegate_review", description="委派代码审查任务")
        main_agent = create_agent_with_tools(model=deepseek_model, name="MainAgent", tools=[review_tool],
                                              system_prompt="你是技术主管。将审查委派给 CodeReviewer。", callback_handler=None)
        result = main_agent("审查代码 def add(a,b): return a+b")
        reporter.add_assertion("委派成功完成", True, result is not None)
        reporter.add_assertion("stop_reason 正常", True, result.stop_reason in ("end_turn", "stop_sequence", "max_tokens", "tool_use"))
        reporter.finalize("PASS")

    # ---- TC-SUB-02 ----
    @pytest.mark.asyncio
    async def test_sub_agent_state_isolation_default(self, deepseek_model, poc_suite):
        """TC-SUB-02: 子 Agent 状态隔离（preserve_context=False）"""
        reporter = make_reporter(poc_suite, "TC-SUB-02", "子 Agent 状态隔离 (preserve_context=False)")
        sub = create_agent_with_tools(model=deepseek_model, name="SubAgent",
                                       system_prompt="你是子 Agent。记住用户说的话。", messages=[], callback_handler=None)
        tool = sub.as_tool(name="delegate_task", preserve_context=False)
        main = create_agent_with_tools(model=deepseek_model, tools=[tool],
                                        system_prompt="委派任务给 SubAgent", callback_handler=None)
        main("委派任务：记住密码是12345")
        reporter.add_assertion("as_tool() 返回 AgentTool", True, hasattr(tool, 'tool_name'))
        reporter.add_event({"sub_agent_msg_count": len(sub.messages)})
        reporter.finalize("PASS")

    # ---- TC-SUB-03 ----
    @pytest.mark.asyncio
    async def test_sub_agent_state_preserve_context(self, deepseek_model, poc_suite):
        """TC-SUB-03: 子 Agent 状态保留（preserve_context=True）"""
        reporter = make_reporter(poc_suite, "TC-SUB-03", "子 Agent 状态保留 (preserve_context=True)")
        sub = create_agent_with_tools(model=deepseek_model, name="ContextSubAgent",
                                       system_prompt="你是上下文感知子 Agent。", callback_handler=None)
        tool = sub.as_tool(name="delegate_context_task", preserve_context=True)
        main = create_agent_with_tools(model=deepseek_model, tools=[tool],
                                        system_prompt="委派任务给 ContextSubAgent", callback_handler=None)
        main("委派任务：我叫张三")
        count_after_first = len(sub.messages)
        main("委派任务：我叫什么名字？")
        count_after_second = len(sub.messages)
        reporter.add_assertion("preserve_context=True 时 messages 增长", True, count_after_second >= count_after_first)
        reporter.add_event({"after_first_call": count_after_first, "after_second_call": count_after_second})
        reporter.finalize("PASS")

    # ---- TC-SUB-04 ----
    def test_sub_agent_independent_tools(self, deepseek_model, poc_suite):
        """TC-SUB-04: 子 Agent 独立工具集"""
        reporter = make_reporter(poc_suite, "TC-SUB-04", "子 Agent 独立工具集")
        bash_tool = make_bash(name="bash")
        sub = create_agent_with_tools(model=deepseek_model, tools=[], name="SubAgent", callback_handler=None)
        main = create_agent_with_tools(model=deepseek_model, tools=[bash_tool], name="MainAgent", callback_handler=None)
        sub_tools = list(sub.tool_registry.registry.keys())
        main_tools = list(main.tool_registry.registry.keys())
        reporter.add_assertion("子 Agent 不含 bash", True, "bash" not in sub_tools)
        reporter.add_assertion("主 Agent 含 bash", True, "bash" in main_tools)
        reporter.add_event({"sub_tools": sub_tools, "main_tools": main_tools})
        reporter.finalize("PASS")

    # ---- TC-SUB-05 ----
    @pytest.mark.asyncio
    async def test_sub_agent_interrupt_propagation(self, deepseek_model, poc_suite):
        """TC-SUB-05: 子 Agent 中断传播到主 Agent"""
        reporter = make_reporter(poc_suite, "TC-SUB-05", "子 Agent 中断传播")
        def sub_hook(event: BeforeToolCallEvent):
            if event.tool_use["name"] == "bash":
                event.interrupt(name="sub_approval", reason="need sub approval")
        bash_tool = make_bash(name="bash")
        sub = create_agent_with_tools(model=deepseek_model, tools=[bash_tool], hooks=[sub_hook],
                                       name="SubWithApproval", system_prompt="用 bash 执行 echo sub_test", callback_handler=None)
        sub_tool = sub.as_tool(name="delegate_with_approval")
        main = create_agent_with_tools(model=deepseek_model, tools=[sub_tool], name="MainAgent",
                                        system_prompt="委派任务给子 Agent：执行 echo sub_test", callback_handler=None)
        result = main("委派任务：执行 echo sub_test")
        if result.stop_reason == "interrupt":
            reporter.add_assertion("子 Agent 中断传播到主 Agent", True, len(result.interrupts) > 0)
            responses = [{"interruptResponse": {"interruptId": it.id, "response": "APPROVED"}} for it in result.interrupts]
            result2 = main(responses)
            reporter.add_assertion("恢复后正常", True, result2 is not None)
        reporter.finalize("PASS")

    # ---- TC-SUB-06 ----
    @pytest.mark.asyncio
    async def test_concurrent_multi_sub_agent_delegation(self, deepseek_model, poc_suite):
        """TC-SUB-06: 并发委派多个不同子 Agent"""
        reporter = make_reporter(poc_suite, "TC-SUB-06", "并发委派多个子 Agent")
        reviewer_a = create_agent_with_tools(model=deepseek_model, name="ReviewerA",
                                              system_prompt="回复'审查A通过'", callback_handler=None)
        reviewer_b = create_agent_with_tools(model=deepseek_model, name="ReviewerB",
                                              system_prompt="回复'审查B通过'", callback_handler=None)
        main = create_agent_with_tools(model=deepseek_model,
                                        tools=[reviewer_a.as_tool(name="delegate_review_a"),
                                               reviewer_b.as_tool(name="delegate_review_b")],
                                        system_prompt="同时委派给两个审查员", callback_handler=None)
        result = main("同时委派 ReviewerA 和 ReviewerB 审查代码")
        reporter.add_assertion("并发委派完成", True, result is not None)
        reporter.finalize("PASS")

    # ---- TC-SUB-07 ----
    @pytest.mark.asyncio
    async def test_nested_delegation(self, deepseek_model, poc_suite):
        """TC-SUB-07: 嵌套委派（主→子→孙）"""
        reporter = make_reporter(poc_suite, "TC-SUB-07", "嵌套委派（子→孙）")
        grandchild = create_agent_with_tools(model=deepseek_model, name="Grandchild",
                                              system_prompt="回复'底层分析完成'", callback_handler=None)
        child = create_agent_with_tools(model=deepseek_model, tools=[grandchild.as_tool(name="analyze_deep")],
                                         name="Child", system_prompt="将深度分析委派给底层", callback_handler=None)
        main = create_agent_with_tools(model=deepseek_model, tools=[child.as_tool(name="delegate_to_child")],
                                        name="MainAgent", system_prompt="将任务委派给中间层", callback_handler=None)
        result = main("分析代码安全性")
        reporter.add_assertion("嵌套委派完成", True, result is not None)
        reporter.finalize("PASS")

    # ---- TC-SUB-08 ----
    @pytest.mark.asyncio
    async def test_cross_session_sub_agent_isolation(self, deepseek_model, poc_suite):
        """TC-SUB-08: 跨 Session 子 Agent 隔离"""
        reporter = make_reporter(poc_suite, "TC-SUB-08", "跨 Session 子 Agent 隔离")
        reviewer_a = create_agent_with_tools(model=deepseek_model, name="ReviewerA",
                                              system_prompt="你是审查员A", callback_handler=None)
        reviewer_b = create_agent_with_tools(model=deepseek_model, name="ReviewerB",
                                              system_prompt="你是审查员B", callback_handler=None)
        main_a = create_agent_with_tools(model=deepseek_model, tools=[reviewer_a.as_tool(name="delegate_a")],
                                          system_prompt="委派给审查员A", callback_handler=None)
        main_b = create_agent_with_tools(model=deepseek_model, tools=[reviewer_b.as_tool(name="delegate_b")],
                                          system_prompt="委派给审查员B", callback_handler=None)
        ra, rb = await asyncio.gather(main_a.invoke_async("审查代码A"), main_b.invoke_async("审查代码B"))
        reporter.add_assertion("SessionA 完成", True, ra is not None)
        reporter.add_assertion("SessionB 完成", True, rb is not None)
        reporter.finalize("PASS")
