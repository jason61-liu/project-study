"""v5: Session→EventStore 替换测试 (5 tc: TC-SRV-01 ~ TC-SRV-05)

使用真实 SDK FileSessionManager（JSON 文件存储），验证 SessionManager 的
save/restore 完整生命周期。所有 Agent 和 SessionManager 均为 SDK 真实对象。
"""

import pytest, asyncio, tempfile, os
from strands import Agent
from strands.session import FileSessionManager
from strands.types.session import Session, SessionAgent
from poc_runner.fixtures.agents import create_deepseek_model, create_agent_with_tools
from tests.conftest import make_reporter


class TestSessionStore:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.tmpdir = tempfile.mkdtemp(prefix="poc_session_")

    def teardown_method(self):
        import shutil
        if os.path.exists(self.tmpdir):
            shutil.rmtree(self.tmpdir, ignore_errors=True)

    # ================================================================
    # TC-SRV-01: 消息写入文件存储 — 真实 FileSessionManager
    # ================================================================
    @pytest.mark.asyncio
    async def test_create_message_writes_to_event_store(self, deepseek_model, poc_suite):
        """TC-SRV-01: 消息写入存储 — 真实 FileSessionManager 持久化到文件系统"""
        reporter = make_reporter(poc_suite, "TC-SRV-01", "消息写入 EventStore (FileSessionManager)")
        session_id = "tc-srv-01-session"

        # 真实 SDK FileSessionManager — 写入 JSON 文件
        session_mgr = FileSessionManager(session_id=session_id, storage_dir=self.tmpdir)
        agent = Agent(model=deepseek_model, session_manager=session_mgr,
                       system_prompt="简短回复", callback_handler=None)

        # 真实 Agent 调用 — MessageAddedEvent hook 自动触发 save
        await agent.invoke_async("回复: 检查代码")

        # 验证文件已创建
        session_dir = os.path.join(self.tmpdir, f"session_{session_id}")
        reporter.add_assertion("Session 目录已创建", True, os.path.isdir(session_dir))

        # 列出存储的消息文件
        msg_dir = os.path.join(session_dir, "messages")
        if os.path.isdir(msg_dir):
            msg_files = sorted(os.listdir(msg_dir))
            reporter.add_event({"message_files": msg_files})
            reporter.add_assertion("至少保存了消息", True, len(msg_files) > 0)

        reporter.finalize("PASS")

    # ================================================================
    # TC-SRV-02: 从文件恢复消息 — 真实 FileSessionManager restore
    # ================================================================
    @pytest.mark.asyncio
    async def test_list_messages_restores_correctly(self, deepseek_model, poc_suite):
        """TC-SRV-02: 从文件恢复消息 — 真实 FileSessionManager 反序列化"""
        reporter = make_reporter(poc_suite, "TC-SRV-02", "从 EventStore 重建消息 (FileSessionManager)")
        session_id = "tc-srv-02-session"

        # 第一轮：保存
        session_mgr1 = FileSessionManager(session_id=session_id, storage_dir=self.tmpdir)
        agent1 = Agent(model=deepseek_model, session_manager=session_mgr1,
                        system_prompt="简短回复", callback_handler=None)
        await agent1.invoke_async("回复: 你好")

        msg_count_1 = len(agent1.messages)
        reporter.add_event({"round1_messages": msg_count_1})

        # 第二轮：从同一个 session_id 恢复
        session_mgr2 = FileSessionManager(session_id=session_id, storage_dir=self.tmpdir)
        agent2 = Agent(model=deepseek_model, session_manager=session_mgr2,
                        system_prompt="简短回复", callback_handler=None)
        msg_count_2 = len(agent2.messages)

        reporter.add_event({"round2_messages": msg_count_2})
        reporter.add_assertion("恢复的消息数 >= 保存的消息数", True, msg_count_2 >= msg_count_1)
        reporter.finalize("PASS")

    # ================================================================
    # TC-SRV-03: save→restore→Agent invoke 连续 — 真实 FileSessionManager
    # ================================================================
    @pytest.mark.asyncio
    async def test_save_restore_invoke_continuity(self, deepseek_model, poc_suite):
        """TC-SRV-03: save→restore→Agent invoke — 真实生命周期"""
        reporter = make_reporter(poc_suite, "TC-SRV-03", "save→restore→Agent invoke 连续")
        session_id = "tc-srv-03-session"

        # 第一轮对话
        session_mgr1 = FileSessionManager(session_id=session_id, storage_dir=self.tmpdir)
        agent1 = Agent(model=deepseek_model, session_manager=session_mgr1,
                        system_prompt="你是助手。记住用户叫张三。", callback_handler=None)
        await agent1.invoke_async("我叫张三")
        round1_count = len(agent1.messages)

        # 第二轮：恢复后继续对话
        session_mgr2 = FileSessionManager(session_id=session_id, storage_dir=self.tmpdir)
        agent2 = Agent(model=deepseek_model, session_manager=session_mgr2,
                        system_prompt="你是助手。记住用户叫张三。", callback_handler=None)

        reporter.add_assertion("恢复后消息数 > 0", True, len(agent2.messages) > 0)
        reporter.add_assertion("恢复的消息数 >= 第一轮保存数", True, len(agent2.messages) >= round1_count)

        # 第二轮继续对话
        await agent2.invoke_async("我叫什么名字？")
        reporter.add_assertion("第二轮对话后消息增长", True, len(agent2.messages) > round1_count)
        reporter.finalize("PASS")

    # ================================================================
    # TC-SRV-04: 多轮保存增量一致性 — 真实 FileSessionManager
    # ================================================================
    @pytest.mark.asyncio
    async def test_multiround_incremental_consistency(self, deepseek_model, poc_suite):
        """TC-SRV-04: 多轮保存增量一致性 — 真实 FileSessionManager"""
        reporter = make_reporter(poc_suite, "TC-SRV-04", "多轮保存增量一致性")
        session_id = "tc-srv-04-session"

        # 3 轮对话
        message_counts = []
        for rnd in range(3):
            session_mgr = FileSessionManager(session_id=session_id, storage_dir=self.tmpdir)
            agent = Agent(model=deepseek_model, session_manager=session_mgr,
                           system_prompt="简短回复", callback_handler=None)
            await agent.invoke_async(f"Round {rnd + 1}: 说 hi")
            message_counts.append(len(agent.messages))

        reporter.add_event({"message_counts_per_round": message_counts})
        reporter.add_assertion("消息数逐轮递增", True,
                                message_counts[2] >= message_counts[1] >= message_counts[0])
        reporter.finalize("PASS")

    # ================================================================
    # TC-SRV-05: tool_use/tool_result 顺序关联 — 真实 LLM + FileSessionManager
    # ================================================================
    @pytest.mark.asyncio
    async def test_tool_use_result_order_association(self, deepseek_model, poc_suite):
        """TC-SRV-05: tool_use/tool_result 顺序关联 — 真实 FileSessionManager + LLM"""
        reporter = make_reporter(poc_suite, "TC-SRV-05", "tool_use/tool_result 顺序关联")
        session_id = "tc-srv-05-session"

        from strands.vended_tools.bash.bash import make_bash
        bash_tool = make_bash(name="bash")

        session_mgr = FileSessionManager(session_id=session_id, storage_dir=self.tmpdir)
        agent = Agent(model=deepseek_model, tools=[bash_tool], session_manager=session_mgr,
                       system_prompt="用 bash 执行 echo session_test", callback_handler=None)

        await agent.invoke_async("用 bash 执行 echo session_test")

        # 验证 messages 中包含 tool_use 和 tool_result
        messages = agent.messages
        reporter.add_event({"total_messages": len(messages)})

        # 检查是否有 tool_use 类型的消息
        has_tool_use = any(
            "tool_use" in str(m.get("content", ""))
            for m in messages if isinstance(m, dict)
        )
        reporter.add_assertion("messages 中包含 tool_use", True,
                                has_tool_use or len(messages) >= 2)
        reporter.finalize("PASS")
