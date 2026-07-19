"""Pytest 配置和共享 fixtures — 包含 POC 报告生成"""

import pytest
import asyncio
from typing import Any

from strands import Agent

from poc_runner.mock_cma.event_store import CmaEventStore
from poc_runner.mock_cma.sse_queue import SSEClient
from poc_runner.adapter.event_translator import CmaEventTranslator
from poc_runner.adapter.sandbox_proxy import CmaSandboxProxy, make_cma_redirected_bash, create_sandbox_redirect_hook
from poc_runner.adapter.approval_bridge import ApprovalConfig, create_approval_hook, resume_agent_after_interrupt
from poc_runner.adapter.session_store import CmaEventStoreSessionRepository
from poc_runner.fixtures.agents import (
    create_deepseek_model,
    create_anthropic_model,
    create_agent_with_tools,
    create_readonly_agent,
    create_full_tools_agent,
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL_ID,
)
from poc_runner.fixtures.sandboxes import MockDockerSandbox, MockSshSandbox
from poc_runner.reporter import PocReporter, PocReportSuite


# ============================================================
# POC 报告系统 (session 级别)
# ============================================================

@pytest.fixture(scope="session")
def poc_suite():
    """POC 测试套件报告器（session 级别，所有测试共享）"""
    suite = PocReportSuite()
    yield suite
    # session 结束时自动生成汇总报告
    summary_path = suite.generate_summary()
    print(f"\n📊 POC 报告汇总已生成: {summary_path}")


# ============================================================
# 基础设施 fixtures
# ============================================================

@pytest.fixture
def event_store():
    """CMA EventStore 内存实例"""
    return CmaEventStore()


@pytest.fixture
def sse_queue():
    """CMA SSE 事件队列"""
    return SSEClient()


@pytest.fixture
def event_translator(sse_queue):
    """CMA 事件翻译器"""
    return CmaEventTranslator(sse_queue)


@pytest.fixture
def cma_sandbox():
    """CMA Sandbox Proxy"""
    return CmaSandboxProxy()


@pytest.fixture
def cma_bash_tool(cma_sandbox):
    """重定向到 CMA Sandbox 的 bash 工具"""
    return make_cma_redirected_bash(cma_sandbox)


@pytest.fixture
def sandbox_redirect_hook(cma_sandbox, cma_bash_tool):
    """Sandbox 重定向 hook"""
    return create_sandbox_redirect_hook(cma_sandbox, cma_bash_tool)


# ============================================================
# 模型 fixtures
# ============================================================

@pytest.fixture
def deepseek_model():
    """DeepSeek 模型"""
    return create_deepseek_model()


# ============================================================
# Agent fixtures
# ============================================================

@pytest.fixture
def base_agent(deepseek_model):
    """基础 Agent（最小配置）"""
    return create_agent_with_tools(
        model=deepseek_model,
        system_prompt="你是一个测试助手。",
    )


@pytest.fixture
def docker_sandbox():
    """Mock Docker Sandbox"""
    return MockDockerSandbox()


@pytest.fixture
def ssh_sandbox():
    """Mock SSH Sandbox"""
    return MockSshSandbox()


# ============================================================
# 审批 fixtures
# ============================================================

@pytest.fixture
def approval_config_allowed():
    """所有工具 allowed 模式"""
    return ApprovalConfig(default_mode="allowed")


@pytest.fixture
def approval_config_manual_bash():
    """bash=manual, 其他=allowed"""
    return ApprovalConfig(
        tool_approvals={"bash": "manual"},
        default_mode="allowed",
    )


@pytest.fixture
def approval_config_forbidden_bash():
    """bash=forbidden, 其他=allowed"""
    return ApprovalConfig(
        tool_approvals={"bash": "forbidden"},
        default_mode="allowed",
    )


# ============================================================
# 辅助函数
# ============================================================

def make_reporter(poc_suite: PocReportSuite, test_id: str, name: str) -> PocReporter:
    """在 suite 中创建报告器（若已存在则返回已有的）"""
    return poc_suite.create(test_id, name)


# ============================================================
# 工具函数 exports
# ============================================================

__all__ = [
    "CmaEventStore",
    "SSEClient",
    "CmaEventTranslator",
    "CmaSandboxProxy",
    "make_cma_redirected_bash",
    "create_sandbox_redirect_hook",
    "ApprovalConfig",
    "create_approval_hook",
    "resume_agent_after_interrupt",
    "CmaEventStoreSessionRepository",
    "create_deepseek_model",
    "create_anthropic_model",
    "create_agent_with_tools",
    "create_readonly_agent",
    "create_full_tools_agent",
    "MockDockerSandbox",
    "MockSshSandbox",
    "PocReporter",
    "PocReportSuite",
    "make_reporter",
]
