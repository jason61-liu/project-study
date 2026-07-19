"""测试夹具：预配置的 Agent 工厂函数"""

import os
from strands import Agent
from strands.models.openai import OpenAIModel

# ---- 模型配置 ----

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL_ID = "deepseek-chat"


def create_deepseek_model(
    api_key: str = DEEPSEEK_API_KEY,
    base_url: str = DEEPSEEK_BASE_URL,
    model_id: str = DEEPSEEK_MODEL_ID,
) -> OpenAIModel:
    """创建 DeepSeek 模型（通过 OpenAI-compatible API）"""
    return OpenAIModel(
        client_args={
            "api_key": api_key,
            "base_url": base_url,
        },
        model_id=model_id,
    )


def create_anthropic_model(
    api_key: str,
    model_id: str = "claude-sonnet-4-6",
):
    """创建 Anthropic 模型（备选）"""
    from strands.models.anthropic import AnthropicModel
    return AnthropicModel(
        client_args={"api_key": api_key},
        model_id=model_id,
    )


# ---- Agent 工厂 ----

def create_agent_with_tools(
    tools: list | None = None,
    system_prompt: str = "",
    messages: list | None = None,
    model=None,
    sandbox=None,
    hooks: list | None = None,
    plugins: list | None = None,
    session_manager=None,
    **kwargs,
) -> Agent:
    """创建通用 Agent 实例"""
    if model is None:
        model = create_deepseek_model()

    agent_kwargs = {
        "model": model,
        "tools": tools or [],
        "system_prompt": system_prompt,
        "messages": messages,
        "sandbox": sandbox,
        "hooks": hooks or [],
        "plugins": plugins or [],
        "session_manager": session_manager,
    }
    # Only set callback_handler if not provided in kwargs
    if "callback_handler" not in kwargs:
        agent_kwargs["callback_handler"] = None  # 禁用默认打印
    agent_kwargs.update(kwargs)
    return Agent(**agent_kwargs)


def create_readonly_agent(**kwargs) -> Agent:
    """创建只读工具 Agent"""
    from strands.vended_tools import bash
    # 只注册 read-only 工具
    return create_agent_with_tools(
        tools=[],  # 无 bash 工具
        system_prompt="你是只读助手，只能读取文件",
        **kwargs,
    )


def create_full_tools_agent(**kwargs) -> Agent:
    """创建完整工具集 Agent（含 bash）"""
    from strands.vended_tools.bash.bash import make_bash

    bash_tool = make_bash(name="bash")
    return create_agent_with_tools(
        tools=[bash_tool],
        system_prompt="你是全功能助手",
        **kwargs,
    )
