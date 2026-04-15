"""
LangGraph Demo - 共享 LLM 配置模块

本模块封装了 LLM 的初始化逻辑，所有示例共用此配置。
使用硅基流动 (SiliconFlow) 平台的 Qwen/Qwen3-32B 模型，
通过 langchain_openai.ChatOpenAI 兼容调用（硅基流动 API 兼容 OpenAI 格式）。
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 加载当前目录下的 .env 文件，读取 API Key 和 Base URL
load_dotenv()


def get_llm(temperature: float = 0.7, streaming: bool = False) -> ChatOpenAI:
    """
    获取 LLM 实例

    Args:
        temperature: 生成温度，越高越随机，越低越确定。默认 0.7
        streaming: 是否启用流式输出。默认 False

    Returns:
        ChatOpenAI 实例，已配置好硅基流动的 API 地址和模型名称
    """
    return ChatOpenAI(
        # 硅基流动的 API 地址（兼容 OpenAI 格式）
        base_url=os.getenv("OPENAI_BASE_URL"),
        # API 密钥
        api_key=os.getenv("OPENAI_API_KEY"),
        # 使用的模型：Qwen3-32B
        model="Qwen/Qwen3-32B",
        temperature=temperature,
        streaming=streaming,
    )
