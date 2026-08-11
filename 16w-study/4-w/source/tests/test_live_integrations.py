"""显式启用的真实 DeepSeek/Mem0 Cloud 集成测试；不使用 Mock。"""

import os

import pytest

from memory_backends import AuthScope, Mem0CloudMemory
from model_client import DeepSeekModel


LIVE = os.getenv("RUN_LIVE_TESTS") == "1"


@pytest.mark.skipif(not LIVE or not os.getenv("DEEPSEEK_API_KEY"), reason="需要 RUN_LIVE_TESTS=1 和 DEEPSEEK_API_KEY")
def test_real_deepseek_v4_pro_returns_usage() -> None:
    """使用无敏感信息的小问题验证真实模型、Token 和费用采集。"""

    result = DeepSeekModel().complete("只回答结果。", "2+2 等于多少？", max_tokens=20)
    assert "4" in result.text
    assert result.model == "deepseek-v4-pro"
    assert result.input_tokens > 0 and result.output_tokens > 0
    assert result.estimated_cost_usd > 0


@pytest.mark.skipif(not LIVE or not os.getenv("MEM0_API_KEY"), reason="需要 RUN_LIVE_TESTS=1 和有效 MEM0_API_KEY")
def test_real_mem0_cloud_lifecycle() -> None:
    """创建隔离测试记忆并在 finally 中按精确 ID 清理。"""

    backend = Mem0CloudMemory(infer=True)
    scope = AuthScope("week4-live-test", "probe-user")
    try:
        memory_id = backend.add(scope, "week4 测试偏好中文")
        assert backend.get(scope, memory_id)
        assert backend.search(scope, "偏好什么语言？", top_k=3)
        backend.update(scope, memory_id, "week4 测试偏好英文")
        assert "英文" in (backend.get(scope, memory_id) or "")
        backend.delete(scope, memory_id)
        assert backend.get(scope, memory_id) is None
    finally:
        backend.close()

