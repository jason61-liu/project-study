"""不使用 Mock，直接测试本地向量和真实 Mem0 OSS/Qdrant。"""

import pytest

from memory_backends import AuthScope, Mem0OssMemory, VectorMemory
from memory_eval import evaluate_memory_backend


@pytest.mark.parametrize("factory", [VectorMemory, Mem0OssMemory])
def test_both_real_backends_support_write_search_update_delete(factory) -> None:
    """同一生命周期契约在两个实际实现上成立。"""

    backend = factory()
    scope = AuthScope("tenant-a", "lifecycle-user")
    try:
        memory_id = backend.add(scope, "项目 Atlas 使用 Go 1.24")
        assert backend.get(scope, memory_id) == "项目 Atlas 使用 Go 1.24"
        assert any("Go 1.24" in hit.text for hit in backend.search(scope, "Atlas 使用什么版本？"))

        backend.update(scope, memory_id, "项目 Atlas 使用 Go 1.25")
        assert backend.get(scope, memory_id) == "项目 Atlas 使用 Go 1.25"

        backend.delete(scope, memory_id)
        assert backend.get(scope, memory_id) is None
        assert all("Go 1.25" not in hit.text for hit in backend.search(scope, "Atlas Go"))
    finally:
        backend.close()


@pytest.mark.parametrize("factory", [VectorMemory, Mem0OssMemory])
def test_both_backends_enforce_tenant_and_user_scope(factory) -> None:
    """相同 user_id 跨租户、相同 tenant 跨用户均不可见。"""

    backend = factory()
    owner = AuthScope("tenant-b", "same-local-user")
    try:
        memory_id = backend.add(owner, "租户 B 的隔离事实是 NEBULA")
        assert backend.get(AuthScope("tenant-a", "same-local-user"), memory_id) is None
        assert backend.get(AuthScope("tenant-b", "other-user"), memory_id) is None
        assert backend.search(AuthScope("tenant-a", "same-local-user"), "NEBULA") == []
    finally:
        backend.close()


@pytest.mark.parametrize("factory", [VectorMemory, Mem0OssMemory])
def test_unauthorized_update_and_delete_are_rejected(factory) -> None:
    """知道 memory ID 不等于获得修改权限。"""

    backend = factory()
    owner = AuthScope("tenant-a", "owner")
    attacker = AuthScope("tenant-b", "owner")
    try:
        memory_id = backend.add(owner, "只能由 owner 修改")
        with pytest.raises(KeyError):
            backend.update(attacker, memory_id, "越权修改")
        with pytest.raises(KeyError):
            backend.delete(attacker, memory_id)
        assert backend.get(owner, memory_id) == "只能由 owner 修改"
    finally:
        backend.close()


@pytest.mark.parametrize("factory", [VectorMemory, Mem0OssMemory])
def test_comparison_metrics_cover_all_requested_dimensions(factory) -> None:
    """统一评测输出写入、召回、错误、隔离、一致性、延迟和 Token。"""

    report = evaluate_memory_backend(factory())
    required = {
        "write_accuracy",
        "recall_at_k",
        "false_memory_rate",
        "tenant_isolation",
        "update_consistency",
        "delete_consistency",
        "write_latency_ms",
        "search_latency_ms",
        "input_tokens",
    }
    assert required <= report.keys()
    assert report["write_accuracy"] == 1.0
    assert report["recall_at_k"] == 1.0
    assert report["tenant_isolation"] is True
    assert report["update_consistency"] is True
    assert report["delete_consistency"] is True
    assert 0.0 <= report["false_memory_rate"] <= 1.0

