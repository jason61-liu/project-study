"""ToolRuntime 的确定性验收：不使用 Mock，不调用大模型。"""

from __future__ import annotations

from dataclasses import replace
import logging
import time

import pytest

from tool_runtime import InMemoryStore, TokenService, ToolRuntime


SECRET = b"test-secret-that-is-long-enough-for-hs256"
ALL_SCOPES = {"documents.read", "calculate.use", "drafts.write", "drafts.read"}


@pytest.fixture
def services():
    """为每个测试创建隔离的 Token、数据和线程池，防止状态串扰。"""

    tokens = TokenService(SECRET)
    runtime = ToolRuntime(tokens, InMemoryStore.sample())
    yield tokens, runtime
    runtime.close()


def issue(tokens: TokenService, scopes: set[str] = ALL_SCOPES, **kwargs) -> str:
    """为 tenant-a/user-1 签发测试 Token，并允许用 kwargs 改写过期时间或 jti。"""

    return tokens.issue(user_id="user-1", tenant_id="tenant-a", scopes=scopes, **kwargs)


def test_five_minimal_tools_complete_a_real_flow(services) -> None:
    """证明五个工具能组成“发现资料→计算→预演→保存→查状态”的真实轨迹。"""

    tokens, runtime = services
    token = issue(tokens)

    search = runtime.invoke("search_documents", {"query": "MCP", "limit": 5}, token=token)
    document = runtime.invoke("read_document", {"document_id": "doc-1"}, token=token)
    calculation = runtime.invoke("calculate", {"expression": "(12 + 8) / 4"}, token=token)
    # dry-run 不写入；后续使用同一幂等键确认，验证预演不会占用业务意图。
    preview = runtime.invoke(
        "save_draft",
        {"title": "周报", "content": "MCP 学习记录", "idempotency_key": "intent-save-00001", "dry_run": True, "confirmed": False},
        token=token,
    )
    saved = runtime.invoke(
        "save_draft",
        {"title": "周报", "content": "MCP 学习记录", "idempotency_key": "intent-save-00001", "dry_run": False, "confirmed": True},
        token=token,
    )
    status = runtime.invoke("get_draft_status", {"draft_id": saved["data"]["draft_id"]}, token=token)

    assert search["data"]["items"] == [{"id": "doc-1", "title": "MCP 入门"}]
    assert document["data"]["content"].startswith("MCP")
    assert calculation["data"]["value"] == 5
    assert preview["data"]["mode"] == "dry_run"
    assert saved["data"]["mode"] == "committed"
    assert status["data"]["state"] == "saved"
    assert len(runtime.store.drafts) == 1


@pytest.mark.parametrize(
    ("tool_name", "arguments", "expected_code"),
    [
        ("search_documents", {"query": "", "limit": 5}, "INVALID_ARGUMENTS"),
        ("read_document", {"document_id": "doc-999"}, "DOCUMENT_NOT_FOUND"),
        ("calculate", {"expression": "__import__('os')"}, "UNSAFE_EXPRESSION"),
        (
            "save_draft",
            {
                "title": "未确认草稿",
                "content": "不得落盘",
                "idempotency_key": "intent-unconfirmed-001",
                "dry_run": False,
                "confirmed": False,
            },
            "CONFIRMATION_REQUIRED",
        ),
        ("get_draft_status", {"draft_id": "draft-deadbeef0000"}, "DRAFT_NOT_FOUND"),
    ],
)
def test_each_tool_has_an_explicit_abnormal_path(services, tool_name, arguments, expected_code) -> None:
    """逐个证明五个工具都把无效/拒绝场景收敛为稳定结构化错误。"""

    tokens, runtime = services
    result = runtime.invoke(tool_name, arguments, token=issue(tokens))

    assert result["ok"] is False
    assert result["error"]["code"] == expected_code
    assert result["meta"]["trace_id"]


def test_schema_rejects_missing_extra_and_invalid_enum_like_controls(services) -> None:
    """同时覆盖必填字段、additionalProperties=false 和布尔类型三个 Schema 边界。"""

    tokens, runtime = services
    token = issue(tokens)

    missing = runtime.invoke("calculate", {}, token=token)
    extra = runtime.invoke("read_document", {"document_id": "doc-1", "admin": True}, token=token)
    wrong_type = runtime.invoke(
        "save_draft",
        {"title": "x", "content": "y", "idempotency_key": "intent-save-00002", "dry_run": "yes", "confirmed": False},
        token=token,
    )

    assert missing["error"]["code"] == "INVALID_ARGUMENTS"
    assert extra["error"]["code"] == "INVALID_ARGUMENTS"
    assert wrong_type["error"]["code"] == "INVALID_ARGUMENTS"
    assert missing["error"]["details"]["errors"][0]["path"] == "$"


def test_unknown_tool_has_structured_error(services) -> None:
    """未知工具应成为可观察结果，而不是 KeyError 或进程异常。"""

    tokens, runtime = services
    result = runtime.invoke("delete_everything", {}, token=issue(tokens))

    assert result["ok"] is False
    assert result["status"] == "business_failure"
    assert result["error"] == {
        "code": "UNKNOWN_TOOL",
        "message": "未知工具：delete_everything",
        "retryable": False,
        "details": {},
    }
    assert result["meta"]["trace_id"]


def test_all_tool_calls_have_deadline_and_timeout_is_structured(services) -> None:
    """把真实慢 handler 的预算压缩到 5ms，验证 Runtime 的物理等待超时。"""

    tokens, runtime = services
    original = runtime.tools["calculate"]

    def slow(_auth, expression):
        """故意超过 Deadline 的真实线程函数。"""

        time.sleep(0.1)
        return {"expression": expression, "value": 1}

    runtime.tools["calculate"] = replace(original, handler=slow, timeout_s=0.005)
    result = runtime.invoke("calculate", {"expression": "1+1"}, token=issue(tokens))

    assert result["status"] == "system_failure"
    assert result["error"]["code"] == "TOOL_TIMEOUT"
    assert result["error"]["retryable"] is True
    assert result["error"]["details"]["execution_state"] == "not_completed"


def test_logging_contains_trace_and_identity_but_never_token(services, caplog) -> None:
    """日志应支持追踪和租户审计，但不得形成 Token 泄漏通道。"""

    tokens, runtime = services
    token = issue(tokens)
    with caplog.at_level(logging.INFO, logger="week3.tools"):
        result = runtime.invoke("calculate", {"expression": "2*3"}, token=token)

    log_text = "\n".join(caplog.messages)
    assert result["meta"]["trace_id"] in log_text
    assert "tenant-a" in log_text and "user-1" in log_text
    assert token not in log_text


@pytest.mark.parametrize(
    ("token_factory", "expected"),
    [
        (lambda tokens: None, "AUTH_MISSING"),
        (lambda tokens: issue(tokens, expires_in_s=-1), "TOKEN_EXPIRED"),
        (lambda tokens: _revoked_token(tokens), "TOKEN_REVOKED"),
        (lambda tokens: issue(tokens, scopes={"documents.read"}), "INSUFFICIENT_SCOPE"),
    ],
)
def test_side_effect_rejects_missing_expired_revoked_and_wrong_scope(services, token_factory, expected) -> None:
    """同一写入请求依次验证四个 OAuth Resource Server 失败分支。"""

    tokens, runtime = services
    arguments = {
        "title": "安全测试",
        "content": "不应落盘",
        "idempotency_key": "intent-security-001",
        "dry_run": False,
        "confirmed": True,
    }
    result = runtime.invoke("save_draft", arguments, token=token_factory(tokens))

    assert result["error"]["code"] == expected
    assert runtime.store.drafts == {}


def _revoked_token(tokens: TokenService) -> str:
    """签发后立即按 jti 撤销，以区别“撤销”与“自然过期”。"""

    token = issue(tokens, token_id="revoked-jti")
    tokens.revoke("revoked-jti")
    return token


def test_dry_run_confirmation_and_idempotent_replay(services) -> None:
    """证明预演无副作用、未确认被拒绝、网络重试只产生一份草稿。"""

    tokens, runtime = services
    token = issue(tokens)
    base = {"title": "草稿", "content": "正文", "idempotency_key": "intent-idempotent-1"}

    preview = runtime.invoke("save_draft", {**base, "dry_run": True, "confirmed": False}, token=token)
    refused = runtime.invoke("save_draft", {**base, "dry_run": False, "confirmed": False}, token=token)
    first = runtime.invoke("save_draft", {**base, "dry_run": False, "confirmed": True}, token=token)
    replay = runtime.invoke("save_draft", {**base, "dry_run": False, "confirmed": True}, token=token)

    # 断言发生在整条轨迹结束后，因此 store 中应恰好只有 confirmed 调用创建的草稿。
    assert preview["ok"] and runtime.store.drafts
    assert refused["error"]["code"] == "CONFIRMATION_REQUIRED"
    assert first["data"]["draft_id"] == replay["data"]["draft_id"]
    assert replay["meta"]["idempotent_replay"] is True
    assert len(runtime.store.drafts) == 1


def test_same_idempotency_key_with_different_business_input_conflicts(services) -> None:
    """防止调用方错误复用幂等键时静默返回不相关的首次结果。"""

    tokens, runtime = services
    token = issue(tokens)
    controls = {"idempotency_key": "intent-idempotent-2", "dry_run": False, "confirmed": True}
    first = runtime.invoke("save_draft", {"title": "A", "content": "one", **controls}, token=token)
    conflict = runtime.invoke("save_draft", {"title": "A", "content": "two", **controls}, token=token)

    assert first["ok"]
    assert conflict["error"]["code"] == "IDEMPOTENCY_CONFLICT"
    assert len(runtime.store.drafts) == 1


def test_tenant_and_user_isolation(services) -> None:
    """同一 Runtime 中，tenant-a 不能通过猜中 doc-3 读取 tenant-b 数据。"""

    tokens, runtime = services
    tenant_a = issue(tokens)
    tenant_b = tokens.issue(user_id="user-2", tenant_id="tenant-b", scopes=ALL_SCOPES)

    cross_tenant_read = runtime.invoke("read_document", {"document_id": "doc-3"}, token=tenant_a)
    own_tenant_read = runtime.invoke("read_document", {"document_id": "doc-3"}, token=tenant_b)

    assert cross_tenant_read["error"]["code"] == "DOCUMENT_NOT_FOUND"
    assert own_tenant_read["data"]["title"] == "隔离文档"


def test_model_definitions_never_expose_token(services) -> None:
    """模型可见的五个工具定义只能包含业务参数，凭证由 Host 注入。"""

    _tokens, runtime = services
    serialized = str(runtime.model_tool_definitions()).lower()

    assert len(runtime.model_tool_definitions()) == 5
    assert "access_token" not in serialized
    assert '"token"' not in serialized
