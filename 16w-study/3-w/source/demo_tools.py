"""依次演示搜索、读取、计算、保存草稿和状态查询五个工具。

这个脚本刻意绕过大模型，直接调用 ToolRuntime，用于观察工具层本身的契约：
输入参数怎样校验、授权上下文怎样产生、dry-run 与真实提交怎样切换，以及写入后
如何查询状态。模型接入后只负责生成相同的工具名和业务参数，不会获得 ``token``。
"""

from __future__ import annotations

import json

from tool_runtime import InMemoryStore, TokenService, ToolRuntime


def show(title: str, result: dict) -> None:
    """以保留中文的 JSON 格式打印一次结构化工具结果。"""

    print(f"\n--- {title} ---")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:
    """构造本地授权环境并执行一条包含五个工具的完整业务轨迹。"""

    # TokenService 是实验用签发器。生产系统中应由外部 OAuth 授权服务器签发，
    # Runtime 只负责验证 Access Token 或接收 Host 已验证的 AuthorizationContext。
    tokens = TokenService(b"week3-demo-secret-must-be-at-least-32-bytes")
    runtime = ToolRuntime(tokens, InMemoryStore.sample())
    # 一枚 Token 同时携带本次演示需要的最小 Scope。这里显式列出，方便观察
    # 任意删除一个 Scope 后，对应工具会返回 INSUFFICIENT_SCOPE。
    token = tokens.issue(
        user_id="user-demo",
        tenant_id="tenant-a",
        scopes={"documents.read", "calculate.use", "drafts.write", "drafts.read"},
    )
    try:
        # 三个只读工具仍经过 Token、Scope、Schema、Deadline、输出 Schema 和日志层，
        # 并不是因为没有副作用就绕开 Runtime 的统一控制。
        show("搜索", runtime.invoke("search_documents", {"query": "MCP", "limit": 5}, token=token))
        show("读取", runtime.invoke("read_document", {"document_id": "doc-1"}, token=token))
        show("计算", runtime.invoke("calculate", {"expression": "(12 + 8) / 4"}, token=token))

        arguments = {
            "title": "第三周学习草稿",
            "content": "完成 MCP 工具契约实验。",
            "idempotency_key": "demo-draft-intent-001",
        }
        # dry-run 使用同一个业务幂等键，但不会落盘或占用幂等记录。用户确认后，
        # Runtime 才允许 dry_run=false 的请求进入真正的 save handler。
        show(
            "保存预演",
            runtime.invoke("save_draft", {**arguments, "dry_run": True, "confirmed": False}, token=token),
        )
        saved = runtime.invoke(
            "save_draft",
            {**arguments, "dry_run": False, "confirmed": True},
            token=token,
        )
        show("确认保存", saved)
        # 状态查询仍会同时校验 tenant_id 和 user_id，其他租户或其他用户不能通过
        # 猜测 draft_id 获取草稿状态，更不会在状态结果里拿到正文。
        show(
            "状态查询",
            runtime.invoke("get_draft_status", {"draft_id": saved["data"]["draft_id"]}, token=token),
        )
    finally:
        # ToolRuntime 持有线程池。显式 close 可以避免教学脚本退出时残留工作线程；
        # Web 服务中通常由应用 lifespan/shutdown hook 负责这一动作。
        runtime.close()


if __name__ == "__main__":
    main()
