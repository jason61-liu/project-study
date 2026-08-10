# 第三周编码实验：安全工具契约与只读 MCP Server

本目录延续 `2-w/source` 的最小结构，包含两层：

```text
模型 / Agent Host
  │ 只看到工具描述和业务参数，不看到 Token
  ▼
ToolRuntime
  ├── JSON Schema 输入/输出校验
  ├── 用户 Token、tenant、Scope 验证
  ├── dry-run、确认和幂等控制
  ├── Deadline、结构化错误与脱敏日志
  └── 5 个最小工具

MCP Client
  │ _meta: tenant_id + user_id
  ▼
只读 MCP Server
  ├── search_documents Tool
  ├── read_document Tool
  └── catalog://documents Resource
```

## 一、文件说明

| 文件 | 作用 |
|---|---|
| `tool_runtime.py` | 5 个工具、完整 Schema、HS256 测试令牌、Scope、超时、日志、幂等和统一结果结构 |
| `mcp_server.py` | 基于官方 Python MCP SDK 的只读 stdio Server |
| `mcp_client_demo.py` | 真实 MCP Client：初始化、发现、调用 Tool、读取 Resource |
| `demo_tools.py` | 依次执行五个工具的本地演示 |
| `tests/test_tool_runtime.py` | 工具正常流程及安全、可靠性边界测试 |
| `tests/test_mcp_server.py` | MCP 发现、调用、Resource、错误、取消和越权上下文测试 |

## 二、五个工具

| 工具 | 类型 | Scope | 关键控制 |
|---|---|---|---|
| `search_documents` | 只读 | `documents.read` | tenant 隔离、limit Schema |
| `read_document` | 只读 | `documents.read` | 文档 ID Schema、越租户返回 not found |
| `calculate` | 只读 | `calculate.use` | AST 白名单，禁止 `eval` 和任意代码 |
| `save_draft` | 写入 | `drafts.write` | 幂等键、dry-run、显式确认、用户 Token |
| `get_draft_status` | 只读 | `drafts.read` | tenant + user 双重隔离，不返回正文 |

所有工具都由 `ToolRuntime.invoke()` 统一执行以下步骤：

1. 查找注册工具；
2. 验证 Token 签名、issuer、audience、用户身份、租户、过期时间、撤销状态和最小 Scope；
3. 使用 JSON Schema 2020-12 校验输入；
4. 对写操作检查 dry-run、确认和幂等键；
5. 在线程池中执行，并受每个 `ToolSpec.timeout_s` Deadline 约束；
6. 校验工具输出 Schema；
7. 返回统一结果并记录脱敏日志。

成功结构：

```json
{
  "ok": true,
  "status": "success",
  "data": {},
  "error": null,
  "meta": {
    "trace_id": "...",
    "duration_ms": 0.2,
    "tenant_id": "tenant-a",
    "user_id": "user-1",
    "idempotent_replay": false
  }
}
```

失败结构：

```json
{
  "ok": false,
  "status": "business_failure",
  "data": null,
  "error": {
    "code": "INSUFFICIENT_SCOPE",
    "message": "Access Token 权限不足",
    "retryable": false,
    "details": {"missing_scopes": ["drafts.write"]}
  },
  "meta": {"trace_id": "..."}
}
```

## 三、幂等、dry-run 与确认

`save_draft` 的推荐调用顺序是：

```text
同一个业务意图生成一次 idempotency_key
        │
        ├── dry_run=true, confirmed=false  → 只返回预演，不落盘
        │
        └── dry_run=false, confirmed=true  → 执行写入并保存幂等结果
                                                   │
                                                   └── 网络重试复用同一 key
```

幂等记录按 `(tenant_id, user_id, tool_name, idempotency_key)` 隔离：

- 相同键和相同业务参数返回第一次结果，并标记 `idempotent_replay=true`；
- 相同键对应不同标题/正文返回 `IDEMPOTENCY_CONFLICT`；
- dry-run 不占用幂等键，所以预演后可以用同一个键确认写入；
- `dry_run`、`confirmed` 是控制参数，不进入业务参数指纹。

线程 Future 超时后，Python 不能强制终止已经开始的同步函数。因此写工具超时会把 `execution_state` 标记为 `unknown`，客户端必须用原幂等键重试或查询状态，不能生成新键盲目重放。

## 四、Token 与授权边界

实验中的 `TokenService` 使用标准库生成 HS256 JWT，目的是确定性验证以下行为：

- Token 缺失：`AUTH_MISSING`；
- 签名、issuer 或 audience 错误：`AUTH_INVALID`；
- Token 过期：`TOKEN_EXPIRED`；
- `jti` 已撤销：`TOKEN_REVOKED`；
- Scope 不足：`INSUFFICIENT_SCOPE`。

生产环境不要使用该本地签发器，应改为：

- 验证授权服务器的非对称 JWT/JWKS；或
- 调用 OAuth Token Introspection；
- 继续保留 Runtime 的 tenant、资源归属、业务参数和幂等校验。

模型可见的 `model_tool_definitions()` 没有 Token 参数。Host 从凭证存储取得 Token，在模型完成 Tool Call 后调用 Runtime；日志仅记录 `trace_id/tenant_id/user_id`。

## 五、MCP 授权上下文如何透传

本实验使用只读 stdio MCP Server。信任边界是：

1. Agent Host 在 MCP 连接外完成 OAuth Token 验证；
2. Host 创建不含 Token 的 `AuthorizationContext`；
3. MCP Client 在请求 `_meta` 中只发送 `tenant_id/user_id`；
4. Server 将 `_meta` 与本次 Host 会话绑定的身份精确比较；
5. 不一致时返回 MCP Tool error，不能通过修改 `_meta` 切换租户；
6. Resource 使用同一 Host 会话上下文进行 tenant 过滤。

stdio 子进程由可信 Host 启动，所以可以使用这一上下文边界。若改成面向不可信网络客户端的 Streamable HTTP，不能相信客户端自报的 `_meta`，应配置 MCP OAuth Token Verifier，并从验证后的令牌 Claims 构造上下文。

写工具 `save_draft` 没有注册到 MCP Server。即使模型猜出该名称，MCP Client 也只会收到 unknown tool，确保“只读 Server”不是靠描述文字约定。

## 六、运行

使用项目虚拟环境：

```bash
source /Users/shiyiliu/workspace/pyproject/.venv/bin/activate
cd /Users/shiyiliu/workspace/pyproject/16w-study/3-w/source
```

演示五个工具：

```bash
python demo_tools.py
```

启动真实 stdio Server，并通过官方 MCP Client 完成发现和调用：

```bash
python mcp_client_demo.py
```

输出应包含：

```text
tools: ['search_documents', 'read_document']
resources: ['catalog://documents']
```

也可以让 MCP Inspector 启动 Server；Inspector 只作为人工查看补充，本项目的自动验收使用官方 MCP Client：

```bash
npx @modelcontextprotocol/inspector \
  /Users/shiyiliu/workspace/pyproject/.venv/bin/python \
  /Users/shiyiliu/workspace/pyproject/16w-study/3-w/source/mcp_server.py
```

## 七、测试

```bash
python -m pytest -q -p no:cacheprovider
```

测试不使用 Mock，覆盖：

- 五个工具的完整正常流程；
- 五个工具各自独立的异常路径；
- 输入 Schema 缺失字段、额外字段和错误类型；
- 未知工具、业务错误、执行超时和结构化错误；
- 日志含 Trace/脱敏身份且不含 Token；
- Token 缺失、过期、撤销和 Scope 越权；
- dry-run、缺失确认、幂等重放和幂等冲突；
- tenant/user 数据隔离；
- MCP Tools/Resource 发现；
- MCP 调用、Schema error、unknown tool；
- MCP 请求取消后会话恢复；
- 伪造 tenant/user 上下文被拒绝。

当前验收结果：`24 passed`。
