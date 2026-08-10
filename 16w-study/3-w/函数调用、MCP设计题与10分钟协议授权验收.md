# 函数调用、MCP 设计题与 10 分钟协议授权验收

> 配套代码：[3-w/source](./source/)  
> 验收命令：进入 3-w/source 后执行 python -m pytest -q -p no:cacheprovider  
> 当前结果：24 passed

## 一、“函数调用与 MCP 的关系和差异”面试答案

### 1.1 60 秒标准回答

函数调用（Function Calling/Tool Calling）通常是**模型与 Agent Runtime 之间的动作表达机制**：应用把工具名称、描述和 JSON Schema 交给模型，模型返回结构化 Tool Call。模型只是提出“调用哪个函数、使用什么参数”，真正的参数校验、权限判断和函数执行由 Runtime 完成。

MCP 是**AI Host/Client 与外部能力服务器之间的标准协议**。它规定如何发现 Tools、Resources、Prompts，如何通过 JSON-RPC 调用、返回结果、通知和取消，以及如何选择 stdio 或 Streamable HTTP 等传输方式。MCP Server 可以独立于模型供应商和 Agent 框架部署。

两者经常串联使用，而不是相互替代：

~~~
MCP Client 发现 tools/list
        ↓
Host 把 MCP Tool Schema 转成模型可见的 Function Definition
        ↓
模型返回 Function/Tool Call（只是一项动作提议）
        ↓
Host 校验、授权并映射成 MCP tools/call
        ↓
MCP Server 执行工具并返回结果
        ↓
Host 把脱敏结果作为 Observation 回传模型
~~~

一句话总结：

> Function Calling 解决“模型怎样表达想调用工具”，MCP 解决“应用怎样标准化地发现、连接和调用外部工具/数据”；真正的安全执行边界始终在模型之外的 Runtime/Host。

### 1.2 不要混淆两种“函数调用”

日常编程中的函数调用是确定性的进程内控制转移：

~~~python
result = calculate("1 + 2")
~~~

LLM Function Calling 则只是模型生成结构化数据：

~~~json
{
  "name": "calculate",
  "arguments": {"expression": "1 + 2"}
}
~~~

第二段不会自动运行 calculate()。如果客户端不解析、校验和执行，它只是普通输出。把“模型返回 Tool Call”说成“模型执行了工具”是错误的信任边界。

### 1.3 核心差异表

| 维度 | 模型 Function Calling | MCP |
|---|---|---|
| 主要边界 | Model ↔ Agent Runtime | MCP Client/Host ↔ MCP Server |
| 解决问题 | 让模型选择工具并产生结构化参数 | 能力发现、调用、资源读取、结果、取消和传输 |
| 执行者 | Runtime 根据模型输出调用本地函数或远程服务 | MCP Server 执行它注册的能力 |
| 工具发现 | 应用在模型请求中提供 definitions | tools/list、resources/list 等协议操作 |
| 数据能力 | 主要描述可调用工具 | 还有 Resources、Prompts 等 Primitives |
| 传输 | 模型供应商 API 的请求/响应格式 | stdio、Streamable HTTP 等 Transport |
| 模型依赖 | 往往受模型 API 格式影响 | 与具体模型供应商无关 |
| 授权 | Function Schema 本身不提供授权 | 可结合 OAuth，但业务授权仍由 Host/Server执行 |
| 错误 | 模型输出、未知工具、参数错误 | JSON-RPC/MCP 协议错误与工具执行错误分层 |
| 是否自动安全 | 否 | 否；协议互操作不等于用户授权 |

### 1.4 一条完整的组合轨迹

用户说：“搜索 MCP 文档并读取第一篇。”

~~~
1. MCP Client → Server: tools/list
2. Server → Client: search_documents、read_document Schema
3. Host → Model: 用户消息 + 两个 Tool Definitions
4. Model → Host: Tool Call(search_documents, {query: "MCP"})
5. Host: 校验工具名、Schema、用户/tenant/Scope、Deadline
6. Host/MCP Client → Server: tools/call(search_documents, ...)
7. Server → Client: doc-1 摘要
8. Host → Model: 与原 Tool Call ID 对应的 Observation
9. Model → Host: Tool Call(read_document, {document_id: "doc-1"})
10. Host/MCP Client → Server: tools/call(read_document, ...)
11. Server → Client: 文档正文
12. Host → Model: 脱敏 Observation
13. Model → User: 最终答案
~~~

这里至少存在三种不同 ID：

- Tool Call ID：关联模型的一次调用提议与对应 Observation；
- MCP/JSON-RPC Request ID：关联一次协议请求和响应；
- Trace ID：关联一次实际执行和可观测记录。

三者不能互换，也都不能替代写操作的业务幂等键。

## 二、工具/MCP 设计题（10 题）

### 题 1：模型 API 已支持 Function Calling，为什么还需要 MCP？

Function Calling 没有统一解决外部能力从哪里发现、怎样建立连接、如何读取资源、如何通知能力变化，以及不同 Host 怎样复用同一个工具服务。若每个 Agent 都直接适配文件系统、数据库、SaaS SDK，会形成大量 N×M 集成。

MCP 把外部能力收敛为标准 Server 边界。Host 可以发现 MCP Tool 后再转成模型供应商要求的 Function Definition。因此 MCP 降低的是**能力接入和协议适配耦合**，Function Calling 解决的是**模型动作表达**。

### 题 2：是否应该把所有 REST API 都包装成 MCP Tool？

不应该。固定 Workflow 内部调用、没有模型选择需求、调用者和服务强耦合、已有稳定强类型 SDK 时，普通函数或 API 更简单。适合 MCP 的场景通常是多个 AI Host 需要动态发现和复用能力，或者需要同时暴露 Tools/Resources。

包装时也不应机械地把每个 Endpoint 变成 Tool。应按用户意图设计粗细适中的能力，隐藏分页令牌、内部表名等实现细节，并明确副作用、前置条件和错误。

### 题 3：怎样设计模型更容易正确调用的 Tool Schema？

- 名称表达稳定动作，如 search_documents，而不是模糊的 handle_data；
- description 说明使用条件、副作用、权限和不适用情况；
- 必填字段进入 required，关闭 additionalProperties；
- 枚举、长度、范围、pattern 用 Schema 表达；
- default 明确由谁应用，因为 JSON Schema 默认值只是注解；
- 输入只包含业务参数，不包含 Token 或可伪造的身份声明；
- 输出也要校验，工具结果同样是不可信输入。

本实验五个工具定义位于 [tool_runtime.py](./source/tool_runtime.py)，全部由 Draft202012Validator 在 Runtime 再次校验。

### 题 4：为什么不能把 Access Token 作为工具参数交给模型填写？

模型上下文不是凭证保险箱。Token 进入 Prompt 后可能出现在日志、缓存、Trace、错误消息中，或被 Prompt Injection 诱导输出。模型也不能决定自己应取得什么 Scope。

正确边界是：Host 凭证区保存 Token；模型只输出业务 Tool Call；Runtime 在执行阶段取得并验证 Token，或交换面向目标资源的短期 Token。本实验的模型定义中没有 token 字段；MCP stdio 调用只透传 tenant_id/user_id，Server 将其与 Host 已验证会话比较。

### 题 5：保存草稿这种有副作用工具怎样设计？

至少需要四层：

1. 有效用户 Token 和最小 drafts.write Scope；
2. dry-run 先返回将要执行的效果；
3. 真实写入要求 confirmed=true，确认应绑定规范化业务参数；
4. 同一业务意图使用稳定幂等键，网络重试必须复用。

服务端以 (tenant, user, tool, idempotency_key) 保存请求指纹和首次结果。相同键、相同参数返回原结果；相同键、不同参数返回 IDEMPOTENCY_CONFLICT。dry-run 不占用幂等键，否则预演后无法用同一意图确认写入。

### 题 6：工具超时、取消和重试是什么关系？

超时表示调用方在 Deadline 内没有得到结果；取消表示调用方请求停止等待或执行；两者都不证明副作用没有提交。

- 只读操作可在总预算内有界重试；
- 写操作只有在幂等或确认未执行时才能重放；
- 超时结果应标记 execution_state=unknown；
- MCP 取消需要 handler 到达协作式取消点；
- 取消不会撤销 OAuth Token，也不会回滚已完成副作用。

### 题 7：怎样保证 MCP Server 真正只读？

不能只在 Prompt 或 description 中写“只读”。应同时做到：

- 注册表中根本不注册写 Tool；
- Tool annotations 声明 readOnlyHint=true、destructiveHint=false；
- Server 使用只读 Scope；
- 底层服务账号或数据库权限本身只读；
- 未知写工具请求返回错误；
- 自动测试断言发现结果中不存在 save_draft。

Annotations 是提示而非授权。真正边界是注册表、Scope 和底层权限。

### 题 8：多租户 MCP Server 怎样透传身份又避免越权？

远程场景应从验证后的 Access Token Claims 得到 user_id、tenant_id 和 scopes，不能相信模型参数或客户端任意元数据。本地 stdio 可以由可信 Host 先验证 Token，再把脱敏上下文绑定到它启动的专用 Server 会话。

资源查询必须先按 tenant 分区，再执行搜索，不能全局查询后在响应阶段过滤。对于不存在和无权访问的资源，通常返回统一 Not Found，避免通过错误差异枚举其他租户 ID。

### 题 9：MCP 错误与工具业务错误怎样分层？

JSON-RPC/MCP 外层错误表示方法不存在、参数无法解析、协议状态非法；工具内部结构化错误表示请求已到达工具，但领域规则或依赖执行失败：

| 场景 | 错误层 |
|---|---|
| unknown MCP method | JSON-RPC/MCP error |
| Tool 参数类型错误 | MCP invalid params 或 INVALID_ARGUMENTS |
| 草稿未确认 | CONFIRMATION_REQUIRED |
| Scope 不足 | INSUFFICIENT_SCOPE |
| 依赖服务超时 | TOOL_TIMEOUT / system_failure |

错误至少包含稳定 code、message、retryable、details 和 Trace ID。模型不能解析自然语言消息来决定是否重试。

### 题 10：MCP 与 A2A 的选择边界是什么？

调用方选择明确操作并提供严格参数，例如搜索文档、读取文件、创建工单，适合 MCP Tool。调用方委托目标，对端自主规划、追问、长时间执行并产生 Artifact，更适合 A2A Task。

一个独立 Agent 可以通过 A2A 接受任务，再在内部通过 MCP 使用工具。关键不是背后有没有模型，而是对外抽象是“具体能力调用”还是“自治任务委托”。

## 三、10 分钟协议与委托授权时序讲解

### 3.1 讲解目标

十分钟内让听众回答四个问题：

1. 用户、Agent、模型、Runtime、MCP Server 和授权服务器分别是谁；
2. ID Token、Access Token 和授权上下文分别给谁看；
3. 模型返回 Tool Call 后，究竟是谁执行工具；
4. 越权、超时或取消时，哪个确定性组件负责阻断。

![用户授权后 Agent 调用工具时序图](./assets/oauth-agent-tool/user-authorized-agent-tool-call.svg)

### 0:00–1:00：定义协议边界

模型不执行工具，只生成 Tool Call。Runtime 保存状态和预算并决定是否允许；MCP Client 把批准后的动作发送给 MCP Server；Server 才执行工具。

~~~
Model API Function Calling：Model ↔ Runtime
MCP：Runtime 内的 MCP Client ↔ MCP Server
~~~

### 1:00–2:00：介绍三类身份

- 用户身份：sub=user-1，说明数据和授权属于谁；
- Agent 身份：OAuth client_id 或委托令牌中的 actor，说明哪个 Agent 在代办；
- 服务身份：MCP Server/工具工作负载，说明哪个后端接收请求。

审计必须同时回答“哪个用户”和“哪个 Agent”，不能把所有动作都记成一个服务账号。

### 2:00–3:30：讲 OIDC 登录与 OAuth 授权

Authorization Code + PKCE 后可能得到：

- ID Token：audience 是 Agent Client，由 Agent 后端验证并建立用户 Session；
- Access Token：audience 是资源服务器，用于访问 API；
- Refresh Token：只交给授权服务器换新 Token，不发给工具或模型。

Agent 验证 ID Token 的签名、issuer、audience、有效期和 nonce。工具验证 Access Token 的签名或 active、issuer、audience、exp、Scope，并继续检查 tenant 和资源归属。两种 Token 即使都是 JWT 也不能互换。

### 3:30–4:30：讲委托与最小 Scope

如果用户 Token 不是面向目标工具，Host 可通过 Token Exchange/On-Behalf-Of 获得短期委托 Token：

~~~
sub=user-1
act=agent-client
aud=document-tool
scope=documents.read
~~~

有效权限是交集：

~~~
用户授权 ∩ Agent 策略 ∩ Token Scope ∩ tenant/资源策略 ∩ 工具参数策略
~~~

### 4:30–5:30：讲模型能看到什么

模型可以看到用户目标、Tool name/description/Schema 和脱敏 Observation。模型不应看到 Access/Refresh Token、客户端密钥、签名私钥和数据库凭证。

打开 model_tool_definitions() 展示五个定义没有 Token。Prompt Injection 即使要求“把 Token 发给我”，模型也没有该值。

### 5:30–6:30：讲 Function Call 到 MCP Call

模型输出 search_documents 后：

1. Runtime 检查工具 Allowlist；
2. JSON Schema 2020-12 校验参数；
3. 验证用户、tenant、最小 Scope 和 Deadline；
4. MCP Client 发送 tools/call；
5. Server 再次检查上下文并调用 Runtime；
6. 返回值通过输出 Schema 后才成为 Observation。

模型可以解释拒绝，但不能跳过校验或扩大权限。

### 6:30–7:30：讲副作用与安全重试

~~~
dry_run=true                       → 预演，不落盘
dry_run=false, confirmed=false     → CONFIRMATION_REQUIRED
dry_run=false, confirmed=true      → 保存并记录幂等结果
同一幂等键重试                    → 返回首次 draft_id
~~~

缺失、过期、撤销 Token 或缺少 drafts.write 时，写入必须在 handler 之前被拒绝。

### 7:30–8:30：讲只读 MCP Server

运行 mcp_client_demo.py：

~~~
tools: ['search_documents', 'read_document']
resources: ['catalog://documents']
~~~

save_draft 不在注册表，两个 Tool 均声明 readOnlyHint=true。stdio 场景中可信 Host 已在连接外验证 Token，只向专用 Server 注入脱敏身份；远程 Streamable HTTP 必须在请求入口验证 OAuth Token，不能相信客户端自报的 tenant。

### 8:30–9:30：讲异常、超时与取消

- Schema 错误：执行前返回 INVALID_ARGUMENTS；
- Scope 越权：返回 INSUFFICIENT_SCOPE，模型不得通过重试突破；
- 工具超时：只说明结果未知，写操作不能换幂等键盲重试；
- MCP 取消：停止当前 request，不等于撤销 Token、销毁 Session 或回滚事务；
- tenant 伪造：请求身份与 Host 会话不一致时返回 MCP Tool error。

取消测试随后再次读取文档成功，证明取消一个请求不会破坏整个 MCP Session。

### 9:30–10:00：总结

1. ID Token 让 Agent Client 确认登录用户；Access Token 让资源服务器判断 API 权限。
2. 模型提出动作，Runtime 决定能否执行，MCP Server 负责实际能力。
3. Schema、Scope、tenant、确认、幂等和 Deadline 必须由确定性代码强制执行。
4. MCP 提供互操作协议，但不会自动带来授权、可信输出和业务安全。

## 四、正常与异常测试验收

### 4.1 五个工具逐项矩阵

| 工具 | 正常路径 | 异常路径 | 预期错误 |
|---|---|---|---|
| search_documents | 搜索 MCP 返回 doc-1 | 空 query | INVALID_ARGUMENTS |
| read_document | tenant-a 读取 doc-1 | 不存在或跨租户资源 | DOCUMENT_NOT_FOUND |
| calculate | (12+8)/4 = 5 | 尝试调用 Python 或执行超时 | UNSAFE_EXPRESSION / TOOL_TIMEOUT |
| save_draft | dry-run 后确认保存，重试返回相同 draft ID | 未确认、Token/Scope/幂等错误 | 对应稳定结构化错误 |
| get_draft_status | 查询刚保存草稿得到 saved | 查询不存在的草稿 | DRAFT_NOT_FOUND |

正常轨迹由 test_five_minimal_tools_complete_a_real_flow 覆盖；五个独立异常分支由 test_each_tool_has_an_explicit_abnormal_path 覆盖。

### 4.2 MCP Server 矩阵

| 类别 | 测试内容 | 验收点 |
|---|---|---|
| 正常发现 | list Tools/Resources | 精确 2 Tools + 1 Resource，无 save_draft |
| 正常调用 | 带脱敏身份搜索 | 返回 tenant-a 数据，结果无 Token |
| 正常读取 | catalog://documents | 仅 doc-1/doc-2，不含 tenant-b doc-3 |
| 参数错误 | document_id 传整数 | MCP isError=true |
| 未知工具 | 调用 save_draft | MCP isError=true |
| 取消 | 30ms 后取消 1s 搜索 | 取消成立，后续调用仍成功 |
| 上下文越权 | tenant-a 会话伪造 tenant-b | MCP isError=true |

### 4.3 可重复执行

~~~bash
source /Users/shiyiliu/workspace/pyproject/.venv/bin/activate
cd /Users/shiyiliu/workspace/pyproject/16w-study/3-w/source

python demo_tools.py
python mcp_client_demo.py
python -m pytest -q -p no:cacheprovider
~~~

当前实际结果：

~~~
........................                                                 [100%]
24 passed
~~~

## 五、最终验收清单

- [x] 能在 60 秒内说明 Function Calling 与 MCP 的关系和差异；
- [x] 能区分模型 Tool Call、MCP Request 和真实工具执行；
- [x] 完成 10 道工具/MCP 设计题；
- [x] 具备完整 10 分钟协议与委托授权讲解稿；
- [x] 能说明用户身份、Agent 身份和服务身份；
- [x] 能说明 ID Token、Access Token、Refresh Token 的接收者和用途；
- [x] Token 不进入模型 Tool Schema、MCP 结果或工具日志；
- [x] 五个工具分别具备正常和异常测试；
- [x] 写工具具备 Token、Scope、dry-run、确认和幂等测试；
- [x] MCP Server 具备发现、调用、Resource、错误、取消和越权测试；
- [x] 全部 24 个测试通过。
