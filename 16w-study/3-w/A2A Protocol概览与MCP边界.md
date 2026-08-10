# A2A Protocol 概览与 MCP 边界

> 原阅读地址 `google-a2a.github.io/A2A/specification/` 当前已失效。A2A 项目已迁移到 Linux Foundation 下的 [a2aproject/A2A](https://github.com/a2aproject/A2A)，本文依据当前 [A2A Specification](https://a2a-protocol.org/latest/specification/) 整理。

## 一、A2A 解决的不是“再造一个工具协议”

A2A（Agent2Agent）连接的是**独立、自治且内部实现不透明的 Agent 应用**。调用方不需要知道远端 Agent 使用什么模型、Prompt、记忆、工作流或工具，只需发现其能力、发送任务、交换消息与制品，并观察长任务状态。

它的核心目标是：

- 跨框架、语言和厂商互操作；
- 通过 Agent Card 发现身份、接口、能力、技能和认证要求；
- 使用 Message、Part、Artifact 交换多模态内容；
- 用 Task 表达有状态、长时、可查询/取消的工作；
- 支持同步响应、流式更新、轮询和异步 push notification；
- 保持远端 Agent 的实现不透明，不要求暴露内部工具和推理轨迹。

因此，A2A 更接近“Agent 应用之间的任务委托协议”，而不是远程函数调用。

## 二、三层架构：数据模型、操作、协议绑定

当前规范将协议拆成三层：

1. **Canonical Data Model**：由 Protocol Buffers 定义规范性数据类型；
2. **Abstract Operations**：定义发送消息、查询任务、取消、订阅等与传输无关的语义；
3. **Protocol Bindings**：把抽象操作映射到 JSON-RPC、gRPC 或 HTTP+JSON/REST。

这意味着“A2A 就是 JSON-RPC over HTTP”已经不准确。JSON-RPC 是一种绑定，核心语义属于上层。实现之间能否互操作，取决于双方是否在 Agent Card 中声明并选择共同支持的 interface/binding。

## 三、核心数据对象

### 3.1 Agent Card：发现和连接契约

Agent Card 通常可从 `/.well-known/agent-card.json` 获取，也可通过注册表或配置分发。它描述：

- Agent 名称、描述、版本、提供方；
- 支持的 interfaces、协议版本和 endpoint；
- streaming、push notification 等能力；
- skills：技能 ID、说明、输入/输出模式和示例；
- 默认输入/输出 MIME types；
- OAuth 2.0、OIDC、API Key、mTLS 等安全方案；
- 可选签名，用于验证来源和完整性。

Agent Card 是**能力与连接元数据**，不是授权结果。声明 `securitySchemes` 不等于调用者已取得令牌；声明某个 skill 也不代表所有用户都能调用它。敏感能力可以放在需要认证的 extended Agent Card 中。

### 3.2 Message、Part 与 Artifact

- `Message` 是用户/Agent 间的一轮通信，带 role、message ID、context/task 关联；
- `Part` 是消息或制品的内容单元，可表示文本、文件、结构化数据等；
- `Artifact` 是任务产生的输出制品，可以分块、追加或作为最终交付物。

Message 更像“沟通内容”，Artifact 更像“工作产物”。例如研究 Agent 先用 Message 回答“已开始调研”，最终用 Artifact 返回报告文件和结构化来源列表。

### 3.3 Task：A2A 的状态核心

如果远端 Agent 能立即给出独立回答，`Send Message` 可以直接返回 Message；若工作需要持续执行，则返回 Task。Task 至少包含稳定 task ID、context ID、状态、历史消息和 artifacts。

常见状态包括：

- `submitted`、`working`：已接收或处理中；
- `input-required`：需要调用方/用户补充内容；
- `auth-required`：需要额外认证或授权；
- `completed`、`failed`、`canceled`、`rejected`：终态。

`input-required` 和 `auth-required` 不是失败终态，而是协议层的控制权交接。客户端应保存 task/context ID，解决输入或授权后继续，而不是创建一个无关联的新任务。

## 四、主要操作及执行语义

| 操作 | 用途 | 关键语义 |
|---|---|---|
| Send Message | 发起或继续一次协作 | 可返回 Message 或 Task |
| Send Streaming Message | 发起任务并接收增量事件 | 通常先收到 Task，再收状态/Artifact 更新，终态结束 |
| Get / List Tasks | 查询与恢复任务 | 支持断线恢复、轮询和运维视图 |
| Cancel Task | 请求取消 | 取消是协议操作，不保证回滚外部副作用 |
| Subscribe to Task | 重新订阅已有任务事件 | 用于流断开后继续观察 |
| Push Notification Config CRUD | 管理异步 webhook | 适合分钟到小时级任务，必须防 SSRF、伪造和重放 |
| Get Extended Agent Card | 认证后获取额外能力 | 避免公开敏感技能和端点 |

长任务有三种观察方式：短连接轮询 `Get Task`、保持流式连接、配置 push webhook。生产系统通常组合使用：流断线后用 Get 恢复真相；超长任务用 push 唤醒客户端；任务状态本身仍由服务端持久化，事件流不是唯一事实来源。

## 五、一次实际协作：采购 Agent 委托供应商 Agent

假设企业采购 Agent 要向供应商 Agent 获取 500 台服务器报价：

1. 采购 Agent 获取并校验供应商的 Agent Card，选择共同支持的 HTTP+JSON 接口和 OAuth 方案；
2. 发送 Message，包含规格、数量、交付地区和结构化约束；
3. 供应商 Agent 返回 Task，状态为 `working`；
4. 供应商 Agent 内部可能调用库存、定价、合规和物流工具，但这些内部实现不对采购 Agent 暴露；
5. 若缺少税务主体，Task 进入 `input-required`，采购 Agent补充信息并继续同一 task/context；
6. 若需要企业授权，Task 进入 `auth-required`，客户端通过声明的外部 OAuth 流程取得凭证；
7. 供应商持续发送状态和报价 Artifact；最终状态 `completed`；
8. 采购 Agent验证 Artifact、记录审计并决定是否接受，而不是因为远端声称完成就自动付款。

这个例子体现了 A2A 的关键价值：委托的是“完成报价任务”的能力，而不是逐个暴露 `query_inventory()`、`calculate_tax()` 等内部函数。

## 六、MCP 与 A2A 的本质边界

| 维度 | MCP：Agent—工具/数据 | A2A：独立 Agent—Agent |
|---|---|---|
| 对端角色 | 提供 tools、resources、prompts 的能力服务器 | 能自主规划和执行的远端 Agent 应用 |
| 交互抽象 | 列表/读取/调用具体能力 | 消息、任务、状态、制品与协作上下文 |
| 调用方认知 | 知道工具名、Schema、资源 URI | 只需知道远端技能和任务契约，不知内部工具 |
| 生命周期 | 多数调用较短；结果直接回传 | 原生支持长任务、输入/授权等待、流式与 push |
| 状态 | 工具应显式返回句柄，不应依赖隐式会话状态 | Task/context 是一等状态对象 |
| 自主性 | Server 执行被选中的明确操作 | 远端 Agent 自主决定如何完成委托 |
| 发现 | `tools/list`、`resources/list`、`prompts/list` 等 | Agent Card + skills + interfaces |
| 安全关注 | 工具参数、副作用、资源权限、Host 审批 | Agent 身份、任务授权、跨域信任、Artifact/消息可信度 |
| 典型例子 | 查数据库、读文件、创建工单 | 让法务 Agent 审查合同并持续协商修改 |

判断方法不是看“网络上有几个进程”，而是看抽象边界：

- 调用方选择一个确定操作并提供严格参数，通常是工具调用；
- 调用方委托一个目标，对端自行规划、追问、产生产物，通常是 Agent 协作；
- 把普通 API 包装成“Agent”不会自动获得 A2A 的任务语义；
- 把远端自治 Agent 的每个内部动作暴露成 MCP Tool，则会破坏封装并放大耦合。

## 七、两者如何组合，而不是二选一

最常见架构是：

```text
用户
  │
  ▼
采购 Agent ───────── A2A ─────────▶ 供应商 Agent
  │                                      │
  ├── MCP → 企业政策资源                 ├── MCP → 库存工具
  └── MCP → 审批工具                     └── MCP → 报价/物流工具
```

A2A 维持组织或应用边界上的自治协作；每个 Agent 在自己的信任域内用 MCP 连接工具和数据。A2A Task ID 与内部 MCP Tool Call ID 不应混为一谈，但可通过 Trace ID 建立因果关联。

## 八、安全、可靠性与幂等

### 8.1 身份与授权

Agent Card 可以声明安全方案，但每个请求仍需认证和授权。收到消息后还需验证调用 Agent、代表的用户/租户、目标 skill 和输入内容。`auth-required` 只是 Task 状态，不定义如何安全传递凭证；凭证应通过声明的 OAuth/OIDC 等外部机制获取，不能放进普通 Message 让模型读取。

### 8.2 消息与 Artifact 不可信

远端 Agent 输出是外部输入，可能含 Prompt Injection、恶意文件或错误事实。接收方应进行 Schema/MIME 校验、恶意内容扫描、来源与签名验证，并在高风险副作用前重新执行本地策略。

### 8.3 幂等和重复事件

规范允许使用 `messageId` 帮助去重，但实现仍应定义持久化范围与期限。Get 操作天然更适合重试；Cancel 应按幂等语义处理；Send Message 在响应丢失时不能通过更换 message ID 盲目重发。流式/push 事件可能重复、乱序或断开，客户端应依据 task ID、事件标识/版本和持久化任务状态去重与恢复。

### 8.4 Push webhook

Server 向客户端提供的 URL 推送状态时，必须防止：

- SSRF：校验 URL、限制协议/端口、阻止内网和云元数据地址；
- 伪造：签名或 OAuth/mTLS 认证推送；
- 重放：时间戳、nonce/event ID 和去重；
- 泄密：只推送必要数据，敏感 Artifact 通过授权下载；
- 投递失败：退避、死信和客户端主动 Get Task 对账。

## 九、何时选择 MCP、A2A 或普通 Workflow

- 需要给一个 Agent 安全接入搜索、文件、数据库、SaaS API：选 MCP；
- 需要跨团队/厂商委托长任务，对端保留自主规划：选 A2A；
- 步骤固定、每步行为可预测、无自治协作必要：普通 Workflow 往往更简单；
- 单次、稳定、强类型的服务调用：普通 HTTP/gRPC API 可能已足够；
- 一个完整系统常同时使用 Workflow 编排、A2A 跨 Agent、MCP 接工具，协议层次不同并不冲突。

## 十、核心总结

1. A2A 的核心对象是 Agent Card、Message、Task、Part 和 Artifact，Task 是长时协作的状态锚点。
2. 当前规范将数据模型、抽象操作和协议绑定分离，并支持 JSON-RPC、gRPC、HTTP+JSON 等绑定。
3. A2A 保护远端 Agent 的不透明性：调用者委托目标，不需要获得其工具、记忆或推理过程。
4. MCP 连接 Agent 与具体工具/数据；A2A 连接独立 Agent。远端 Agent 完全可以在内部使用 MCP。
5. 能力发现不等于授权，任务完成也不等于结果可信；身份、策略、内容校验和审计必须独立实现。
6. 流式、轮询和 push 是任务观察机制，持久化 Task 才是恢复和对账的事实来源。

## 参考资料

- [A2A Protocol Specification](https://a2a-protocol.org/latest/specification/)
- [A2A Protocol Documentation](https://a2a-protocol.org/latest/)
- [A2A GitHub repository](https://github.com/a2aproject/A2A)
- [MCP Architecture](https://modelcontextprotocol.io/docs/learn/architecture)
