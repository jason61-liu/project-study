# MCP JSON-RPC、STDIO、Streamable HTTP、通知、取消与错误响应

> 基于 MCP `2026-07-28` 版本整理。
>
> 这一版本是无状态协议：每个请求携带协议版本和客户端能力。Streamable HTTP 不再使用旧版独立 GET 事件流、`Mcp-Session-Id` 或 `Last-Event-ID` 恢复机制；不要把不同版本的实现细节混在一起。

## 一、先建立整体模型

这六个概念位于不同层次：

| 层次 | 概念 | 解决的问题 |
|---|---|---|
| 消息语义 | JSON-RPC 2.0 | 请求、响应、错误和通知如何表示与关联 |
| 本地传输 | STDIO | JSON-RPC 消息如何通过子进程标准流传递 |
| 远程传输 | Streamable HTTP | JSON-RPC 如何通过 HTTP POST、JSON 响应或 SSE 传递 |
| 异步事件 | Notification | 不要求响应的单向事件如何表达 |
| 生命周期控制 | Cancellation | 调用方如何表达“不再需要这个仍在执行的请求” |
| 失败语义 | Error Response | 如何区分协议错误、业务失败、传输失败和本地超时 |

它们的关系可以概括为：

```text
MCP 方法：tools/call、resources/read、prompts/get ...
        ↓ 封装为
JSON-RPC：request / result / error / notification
        ↓ 承载于
STDIO：一行一个 JSON-RPC 消息
或
Streamable HTTP：一次 POST + JSON 或请求专属 SSE
```

JSON-RPC 定义“消息是什么意思”，Transport 定义“消息如何到达”。通知、取消和错误都要同时遵守 JSON-RPC 语义与具体 Transport 的传递规则。

## 二、JSON-RPC：MCP 的消息骨架

### 2.1 Request

Request 表示 Client 希望 Server 执行一个操作：

```json
{
  "jsonrpc": "2.0",
  "id": "req-42",
  "method": "tools/call",
  "params": {
    "name": "get_weather",
    "arguments": {
      "city": "Shanghai"
    },
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {}
    }
  }
}
```

关键约束：

- `jsonrpc` 固定为 `"2.0"`；
- `id` 必须是字符串或整数，MCP 不允许 `null`；
- 同一发送方尚未收到响应的 Request 之间，`id` 不得重复；
- `method` 决定协议操作；
- `params` 承载业务参数和 MCP `_meta`；
- 当前 MCP 是无状态的，Server 不能从连接历史推断协议版本或 Client 能力。

`id` 是并发关联键，不是 Trace ID。一个 Trace 可以包含多次模型调用和多个 JSON-RPC Request；反过来，请求 ID 通常只在一条连接或一个 Client 实例的关联表中有意义。工程上应同时维护：

```text
trace_id       跨模型、Host、Client 和 Server 追踪整条任务
request_id     关联一次 JSON-RPC Request 与 Response
subscriptionId 关联长期订阅产生的多条 Notification
```

### 2.2 Result Response

成功响应必须复用请求的 `id`：

```json
{
  "jsonrpc": "2.0",
  "id": "req-42",
  "result": {
    "resultType": "complete",
    "content": [
      {
        "type": "text",
        "text": "Shanghai: 31°C"
      }
    ]
  }
}
```

当前规范要求结果包含 `resultType`：

- `complete`：操作已经完成；
- `input_required`：操作尚未完成，需要 Client 提供额外输入，再以新的 JSON-RPC ID 重试原操作；
- 扩展可以定义其他协商过的类型；Client 遇到未知类型应判定为无效结果。

为兼容旧 Server，缺少 `resultType` 时，当前 Client 需要把它视为 `complete`。这是一种协议兼容规则，不代表新 Server 可以继续省略该字段。

### 2.3 Error Response

JSON-RPC 错误表示请求无法按协议或方法语义正常完成：

```json
{
  "jsonrpc": "2.0",
  "id": "req-42",
  "error": {
    "code": -32602,
    "message": "Invalid params",
    "data": {
      "field": "city"
    }
  }
}
```

原则上错误响应使用原 Request 的 `id`。只有消息严重损坏、接收方无法读取 ID 时，错误才可能没有可关联的 `id`。

### 2.4 Notification

Notification 是无需响应的单向 JSON-RPC 消息：

```json
{
  "jsonrpc": "2.0",
  "method": "notifications/tools/list_changed"
}
```

Notification 与 Request 的决定性区别是：

- 没有 `id`；
- 接收方不得返回 JSON-RPC Response；
- 发送方无法通过协议响应确认业务处理成功；
- 格式错误、未知事件或竞态事件通常只能记录或忽略。

如果业务必须知道对端是否处理成功，就不应该设计成 Notification，而应使用 Request/Response。

### 2.5 并发和乱序

同一 Transport 上可以存在多个并发请求。响应不必按照请求发送顺序到达，所以实现不能使用 FIFO 队列简单匹配：

```text
发送 req-1 ──► 慢查询
发送 req-2 ──► 快查询
收到 req-2 ◄── 先完成
收到 req-1 ◄── 后完成
```

Client 应维护类似 `pending[request_id] -> Future/Promise` 的表。收到响应后按 `id` 完成对应 Future，而不是完成“最早发送”的请求。

## 三、STDIO Transport：共享字节流上的消息复用

### 3.1 进程与通道

在 STDIO 模式下，Client 启动 MCP Server 子进程：

```text
MCP Client                      MCP Server 子进程
    │                                  │
    ├── 写入 Server stdin ────────────►│ JSON-RPC Request/Notification
    │                                  │
    │◄────────── 读取 Server stdout ──┤ JSON-RPC Response/Notification
    │                                  │
    │◄────────── 可选读取 stderr ─────┤ 日志与诊断
```

STDIO 的核心约束是：

- 一条物理双向通道承载所有请求、响应和通知；
- 每个 JSON-RPC 消息占一行，以换行符分隔；
- 消息不能包含未转义的实际换行；JSON 字符串里的换行必须编码为 `\n`；
- Server 的 `stdout` 只能输出合法 MCP 消息；
- 日志必须写到 `stderr`，不能用 `print()` 污染 `stdout`；
- `stderr` 出现内容不一定代表请求失败，它只是诊断通道。

常见故障是 Server 在 stdout 输出启动横幅：

```text
Server started successfully!
{"jsonrpc":"2.0", ...}
```

第一行不是 JSON-RPC，Client 的帧解析器会失败。正确做法是把启动日志写入 stderr。

### 3.2 帧解析不能假设一次 read 就是一条消息

STDIO 是字节流。一次系统调用可能读到半条消息，也可能一次读到多条消息。正确解析方式是：

1. 将读取到的字节追加到缓冲区；
2. 查找换行分隔符；
3. 每找到一行就按 UTF-8 解码并解析一个 JSON 对象；
4. 保留最后一段未完成数据等待下次读取；
5. 对单行最大长度和缓冲区大小设置限制。

这可以避免半包、粘包以及恶意超长消息导致的内存耗尽。

### 3.3 STDIO 中如何关联消息

因为所有消息共用 stdout：

- Response 使用 JSON-RPC `id` 关联 Request；
- 请求执行期间的 Progress/Message Notification 依靠请求相关元数据关联；
- 长期订阅 Notification 使用 `_meta.io.modelcontextprotocol/subscriptionId`；
- 不能根据消息出现位置判断它属于哪个请求。

### 3.4 STDIO 的认证与安全边界

STDIO 没有 HTTP Header 和 OAuth 交换。凭证通常由 Host 通过受控环境变量或进程配置提供给 Server。需要注意：

- 不要把密钥放入模型上下文或 Tool 参数；
- 控制子进程继承的环境变量，只传递必要项；
- 限制 Server 的文件、网络和操作系统权限；
- 本地进程不天然可信，第三方 Server 仍可能读取本机数据或执行恶意代码。

### 3.5 关闭与故障恢复

正常关闭通常是：Client 关闭 Server stdin，等待子进程退出；超时后再升级为强制终止。Server 应在 stdin EOF 后及时退出。

进程意外退出时：

- 所有 in-flight Request 都失去响应；
- Client 可以重启 Server；
- 长期订阅必须重新建立；
- 是否重试普通请求取决于幂等性，不能因为 MCP 无状态就盲目重放有副作用的 Tool。

协议“无状态”只表示 Server 不依赖连接历史，不表示业务操作没有状态或失败后一定可安全重试。

## 四、Streamable HTTP：每个请求独立 POST，可选 SSE 响应

### 4.1 当前版本的基本形态

Server 暴露一个 MCP Endpoint，例如：

```text
https://example.com/mcp
```

Client 发送的每个 JSON-RPC Request 都是新的 HTTP POST：

```http
POST /mcp HTTP/1.1
Content-Type: application/json
Accept: application/json, text/event-stream
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tools/call
Mcp-Name: get_weather

{"jsonrpc":"2.0","id":"req-42","method":"tools/call","params":{...}}
```

Server 可以按单次请求选择两种响应：

1. `application/json`：直接返回一个 JSON-RPC Result 或 Error；
2. `text/event-stream`：先发送与本请求有关的 Notification，最后发送 JSON-RPC Response 并结束流。

因此 Streamable HTTP 的“Streamable”不是说所有请求都必须流式，也不是一个全局事件连接。它表示某次 POST 的响应可以升级为该请求专属的 SSE 流。

### 4.2 请求专属 SSE

长任务可能返回：

```text
POST tools/call (id=req-42)
  ◄── SSE: notifications/progress
  ◄── SSE: notifications/progress
  ◄── SSE: final JSON-RPC response (id=req-42)
  ◄── stream closes
```

约束包括：

- 流中的 Notification 必须与原 Request 有关；
- Server 不能在这个流上独立发起新的 JSON-RPC Request；
- 需要用户输入等反向交互时，Server 使用 `InputRequiredResult`，Client 获取输入后以新 ID 重试；
- 最终 Response 应结束流；
- Client 必须同时支持 JSON 与 SSE，不能假设某个方法固定使用其中一种。

### 4.3 长期通知流

长期变化通知由 `subscriptions/listen` 创建。它也是一个普通 JSON-RPC Request，但其 HTTP Response 是保持打开的 SSE：

```text
POST subscriptions/listen (id=sub-7)
  ◄── acknowledged, subscriptionId=sub-7
  ◄── notifications/tools/list_changed
  ◄── notifications/resources/updated
  ◄── ... 长期保持 ...
```

它与普通请求 SSE 的区别是：

| 请求专属 SSE | 订阅 SSE |
|---|---|
| 生命周期通常较短 | 长期保持 |
| 发送 Progress、Message 等请求相关事件 | 发送用户订阅的列表或资源变化事件 |
| 最终 Response 后关闭 | 直到取消、Server 结束或 Transport 断开 |

### 4.4 HTTP Header 与 Body 双重信息

当前版本要求把部分 Body 信息镜像到 Header，便于网关、路由器和限流器无需解析 JSON 就能决策：

- `MCP-Protocol-Version`：必须与 `_meta` 中版本一致；
- `Mcp-Method`：对应 JSON-RPC `method`；
- `Mcp-Name`：`tools/call`、`resources/read`、`prompts/get` 等请求对应名称或 URI；
- `Mcp-Param-*`：Tool Schema 通过 `x-mcp-header` 指定的可选参数镜像。

Server 必须核对 Header 与 Body 一致，避免网关按 Header 判断“只读工具”，而后端按 Body 实际执行“删除工具”的请求走私问题。不一致时使用 HTTP `400` 和 MCP `HeaderMismatch (-32020)`。

### 4.5 代理和连接工程

SSE 经常受到反向代理影响：

- Server 应考虑返回 `X-Accel-Buffering: no`，避免 nginx 缓冲事件；
- 长期安静的订阅流可定期发送 SSE Comment 作为 Keep-alive；
- Client 必须忽略以冒号开头的 SSE Comment；
- 当前版本不支持 `Last-Event-ID` 恢复，断线后需要重新订阅并主动刷新状态；
- 设置连接超时、读取空闲超时和绝对最大时长时，要区分普通请求与长期订阅。

### 4.6 HTTP 安全

Server 必须验证 `Origin`，防止网页利用 DNS Rebinding 访问本机 MCP 服务。本地 HTTP Server 应只监听 loopback，而不是默认绑定 `0.0.0.0`。远程场景还需要认证、授权、TLS、速率限制和请求大小限制。

### 4.7 与旧版 Streamable HTTP 的区别

2025-03-26 到 2025-11-25 的 Streamable HTTP 曾包含：

- `Mcp-Session-Id`；
- HTTP GET 打开的独立 SSE；
- HTTP DELETE 结束 Session；
- Server 在 SSE 上发送 JSON-RPC Request；
- `Last-Event-ID` 恢复流。

这些都不属于 2026-07-28。新实现必须按协商出的协议版本选择行为，而不能只看到“Streamable HTTP”这个名称就假设线协议相同。

## 五、STDIO 与 Streamable HTTP 对比

| 维度 | STDIO | Streamable HTTP 2026-07-28 |
|---|---|---|
| 部署位置 | 通常本地子进程 | 通常远程独立服务，也可本地 |
| 消息边界 | 一行一个 JSON-RPC 消息 | 一个 POST Body 一个 JSON-RPC 消息；SSE 事件承载返回消息 |
| 并发复用 | 全部请求共用一条 stdout | 每个 Request 有独立 HTTP Response |
| 日志 | stderr | 服务日志系统或可观测性平台 |
| 普通响应 | stdout 上的 JSON-RPC Response | 单个 JSON 或请求专属 SSE |
| 长期通知 | 同一 stdout，以 subscriptionId 解复用 | `subscriptions/listen` 的长期 SSE |
| Client 取消 | 发送 `notifications/cancelled` | 关闭该 Request 的 SSE Response |
| 认证 | 通常进程环境和 OS 权限 | HTTP Authorization/OAuth 等 |
| 断线恢复 | 重启进程并重建订阅 | 重连、重订阅；无 Last-Event-ID 恢复 |
| 主要风险 | stdout 污染、子进程权限过大 | Origin、认证、代理缓冲、Header/Body 不一致 |

## 六、Notification：事件，不是“没有结果的 Request”

### 6.1 两类 Server Notification

第一类是与某个正在处理的请求关联的事件：

- `notifications/progress`；
- 请求范围内的消息或日志。

它们只描述当前请求进展，不能代替最终 Response。收到 90% Progress 后连接断开，Client 仍不能把该请求判定为成功。

第二类来自长期订阅：

- `notifications/tools/list_changed`；
- `notifications/prompts/list_changed`；
- `notifications/resources/list_changed`；
- `notifications/resources/updated`。

它们必须属于 Client 显式请求且 Server 确认接受的订阅范围。

### 6.2 订阅确认与关联

Client 发出：

```json
{
  "jsonrpc": "2.0",
  "id": "sub-7",
  "method": "subscriptions/listen",
  "params": {
    "notifications": {
      "toolsListChanged": true,
      "resourceSubscriptions": ["file:///project/config.json"]
    },
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {}
    }
  }
}
```

Server 必须先发送 `notifications/subscriptions/acknowledged`，其中返回实际接受的过滤器。后续每个事件在 `_meta` 中携带：

```json
{
  "io.modelcontextprotocol/subscriptionId": "sub-7"
}
```

`subscriptionId` 的值就是原 `subscriptions/listen` Request 的 JSON-RPC ID。Client 应以它解复用多个并发订阅。

### 6.3 通知是 best-effort

通知可能因断线、进程重启或代理超时丢失。因此：

- `list_changed` 表示“目录可能变化了”，Client 应重新调用对应 `*/list`；
- `resources/updated` 表示“资源可能变化了”，Client 应重新读取；
- 重连后应重新订阅，并主动刷新关键目录或资源；
- 不能把通知当成完整事件日志或 Exactly-once 消息队列。

## 七、Cancellation：终止意图，不是事务回滚

### 7.1 两种 Transport 的取消方式

STDIO 没有每个请求独享的流，因此发送：

```json
{
  "jsonrpc": "2.0",
  "method": "notifications/cancelled",
  "params": {
    "requestId": "req-42",
    "reason": "User requested cancellation"
  }
}
```

Streamable HTTP 中，每个流式请求拥有独立 SSE Response，Client 关闭该流即表示取消，不再额外发送 `notifications/cancelled`。

### 7.2 取消是协作式的

Server 收到取消后应尽快：

- 停止计算或中断下游调用；
- 释放文件、连接、锁和内存；
- 不再为该 Request 发送后续消息或 Response。

但 Server 可以在以下情况忽略取消：请求未知、已经完成或底层操作不可取消。取消因此不是“立刻杀死”的强保证。

### 7.3 必须处理取消竞态

典型竞态是：Server 已经完成并发送 Response，但取消消息还在路上。

```text
Client                         Server
  ├── Request req-42 ─────────► 开始执行
  │                            已完成并发送结果
  ├── Cancel req-42 ──────────► 取消到达过晚
  │◄──────── Response req-42 ── 网络中迟到的响应
```

规则是：

- Client 只取消自己发出且认为仍在进行的 Request；
- Server 对未知、已完成或格式错误的取消直接忽略；
- Client 在本地把 Request 标记为 cancelled 后，应忽略之后到达的 Response；
- Request ID 不应过早复用，否则迟到 Response 可能错误完成另一个请求。

### 7.4 取消不等于撤销副作用

如果 `send_email` 已提交给邮件服务，取消只能阻止后续工作，不能让邮件自动消失。类似地：

- 数据库可能已经提交事务；
- 支付可能已经创建；
- 文件可能写入了一半；
- 第三方 API 可能执行成功，但响应在取消时丢失。

因此有副作用的 Tool 还需要：

- Idempotency Key；
- 操作状态查询接口；
- 明确的提交点；
- 补偿操作；
- 审计记录。

### 7.5 Timeout 应触发取消，但两者不是同义词

Timeout 是调用方本地的等待策略；Cancellation 是向执行方表达停止意图。推荐处理：

```text
达到请求超时
  → 本地状态改为 timing_out
  → 按 Transport 发送取消或关闭 SSE
  → 释放本地等待者
  → 状态改为 timed_out
  → 忽略迟到响应
```

可以维护两类计时器：

- **软超时/空闲超时**：收到 Progress 后可选择重置；
- **硬性最大超时**：无论是否持续收到 Progress 都不能超过。

MCP 没有为 SDK 本地产生的 Timeout 分配标准 JSON-RPC Error Code。实现必须区分“对端返回的错误”与“本地等待超时”。

### 7.6 长期订阅的结束

`subscriptions/listen` 本质上仍是一个尚未返回最终结果的长生命周期 Request：

- Client 在 STDIO 上取消订阅时，发送 `notifications/cancelled`，引用原 listen Request ID；
- Client 在 HTTP 上取消订阅时，关闭对应 SSE；
- Server 主动优雅结束订阅时，应返回原 listen Request 的空 `complete` Result，再关闭流；
- Server 发送取消通知的权限只用于终止 `subscriptions/listen`，不能用来随意取消 Client 的其他普通 Request；
- Transport 意外断开没有最终 Result，Client 应视为非正常结束，并在重连后重新订阅。

## 八、错误响应：必须区分六个失败平面

### 8.1 Transport Failure

例如：

- 子进程退出、管道 EOF；
- DNS、TLS 或 TCP 失败；
- HTTP 代理中断 SSE；
- Response Body 超限；
- 本地 Timeout 或用户取消。

这些情况可能根本没有 JSON-RPC Error。它们说明“没有可靠获得协议结果”，并不一定说明 Server 没执行操作。

### 8.2 HTTP Failure

HTTP 状态属于 Streamable HTTP 传输语义。例如当前规范明确规定：

- 合法 Notification 被接受：`202 Accepted`，无 Body；
- Origin 无效：`403 Forbidden`；
- 请求缺少必要 `_meta`、版本不支持或 Header 不匹配：`400 Bad Request`；
- MCP 方法不存在：`404 Not Found`，并带 `-32601` JSON-RPC Error；
- 旧版 GET/DELETE 发到仅支持当前版本的 Endpoint：`405 Method Not Allowed`。

Client 不能只检查 HTTP 状态，也不能只检查 JSON Body。现代 MCP Server 的 `400` Body 可能告诉 Client 应修正 Header、声明能力或选择共同版本，而不是回退到旧 `initialize`。

### 8.3 JSON-RPC Protocol Error

常见标准错误码：

| Code | 含义 | 典型原因 | 推荐处理 |
|---|---|---|---|
| `-32700` | Parse error | JSON 语法损坏 | 修复编码或帧，不重试原字节 |
| `-32600` | Invalid Request | JSON-RPC 结构不合法 | 修复协议结构 |
| `-32601` | Method not found | 方法不存在或版本不匹配 | 检查版本、能力和方法名 |
| `-32602` | Invalid params | 参数或必需 `_meta` 不合法 | 按 Schema 修正参数 |
| `-32603` | Internal error | Server 内部异常 | 记录诊断；仅对可安全重试操作退避重试 |

### 8.4 MCP 定义的协议错误

当前版本在 JSON-RPC Server Error 保留区定义：

| Code | 名称 | 含义 |
|---|---|---|
| `-32020` | `HeaderMismatch` | HTTP Header 缺失、格式错误或与 Body 不一致 |
| `-32021` | `MissingRequiredClientCapability` | Request 需要 Client 未声明的能力 |
| `-32022` | `UnsupportedProtocolVersion` | Server 不支持本请求版本，并返回支持版本 |

旧版本曾使用 `-32002` 表示 Resource Not Found；2026-07-28 Server 不应再发出它，但 Client 为兼容旧 Server 仍应接受。当前 Resource 不存在使用 `-32602`。

### 8.5 业务或 Tool Execution Error

Tool 已被正确找到并执行，但业务条件失败时，规范建议返回正常 JSON-RPC Result，并在 Tool 结果中设置 `isError: true`：

```json
{
  "jsonrpc": "2.0",
  "id": "req-42",
  "result": {
    "resultType": "complete",
    "content": [
      {
        "type": "text",
        "text": "departure_date must be in the future"
      }
    ],
    "isError": true
  }
}
```

这与 `-32602 Invalid params` 不同：

- Request 结构、工具名或协议参数不合法，属于 JSON-RPC Error；
- 日期虽是合法字符串，但违反“必须是未来日期”的业务规则，属于 Tool Execution Error；
- 执行错误适合反馈给模型，让模型修改输入后自我纠正；
- Protocol Error 通常说明 Client 实现或协议使用有问题。

### 8.6 Local Client Error

SDK 还会产生不来自 Server 的错误：

- `TimeoutError`；
- `CancelledError`；
- JSON 解码失败；
- Schema 校验失败；
- 响应 ID 未知或重复；
- 本地队列、进程启动或网络库异常。

不要随意把它们伪造成 Server 返回的 JSON-RPC Error，否则监控会错误归因。建议错误对象包含：

```text
origin: local | transport | peer
phase: connect | send | receive | decode | validate | execute
retryable: true | false | unknown
request_id
trace_id
http_status（如有）
jsonrpc_code（如有）
```

## 九、错误与重试决策

| 失败 | 能否重试 | 注意事项 |
|---|---|---|
| Parse/Invalid Request | 修复后再发 | 原请求不变地重发没有意义 |
| Method Not Found | 条件性 | 重新发现版本和能力，避免无限重试 |
| Invalid Params | 修复后再发 | 先按 Schema 校验，避免把服务当验证器 |
| Unsupported Version | 可以 | 从双方版本交集选择新版本 |
| Missing Capability | 条件性 | 只有 Client 真能提供该能力时才能声明后重试 |
| Header Mismatch | 可以 | 重新获取 Tool Schema，修正 Header/Body 后再试 |
| Internal Error/HTTP 503 | 条件性 | 指数退避、抖动、熔断；有副作用时检查幂等性 |
| HTTP 429 | 延迟后 | 遵循 Retry-After，并进入速率限制预算 |
| Timeout/断线 | 结果未知 | 查询操作状态，不能默认 Server 未执行 |
| Tool Execution Error | 视错误而定 | 把可行动信息交给模型，限制自我修正次数 |
| Cancelled | 通常不自动重试 | 取消表达用户或上层已经不再需要结果 |

最危险的错误是“结果未知”：请求可能已经执行成功，只是 Response 丢失。对支付、发信、建单等操作，必须依赖 Idempotency Key 或查询接口解决，而不是简单重发。

## 十、一个完整的健壮调用状态机

```text
CREATED
  → SENDING
  → IN_FLIGHT
      ├─ Progress/Message → IN_FLIGHT
      ├─ Result           → SUCCEEDED
      ├─ JSON-RPC Error   → FAILED_REMOTE
      ├─ Tool isError     → FAILED_DOMAIN
      ├─ Timeout          → CANCELLING → TIMED_OUT
      ├─ User Cancel      → CANCELLING → CANCELLED
      └─ Transport Close  → FAILED_TRANSPORT / OUTCOME_UNKNOWN
```

实现时需要保证：

- 每个终态只能写入一次；
- 进入取消或超时终态后忽略迟到 Response；
- 清理 `pending` 表、Timer、SSE Reader 和子进程资源；
- 记录“是否可能已经产生副作用”；
- 重试产生新的 Request ID，但沿用同一个 Trace ID；
- 对有幂等键的重试沿用同一业务 Idempotency Key。

## 十一、实现检查清单

### JSON-RPC

- Request ID 在 in-flight 集合内唯一；
- 按 ID 关联乱序 Response；
- Notification 无 ID 且绝不返回 Response；
- 检查 `jsonrpc: "2.0"`、Result/Error 互斥和 `resultType`；
- 未知 Response ID 记录告警并丢弃。

### STDIO

- 按 UTF-8 和换行做增量解帧；
- stdout 仅允许协议消息，stderr 承载日志；
- 设置单帧、缓冲区和并发上限；
- stdin EOF 后执行优雅退出；
- 子进程重启后重新发现能力和建立订阅。

### Streamable HTTP

- 每个 JSON-RPC 消息独立 POST；
- `Accept` 同时声明 JSON 和 SSE；
- 验证协议版本与方法 Header；
- 同时处理 HTTP 状态和 JSON-RPC Error；
- 禁用代理缓冲并设计 Keep-alive；
- 断线后重新订阅，不依赖 Last-Event-ID；
- 校验 Origin、认证、Body 大小和 Header/Body 一致性。

### 取消与错误

- 每个请求有可配置 Timeout 和绝对最大时长；
- STDIO 发取消通知，HTTP 关闭请求 SSE；
- 处理 Cancel/Complete 竞态和迟到 Response；
- 区分 Transport、HTTP、JSON-RPC、Domain 和 Local Error；
- 只有满足幂等或有状态查询保障时才自动重试。

## 十二、建议测试用例

至少覆盖：

1. 两个并发请求乱序返回，仍按 ID 正确完成；
2. STDIO 一条 JSON 被拆成多次读取；
3. STDIO 一次读取包含多行 JSON；
4. Server stdout 混入日志时拒绝并给出诊断；
5. HTTP JSON Response 和 SSE Response 两种路径；
6. SSE 在最终 Response 前发送多条 Progress；
7. 多个订阅事件按 `subscriptionId` 正确解复用；
8. STDIO 取消后忽略迟到 Response；
9. HTTP 关闭 SSE 后释放 Server 工作；
10. Timeout 即使持续收到 Progress 也受硬上限约束；
11. JSON-RPC Error 与 Tool `isError` 被分到不同错误类型；
12. 有副作用请求在断线后进入 `OUTCOME_UNKNOWN`，不会自动重放；
13. Header 与 Body 不一致时返回 `-32020`；
14. 进程或 HTTP 重连后重新建立订阅并刷新目录。

## 十三、总结

掌握这部分协议的关键不是记住名词，而是形成分层判断：

1. JSON-RPC 用 `id` 建立 Request/Response 关联，用无 `id` 的 Notification 表达单向事件。
2. STDIO 是共享流，需要换行解帧、ID 解复用和 stdout/stderr 严格隔离。
3. Streamable HTTP 当前版本是每消息一个 POST，Response 可为 JSON 或该请求专属 SSE；长期事件通过 `subscriptions/listen`。
4. Notification 不保证处理确认，也不是可靠事件日志，重连后需要刷新和重订阅。
5. Cancellation 是尽力停止未来工作，不回滚已经发生的副作用，并存在完成与取消竞态。
6. Error 必须区分 Transport、HTTP、JSON-RPC、MCP 协议、Tool 业务和本地 SDK 六种来源，重试策略由错误来源和幂等性共同决定。

一句话概括：

> 用 JSON-RPC 表达意图，用 Transport 传递消息，用 ID 关联并发，用 Notification 传递事件，用 Cancellation 收敛无用工作，用分层 Error 模型决定恢复方式。

## 参考资料

- [MCP Base Protocol](https://modelcontextprotocol.io/specification/2026-07-28/basic)
- [Transport Overview](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports)
- [STDIO Transport](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/stdio)
- [Streamable HTTP Transport](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http)
- [Subscriptions](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/subscriptions)
- [Cancellation](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/cancellation)
- [JSON-RPC 2.0 Specification](https://www.jsonrpc.org/specification)
