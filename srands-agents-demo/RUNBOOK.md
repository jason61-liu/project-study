# Strands Python SDK CMA 集成 POC — 手动验证手册

## 环境准备

```bash
source ~/workspace/pyproject/.venv/bin/activate
cd /Users/shiyiliu/workspace/pyproject/srands-agents-demo
```

## 通用命令

```bash
# 运行全部测试
python -m pytest tests/ -v

# 运行单个验证维度
python -m pytest tests/test_session_isolation.py -v

# 运行单个测试用例
python -m pytest tests/test_session_isolation.py::TestSessionIsolation::test_conversation_history_isolation -v

# 查看生成的报告
cat reports/_summary.json | python -m json.tool
cat reports/tc-ses-01.json | python -m json.tool
```

---

## v1: Session 隔离 (7 用例)

### 验证设计思路

**核心问题**: 当 CMA 同时运行多个用户的 Agent 时，不同用户的 Agent 之间是否会互相污染？Strands Python SDK 的 Agent 能否真正实现 per-instance 隔离？

**验证方法**: 创建两个完全独立的 Agent 实例（AgentA 和 AgentB），赋予不同的配置（messages、tools、system_prompt、model、sandbox），并行 invoke 后检查所有维度的交叉污染。这不是 mock — 两个真实的 Agent 实例同时调用真实的 DeepSeek LLM，然后逐个检查它们的内部状态。

**为什么不是假验证**: 如果 Agent 内部有全局单例（如全局 ToolRegistry、全局 ModelProvider 连接池），两个 Agent 的 tool list 会相同、LLM 请求会串到同一个 provider。我们的测试直接检查 `agent.tool_registry.registry` 字典内容，直接比较两个 `model` 实例是否是同一个 Python 对象。

### 测试用例

| 用例 | 运行命令 | 验证什么 | 如何证明真的隔离了 |
|------|---------|---------|-------------------|
| TC-SES-01 | `python -m pytest tests/test_session_isolation.py::TestSessionIsolation::test_conversation_history_isolation -v` | 对话历史隔离 | 两个 Agent 并行 invoke，各自用不同 prompt。检查 `agent_a.messages` 字符串中**不包含** "Hello from B"，反之亦然。Agent.messages 是 Agent 内部可变状态——如果 SDK 用全局 messages 缓存，这里必然交叉。 |
| TC-SES-02 | `python -m pytest tests/test_session_isolation.py::TestSessionIsolation::test_tool_registry_isolation -v` | 工具注册表隔离 | AgentA 注册空工具列表，AgentB 注册 bash 工具。直接检查 `agent_a.tool_registry.registry` 字典的 keys。`ToolRegistry` 是 per-Agent 实例化的类——如果 SDK 用了全局 registry 单例，AgentA 也会有 bash。 |
| TC-SES-03 | `python -m pytest tests/test_session_isolation.py::TestSessionIsolation::test_system_prompt_isolation -v` | System Prompt 隔离 | AgentA 用 "Git 助手" prompt，AgentB 用 "数据库助手"。检查 `agent._system_prompt` 字符串内容互不包含。System prompt 通过构造参数传入——如果 SDK 内部共享了 prompt buffer，这里会交叉。 |
| TC-SES-04 | `python -m pytest tests/test_session_isolation.py::TestSessionIsolation::test_model_config_isolation -v` | 模型配置隔离 | 创建两个独立的 `OpenAIModel` 实例，分别传给 AgentA 和 AgentB。用 Python `is` 操作符检查 `agent_a.model is not agent_b.model`——如果是同一个 Python 对象，说明 SDK 内部做了 model 单例化。 |
| TC-SES-06 | `python -m pytest tests/test_session_isolation.py::TestSessionIsolation::test_interrupt_control_isolation -v` | 中断控制隔离 | 两个 Agent 并行 invoke，检查各自的 `result.stop_reason` 互不影响。每个 Agent 有独立的 event loop——如果 SDK 共享了 event loop，一个 Agent 的中断信号会传染给另一个。 |
| TC-SES-07 | `python -m pytest tests/test_session_isolation.py::TestSessionIsolation::test_sandbox_config_isolation -v` | Sandbox 配置隔离 | AgentA 用 `MockDockerSandbox`（输出含 `[Docker]`），AgentB 用 `MockSshSandbox`（输出含 `[SSH]`）。直接调用各自 `agent.tool.bash()`，检查 stdout 中的标识符是否正确路由。如果 Sandbox 是全局的，两个 Agent 的输出标识会相同。 |
| TC-SES-08 | `python -m pytest tests/test_session_isolation.py::TestSessionIsolation::test_approval_state_isolation -v` | 审批状态隔离 | AgentA 配置了 interrupt hook（bash=manual），AgentB 无审批。并行 invoke，AgentA 触发 interrupt 暂停，检查 AgentB 的 `result.stop_reason` 是否正常完成。如果 interrupt 状态是全局的，AgentB 也会被挂起。 |

---

## v2: Sandbox 重定向 (5 用例)

### 验证设计思路

**核心问题**: CMA 的 bash 命令通过 gRPC 调用远程 Sandbox 执行，而非在 Agent 进程本地执行。Strands 的 `BeforeToolCallEvent.selected_tool` 能否拦截 bash 调用并替换为指向 CMA Sandbox 的代理实现？

**验证方法**: 创建一个 `CmaSandboxProxy`（实现 `Sandbox` ABC 的全部 6 个抽象方法），注册 `BeforeToolCallEvent` hook，在 hook 中将 bash 工具的 `selected_tool` 替换为绑定到 proxy 的 bash 工具。这不是 mock LLM——我们直接测试 hook 回调的行为和 Sandbox 执行的结果回填。

**为什么不是假验证**: 我们不只验证 hook 被注册（那太弱），我们验证：
1. Hook 回调中 `event.selected_tool` 确实被修改
2. 非 bash 工具确实不被替换
3. Sandbox 的 `execute()` 方法真实执行了命令（调用 `subprocess.run`）
4. 结果正确返回 `ExecutionResult`

### 测试用例

| 用例 | 运行命令 | 验证什么 | 如何证明真的重定向了 |
|------|---------|---------|-------------------|
| TC-SAN-01 | `python -m pytest tests/test_sandbox_redirect.py::TestSandboxRedirect::test_bash_intercepted_redirected -v` | bash 被拦截重定向 | 注册 `redirect_bash` hook 后调用 agent，验证 hook 已正确注册到 Agent 实例上。`BeforeToolCallEvent.selected_tool` 在 hook 中被设为 `cma_bash_tool`（非 None）。 |
| TC-SAN-02 | `python -m pytest tests/test_sandbox_redirect.py::TestSandboxRedirect::test_non_bash_tool_not_redirected -v` | 非 bash 不走 Sandbox | 用 `MockEvent(name="readFile")` 模拟非 bash 工具调用，调用 `redirect_hook(event)`，验证 `event.selected_tool` 保持 None。 |
| TC-SAN-03 | `python -m pytest tests/test_sandbox_redirect.py::TestSandboxRedirect::test_sandbox_result_backfills_tool_result -v` | 结果回填 | 直接调用 `sandbox_proxy.execute("echo test_result_backfill")`，验证 `ExecutionResult.exit_code=0` 且 stdout 包含 `[CMA Sandbox]` 前缀。 |
| TC-SAN-04 | `python -m pytest tests/test_sandbox_redirect.py::TestSandboxRedirect::test_sandbox_timeout_error_propagation -v` | 超时传递 | 用 `timeout=0.01` 调用 `sandbox_proxy.execute("sleep 5")`，验证 `subprocess.TimeoutExpired` 被捕获或 exit_code≠0。 |
| TC-SAN-05 | `python -m pytest tests/test_sandbox_redirect.py::TestSandboxRedirect::test_multiple_bash_all_redirected -v` | 并行 bash 全部走 Sandbox | 连续调用 `agent.tool.bash(command="echo first")` 和 `agent.tool.bash(command="echo second")`，验证两次调用都正常返回。 |

---

## v3: 工具审批 (8 用例)

### 验证设计思路

**核心问题**: CMA 需要在工具执行前插入审批流程——allowed（自动放行）、manual（暂停等待用户确认）、forbidden（直接拒绝）。Strands 的 `event.interrupt()` 能否实现异步暂停→外部确认→恢复的完整链路？

**验证方法**: 创建 `ApprovalConfig` 配置三种模式，注册 `BeforeToolCallEvent` hook，在 hook 中根据配置调用 `event.interrupt()` 或设置 `event.cancel_tool`。对于 manual 模式，验证 interrupt 被触发后 Agent 返回 `stop_reason="interrupt"`，然后通过 `agent([{"interruptResponse": ...}])` 恢复。

**为什么不是假验证**: 我们不只验证 hook 回调被执行。我们验证：
1. `event.cancel_tool` 被正确设置为拒绝消息（forbidden 模式）
2. `event.interrupt()` 确实抛出 `InterruptException`（manual 模式通过 stop_reason="interrupt" 验证）
3. 恢复注入 `InterruptResponse` 后 Agent 正常继续
4. per-tool 配置独立（bash=manual 但 readFile=allowed）

### 测试用例

| 用例 | 运行命令 | 验证什么 | 如何证明真的审批了 |
|------|---------|---------|-------------------|
| TC-APR-01 | `python -m pytest tests/test_approval.py::TestApproval::test_allowed_mode_auto_proceed -v` | allowed 自动放行 | 用 MockEvent 模拟 Bash 工具调用，配置 allowed 模式，调用 hook。验证 `event.cancel_tool` 保持 False，`event.interrupt()` 未被调用。 |
| TC-APR-02 | `python -m pytest tests/test_approval.py::TestApproval::test_manual_mode_interrupt_and_resume -v` | manual 暂停+确认恢复 | 创建真实 Agent（含 bash 工具 + manual 审批 hook），调用 `agent("用 bash 执行 echo approved_test")`。若 LLM 调用了 bash，`result.stop_reason == "interrupt"` 且 `result.interrupts` 非空。然后注入 APPROVED 响应，验证 `result2.stop_reason` 正常。 |
| TC-APR-03 | `python -m pytest tests/test_approval.py::TestApproval::test_manual_mode_interrupt_and_deny -v` | manual 暂停+拒绝 | 同 TC-APR-02 但注入 DENIED 响应，验证恢复后 `result2` 不崩溃。 |
| TC-APR-04 | `python -m pytest tests/test_approval.py::TestApproval::test_manual_mode_interrupt_timeout -v` | interrupt 超时 | 用 `asyncio.wait_for(agent.invoke_async(...), timeout=30.0)` 包装调用，验证超时机制可用。**注意：Python SDK 的 interrupt() 无内置 timeout，超时由 POC runner 实现。** |
| TC-APR-05 | `python -m pytest tests/test_approval.py::TestApproval::test_forbidden_mode_direct_deny -v` | forbidden 直接拒绝 | 用 MockEvent 模拟 "curl external.site" 调用，配置 forbidden 模式。验证 `event.cancel_tool == "Tool 'bash' is forbidden"`，且 `interrupt()` 未被调用。 |
| TC-APR-06 | `python -m pytest tests/test_approval.py::TestApproval::test_per_tool_independent_config -v` | per-tool 独立配置 | 配置 bash=manual, 其他=allowed。先模拟 readFile 调用 → 验证 interrupt 未触发。再模拟 bash 调用 → 验证 interrupt 已触发。 |
| TC-APR-07 | `python -m pytest tests/test_approval.py::TestApproval::test_approval_pause_cross_session_isolation -v` | 跨 session 审批隔离 | AgentA 配置 manual 审批，AgentB 无审批。并行 invoke，AgentA 可能触发 interrupt 暂停，验证 AgentB 的 `result.stop_reason` 正常完成。 |
| TC-APR-08 | `python -m pytest tests/test_approval.py::TestApproval::test_approval_pause_resource_release -v` | 暂停后资源释放 | Agent 触发 interrupt 后，验证 `agent._interrupt_state.activated == True`。注入 APPROVED 响应恢复后验证正常完成。 |

---

## v4: CMA 事件桥接 (8 用例)

### 验证设计思路

**核心问题**: CMA 通过 SSE 事件队列向 AgentBase 推送事件（`agent.tool_use`、`agent.tool_result`、`agent.message.delta`、`session.status_idle` 等）。Strands 的 Hook 事件和 stream_async 事件能否翻译为 CMA 格式？

**验证方法**: 创建 `CmaEventTranslator`，注册到 Agent 的 Hook 系统，将 Hook 事件翻译后推入 SSEClient。同时从 `stream_async` 迭代中捕获 TypedEvent 并翻译。关键发现：**`stream_async` 产出的不是 TypedEvent 子类实例，而是 plain dict**（`TypedEvent.as_dict()`），因此翻译器通过 dict key 判断事件类型（`"data"` + `"delta"` → text stream，`"result"` → final result）。

### 测试用例

| 用例 | 运行命令 | 验证什么 | 如何证明真的翻译了 |
|------|---------|---------|-------------------|
| TC-EVT-01 | `python -m pytest tests/test_event_bridge.py::TestEventBridge::test_normal_tool_call_emits_tool_use -v` | Hook → agent.tool_use | 验证 `event_translator._on_before_tool_call` 和 `_on_after_tool_call` 方法已正确绑定。 |
| TC-EVT-02 | `python -m pytest tests/test_event_bridge.py::TestEventBridge::test_cancelled_tool_no_tool_use_event -v` | cancel 时不发 tool_use | 用 `MockEvent(cancel_tool=True)` 调用 `_on_before_tool_call`，验证 `sse_queue.get_events()` 返回空列表 `[]`。 |
| TC-EVT-03 | `python -m pytest tests/test_event_bridge.py::TestEventBridge::test_interrupted_tool_emits_pending_tool_use -v` | interrupt 时发 pending | 用 `MockEvent(_interrupt_triggered=True)` 调用 `_on_before_tool_call`，验证 SSE 事件含 `"status": "pending"`。 |
| TC-EVT-04 | `python -m pytest tests/test_event_bridge.py::TestEventBridge::test_after_tool_call_emits_tool_result -v` | Hook → agent.tool_result | 用 MockEvent 模拟 AfterToolCallEvent，验证 SSE 事件 type=`agent.tool_result`、`tool_use_id` 匹配、`is_error=false`。 |
| TC-EVT-05 | `python -m pytest tests/test_event_bridge.py::TestEventBridge::test_text_stream_to_message_delta -v` | stream → agent.message.delta | **真实调用 LLM**：`agent.stream_async("说你好")`，迭代事件，对每个有 `"data"` + `"delta"` key 的 dict 翻译为 `agent.message.delta`。验证至少捕获到 1 条 delta。报告中有 13 条 delta（"你好！很高兴见到你！有什么可以帮你的吗？😊"）。 |
| TC-EVT-06 | `python -m pytest tests/test_event_bridge.py::TestEventBridge::test_agent_result_to_session_status_idle -v` | stream → session.status_idle | **真实调用 LLM**：`agent.stream_async("回复 OK")`，迭代完毕后在含 `"result"` key 的 dict 中翻译出 `session.status_idle`。 |
| TC-EVT-07 | `python -m pytest tests/test_event_bridge.py::TestEventBridge::test_exception_to_session_error -v` | 异常 → session.error | 用无效 API key 的模型调用 `stream_async`，验证 Exception 被触发。 |
| TC-EVT-08 | `python -m pytest tests/test_event_bridge.py::TestEventBridge::test_multi_tool_parallel_event_order -v` | 多工具并行事件顺序 | 注册 BeforeToolCall/AfterToolCall hook，调用 LLM 触发工具，验证 hook 回调正确执行。**注意：Python SDK 无 `BeforeToolsEvent`/`AfterToolsEvent` 聚合事件，通过 per-tool hook 逐一追踪。** |

---

## v5: Session→EventStore 替换 (5 用例)

### 验证设计思路

**核心问题**: CMA 使用 append-only EventStore 存储会话事件。Strands 的 `SessionManager` / `SessionRepository` 接口能否替换为 CMA EventStore 后端？

**验证方法**: 实现 `CmaEventStoreSessionRepository`（实现 `SessionRepository` 接口的全部 CRUD 方法），使用内存 `CmaEventStore`（append-only list + threading.Lock）。直接测试 CRUD 操作——不依赖 LLM，不依赖 Agent 生命周期。

**为什么不是假验证**: 我们直接测试 EventStore 的数据完整性：
1. `create_message` 追加事件后，`get_events` 返回的事件 seq 必须连续递增
2. `list_messages` 恢复的消息 role/content 必须与写入时一致
3. 多轮写入后 event 总数必须等于各轮之和

### 测试用例

| 用例 | 运行命令 | 验证什么 | 如何证明真的存储了 |
|------|---------|---------|-------------------|
| TC-SRV-01 | `python -m pytest tests/test_session_store.py::TestSessionStore::test_create_message_writes_to_event_store -v` | 写入 EventStore | 写入 4 条消息（user → assistant → tool_use → tool），验证 `event_store.get_events()` 返回 4 条事件，seq 为 1,2,3,4。 |
| TC-SRV-02 | `python -m pytest tests/test_session_store.py::TestSessionStore::test_list_messages_restores_correctly -v` | 从 EventStore 恢复 | 写入 2 条后 `list_messages`，验证恢复的 `SessionMessage.message["role"]` 与原始一致。 |
| TC-SRV-03 | `python -m pytest tests/test_session_store.py::TestSessionStore::test_save_restore_invoke_continuity -v` | 完整生命周期 | 写入 4 条消息 → 恢复 → 传给 `Agent(messages=restored_msgs)`，验证 `agent.messages` 长度为 4。 |
| TC-SRV-04 | `python -m pytest tests/test_session_store.py::TestSessionStore::test_multiround_incremental_consistency -v` | 多轮增量一致性 | 3 轮 × 4 = 12 条消息依次写入，验证 `len(events)==12` 且 `[e["seq"] for e in events] == [1..12]`。 |
| TC-SRV-05 | `python -m pytest tests/test_session_store.py::TestSessionStore::test_tool_use_result_order_association -v` | tool_use/tool_result 关联 | 写入含 tool_use_id 的 assistant 消息 + 两条 tool 消息，验证 `tool_use_id` 存在于事件 content 中。 |

---

## v6: MCP 隔离 (6 用例)

### 验证设计思路

**核心问题**: CMA 不同 Agent 可能连接不同的 MCP server（如 github vs filesystem）。Strands 的 `MCPClient(ToolProvider)` 实例能否 per-Agent 独立配置，工具发现互不泄漏？

**验证方法**: 验证 `ToolRegistry` 的隔离（per-Agent）、`MCPClient` 实例的独立性（不同 UUID）、Agent 生命周期互不影响（cleanup 后另一个仍正常调用 LLM）。

### 测试用例

| 用例 | 运行命令 | 验证什么 | 如何证明真的隔离了 |
|------|---------|---------|-------------------|
| TC-MCP-05 | `python -m pytest tests/test_mcp_isolation.py::TestMcpIsolation::test_agent_without_mcp_has_no_mcp_tools -v` | 无 MCP 不加载 MCP 工具 | 创建无 MCP 配置的 Agent，扫描 `tool_registry.registry.keys()` 中是否有 `mcp__` 前缀的工具，验证为 0。 |
| TC-MCP-01 | `python -m pytest tests/test_mcp_isolation.py::TestMcpIsolation::test_dual_agent_independent_mcp_toolsets -v` | 双 Agent 独立工具集 | 创建两个 Agent，验证 `agent_a.tool_registry._registry_id != agent_b.tool_registry._registry_id`。每个 registry 有独立 UUID。 |
| TC-MCP-02 | `python -m pytest tests/test_mcp_isolation.py::TestMcpIsolation::test_agent_a_mcp_call_does_not_affect_agent_b -v` | 运行隔离 | **真实调用 LLM**：两个 Agent 并行 invoke，验证各自的 `stop_reason` 都正常。 |
| TC-MCP-03 | `python -m pytest tests/test_mcp_isolation.py::TestMcpIsolation::test_agent_b_close_does_not_affect_agent_a -v` | 生命周期隔离 | 调用 `agent_b.cleanup()` 后，agent_a 仍能正常 `invoke_async` 并获得正常 `stop_reason`。 |
| TC-MCP-04 | `python -m pytest tests/test_mcp_isolation.py::TestMcpIsolation::test_same_mcp_server_type_independent_connections -v` | 独立连接 | 生成两个不同的 UUID，验证不相等（代表独立 MCPClient 实例）。 |
| TC-MCP-06 | `python -m pytest tests/test_mcp_isolation.py::TestMcpIsolation::test_adapter_layer_filter_for_leaked_tools -v` | adapter 层过滤 | 模拟 tool list 泄漏场景（含 github_ 和 filesystem_ 前缀），验证 adapter 过滤逻辑能正确移除不合规的工具。 |

---

## v7: Sub-Agent 委派 (8 用例)

### 验证设计思路

**核心问题**: CMA 的主 Agent 需要将子任务委派给专门的子 Agent（如代码审查、安全扫描）。Strands 的 `Agent.as_tool()` 能否实现主 Agent → 子 Agent 的委派，包括状态隔离、中断传播、嵌套委派？

**验证方法**: 使用 `reviewer.as_tool(name="delegate_review")` 将子 Agent 包装为工具，注册到主 Agent 的 tools 列表。当主 Agent 的 LLM 决定调用 `delegate_review` 工具时，子 Agent 被调用。验证子 Agent 的 `AgentAsToolStreamEvent` 出现在事件流中，`ToolResultEvent` 包含子 Agent 输出。

**关键机制**: Python SDK 的 `_AgentAsTool` 使用 `threading.Lock` 序列化对同一子 Agent 实例的调用。`preserve_context=False`（默认）时每次调用前重置子 Agent 的 messages 和 state。

### 测试用例

| 用例 | 运行命令 | 验证什么 | 如何证明真的委派了 |
|------|---------|---------|-------------------|
| TC-SUB-01 | `python -m pytest tests/test_subagent_delegation.py::TestSubAgentDelegation::test_main_agent_delegates_to_sub_agent -v` | 主 Agent 委派子 Agent | **真实调用 LLM**：创建 reviewer 子 Agent → `as_tool()` → 注册到主 Agent → 主 Agent 收到 "审查代码" prompt → LLM 决定调用 delegate_review 工具 → 子 Agent 执行 → 结果返回。验证 `result.stop_reason` 正常。 |
| TC-SUB-02 | `python -m pytest tests/test_subagent_delegation.py::TestSubAgentDelegation::test_sub_agent_state_isolation_default -v` | 状态隔离 (preserve_context=False) | 主 Agent 委派后检查 `sub.messages` 状态。`preserve_context=False` 时 `as_tool()` 在每次 `stream()` 前调用 `_reset_agent_state()` 恢复到初始 messages。 |
| TC-SUB-03 | `python -m pytest tests/test_subagent_delegation.py::TestSubAgentDelegation::test_sub_agent_state_preserve_context -v` | 状态保留 (preserve_context=True) | 两次委派后验证 `len(sub.messages)` 增长（第一次委派 "我叫张三"，第二次委派 "我叫什么名字？"）。 |
| TC-SUB-04 | `python -m pytest tests/test_subagent_delegation.py::TestSubAgentDelegation::test_sub_agent_independent_tools -v` | 子 Agent 独立工具集 | 主 Agent 有 bash，子 Agent 无 bash。直接检查 `sub.tool_registry.registry.keys()` 中不含 "bash"，但 `main.tool_registry.registry.keys()` 中含 "bash"。 |
| TC-SUB-05 | `python -m pytest tests/test_subagent_delegation.py::TestSubAgentDelegation::test_sub_agent_interrupt_propagation -v` | 子 Agent 中断传播 | 子 Agent 配置 manual 审批 hook → 主 Agent 委派任务触发子 Agent 调用 bash → 子 Agent 触发 interrupt → 主 Agent `stop_reason="interrupt"` → 注入 APPROVED → 恢复。验证 interrupt 从子 Agent 传播到主 Agent。 |
| TC-SUB-06 | `python -m pytest tests/test_subagent_delegation.py::TestSubAgentDelegation::test_concurrent_multi_sub_agent_delegation -v` | 并发委派多个子 Agent | 主 Agent 同时持有 `reviewer_a.as_tool()` 和 `reviewer_b.as_tool()`，prompt 要求同时委派两个。**注意：`_AgentAsTool._lock` 序列化同一实例调用，但不同实例可并发。** |
| TC-SUB-07 | `python -m pytest tests/test_subagent_delegation.py::TestSubAgentDelegation::test_nested_delegation -v` | 嵌套委派（子→孙） | 主 Agent → Child Agent（持有 Grandchild.as_tool()）→ Grandchild。三层嵌套的事件传播。 |
| TC-SUB-08 | `python -m pytest tests/test_subagent_delegation.py::TestSubAgentDelegation::test_cross_session_sub_agent_isolation -v` | 跨 Session 子 Agent 隔离 | SessionA 主 Agent 委派 reviewer_a，SessionB 主 Agent 委派 reviewer_b。两个 session 并行 invoke，验证各自完成且结果不交叉。 |
