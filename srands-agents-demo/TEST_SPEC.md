# POC 测试用例详细规格说明书

> 每个测试用例包含：测试目标、测试思路、执行流程、验证断言、失败含义

---

## v1: Session 隔离 (7 用例)

### TC-SES-01: 对话历史隔离

**测试目标**: 验证两个独立 Agent 实例的对话历史（`agent.messages`）互不污染。这是最基础的隔离验证——如果对话历史都隔离不了，其他维度无从谈起。

**测试思路**: 创建 AgentA 和 AgentB，各自赋不同的 system_prompt，并行调用真实 DeepSeek LLM（`asyncio.gather`）。AgentA 回复 "Hello from A"，AgentB 回复 "Hello from B"。检查 `agent_a.messages` 的字符串表示中不包含 AgentB 的回复内容。

**执行流程**:
1. `Agent(model=deepseek, system_prompt="你是 Agent A")` → 独立实例
2. `Agent(model=deepseek, system_prompt="你是 Agent B")` → 独立实例
3. `asyncio.gather(a.invoke_async("回复'Hello from A'"), b.invoke_async("回复'Hello from B'"))`
4. 检查 `str(agent_a.messages)` 不包含 `"Hello from B"`
5. 检查 `str(agent_b.messages)` 不包含 `"Hello from A"`

**验证断言**:
- `agent_a.messages` 字符串不含 "Hello from B"
- `agent_b.messages` 字符串不含 "Hello from A"

**失败含义**: Agent 内部有全局 messages 缓存。CMA 多用户场景下，用户 A 的对话历史会泄露给用户 B。

---

### TC-SES-02: 工具注册表隔离

**测试目标**: 验证两个 Agent 的 `ToolRegistry` 独立——AgentA 注册空工具列表，AgentB 注册 bash 工具，AgentA 的工具列表中不应出现 bash。

**测试思路**: 不调 LLM。直接检查 `agent.tool_registry.registry` 字典。`ToolRegistry` 在 `Agent.__init__()` 中通过 `self.tool_registry = ToolRegistry()` 创建——如果 SDK 有全局 registry 单例，两个 Agent 的 registry 会是同一个对象。

**执行流程**:
1. AgentA: `tools=[]`
2. AgentB: `tools=[make_bash(name="bash")]`
3. 检查 `agent_a.tool_registry.registry.keys()` 不含 `"bash"`
4. 检查 `agent_b.tool_registry.registry.keys()` 含 `"bash"`

**验证断言**:
- `"bash" not in agent_a.tool_registry.registry.keys()`
- `"bash" in agent_b.tool_registry.registry.keys()`

**失败含义**: SDK 的 ToolRegistry 是全局共享的。AgentA 的 LLM 会看到本不属于它的 bash 工具，造成越权操作风险。

---

### TC-SES-03: System Prompt 隔离

**测试目标**: 验证两个 Agent 的 system prompt 互不包含对方的内容。

**测试思路**: AgentA 的 system_prompt 是 "你是 Git 助手"，AgentB 的是 "你是数据库助手"。直接检查 `agent._system_prompt` 字符串。System prompt 通过构造参数传入，存储为实例属性。

**执行流程**:
1. AgentA: `system_prompt="你是 Git 助手"`
2. AgentB: `system_prompt="你是数据库助手"`
3. 检查 `"Git 助手" in agent_a._system_prompt`
4. 检查 `"数据库助手" in agent_b._system_prompt`
5. 检查 `"数据库" not in agent_a._system_prompt`

**验证断言**:
- AgentA 含 "Git 助手"
- AgentB 含 "数据库助手"
- AgentA 不含 "数据库"

**失败含义**: System prompt 被共享或串扰，Agent 的行为指令混乱。

---

### TC-SES-04: 模型配置隔离

**测试目标**: 验证两个 Agent 的 Model 实例是独立对象，不会共享 provider 连接。

**测试思路**: 创建两个独立的 `OpenAIModel` 实例，用 Python `is` 操作符比较 `agent_a.model is not agent_b.model`。如果是同一个对象，说明 SDK 做了 model 单例化。

**执行流程**:
1. `model_a = create_deepseek_model()`
2. `model_b = create_deepseek_model()`（另一个实例）
3. AgentA 用 model_a，AgentB 用 model_b
4. 检查 `agent_a.model is not agent_b.model`
5. 检查两者 `model_id` 都是 `"deepseek-chat"`

**验证断言**:
- `agent_a.model is not agent_b.model`
- 两者 `get_config()["model_id"]` 均为 `"deepseek-chat"`

**失败含义**: Model 实例被 SDK 内部共享为单例。不同 Agent 的 LLM 请求会串到同一个 provider 连接，HTTP 请求可能互相阻塞。

---

### TC-SES-06: 中断控制隔离

**测试目标**: 验证两个 Agent 的 event loop 独立——AgentA 的中断信号不会影响 AgentB。

**测试思路**: 两个 Agent 并行 invoke 真实 LLM，各自正常完成。如果 SDK 共享了 event loop 或 AbortSignal，AgentA 的取消信号会传染给 AgentB。

**执行流程**:
1. AgentA 和 AgentB 并行 `invoke_async`
2. 检查 `result_b.stop_reason` 正常
3. 检查 `result_a.stop_reason` 正常

**验证断言**:
- `result_b.stop_reason in ("end_turn", "stop_sequence", "max_tokens")`
- `result_a.stop_reason in ("end_turn", "stop_sequence", "max_tokens")`

**失败含义**: Event loop 共享，Agent 之间互相干扰。

---

### TC-SES-07: Sandbox 配置隔离

**测试目标**: 验证两个 Agent 各自绑定到不同的 Sandbox 实现，命令执行路由到正确的 Sandbox。

**测试思路**: 创建两个真实 `Sandbox` ABC 子类——`SandboxAlpha`（输出带 `[Sandbox-Alpha]` 标记）和 `SandboxBeta`（输出带 `[Sandbox-Beta]` 标记）。AgentA 绑定 Alpha，AgentB 绑定 Beta。各自调用 `agent.tool.bash()`（真实 subprocess.run 执行），验证输出的标记正确。

**执行流程**:
1. 定义 `SandboxAlpha(Sandbox)` 和 `SandboxBeta(Sandbox)`，各自实现 6 个抽象方法
2. `make_bash(sandbox=sandbox_a)` → bash 工具绑定到 Alpha
3. `agent_a.tool.bash(command="echo alpha_test")` → `subprocess.run` 真实执行
4. 检查输出含 `[Sandbox-Alpha]`

**验证断言**:
- `"[Sandbox-Alpha]" in str(result_a)`
- `"[Sandbox-Beta]" in str(result_b)`
- `"[Sandbox-Beta]" not in str(result_a)`

**失败含义**: Sandbox 是全局单例，所有 Agent 共享同一个执行环境。

---

### TC-SES-08: 审批状态隔离

**测试目标**: 验证 interrupt 状态绑定到 Agent 实例——AgentA 触发审批暂停时 AgentB 不受影响。

**测试思路**: AgentA 注册 manual 审批 hook（bash 调用时触发 `event.interrupt()`），AgentB 无审批。并行 invoke 真实 LLM。AgentA 可能因 LLM 调用 bash 而触发 interrupt，AgentB 应正常完成。

**执行流程**:
1. AgentA: bash + manual hook → LLM 可能触发 interrupt
2. AgentB: 无工具无 hook → LLM 直接回复
3. `asyncio.gather(a.invoke_async(), b.invoke_async())`
4. 检查 `result_b.stop_reason` 正常

**验证断言**:
- `result_b.stop_reason in ("end_turn", "stop_sequence", "max_tokens")`

**失败含义**: interrupt 状态是全局的，AgentA 的暂停会阻塞 AgentB。

---

## v2: Sandbox 重定向 (5 用例)

### TC-SAN-01: bash 被拦截重定向

**测试目标**: 验证 `BeforeToolCallEvent.selected_tool` 能在 hook 中被替换，将 bash 调用重定向到 CMA Sandbox Proxy。

**测试思路**: 创建 CmaSandboxProxy（真实 Sandbox ABC 实现）和 `make_cma_redirected_bash(sandbox_proxy)`。注册 `BeforeToolCallEvent` hook：当 `tool_use["name"] == "bash"` 时设置 `event.selected_tool = cma_bash_tool`。创建真实 Agent 并调用 LLM。

**执行流程**:
1. 创建 `CmaSandboxProxy`（实现 Sandbox ABC）
2. `cma_bash_tool = make_cma_redirected_bash(proxy)`
3. 注册 hook: `if tool == "bash": event.selected_tool = cma_bash_tool`
4. 创建 Agent + 调用 LLM

**验证断言**:
- `redirect_hook is not None`
- `sandbox_proxy is not None`

**失败含义**: `selected_tool` 机制不可用，无法实现 Sandbox 重定向。

---

### TC-SAN-02: 非 bash 工具不走 Sandbox

**测试目标**: 验证 redirect hook 不会误替换非 bash 工具。

**测试思路**: 注册 redirect hook 但不给 Agent 任何 bash 工具。调用真实 LLM（prompt 要求只回复不调工具）。检查 `sandbox_proxy.execute_log` 为空——证明没有 Sandbox 调用发生。

**执行流程**:
1. Agent 不注册 bash 工具，只注册 redirect hook
2. LLM 调用: `"回复 OK"`
3. 检查 `len(sandbox_proxy.execute_log) == 0`

**验证断言**:
- `sandbox_proxy.execute_log` 长度为 0

**失败含义**: redirect hook 误拦截了非 bash 调用，导致正常工具异常。

---

### TC-SAN-03: Sandbox 执行结果正确回填

**测试目标**: 验证 CmaSandboxProxy.execute() 返回的 `ExecutionResult` 正确——exit_code、stdout。

**测试思路**: 直接调用 `sandbox_proxy.execute("echo test_result_backfill")`，它是真实 `subprocess.run` 执行。验证返回的 `ExecutionResult.exit_code == 0` 且 stdout 含预期内容。

**执行流程**:
1. `exec_result = await sandbox_proxy.execute("echo test_result_backfill")`
2. 检查 `exec_result.exit_code == 0`
3. 检查 stdout 含命令输出

**验证断言**:
- `exit_code == 0`
- stdout 含 `[CMA Sandbox]` 或命令输出

**失败含义**: Sandbox Proxy 的 execute 方法实现有误，结果无法正确返回。

---

### TC-SAN-04: 超时/错误传递

**测试目标**: 验证 Sandbox 执行超时时正确处理。

**测试思路**: 用极短超时 `timeout=0.01` 执行 `sleep 5`，验证 `subprocess.TimeoutExpired` 被捕获或 exit_code != 0。

**执行流程**:
1. `sandbox_proxy.execute("sleep 5", timeout=0.01)`
2. 验证超时被处理（TimeoutExpired 或 exit_code != 0）

**验证断言**:
- 超时被正确处理（不崩溃）

**失败含义**: 超时未被正确处理，长时间命令会永久阻塞 Agent。

---

### TC-SAN-05: 多个 bash 并行全部走 Sandbox

**测试目标**: 验证 agent.tool.bash() 可以连续多次调用，各自返回正确结果。

**测试思路**: 通过 `agent.tool.bash(command="echo first")` 和 `agent.tool.bash(command="echo second")` 连续两次真实 SDK 工具调用。

**执行流程**:
1. `r1 = agent.tool.bash(command="echo first")`
2. `r2 = agent.tool.bash(command="echo second")`
3. 验证两个结果均非 None

**验证断言**:
- r1 is not None
- r2 is not None

**失败含义**: 连续工具调用有状态污染。

---

## v3: 工具审批 (8 用例)

### TC-APR-01: allowed 模式自动放行

**测试目标**: 验证 allowed 审批模式下工具正常执行，不被取消。

**测试思路**: 创建 Agent + bash 工具 + allowed approval hook。通过 `agent.tool.bash(command="echo allowed_test")` 调用真实 SDK 工具执行链路。allowed 模式下 hook 不设 `cancel_tool`，不调 `interrupt()`。工具应正常执行，返回 `status: "success"`。

**代码路径**: `agent.tool.bash()` → `ToolCaller.__getattr__` → `ToolExecutor._stream()` → `_invoke_before_tool_call_hook()` → `HookRegistry.invoke_callbacks_async()` → 我们的 hook 回调（不设置 cancel/不调 interrupt）→ 工具真实执行（`subprocess.run`）→ 返回结果

**执行流程**:
1. `create_approval_hook(ApprovalConfig(default_mode="allowed"))`
2. `agent = Agent(tools=[bash], hooks=[hook])`
3. `result = agent.tool.bash(command="echo allowed_test")` ← 真实工具执行
4. 检查 `result["status"] == "success"`

**验证断言**:
- `result["status"] == "success"`

**失败含义**: allowed 模式实现有 bug——即使配置为 allowed，工具仍然被意外取消或中断。

---

### TC-APR-02: manual 模式 interrupt 暂停 + 确认恢复

**测试目标**: 验证 SDK 的 `event.interrupt()` → `InterruptException` → `stop_reason="interrupt"` → 注入 `InterruptResponse` → 恢复执行的完整链路。这是整个审批机制的核心。

**测试思路**: 创建 Agent + bash 工具 + manual approval hook。真实 LLM 调用，如果 LLM 决定调用 bash → hook 调 `event.interrupt(name, reason)` → SDK 抛出 `InterruptException` → Agent event loop 捕获 → 设置 `stop_reason="interrupt"` → 返回 `AgentResult(interrupts=[...])`。然后注入 `[{"interruptResponse": {"interruptId": ..., "response": "APPROVED"}}]` 重新 `agent(responses)` → SDK 的 `_InterruptState.resume()` 匹配响应 → 再次经过 hook 时 `interrupt_.response` 不为 None → 返回 response 而不再抛异常 → 工具继续执行。

**代码路径**:
1. 第一轮: `agent("用 bash 执行...")` → LLM 决定调用 bash → BeforeToolCallEvent → hook 调 `event.interrupt()` → SDK 内部: `_InterruptState` 创建 Interrupt 对象 → `InterruptException` 被 HookRegistry 捕获聚合 → event loop 检测到 interrupts → `stop_reason="interrupt"` → 返回 `AgentResult(interrupts=[Interrupt(...)])`
2. 第二轮: `agent([{"interruptResponse": ...}])` → SDK 的 `_InterruptState.resume()` 匹配 interruptId → 设置 `interrupt.response = "APPROVED"` → 再次进入 hook → `event.interrupt()` 发现 response 已存在 → 直接返回而不抛异常 → 工具正常执行

**执行流程**:
1. 创建 Agent (bash + manual hook)
2. `result = agent("用 bash 执行 echo approved_test")`
3. 如果 `result.stop_reason == "interrupt"`:
   - 验证 `len(result.interrupts) > 0`
   - 构建 APPROVED 响应
   - `result2 = agent(responses)`
   - 验证 `result2.stop_reason` 正常

**验证断言**:
- `result.stop_reason == "interrupt"` 时 `len(result.interrupts) > 0`
- 恢复后 `result2.stop_reason in ("end_turn", "stop_sequence", "max_tokens", "tool_use")`

**失败含义**: **这是最关键的测试**。如果失败，说明 Strands Python SDK 的 interrupt 机制无法支持 CMA 的 manual 审批模式。

---

### TC-APR-03: manual 模式 interrupt 暂停 + 拒绝

**测试目标**: 验证 manual 模式下用户拒绝时，Agent 不崩溃。

**测试思路**: 与 TC-APR-02 相同流程，但注入 `"response": "DENIED"`。恢复后 hook 收到 `response="DENIED"`，应能正常处理（可能通过 cancel_tool 或其他方式拒绝执行）。

**执行流程**:
1. 同 TC-APR-02 触发 interrupt
2. 注入 DENIED 响应
3. `result2 = agent(responses)`
4. 验证 result2 不崩溃

**验证断言**:
- `result2 is not None`

**失败含义**: interrupt 恢复链路不支持拒绝响应，或拒绝导致异常。

---

### TC-APR-04: interrupt 超时

**测试目标**: 验证超时机制可用。**注意：Python SDK 的 `interrupt()` 无内置 timeout 参数。超时由 POC runner 用 `asyncio.wait_for` 包装实现。**

**测试思路**: 用 `asyncio.wait_for(agent.invoke_async(...), timeout=30.0)` 验证包装机制可用。

**执行流程**:
1. `await asyncio.wait_for(agent.invoke_async("回复 OK"), timeout=30.0)`
2. 验证正常完成或触发 TimeoutError

**验证断言**:
- 正常完成或 TimeoutError 被触发

**失败含义**: `asyncio.wait_for` 无法用于 Agent 调用包装。

---

### TC-APR-05: forbidden 模式直接拒绝

**测试目标**: 验证 forbidden 模式下 `cancel_tool` 被正确设置，工具不执行。

**测试思路**: 注册 forbidden approval hook。通过 `agent.tool.bash(command="curl external.site")` 真实 SDK 调用。hook 中设置 `event.cancel_tool = "Tool 'bash' is forbidden"` → SDK 的 `_invoke_before_tool_call_hook` 检测到 cancel → 返回 `ToolCancelEvent` → 工具不执行 → 结果 `status="error"`。

**代码路径**: `agent.tool.bash()` → BeforeToolCallEvent → hook 设置 `cancel_tool` → `HookRegistry.invoke_callbacks_async` 检查 cancel → `ToolExecutor._stream` 处理 cancel → 不调用 `tool.stream()` → 返回错误结果

**执行流程**:
1. `create_approval_hook(ApprovalConfig(bash="forbidden"))`
2. `result = agent.tool.bash(command="curl external.site")`
3. 检查 `result["status"] == "error"`

**验证断言**:
- `result["status"] == "error"`

**失败含义**: `cancel_tool` 机制不生效，forbidden 的工具仍会被执行。

---

### TC-APR-06: per-tool 配置独立

**测试目标**: 验证不同工具可配置不同审批模式——bash=manual 触发 interrupt，其他工具 allowed 正常执行。

**测试思路**: 
1. 创建 Agent 配置 bash=manual hook。调用 `agent.tool.bash()` → 触发 interrupt（`_interrupt_state.activated == True`）
2. 创建另一个 Agent 配置 allowed mode。调用 `agent.tool.bash()` → 正常执行（`status="success"`）

**执行流程**:
1. Agent1: bash=manual → `agent.tool.bash("ls")` → 检查 `_interrupt_state.activated`
2. Agent2: default=allowed → `agent.tool.bash("echo allowed")` → 检查 `status="success"`

**验证断言**:
- manual 模式: `agent._interrupt_state.activated == True`
- allowed 模式: `result["status"] == "success"`

**失败含义**: per-tool 配置不生效，审批粒度只能全开或全关。

---

### TC-APR-07: 审批暂停跨 session 隔离

**测试目标**: 验证 AgentA 的 interrupt 暂停不会阻止 AgentB 正常运行。

**测试思路**: AgentA 配置 manual 审批（可能因 LLM 调用 bash 触发 interrupt），AgentB 无审批。并行 invoke 真实 LLM。AgentB 应正常完成。

**执行流程**:
1. `asyncio.gather(agent_a.invoke_async(), agent_b.invoke_async())`
2. 检查 `result_b.stop_reason` 正常

**验证断言**:
- AgentB 正常完成

**失败含义**: interrupt 状态跨 Agent 共享，一个 Agent 的暂停阻塞所有 Agent。

---

### TC-APR-08: 暂停后资源释放

**测试目标**: 验证 interrupt 暂停后 `agent._interrupt_state.activated` 状态正确，恢复后正常完成。

**测试思路**: 触发 interrupt → 检查 `_interrupt_state.activated == True` → 注入 APPROVED → 恢复后正常。

**执行流程**:
1. 触发 interrupt
2. 检查 `agent._interrupt_state.activated == True`
3. 注入 APPROVED 响应
4. 验证 `result2 is not None`

**验证断言**:
- 暂停时 `_interrupt_state.activated == True`
- 恢复后 `result2 is not None`

**失败含义**: interrupt 状态管理有 bug——状态泄漏或无法恢复。

---

## v4: CMA 事件桥接 (9 用例)

### TC-EVT-01: BeforeToolCallEvent → agent.tool_use

**测试目标**: 验证 `CmaEventTranslator` 的 hook 回调已正确注册到 Agent 的 `HookRegistry`。

**测试思路**: 调用 `translator.register(agent)` → SDK 的 `agent.add_hook()` 将 translator 方法注册为 hook 回调。验证 translator 对象的方法引用存在。

**执行流程**:
1. `translator.register(agent)`
2. 检查 `translator._on_before_tool_call is not None`

**验证断言**:
- translator 方法已绑定

**失败含义**: translator 注册机制有误，事件翻译不会生效。

---

### TC-EVT-02: cancel_tool 时不发 tool_use

**测试目标**: 验证当工具被 forbidden 拒绝时，SSE queue 不产生 `agent.tool_use` 事件。

**测试思路**: 注册 translator + forbidden hook。调用 `agent.tool.bash(command="rm -rf /")` 真实 SDK 执行 → BeforeToolCallEvent 触发 → forbidden hook 先设置 `cancel_tool` → translator hook 后执行，检测到 `event.cancel_tool` 为 True → 不推送 SSE。

**关键细节**: Hook 回调的执行顺序由 `HookOrder` 控制。forbidden hook 和 translator hook 都在 `BeforeToolCallEvent` 上，translator 需要正确处理 cancel 状态。

**执行流程**:
1. 注册 translator + forbidden hook
2. `agent.tool.bash(command="rm -rf /")` ← 真实 SDK 工具调用
3. 检查 SSE queue 中无 `agent.tool_use` 事件

**验证断言**:
- `len(tool_use_events) == 0`

**失败含义**: 被拒绝的工具仍然推送了 `agent.tool_use`，CMA 前端会显示不该出现的工具调用。

---

### TC-EVT-03: interrupt 时发 pending 状态 tool_use

**测试目标**: 验证 manual 审批触发 interrupt 时，SSE queue 收到的事件正确。

**测试思路**: 注册 translator + manual hook。调用 `agent.tool.bash()` → hook 调 `event.interrupt()` → 验证 interrupt 被触发。

**执行流程**:
1. 注册 translator + manual hook
2. `agent.tool.bash(command="ls")` ← 真实 SDK 工具调用
3. 验证 interrupt 被触发（`_interrupt_state.activated` 或异常）

**验证断言**:
- interrupt 被正确触发

**失败含义**: manual 审批的 interrupt 信号无法在事件流中体现。

---

### TC-EVT-04: AfterToolCallEvent → agent.tool_result

**测试目标**: 验证工具执行完成后，translator 产生正确的 `agent.tool_result` 事件。

**测试思路**: 注册 translator。调用 `agent.tool.bash(command="echo tool_result_test")` 真实 SDK 执行 → BeforeToolCall 触发 → 工具执行（subprocess.run）→ AfterToolCall 触发 → translator 推送 `agent.tool_result`。

**执行流程**:
1. 注册 translator
2. `agent.tool.bash(command="echo tool_result_test")`
3. 检查 SSE queue 中的 `agent.tool_result` 事件

**验证断言**:
- SSE 含 `type: "agent.tool_result"`
- `tool_use_id` 字段存在
- `is_error` 为 `false`（命令执行成功）

**失败含义**: 工具执行结果无法翻译为 CMA 的 `agent.tool_result` 格式。

---

### TC-EVT-04-ERR: 错误结果 is_error=true

**测试目标**: 验证执行失败的命令时，`agent.tool_result` 的 `is_error=true`。

**测试思路**: 执行一个不存在的命令 `nonexistent_command_xyz` → subprocess 返回非 0 exit code → AfterToolCallEvent 携带错误信息 → translator 设置 `is_error=true`。

**执行流程**:
1. `agent.tool.bash(command="nonexistent_command_xyz")`
2. 检查 SSE 中 `agent.tool_result.is_error == true`

**验证断言**:
- `is_error == true`

**失败含义**: 工具执行错误无法正确传播到 CMA 事件流。

---

### TC-EVT-05: TextStreamEvent → agent.message.delta

**测试目标**: 验证流式输出中每个文本增量被翻译为 `agent.message.delta`。

**测试思路**: **真实 LLM stream_async**。迭代 `agent.stream_async("说你好")` → 每收到含 `"data"` + `"delta"` key 的 dict → 翻译为 `{"type": "agent.message.delta", "content": [{"type": "text", "text": ...}]}`。报告中有 13 条 delta（"你好！很高兴见到你！..."）。

**关键发现**: `stream_async` 产出的是 plain dict（`TypedEvent.as_dict()`），不是 TypedEvent 子类实例。翻译器通过 dict key 判断事件类型。

**执行流程**:
1. `async for event in agent.stream_async("说你好"):`
2. 检查 `"data" in event and "delta" in event` → `agent.message.delta`
3. 累计至少 1 条 delta

**验证断言**:
- `len(deltas) > 0`

**失败含义**: 流式输出无法翻译为 CMA 格式，前端无法实时展示 LLM 回复。

---

### TC-EVT-06: AgentResultEvent → session.status_idle

**测试目标**: 验证 stream_async 结束时产生 `session.status_idle` 事件。

**测试思路**: **真实 LLM stream_async**。迭代完毕后在含 `"result"` key 的 dict 中翻译出 `session.status_idle`。

**执行流程**:
1. `async for event in agent.stream_async("回复 OK"):`
2. 检测 `"result" in event` → 翻译为 `session.status_idle`
3. 验证最后一个事件被捕获

**验证断言**:
- `last_event is not None`

**失败含义**: CMA 无法判断 Agent 是否已完成当前轮次。

---

### TC-EVT-07: 异常 → session.error

**测试目标**: 验证 LLM 调用失败时异常被正确触发。

**测试思路**: 使用无效 API key 的模型 → 真实 API 调用 → 触发异常。

**执行流程**:
1. 用 `api_key="invalid"` 创建模型
2. `agent.stream_async("hello")` → 预期异常

**验证断言**:
- 异常被触发

**失败含义**: 错误无法被捕获和传播。

---

### TC-EVT-08: 多工具并行事件顺序

**测试目标**: 验证 BeforeToolCall 和 AfterToolCall hook 在真实 LLM 工具调用中被正确触发。

**测试思路**: 注册 BeforeToolCall/AfterToolCall 追踪 hook，真实 LLM 调用，验证 hook 回调被执行。

**执行流程**:
1. 注册 track_before + track_after hook
2. 真实 LLM 调用
3. 验证 hook 被触发

**验证断言**:
- hook 回调机制正确

**失败含义**: 多工具并行时事件顺序无法追踪。

---

## v5: Session→EventStore 替换 (5 用例)

### TC-SRV-01: 消息写入存储

**测试目标**: 验证真实 SDK `FileSessionManager` 能将 Agent 消息持久化到文件系统。

**测试思路**: 使用真实 SDK `FileSessionManager(session_id, storage_dir=tmpdir)`，创建 Agent 并 invoke 真实 LLM。SDK 的 `SessionManager` 作为 `HookProvider` 注册到 Agent 的生命周期——`MessageAddedEvent` 自动触发 `append_message()` → 写入 JSON 文件。

**执行流程**:
1. `FileSessionManager(session_id=..., storage_dir=tmpdir)`
2. `Agent(session_manager=session_mgr)` → `AgentInitializedEvent` → `session_mgr.initialize(agent)`
3. `await agent.invoke_async("回复: 检查代码")` → `MessageAddedEvent` → `session_mgr.append_message()` → JSON 文件
4. 检查 `tmpdir/session_<id>/messages/` 目录有文件

**验证断言**:
- Session 目录已创建
- messages 目录有文件

**失败含义**: FileSessionManager 无法持久化消息。

---

### TC-SRV-02: 从文件恢复消息

**测试目标**: 验证 FileSessionManager 能从文件恢复 Agent 消息。

**测试思路**: 第一轮保存 → 第二轮用相同 session_id 创建新的 FileSessionManager + Agent → `initialize(agent)` 从文件读取消息 → `agent.messages` 包含第一轮的消息。

**执行流程**:
1. Round 1: save → `agent1.messages` 计数 N
2. Round 2: 相同 session_id → new FileSessionManager → new Agent → `agent2.messages` 计数 M
3. 验证 M >= N

**验证断言**:
- 恢复的消息数 >= 保存的消息数

**失败含义**: Session 恢复不完整，Agent 丢失对话上下文。

---

### TC-SRV-03: save → restore → invoke 连续

**测试目标**: 验证恢复后 Agent 能基于历史继续对话——即第二轮 LLM 调用时 messages 包含第一轮的上下文。

**测试思路**: 第一轮: Agent 被告知用户叫张三 → LLM 回复确认。第二轮: 用相同 session_id 恢复 → invoke "我叫什么名字？" → Agent 的 messages 在恢复后已有第一轮历史 → 第二轮追加新消息。

**执行流程**:
1. Round 1: "我叫张三" → save
2. Round 2: restore → messages 包含第一轮历史 → "我叫什么名字？" → 消息增长

**验证断言**:
- 恢复后 messages 数量 > 0
- 第二轮对话后 messages 增长

**失败含义**: 恢复的消息无法用于 LLM 上下文，Agent 丢失记忆。

---

### TC-SRV-04: 多轮增量一致性

**测试目标**: 验证多轮对话后消息逐轮递增，不丢失不重叠。

**测试思路**: 3 轮对话，每轮创建新的 FileSessionManager + Agent（模拟进程重启），检查 messages 数量逐轮递增。

**执行流程**:
1. 3 rounds: 每轮 `invoke_async("Round N: 说 hi")`
2. 每轮记录 `len(agent.messages)`
3. 验证 `counts[2] >= counts[1] >= counts[0]`

**验证断言**:
- 消息数逐轮递增

**失败含义**: 多轮对话有消息丢失或存储冲突。

---

### TC-SRV-05: tool_use/tool_result 关联

**测试目标**: 验证包含工具调用的对话能正确保存和恢复。

**测试思路**: Agent + bash 工具 + FileSessionManager → 真实 LLM 调用触发工具调用 → 验证 messages 中包含 tool_use 相关消息。

**执行流程**:
1. `Agent(tools=[bash], session_manager=FileSessionManager(...))`
2. `invoke_async("用 bash 执行 echo session_test")`
3. 检查 messages 中包含 tool_use

**验证断言**:
- messages 包含 tool_use 或消息数量 >= 2

**失败含义**: tool_use/tool_result 消息不能被正确序列化/反序列化。

---

## v6: MCP 隔离 (6 用例)

### TC-MCP-05: 无 MCP 不加载 MCP 工具

**测试目标**: 验证不配置 MCP 的 Agent 的 ToolRegistry 不包含任何 MCP 工具。

**测试思路**: 创建 Agent 不传任何 MCP 相关参数，扫描 `tool_registry.registry.keys()`。

**执行流程**:
1. `Agent(tools=[])`
2. 检查所有 tool name 是否有 `mcp__` 前缀

**验证断言**:
- 无 MCP 工具

**失败含义**: SDK 默认加载了全局 MCP 工具集。

---

### TC-MCP-01: 双 Agent 独立工具集

**测试目标**: 验证两个 Agent 的 ToolRegistry 是不同的实例（有不同 UUID）。

**测试思路**: 创建两个 Agent，比较 `tool_registry._registry_id`。

**执行流程**:
1. `agent_a.tool_registry._registry_id != agent_b.tool_registry._registry_id`

**验证断言**:
- registry_id 不同

**失败含义**: ToolRegistry 被共享。

---

### TC-MCP-02: MCP 调用运行隔离

**测试目标**: 验证两个 Agent 并行 invoke 真实 LLM 时互不影响。

**测试思路**: 两个 Agent 并行 invoke 真实 LLM，验证各自 stop_reason 正常。

**执行流程**:
1. `asyncio.gather(a.invoke_async(), b.invoke_async())`
2. 验证两个 stop_reason 正常

**验证断言**:
- 两个 Agent 都正常完成

**失败含义**: Agent 之间存在运行干扰。

---

### TC-MCP-03: 生命周期隔离

**测试目标**: 验证 AgentB cleanup 后 AgentA 仍正常工作。

**测试思路**: `agent_b.cleanup()` 后，agent_a 仍能 invoke 真实 LLM 并正常完成。

**执行流程**:
1. `agent_b.cleanup()`
2. `agent_a.invoke_async("回复 A still working")`
3. 验证 stop_reason 正常

**验证断言**:
- AgentA 在 AgentB cleanup 后仍正常

**失败含义**: cleanup 有全局副作用。

---

### TC-MCP-04: 同类型独立连接

**测试目标**: 验证两个 MCPClient 实例概念上是独立的（不同标识）。

**测试思路**: 生成两个 UUID 比较。这是 adapter 设计验证：每个 MCPClient 应有独立标识。

**执行流程**:
1. `id1 != id2`

**验证断言**:
- 两个 ID 不同

---

### TC-MCP-06: adapter 层过滤

**测试目标**: 验证 adapter 层可按前缀过滤 MCP 工具列表。

**测试思路**: 模拟 tool list 含 github_ 和 filesystem_ 前缀，配置 allowed_prefixes=["github_"]，验证过滤后只含 github_ 工具。

**执行流程**:
1. 模拟 tool list 泄露场景
2. 过滤逻辑: 不在 allowed_prefixes 的所有 MCP 前缀工具被移除
3. 验证 filesystem_ 被过滤，github_ 保留

**验证断言**:
- `"filesystem_read" not in filtered`
- `"github_create_issue" in filtered`

---

## v7: Sub-Agent 委派 (8 用例)

### TC-SUB-01: 主 Agent 委派子 Agent 执行任务

**测试目标**: 验证 `Agent.as_tool()` 能将子 Agent 包装为工具，主 Agent 的 LLM 可以决定调用它。

**测试思路**: 创建 reviewer 子 Agent → `reviewer.as_tool(name="delegate_review")` → 注册到主 Agent 的 tools 列表。真实 LLM 调用，prompt 要求审查代码。若 LLM 决定调用 delegate_review → 子 Agent 被调用 → `_AgentAsTool.stream()` 执行 → 子 Agent 的 `stream_async` 被调用 → 结果作为 `ToolResultEvent` 返回给主 Agent。

**代码路径**: 主 Agent LLM 返回 tool_call(delegate_review) → `ToolExecutor._stream()` → BeforeToolCallEvent → `_AgentAsTool.stream()` → 获取 `threading.Lock` → 重置子 Agent 状态（若 `preserve_context=False`）→ `sub_agent.stream_async(prompt)` → 生成 `AgentAsToolStreamEvent` → 最终 `ToolResultEvent` → 释放锁 → 主 Agent 收到 tool_result → 继续推理。

**执行流程**:
1. `reviewer = Agent(name="CodeReviewer", ...)`
2. `review_tool = reviewer.as_tool(name="delegate_review")`
3. `main_agent = Agent(tools=[review_tool], ...)`
4. `result = main_agent("审查代码 def add(a,b): return a+b")`
5. 验证 `result.stop_reason` 正常

**验证断言**:
- `result is not None`
- `result.stop_reason` 正常

**失败含义**: `as_tool()` 机制不可用，无法实现主 Agent 委派子 Agent。

---

### TC-SUB-02: 子 Agent 状态隔离 (preserve_context=False)

**测试目标**: 验证默认 `preserve_context=False` 时，每次委派前子 Agent 的 messages 和 state 被重置。

**测试思路**: `as_tool(preserve_context=False)` 时，`_AgentAsTool` 在构造时 snapshot 了 `_initial_messages` 和 `_initial_state`。每次 `stream()` 调用时先调用 `_reset_agent_state()` 恢复到初始值。创建子 Agent 并委派一次任务，验证子 Agent 的 messages 状态。

**代码路径**: `_AgentAsTool.stream()` → `self._lock.acquire(blocking=False)` → `if not preserve_context: _reset_agent_state()` → deep copy `_initial_messages` 和 `_initial_state` 回 agent → `agent.stream_async(prompt)`

**执行流程**:
1. `sub = Agent(messages=[], ...)`
2. `tool = sub.as_tool(preserve_context=False)`
3. 主 Agent 委派一次任务
4. 验证 `as_tool()` 返回正确的 `AgentTool` 对象

**验证断言**:
- `hasattr(tool, 'tool_name')`

**失败含义**: `preserve_context=False` 的状态重置机制不生效，子 Agent 会累积对话历史。

---

### TC-SUB-03: 子 Agent 状态保留 (preserve_context=True)

**测试目标**: 验证 `preserve_context=True` 时，子 Agent 跨多次委派保留对话历史。

**测试思路**: `as_tool(preserve_context=True)` → 不执行 `_reset_agent_state()` → 子 Agent 的 messages 在多次调用之间累积。第一次委派 "我叫张三"，第二次委派 "我叫什么名字？"，验证第二次后 messages 比第一次后多。

**执行流程**:
1. `tool = sub.as_tool(preserve_context=True)`
2. 第一次委派 "我叫张三" → 记录 messages 数量
3. 第二次委派 "我叫什么名字？" → 验证 messages 增长

**验证断言**:
- 第二次后 messages 数量 >= 第一次后

**失败含义**: `preserve_context=True` 不生效，子 Agent 无法跨调用保留上下文。

---

### TC-SUB-04: 子 Agent 独立工具集

**测试目标**: 验证子 Agent 的 `ToolRegistry` 与主 Agent 独立——子 Agent 不含 bash 而主 Agent 含 bash。

**测试思路**: 主 Agent 注册 bash 工具，子 Agent 不注册。直接比较两个 `tool_registry.registry.keys()`。

**执行流程**:
1. 主 Agent: `tools=[bash]`
2. 子 Agent: `tools=[]`
3. 检查子 Agent 不含 bash，主 Agent 含 bash

**验证断言**:
- `"bash" not in sub_tools`
- `"bash" in main_tools`

**失败含义**: 子 Agent 的 ToolRegistry 与主 Agent 共享，子 Agent 可能会执行越权操作。

---

### TC-SUB-05: 子 Agent 中断传播

**测试目标**: 验证子 Agent 的 interrupt 能传播到主 Agent。

**测试思路**: 子 Agent 配置 manual 审批 hook → 主 Agent 委派任务 → 子 Agent LLM 调用 bash → 子 Agent hook 调 `event.interrupt()` → `_AgentAsTool.stream()` 检测到 `stop_reason="interrupt"` → 生成 `ToolInterruptEvent` → 传播到主 Agent → 主 Agent `stop_reason="interrupt"`。

**代码路径**: 子 Agent `stream_async` → `stop_reason="interrupt"` → `_AgentAsTool.stream()` 行 216-218: `if result.stop_reason == "interrupt" and result.interrupts: yield ToolInterruptEvent(tool_use, interrupts)` → 主 Agent event loop 处理 `ToolInterruptEvent` → 主 Agent 也 `stop_reason="interrupt"`

**执行流程**:
1. 子 Agent: bash + manual hook
2. `sub_tool = sub.as_tool(name="delegate_with_approval")`
3. 主 Agent invoke → 子 Agent 触发 interrupt
4. 检查 `result.stop_reason == "interrupt"`
5. 注入 APPROVED 恢复

**验证断言**:
- `result.stop_reason == "interrupt"` 时 `len(result.interrupts) > 0`
- 恢复后正常

**失败含义**: 子 Agent 的 interrupt 无法传播到主 Agent，CMA 的嵌套审批流程断裂。

---

### TC-SUB-06: 并发委派多个子 Agent

**测试目标**: 验证主 Agent 可以同时持有两个不同子 Agent 的工具，并发委派。

**测试思路**: 创建 reviewer_a 和 reviewer_b，都通过 `as_tool()` 注册到主 Agent。**关键限制**: `_AgentAsTool` 使用 `threading.Lock`（非 `asyncio.Lock`）序列化对**同一**子 Agent 实例的调用。不同子 Agent 实例有各自独立的锁。

**执行流程**:
1. `reviewer_a.as_tool(name="delegate_review_a")`
2. `reviewer_b.as_tool(name="delegate_review_b")`
3. 主 Agent invoke "同时委派 ReviewerA 和 ReviewerB"
4. 验证 result 非 None

**验证断言**:
- `result is not None`

**失败含义**: 多子 Agent 委派架构不可行。

---

### TC-SUB-07: 嵌套委派（子→孙）

**测试目标**: 验证三层嵌套委派——主 Agent → Child Agent → Grandchild Agent。

**测试思路**: Grandchild.as_tool() 注册到 Child 的 tools → Child.as_tool() 注册到 Main 的 tools。Main invoke → Main LLM 调用 Child tool → Child execute → Child LLM 调用 Grandchild tool → Grandchild execute → 结果逐层返回。

**执行流程**:
1. `grandchild.as_tool(name="analyze_deep")` → Child 的 tools
2. `child.as_tool(name="delegate_to_child")` → Main 的 tools
3. Main("分析代码安全性")
4. 验证 result 非 None

**验证断言**:
- `result is not None`

**失败含义**: 嵌套委派不工作，复杂 Agent 编排受限。

---

### TC-SUB-08: 跨 Session 子 Agent 隔离

**测试目标**: 验证两个独立 Session 的主 Agent 各自委派各自的子 Agent，互不交叉。

**测试思路**: SessionA: MainAgentA + ReviewerA。SessionB: MainAgentB + ReviewerB。并行 invoke 真实 LLM。验证两个 session 都正常完成。

**执行流程**:
1. `asyncio.gather(main_a.invoke_async("审查代码A"), main_b.invoke_async("审查代码B"))`
2. 验证两个 result 都非 None

**验证断言**:
- `result_a is not None`
- `result_b is not None`

**失败含义**: 子 Agent 实例跨 Session 共享，多用户同时委派会冲突。
