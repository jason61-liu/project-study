# Agent 边界、Agent Loop 设计题与 10 分钟验收

## 1. Agent 与 Workflow、Chatbot、RAG Pipeline 的边界

最有区分度的问题不是“是否调用了大模型或工具”，而是：**下一步控制流由谁决定，是否会根据外部 Observation 动态改变。**

| 系统 | 下一步由谁决定 | 控制流 | 是否必须调用工具 | 典型终止方式 |
|---|---|---|---:|---|
| Chatbot | 对话程序与模型共同决定回复内容 | 通常是一问一答或多轮消息 | 否 | 当前回复生成完毕 |
| RAG Pipeline | 程序预先规定 | 通常固定为检索、拼接、生成 | 检索是固定步骤 | 生成最终回答 |
| Workflow | 代码、规则或预定义 DAG | 构建时基本已知，可包含条件分支 | 可选 | 到达指定结束节点 |
| Agent | 模型基于状态和 Observation 提议下一动作，Runtime 审批执行 | 运行时动态形成 | 通常有，但定义上不是必须 | 目标谓词满足或预算/策略终止 |

### 1.1 Chatbot 的边界

普通 Chatbot 的核心能力是根据消息历史生成下一条回复。即使它保留多轮上下文，也不一定是 Agent，因为它可能没有 Action、Observation 和自主循环。

```text
User → Model → Assistant Message
```

加入一次固定的敏感词检测或历史查询，也不会自动变成 Agent。只有当模型能够根据结果动态选择下一动作，并可能重复“决策—执行—观察”时，才进入 Agent 范畴。

### 1.2 RAG Pipeline 的边界

经典 RAG 的路径在开发时已经确定：

```text
Query → Rewrite（可选）→ Retrieve → Rerank（可选）→ Generate
```

模型可能参与 Query Rewrite 和最终生成，但通常不能决定跳过检索、改用数据库、再次检索或调用其他工具，所以它仍是 Pipeline。Agentic RAG 则把检索器作为 Tool，模型可以根据 Observation 决定是否改写问题、换数据源或再次检索。

### 1.3 Workflow 的边界

Workflow 可以非常复杂，也可以调用多个模型，但只要主要控制流由代码或 DAG 预先规定，它仍然是 Workflow：

```text
提交申请 → 规则校验 → 金额分支 → 人工审批 → 入账
```

Workflow 的优势是可预测、易测试、易审计；代价是对开放式任务适应性较弱。模型作为 Workflow 中的一个节点，不会让整个系统自动成为 Agent。

### 1.4 Agent 的边界

Agent 至少包含以下闭环：

```text
State → Model Decision → Action → Tool Execution → Observation → State
          ↑                                           │
          └───────────────────────────────────────────┘
```

模型负责提出动作或完成声明，Runtime 负责 Schema 校验、权限、预算、超时、工具执行和最终终止判定。真正的 Agent 不是“让模型拥有无限控制权”，而是**在确定性边界内允许模型动态选择路径**。

如果业务要求“必须先取得工具证据”，不能只在 Prompt 中写“必须调用”。模型仍可能跳过工具并生成看似合理的答案。应由 Runtime 在首轮设置 `tool_choice="required"`，并且只发送 Tool Schema；获得 Observation 后设置 `tool_choice="none"`，再启用最终 JSON Schema。Prompt 用于表达语义，协议与代码用于执行约束。

### 1.5 一个实用判断法

依次问四个问题：

1. 下一步在开发时是否已经确定？如果是，优先视为 Workflow/Pipeline。
2. 模型是否能根据外部结果选择不同工具或再次行动？如果否，通常不是 Agent。
3. 是否存在 `Action → Observation → 再决策` 的反馈循环？如果是，具备 Agent 特征。
4. 是否有模型之外的确定性终止、权限和预算控制？如果没有，它是不安全的循环，而不是可靠 Agent。

实际系统经常是混合架构：外层 Workflow 保证关键业务顺序，某个开放式节点内部使用 Agent；RAG Retriever 也可以作为 Agent 的一个只读工具。

## 2. Agent Loop 设计题

### 题 1：一个最小 Agent Loop 需要保存哪些状态？

至少需要：消息或决策上下文、当前 Step、模型调用数、工具调用数、累计 Token/费用、绝对 Deadline、待处理 Tool Call、Observation、完成证据和终止原因。只保存消息列表不足以恢复任务，因为消息不能可靠表达预算、审批、幂等记录和工具执行状态。

本实验把模型与工具轨迹记录为：

```text
run_id
  ├─ model trace_id: started_at / ended_at / status
  ├─ tool trace_id:  started_at / ended_at / status
  └─ model trace_id: started_at / ended_at / status
```

### 题 2：为什么只设置 `max_steps` 不够？

因为一个 Step 可能调用多个并行工具、携带很长上下文或等待一个极慢工具。最大步数只约束循环深度，不能约束扇出、Token、费用和墙钟时间。可靠 Runtime 至少同时控制：

- 最大 Agent Step；
- 模型和工具调用总数；
- 单步工具扇出；
- Token 与费用预算；
- 单次模型/工具超时；
- 总体绝对 Deadline；
- 最终回答和清理资源预留。

### 题 3：工具失败后应该直接结束 Agent 吗？

不应一概而论。工具错误应先转成稳定 Observation，例如：

```json
{
  "ok": false,
  "error": {
    "code": "TOOL_TIMEOUT",
    "message": "tool exceeded 1.0s",
    "retryable": true
  }
}
```

Runtime 根据幂等性、剩余预算和错误类别决定是否允许恢复；模型可以解释失败、修正参数或选择替代工具。权限失败、未知工具和永久参数错误不能靠模型无限重试绕过。

### 题 4：模型为什么不能直接执行它生成的 Tool Call？

模型输出只是非可信的动作提议。执行前必须经过工具 Allowlist、JSON/Schema 校验、业务校验、身份与权限、审批、幂等和超时控制。否则 Prompt Injection 或模型幻觉可能把一段 Token 输出升级成真实副作用。

正确边界是：

```text
Model proposes → Runtime validates/authorizes → Tool executes
```

### 题 5：怎样判断 Agent 真正完成，而不是“模型停止输出”？

EOS、`response.completed` 和业务完成是三件事。`response.completed` 只说明本次模型响应完成；它可能只返回一个 Tool Call。业务完成应由可判定谓词复核：

$$
complete(state)=goal\_satisfied\land evidence\_ready\land no\_pending\_actions\land output\_valid
$$

例如订单查询任务必须同时具备订单工具成功回执、匹配的订单号、无待执行调用以及通过 Schema 的最终回答。

### 题 6：流式 Tool Call 为什么不能边接收边执行？

`arguments.delta` 可能在任意字符处切分，中间状态经常不是合法 JSON；后续 Delta 还可能改变金额、路径或收件人。必须按 `output_index/call_id` 分别缓冲，等待参数 Done，再执行 JSON 解析、Schema 校验和授权。

### 题 7：工具超时后为什么不能直接重试？

客户端超时只表示没有及时收到结果，不证明工具没有执行。扣款、发信等非幂等操作可能已经成功但响应丢失。正确策略是使用业务幂等键和执行记录，超时后先查询状态；只有读操作或具备幂等保证的写操作才能有界重试。

### 题 8：什么时候应该用 Workflow 而不是 Agent？

当步骤、分支和验收规则在开发时已经清楚，尤其涉及合规、资金或严格 SLA 时，优先 Workflow。只有任务路径开放、需要根据未知 Observation 动态选择行动，并且这种适应性收益大于额外成本和风险时，才使用 Agent。资深设计的默认答案不是“都用 Agent”，而是选择最弱但足够的控制机制。

## 3. 10 分钟演示脚本

### 0:00–1:00：定义目标

说明本实验不用 Agent 框架，真实调用 DeepSeek-V4-Pro，完成订单查询，并展示结构化输出、流式响应、Tool Call、Observation、Trace 和显式终止。

### 1:00–2:00：画出边界

```text
User
  → AgentRuntime
  → ModelAdapter
  → DeepSeek-V4-Pro
  → function_call(get_order)
  → Schema Validation
  → Local Tool
  → function_call_output
  → DeepSeek-V4-Pro
  → Structured Final Answer
```

强调模型只提出 Tool Call，Runtime 才是真正执行者。

### 2:00–4:00：讲核心代码

打开 `agent_runtime.py`，只讲四个入口：

1. `ModelAdapter.call()`：屏蔽流式、非流式和供应商差异；
2. `AgentRuntime.run()`：有界循环；
3. `_execute_tool()`：Allowlist、Schema、Deadline、异常包装；
4. `TraceRecorder`：记录每次真实调用的生命周期。

### 4:00–6:00：运行真实演示

```bash
source /Users/shiyiliu/workspace/pyproject/.venv/bin/activate
export OPENAI_API_KEY="你的密钥"
export OPENAI_BASE_URL="https://api.siliconflow.cn/v1"
export AGENT_TEST_MODEL="deepseek-ai/DeepSeek-V4-Pro"
python demo_agent_trace.py
```

预期终态是 `completed`，轨迹应包含两次模型调用和一次工具调用。

### 6:00–8:30：逐步解释完整轨迹

1. 第一次 Model Trace：读取用户目标和工具 Schema，返回 `get_order` Tool Call；
2. Tool Trace：Runtime 检查工具名，校验 `ORD-1001`，执行本地查询并返回 Observation；
3. 第二次 Model Trace：模型读取与原 `call_id` 匹配的工具结果，生成符合最终 Schema 的回答；
4. Runtime 确认没有新 Tool Call，结构化输出通过校验，终态设为 `completed`。

每条 Trace 都要指出：`trace_id`、`kind`、`name`、`started_at`、`ended_at`、`status` 和结果摘要。`run_id` 串联整条轨迹，`call_id` 串联一个 Tool Call 及其 Observation，它们不能混用。

### 8:30–9:30：演示失败边界

快速展示测试名称：未知工具、非法参数、工具异常、工具超时、最大步数和提前终止。说明错误不会直接变成未经处理的异常，而是结构化 Observation 或明确 RunStatus。

### 9:30–10:00：总结取舍

核心循环保持简单，供应商差异放在 Transport，动作安全放在 Runtime，业务逻辑放在 Tool。当前实现是教学基线；生产系统仍需增加鉴权、持久化幂等、完整 JSON Schema、取消传播和可恢复 Checkpoint。

## 4. 验收清单

- [x] 能区分 Agent、Workflow、Chatbot 和 RAG Pipeline；
- [x] 完成 8 道 Agent Loop 设计题；
- [x] 提供可重复的 10 分钟演示脚本；
- [x] 演示能够逐步打印并解释一条完整 Trace；
- [x] 核心循环不依赖 Agent 框架，可独立运行；
- [x] 测试覆盖正常完成、最大步数、错误恢复和提前终止。
