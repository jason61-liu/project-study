# 第 7 周：Agent 框架面试与验收总结

> 核对日期：2026-08-13。框架演进很快，选型结论必须和具体版本、部署方式、模型供应商一起评审，不能把框架名当成永久能力承诺。

## 1. 先给验收结论

本周不是简单地写了四份“调用模型”的 Demo，而是用同一研究任务拆开四层责任：

```text
业务正确性：同一任务、语料、工具返回结构、引用要求和完成谓词
执行控制：下一步执行什么、何时停止、怎样处理失败
持久恢复：Checkpoint、Interrupt、Resume、重复提交
安全边界：审批、Guardrail、工具权限、幂等副作用
可观测性：Trace、Span、模型/工具次数、Token、延迟和失败类型
```

本周交付物的角色如下：

| 角色 | 实现 | 验证目标 |
|---|---|---|
| 原生对照基线 | [`source/native_baseline.py`](./source/native_baseline.py) | 展开框架隐藏的状态机、持久化、审批和幂等责任 |
| 主实现 | [`source/langgraph_workflow.py`](./source/langgraph_workflow.py) | 用 StateGraph、SQLite Checkpoint、Interrupt/Resume 重构基线 |
| SDK 对照实现 | [`source/agents_sdk_workflow.py`](./source/agents_sdk_workflow.py) | 验证 Runner、Tool、Handoff、Guardrail 和工具审批 |
| Harness 最小版本 | [`source/deep_agents_workflow.py`](./source/deep_agents_workflow.py) | 验证主 Agent 委派、子 Agent 独立上下文和有界工具集合 |
| 统一实验入口 | [`source/run_comparison.py`](./source/run_comparison.py) | 用同一任务与指标结构运行四种架构 |
| 正式结果 | [`source/artifacts/final/architecture-comparison.md`](./source/artifacts/final/architecture-comparison.md) | 三次重复的成功率、步骤、工具、模型、Token 和延迟 |

当前结果：16 个测试全部通过；正式实验包含 12 条运行记录；四种实现三轮成功率均为 100%；Deep Agents 三轮上下文隔离均通过；每轮具有副作用的实现最多产生一条发布记录。

### 1.1 “关键行为一致”具体指什么

不能笼统地说“结果差不多，所以等价”。本项目按下面三层定义等价：

| 等价层级 | Native | LangGraph | Agents SDK | Deep Agents 最小版 |
|---|---:|---:|---:|---:|
| 同一任务、语料、检索/读取工具和完成谓词 | 是 | 是 | 是 | 是 |
| 最终答案含必需术语和四个来源 ID | 是 | 是 | 是 | 是 |
| 统一 Trace/指标结构 | 是 | 是 | 是 | 是 |
| 草稿、审批后发布、拒绝不发布、发布幂等 | 是 | 是 | 是 | 否，刻意不重复实现 |
| 持久化后恢复 | JSON 状态 | SQLite Checkpoint | 序列化 RunState + SQLite 账本 | 否，非本次 Harness 消融变量 |
| 子 Agent 上下文隔离 | 不适用 | 不适用 | Handoff 是控制权转移，不是子上下文实验 | 是 |

因此，LangGraph 和 Agents SDK 与原生基线具有“发布协议等价”；Deep Agents 与基线只有“研究结果与可观测性等价”。这符合“所选 Harness 只做最小版本，不重复实现全部框架”的实验约束。如果验收标准要求 Harness 也具备持久审批发布，就必须继续实现，而不能在报告里把缺失能力写成已完成。

## 2. 框架选型前必须先理解两个风险

### 2.1 抽象泄漏是什么

框架希望开发者只面对 `Agent`、`Node` 或 `Runner`，但底层问题仍会穿透抽象，迫使业务代码理解它们，这就是抽象泄漏。例如：

- LangGraph 隐藏了调度循环，但恢复错误时仍必须理解 checkpoint、thread、node 重放和 reducer；
- Agents SDK 隐藏了 tool loop，但模型若在 Handoff 后调用旧 Agent 的工具，应用仍会收到协议级失败；
- Harness 提供文件、Shell 和子 Agent，但权限、工作目录、进程取消和工具副作用仍由宿主负责；
- “支持持久化”不代表外部 API exactly-once，支付、发布、发信仍需要业务幂等键。

评审框架时，应问：“最坏情况下，我需要下钻到哪一层才能解释故障？”而不是只问“Hello World 有几行代码”。

### 2.2 锁定风险不是只有模型厂商锁定

锁定至少有四种：

1. **模型锁定**：Prompt、工具调用、推理参数依赖某个模型或 API。
2. **代码锁定**：业务逻辑直接继承框架 Agent/Node/Message 类型。
3. **状态锁定**：Checkpoint、Session、事件日志无法稳定导出并由其他 Runtime 恢复。
4. **运维锁定**：Tracing、部署、权限、沙箱或托管服务只能在特定平台工作。

最危险的通常不是换一个模型 Client，而是已有数百万条会话状态只能由原 Runtime 解释。

## 3. 九个框架的定位与选型表

风险等级是工程判断，不是官方评级。`低` 不代表零成本，`高` 也不代表不能采用；它表示迁移时需要替换多少执行语义和持久状态。

| 框架 | 核心抽象与定位 | 最适合 | 主要抽象泄漏 | 锁定风险 | 不应优先选择的情况 |
|---|---|---|---|---|---|
| **LangGraph** | 状态图 Runtime：State、Node、Edge、Reducer、Checkpoint、Interrupt | 长任务、复杂分支、HITL、失败恢复、可检查状态 | Node 可能重放；并发更新必须理解 Reducer；Checkpoint 不等于副作用幂等；Thread/Command 语义会进入业务层 | **中高**：模型锁定低，图状态和恢复语义锁定中高 | 只有两三个固定步骤，普通函数或队列已足够 |
| **OpenAI Agents SDK** | 轻量 Agent loop：Agent 配置 + Runner + Tool + Handoff + Guardrail + Trace | 工具型 Agent、客服分流、少量 Specialist 转交、快速接入 tracing | Provider 对工具/结构化输出兼容性不同；Handoff 后工具集合变化；Guardrail 触发和 RunState 恢复仍需宿主处理 | **中**：代码对 SDK 类型有依赖；可换模型，但行为兼容不自动成立 | 需要显式 DAG、复杂并行合并、长周期业务状态机 |
| **Deep Agents** | 建立在 LangGraph 上的 batteries-included Harness：规划、文件系统、子 Agent、上下文管理 | 研究、代码、文件密集型长任务；希望快速得到 Harness 能力 | 同时泄漏 Agent Prompt、`task` 委派、文件系统和底层 LangGraph 状态语义；工具输出大小会影响上下文 | **高**：上层 Harness 与下层 LangGraph 双层耦合 | 受严格流程约束的短交易；团队不需要文件/子 Agent |
| **Claude Agent SDK** | 把 Claude Code 的 agent loop、内置工具、权限、Hooks、Session、Subagent 暴露给应用 | 编码 Agent、仓库操作、Shell/文件工作流和强权限交互 | SDK 外观下仍是 Claude Code Runtime；subprocess、Session、权限模式、工具名和版本行为会穿透 | **高**：Claude 模型与 Claude Code Runtime 双重锁定 | 需要模型中立的通用业务编排，或不能运行相应 Runtime |
| **Hermes Agent** | 开源产品化 Agent Harness：CLI/服务入口、工具、Skills、记忆、渠道与完整循环 | 研究完整 Agent 产品怎样装配；快速获得多提供商 Harness | 配置、工具注册、会话、Gateway、渠道和产品约定都会进入业务；仓库变化快、表面积大 | **中高**：厂商锁定相对低，产品架构锁定高 | 只需要可嵌入的小型 Python Agent loop |
| **Kimi Agent SDK** | Go/Node/Python 薄客户端，把 Kimi CLI Runtime 嵌入应用；流式返回工具、审批和 Session 事件 | 已选择 Kimi Code 能力，要嵌入产品或自动化任务 | 真正行为在 Kimi CLI；wire protocol、CLI 安装、配置、Skills、MCP 与审批事件会穿透 SDK | **高**：对 Kimi CLI Runtime 和协议版本依赖明显 | 想完全掌控底层循环，或需要供应商中立状态格式 |
| **PydanticAI** | 类型优先的 Agent：泛型依赖、Pydantic 结构化输出、Tool、Usage、Evals；可接 durable execution 系统 | Python 服务、强 Schema/DI、结构化输出和可测试性 | Provider 并不完全支持相同 Schema；stream end strategy、重试和消息模型会影响业务；durability 来自 Temporal/DBOS/Prefect/Restate 等集成 | **中**：模型锁定低；代码对 Pydantic 类型低中度依赖；若使用 durable backend 还会增加状态锁定 | 主要需求是分布式多 Agent 消息系统而不是类型安全单 Agent |
| **AutoGen** | AgentChat 是高层会话/Team API；Core 是 Actor 风格、事件驱动的 Agent Runtime | 多 Agent 研究、异步消息、分布式 Agent、Python/.NET 协作 | 消息类型、AgentId、订阅、Runtime 生命周期、终止条件、Group Chat 状态会进入设计 | **高**：不是模型锁定，而是 Actor/消息架构和状态模型锁定 | 单 Agent 加几个工具；没有分布式或多 Agent 必要性 |
| **Google ADK** | Agent + Workflow + Runtime：模型 Agent、确定性 Workflow、Session/Event/Artifact/Memory、A2A 与部署工具链 | Google Cloud 生态、多语言 Agent、混合确定性/非确定性流程、A2A | Event loop、Session service、Callback、Artifact/Memory、部署与评测语义会进入应用 | **中高**：支持多模型，但采用托管 Runtime/Google 服务后运维锁定升高 | 简单本地 Python loop，或团队无 Google Cloud/ADK 生态需求 |

官方资料显示：PydanticAI 的 `Agent` 将 instructions、tools、structured output、typed dependencies 和 model 组合在一起，并把长期执行交给 Temporal、DBOS、Prefect、Restate 等集成；AutoGen 明确区分便于上手的 AgentChat 和面向事件驱动/分布式系统的 Core Runtime；Google ADK 同时提供模型 Agent、确定性/多 Agent Workflow、Session/Event/Artifact/Memory 和 Runtime。这三者不能只用“都支持工具调用”来比较。[PydanticAI Agents](https://pydantic.dev/docs/ai/core-concepts/agent/)、[PydanticAI Durable Execution](https://pydantic.dev/docs/ai/capabilities/durable_execution/overview/)、[AutoGen Runtime](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/framework/agent-and-agent-runtime.html)、[Google ADK Agents](https://adk.dev/agents/)、[Google ADK Runtime](https://adk.dev/runtime/)

### 3.1 快速决策规则

```text
只是固定步骤？
  └─ 是：普通 Workflow/函数，不要先引入 Agent 框架

需要跨进程恢复、HITL、状态检查和复杂路由？
  └─ 是：LangGraph；强类型 Python 服务也评估 PydanticAI + durable backend

主要是模型驱动的 Tool/Handoff/Guardrail，图不复杂？
  └─ 是：OpenAI Agents SDK

任务天然围绕仓库、Shell、文件和子 Agent？
  ├─ Claude 技术栈：Claude Agent SDK
  ├─ Kimi CLI 技术栈：Kimi Agent SDK
  └─ 模型中立、想直接使用 LangGraph Harness：Deep Agents

确实需要异步分布式多 Agent 消息系统？
  └─ 是：AutoGen Core；Google 生态/A2A/多语言部署则评估 ADK

想研究或二次开发完整开源 Agent 产品？
  └─ 是：Hermes Agent
```

## 4. Agent 框架面试题（12 道）

### 题 1：LangGraph 和普通 Workflow 的本质区别是什么？

**参考回答：**

普通 Workflow 也能表达步骤和分支。LangGraph 的核心价值不是“能画图”，而是把状态、节点激活、Checkpoint、Thread、Interrupt 和恢复协议统一成 Runtime。只有当任务需要跨请求持续、暂停等待人工、失败后从明确位置恢复、检查历史状态时，这层 Runtime 才明显有价值。

如果流程是 `读取 → 转换 → 保存` 三步且一次请求内完成，普通函数更清晰。把简单流程写成图只增加序列化、调试和版本迁移成本。

### 题 2：有了 LangGraph Checkpoint，为什么发布工具仍需幂等键？

**参考回答：**

Checkpoint 保证的是“图状态可恢复”，不是“外部世界只改变一次”。典型故障时序是：

```text
publish API 已成功
→ 进程在写入下一 Checkpoint 前崩溃
→ 恢复后 Runtime 认为 publish Node 尚未完成
→ Node 被再次执行
```

所以外部副作用必须用业务幂等键或唯一约束去重。本项目用 `run_id:publish:v1` 和数据库唯一键保证最多生成一个发布收据。Checkpoint 解决 at-least-once 重放，业务账本把重放转化为 exactly-once effect。

### 题 3：OpenAI Agents SDK 中 Agent、Runner、Tool 和模型分别控制什么？

**参考回答：**

- `Agent` 是配置：指令、模型、工具、Handoff、Guardrail 和输出约束；
- `Runner` 推进循环：调用模型、识别 Tool/Handoff、执行分派、回传 observation、判断是否结束；
- 模型只能生成“建议调用什么”的结构化输出；
- Tool Runtime 才真正执行代码、校验权限、超时和副作用；
- 宿主应用拥有最终授权、持久化、取消和预算控制。

因此不能说“模型调用了数据库”。准确说法是：模型返回 Tool Call，Runner/宿主校验并执行工具，再把 Tool Result 作为 observation 回传。

### 题 4：Handoff 和 Agent-as-Tool 的控制权差异是什么？

**参考回答：**

Handoff 会更换当前 Agent。后续对话使用目标 Agent 的 instructions、tools、guardrails 和输出规则，直到目标结束或再次 Handoff。Agent-as-Tool 则像函数调用：父 Agent 保持控制权，子 Agent 完成一个有界任务并返回结果。

若 Specialist 要持续和用户沟通，选择 Handoff；若只想隔离一次检索或审查，选择 Agent-as-Tool/Subagent。本周真实调试也说明 Handoff 会改变工具边界：转交后的 Agent 若尝试调用旧 Agent 工具，Runtime 应报协议错误，而不是偷偷放宽权限。

### 题 5：Guardrail 和普通输入/输出 Schema 校验有什么区别？

**参考回答：**

Schema 回答“结构是否合法”，例如 citations 是否为字符串数组；Guardrail 回答“在业务或安全语义上是否允许继续”，例如引用是否覆盖强制来源、请求是否越权。两者应组合：先做确定性 Schema 校验，再做确定性策略检查，必要时才用模型型 Guardrail。

Guardrail 失败是预期运行结果，不应使整个批处理进程崩溃。应记录 tripwire 类型、终止该 Run、阻止副作用并继续其他任务。

### 题 6：怎样证明子 Agent 真的实现了上下文隔离？

**参考回答：**

“定义了一个 subagent”不是证据。至少需要验证：

1. Trace 中出现真实子 Agent/Agent Tool 调用；
2. 子 Agent 工具事件具有独立 actor 或 parent span；
3. 父上下文 canary 没有进入子任务或最终结果；
4. 子 Agent 只拥有声明的只读工具；
5. 父 Agent 只收到子 Agent 的结果，而不是完整中间轨迹；
6. Token、并发数、委派深度和超时有上限。

上下文隔离不自动等于进程、文件系统、网络和身份隔离。若子 Agent 与父 Agent 共用宿主权限，它仍可能访问同一资源。

### 题 7：为什么不能根据本实验 Token 表直接说 LangGraph 比 Agents SDK 省 Token？

**参考回答：**

本实验的 Native 和 LangGraph 使用确定性 `build_answer()`，没有模型节点，因此 `model_calls=0`、Token 为 0；Agents SDK 和 Deep Agents 调用真实模型。两组不是相同生成算法，Token 不可直接比较。

公平的 Token 消融必须让四者使用相同模型、Prompt、工具、历史裁剪、最大步骤、停止条件和重试策略。否则测到的是“是否调用模型”和 Harness Prompt 大小，而不只是框架开销。

### 题 8：如何判断一个框架的持久化是否能迁移？

**参考回答：**

不要只看有没有 `save_state()`，要检查：

- 能否导出版本化、稳定、可读的业务状态；
- 状态中是否混入框架内部对象、序列化类名或不可重建的 Client；
- 工具调用、待审批项和当前控制点能否精确恢复；
- Schema 升级怎样迁移旧 Checkpoint；
- 能否在新版本 Runtime 中回放旧状态；
- 即使无法恢复执行，能否导出消息和业务 Artifact。

若只有原框架能解释 Checkpoint，状态锁定通常比模型锁定更严重。

### 题 9：Deep Agents、Claude Agent SDK 和 Kimi Agent SDK 都像 Harness，怎样区分？

**参考回答：**

Deep Agents 是建立在 LangGraph 之上的通用 Python Harness，重点是 planning、filesystem、subagent 和上下文管理；Claude Agent SDK 把 Claude Code 的成熟循环、内置工具、权限、Hooks 和 Session 暴露给应用；Kimi Agent SDK 是复用 Kimi CLI Runtime 的多语言薄客户端。

三者都提高开箱即用程度，但控制面不同：Deep Agents 更接近可组合框架；Claude/Kimi SDK 更接近产品 Runtime 的编程接口。越接近产品 Harness，开发越快，但工具协议、Session 和运行环境的锁定通常越强。[Claude Agent SDK](https://code.claude.com/docs/en/agent-sdk/overview)、[Kimi Agent SDK](https://github.com/MoonshotAI/kimi-agent-sdk)、[Deep Agents](https://docs.langchain.com/oss/python/deepagents/overview)

### 题 10：AutoGen 适合什么任务，为什么不应默认用于多 Agent？

**参考回答：**

AutoGen Core 的优势是 Actor 风格、异步消息、Agent 生命周期、订阅和可扩展 Runtime，适合真的需要事件驱动或分布式 Agent 的系统。AgentChat 则提供更高层的 Agent/Team 模式。

但多数任务的问题是工具可靠性、上下文质量或完成条件，而不是缺少更多 Agent。多 Agent 会增加消息复制、路由错误、终止判断、Token、并发和调试难度。如果单 Agent 加确定性工具能完成，就不应为了“架构先进”引入消息网络。

### 题 11：PydanticAI 相比通用 Agent SDK 的核心优势是什么？

**参考回答：**

优势不是“也能调用工具”，而是把依赖类型和输出类型放进 Agent 泛型，用 Pydantic 统一运行时校验、IDE 类型提示和结构化结果。这适合本来就以 FastAPI/Pydantic 为核心的 Python 服务。

但类型安全只覆盖进入/离开边界的结构，不能证明语义正确；不同 Provider 对 JSON Schema、并行 Tool Call 和流式结构化输出的支持也不一致。耐久执行通常还依赖 Temporal、DBOS、Prefect 或 Restate，不能把集成能力误认为内置单体 Runtime。

### 题 12：Google ADK 与 LangGraph 的选型分界是什么？

**参考回答：**

LangGraph 强项是以显式状态图控制执行与恢复，适合希望精确掌握 State/Node/Checkpoint 的 Python 系统。ADK 覆盖范围更宽：Agent、确定性与多 Agent Workflow、Session/Event/Artifact/Memory、Runtime、A2A、评测和多语言/部署生态。

若主要问题是复杂状态机和可恢复执行，LangGraph 更聚焦；若团队需要 Google Cloud 部署、多语言 Agent、A2A 和一体化运行服务，ADK 值得优先评估。选择 ADK 后，应提前验证 Session/Artifact 可导出性，避免运维层锁定。

## 5. 15 分钟源码入口与运行时流程讲解

下面不是把代码从头念到尾，而是按“公共契约 → 原生基线 → 框架替换 → 实验事实”建立因果链。

### 0:00–1:30：先讲任务和不变量

打开 [`source/common.py`](./source/common.py)，说明四个实现共享：

- `ResearchTask`：同一输入和必需来源；
- `ToolResult`：成功、错误、是否可重试的稳定结构；
- `TraceEvent/TraceRecorder`：统一 Model/Tool/Control/Approval/Subagent Span；
- `ArtifactLedger`：草稿、审批提交和发布收据；
- `ResearchToolRuntime`：搜索、读取、保存、发布的实际执行者；
- `score()`：完成状态之外，还要满足术语和来源要求。

讲解重点：框架可以替换，但业务契约、权限和完成谓词不能散落在框架 Prompt 中。

### 1:30–4:00：原生基线展开所有隐藏责任

打开 [`source/native_baseline.py`](./source/native_baseline.py)：

```text
start(task)
  → search_documents
  → read_document × N（瞬时错误最多重试一次）
  → build_answer
  → save_draft(idempotency_key)
  → 原子写 JSON 状态
  → waiting_approval

resume(run_id, decision, submission_id)
  → 读取 JSON 与中断前 Trace
  → claim_submission 去重
  → reject：确定性终止
  → approve：publish_report(idempotency_key)
  → completed
```

强调两个键：`submission_id` 防止同一审批请求被消费两次；`publish idempotency_key` 防止恢复重放造成二次发布。它们解决的是不同层的问题。

### 4:00–7:00：LangGraph 主实现怎样替换控制面

打开 [`source/langgraph_workflow.py`](./source/langgraph_workflow.py)，从 `_build_graph()` 开始：

```text
START → search → read ──无证据──→ END
                         │
                       有证据
                         ↓
                       draft → approval(interrupt)
                                         │
                    reject → END ← decision → approve → publish → END
```

解释：

- `GraphState` 是可序列化业务状态，而不是随便塞对象的字典；
- Node 是可能因恢复而重放的工作单元，因此外部副作用仍需幂等；
- `thread_id` 标识一条可持续执行线；
- `interrupt()` 把待审批信息写进 Checkpoint；
- `Command(resume=...)` 注入人工决定并从控制点继续；
- SQLite Checkpointer 保存图状态，`ArtifactLedger` 保存业务副作用，两者不能合并成一个概念。

### 7:00–10:00：Agents SDK 的 Agent loop、Handoff 和 Guardrail

打开 [`source/agents_sdk_workflow.py`](./source/agents_sdk_workflow.py)，沿这条路径讲：

```text
Runner.run(researcher)
  → 模型决定 search/read/save_draft
  → Runner 执行 function_tool 并回传 observation
  → handoff(compliance_specialist)
  → 模型请求 publish_report
  → needs_approval=True 产生 interruption
  → RunState.to_json() 落盘

丢弃原 Workflow 对象，模拟进程内对象全部丢失
  → 从 metadata + RunState + SQLite 重建 Agent/Runtime
  → approve/reject interruption
  → Runner.run(state) 继续
  → output guardrail 校验强制引用
```

强调三条边界：

1. Token/Client 放在宿主 Context，不进入模型输入和持久状态；
2. Handoff 之后工具集合随当前 Agent 改变；
3. Guardrail、模型行为错误和最大轮数都要变成结构化 Run 失败，不能炸掉整个批次。

官方 Runner 语义也是“模型 → tool/handoff → 再次模型 → final output”的循环。[OpenAI Agents SDK — Running agents](https://openai.github.io/openai-agents-python/running_agents/)

### 10:00–12:00：Deep Agents 最小 Harness 与隔离证明

打开 [`source/deep_agents_workflow.py`](./source/deep_agents_workflow.py)：

- `create_deep_agent()` 建立主 Agent；
- `evidence-researcher` 只拿到 search/read 两个只读工具；
- 主 Agent 用 `task` 类委派把有界任务交给子 Agent；
- Tool Trace 的 actor 必须是 `research_subagent`；
- 父上下文 canary `PARENT_ONLY_7W` 不得出现在最终答案；
- 子 Agent 返回证据后，主 Agent组合答案。

说明隔离边界：独立对话减少上下文污染，但当前版本仍共享 Python 进程和工具 Runtime；它不是操作系统安全沙箱。

### 12:00–14:00：统一实验怎样避免不可比

打开 [`source/run_comparison.py`](./source/run_comparison.py)：

- 四种实现加载同一 `tasks.json`；
- 每个 repetition 使用独立目录；
- 已存在 `runs/` 时直接拒绝运行，防止旧 Checkpoint/幂等缓存污染；
- SDK 在审批前重新创建 Workflow 对象，覆盖磁盘恢复路径；
- 失败不会被自动改写为成功，而是记录 `error_type`；
- 汇总成功率、步骤、工具、模型、Token、延迟和标准差。

然后打开正式报告解释 Token 边界：Native/LangGraph 没有模型节点，所以 Token 为 0；不能拿它证明图框架比真实 Agent loop 更便宜。

### 14:00–15:00：给出结论，而不是框架排名

最后一分钟可以这样总结：

> 原生基线让所有责任可见；LangGraph 把状态推进、Checkpoint 和 Interrupt 交给图 Runtime；Agents SDK 把模型—工具—Handoff 循环交给 Runner；Deep Agents 再增加规划、文件和子 Agent Harness。框架越高层，上手越快，但抽象泄漏和状态锁定越强。无论选择哪一层，权限、幂等、完成谓词、失败分类和业务数据仍必须由确定性系统负责。

## 6. 验收矩阵与复现命令

### 6.1 关键行为验收

| 验收项 | 证据 | 结果 |
|---|---|---:|
| 原生基线独立运行 | `NativeBaseline.start/resume` | 通过 |
| LangGraph 持久化、Interrupt、恢复 | SQLite Checkpointer + `interrupt/Command(resume)` | 通过 |
| 新 LangGraph 对象恢复旧 Thread | `test_langgraph_state_survives_new_process_object` | 通过 |
| 重复审批不重复发布 | `submission_id` 唯一约束 + 参数化测试 | 通过 |
| 审批拒绝不发布 | Native/LangGraph 拒绝测试 | 通过 |
| 工具瞬时异常有界恢复 | 首次 read 失败、第二次成功 | 通过 |
| Agents SDK Tool/Handoff/Guardrail | 原生 SDK primitives + 真实 DeepSeek 运行 | 通过 |
| Agents SDK 状态持久与对象丢失后恢复 | `RunState.to_json/from_json` + 重建 Runtime | 通过 |
| Deep Agents 真实子 Agent 调用 | `create_deep_agent` + 子 actor Tool Trace | 通过 |
| Deep Agents 上下文隔离 | canary 不泄漏且三轮 `context_isolated=true` | 通过 |
| 四架构统一 Trace 与指标 | `RunReport` + 12 条正式 Run | 通过 |
| 密钥不进入源码/Trace | 凭据扫描 | 通过 |

### 6.2 本地测试

```bash
source /Users/shiyiliu/workspace/pyproject/.venv/bin/activate
python -m pytest -q \
  -o cache_dir=/tmp/week7-pytest-cache \
  7-w/source/tests
```

验收结果：

```text
16 passed
```

### 6.3 真实模型三轮复现

每次必须使用一个新的输出目录；运行器会拒绝复用已有 `runs/`：

```bash
source /Users/shiyiliu/workspace/pyproject/.venv/bin/activate
export DEEPSEEK_API_KEY='从密钥管理器读取，不写入仓库'
export OPENAI_BASE_URL='https://api.deepseek.com'
export AGENT_TEST_MODEL='deepseek-v4-pro'

python 7-w/source/run_comparison.py \
  --output 7-w/source/artifacts/reproduction-01 \
  --repeats 3
```

本次正式结果：

| 架构 | 成功率 | 平均步骤 | 平均工具数 | 平均模型数 | 总 Token（均值±σ） | 延迟 ms（均值±σ） |
|---|---:|---:|---:|---:|---:|---:|
| Native | 100% | 8 | 7 | 0 | 0±0 | 4.47±1.59 |
| LangGraph | 100% | 8 | 7 | 0 | 0±0 | 10.33±2.58 |
| Agents SDK | 100% | 16 | 8 | 6 | 14,590±477 | 28,525.02±1,426.99 |
| Deep Agents | 100% | 14 | 8 | 5 | 16,115±33 | 26,890.15±756.13 |

### 6.4 最终验收边界

本阶段可以通过，但结论必须写完整：

- **已通过**：主实现、原生对照、Agents SDK 对照和所选 Harness 最小版本均可运行；共享研究任务、工具契约、完成谓词和 Trace；主实现与 SDK 实现覆盖审批、恢复和幂等关键行为。
- **刻意未做**：没有在 Deep Agents 中重复实现一套审批/Checkpoint/发布状态机；它只验证 Harness 的子 Agent 与上下文隔离变量。
- **不能推出**：不能用 Native/LangGraph 的零 Token 证明它们比 Agents SDK/Deep Agents 省 Token，因为前两者没有调用模型。
- **生产前仍需补充**：Checkpoint Schema 迁移、并发审批竞争、真实外部副作用补偿、分布式锁、Provider 限流、长期负载和安全审计。

## 7. 官方资料

- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview)
- [LangGraph persistence](https://docs.langchain.com/oss/python/langgraph/persistence)
- [LangGraph interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts)
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/)
- [Deep Agents overview](https://docs.langchain.com/oss/python/deepagents/overview)
- [Claude Agent SDK overview](https://code.claude.com/docs/en/agent-sdk/overview)
- [Claude Agent SDK subagents](https://code.claude.com/docs/en/agent-sdk/subagents)
- [Hermes Agent](https://github.com/NousResearch/hermes-agent)
- [Kimi Agent SDK](https://github.com/MoonshotAI/kimi-agent-sdk)
- [PydanticAI Agents](https://pydantic.dev/docs/ai/core-concepts/agent/)
- [PydanticAI Durable Execution](https://pydantic.dev/docs/ai/capabilities/durable_execution/overview/)
- [AutoGen](https://microsoft.github.io/autogen/stable/)
- [AutoGen Agent and Runtime](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/framework/agent-and-agent-runtime.html)
- [Google ADK Agents](https://adk.dev/agents/)
- [Google ADK Runtime](https://adk.dev/runtime/)
