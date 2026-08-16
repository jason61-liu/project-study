# AI 深耕学习计划：Agent 主线 × 模型推理

> 周期：16 周  
> 强度：每周 16–18 小时  
> 个人基线：硕士、10 年工程经验、近 2 年学习与实践 Agent  
> 求职定位：资深 AI Agent / 大模型应用架构与工程岗位  
> 主线比例：Agent 约 70%，模型推理约 30%；基础原理均为必修
> 技术生态快照：2026-07-17；框架会变化，学习重点按可迁移机制而非项目热度排序

> 科技路线图将在本计划确认后生成并嵌入。
>
> 制定依据：[AI Agent 与模型推理岗位市场调研](./ai-agent-job-market-research.md)

## 使用方法

1. 每周开始时填写计划日期，并浏览本周全部任务。
2. 完成任务后将 `- [ ]` 改为 `- [x]`；只有达到任务描述中的可验证结果才算完成。
3. 每周至少保留一份代码、一份测试或数据记录，以及一份书面复盘。
4. 每四周执行一次阶段关卡。关卡未通过时，优先补缺，不继续叠加框架。
5. 预计每周分配：原理与论文 5 小时、编码与实验 7 小时、面试表达 3 小时、复盘与补缺 1–3 小时。
6. 10 年经验不再用入门 Demo 证明能力；每周产出优先采用架构决策记录、评测数据、故障复盘、容量估算或可运行系统。

## 最终目标

完成 16 周后，应能够：

- 从 Transformer 推理过程解释 Agent 每一次模型调用的成本、延迟和上下文限制。
- 不依赖框架实现具备工具、状态、重试、终止和轨迹记录的 Agent Loop。
- 在 Workflow、单 Agent、多 Agent、Manager、Handoff 等架构之间做出有证据的选择。
- 设计可复现的 Agent 评测集，区分模型、检索、工具、上下文和编排故障。
- 建立最小权限、人工审批、沙箱、审计和 Prompt Injection 防护。
- 使用 TTFT、TPOT、吞吐、Goodput、P95/P99 和单位任务成本评估系统。
- 展示一个包含架构图、评测报告、威胁模型、性能数据和演示的深度研究 Agent。
- 主持一次 30 分钟资深岗位级设计评审，清楚说明范围、权衡、SLO、容量、成本、风险和演进路径。

## 知识优先级

### 必须精通

- Agent Loop、状态机、终止条件、重试、幂等、结构化输出和工具调用。
- Tool Schema、错误语义、超时、权限边界、并发和不可控外部结果处理。
- RAG、上下文选择、Token 预算、摘要压缩、状态持久化和记忆淘汰。
- 区分运行状态、Checkpoint、语义/情景/程序性记忆、Prompt Cache、KV Cache 与业务事实库，能为每一层定义写入、检索、淘汰和评测策略。
- Workflow/Agent 边界、Routing、ReAct、Plan-and-Execute、Manager/Handoff 的取舍。
- 评测集、Rubric、轨迹、最终状态、回归测试、成本和延迟评测。
- Prompt Injection、最小权限、人工审批、沙箱、敏感信息和审计。
- OAuth 2.0/OIDC、用户委托授权、RBAC/ABAC、Scope、Token 撤销、Secret 轮换、多租户隔离、PII 脱敏与数据删除传播。
- Transformer、Prefill/Decode、KV Cache、采样和主要推理性能指标。
- Decoder-only Transformer 的数据流、Attention 复杂度、RoPE、Tokenizer、MHA/MQA/GQA，以及这些机制如何约束 Agent 的上下文、延迟与成本。
- 生产系统的 SLO、容量估算、尾延迟、回压、故障恢复、可观测性和架构决策表达。
- Prompt/模型/Tool Schema/MCP/数据集版本化、Eval Gate、Shadow/Canary、回滚、线上反馈和业务价值度量。

### 需要熟练

- MCP Host/Client/Server、生命周期、能力协商、Tools/Resources/Prompts。
- LangGraph Persistence/Interrupt/HITL 与 OpenAI Agents SDK Runner/Tool/Handoff/Guardrail/Tracing。
- 混合检索、重排、分层摘要、短期/长期记忆、Checkpoint 和故障恢复。
- 异步、限流、指数退避、熔断、队列、可观测性和模型路由。
- Continuous Batching、PagedAttention、Prefix Cache、Chunked Prefill、量化和 Speculative Decoding。
- vLLM、SGLang 的部署、配置、Benchmark 和取舍；LMCache 的分层 KV Cache 与跨实例复用。
- DeepSeek 的 MLA、DeepSeekMoE/MoE、MTP、FP8、GRPO、DSA，以及 Kimi 的 MoE、MuonClip、KDA/混合线性注意力、长上下文 RL 和 Mooncake KV-centric Serving 的设计动机。
- Mem0 的抽取、存储、检索、更新、删除和隔离边界；Deep Agents、Claude Agent SDK、Hermes Agent、Kimi Agent SDK 等 Agent Harness 的职责与差异。
- LoRA、SFT、DPO、蒸馏和 Agentic RL 的目标、数据、训练/推理差异、评测风险与采用条件。

### 了解即可

- CUDA、Triton、算子融合、NCCL、张量并行、流水线并行和专家并行的实现细节。
- TensorRT-LLM、NVIDIA Dynamo、llm-d、Mooncake 的集群级部署细节；DeepGEMM、FlashMLA、DualPipe、3FS、FlashKDA 的内核或训练基础设施实现。
- Debate、Swarm、大规模自治多 Agent、复杂认知记忆和具身 Agent。
- 预训练数据工程、分布式训练和 Agentic RL 的完整训练流水线。
- Dify、Coze 等低代码平台的能力边界；能快速评估，不把平台使用等同于 Agent 架构能力。

## 技术生态学习矩阵

这份计划不要求“精通所有框架”。精通的是可迁移机制；每一层选择 1–2 个代表实现动手，其余通过源码入口、最小样例和选型表达到熟悉。

| 能力层 | 必须掌握的机制 | 必须动手的代表实现 | 熟悉与对比 | 只做前沿跟踪 |
|---|---|---|---|---|
| 模型架构与推理 | Attention/KV Cache、MLA、MoE、稀疏/线性注意力、推理时计算、量化 | PyTorch 最小 Attention/KV 实验；小模型推理 Benchmark | DeepSeek V3/R1/V3.2、Kimi K2/K2.5/Kimi Linear 的技术报告与公开实现 | DeepGEMM、FlashMLA、FlashKDA、DualPipe、3FS 的内核/集群细节 |
| 单机 Serving | 调度、PagedAttention、Radix/Prefix Cache、Chunked Prefill、抢占 | 同模型、同负载实测 vLLM 与 SGLang | TensorRT-LLM、llama.cpp/MLX 的适用边界 | 新后端、新硬件的版本特性 |
| 分布式推理与缓存 | KV Cache 分层、跨实例复用、KV-aware Routing、Prefill/Decode 解耦 | vLLM/SGLang + LMCache 最小复用实验 | Mooncake、NVIDIA Dynamo、llm-d 的架构与成本条件 | 多节点 RDMA/NIXL/RoCE 调优细节 |
| Agent Runtime/Harness | Loop、状态、恢复、HITL、Handoff、子 Agent、Skills、Tracing | 原生 Loop → LangGraph → OpenAI Agents SDK | Deep Agents、Claude Agent SDK/Claude Code 子 Agent、Hermes Agent、Kimi Agent SDK | PydanticAI、AutoGen、Google ADK 等做季度选型复核 |
| 协议与工具 | Tool Schema、MCP、身份、授权、幂等、错误语义 | 自建 MCP Server/Client | A2A 的 Agent Card、Task、Artifact 与 MCP 的边界 | 协议生态与互操作成熟度 |
| 上下文与记忆 | 选择、压缩、隔离、溯源；语义/情景/程序性记忆；热路径/后台更新 | 自建记忆基线 + Mem0 对照实验 | LangGraph Store、Deep Agents 文件系统上下文 | 图记忆、学习型记忆及长期基准进展 |
| 隔离执行 | 容器/微虚机边界、网络出口、文件系统、Secret、资源限额、快照和审计 | E2B 或 OpenSandbox 完成一次真实工具执行 | Agent Sandbox、CubeSandbox 的部署、E2B 兼容性和隔离模型 | 大规模多租户与 RL Sandbox 调度 |
| 身份与数据治理 | OAuth/OIDC、委托授权、RBAC/ABAC、租户/用户隔离、PII、保留与删除 | 双租户工具、RAG、记忆和日志越权测试 | OPA/云 IAM/企业 IdP 的集成模式 | 行业合规与跨域身份协议 |
| LLMOps 与反馈 | 全链路版本、CI Eval Gate、Shadow/Canary、回滚、人工升级、Offline/Online 指标 | 一次回归阻断和一次灰度回滚演练 | OpenTelemetry GenAI、实验平台和反馈数据闭环 | 自动 Prompt/策略优化 |

### 代表技术为什么进入计划

- DeepSeek 线用于理解“模型结构如何降低训练与推理成本”：MLA 压缩 KV、稀疏 MoE 控制激活参数、MTP 扩充训练信号并可辅助推测解码、DSA 面向长上下文稀疏计算，R1 用 RL 与蒸馏塑造推理能力。
- Kimi 线用于理解“长上下文、Agent 能力与 Serving 如何协同”：K2/K2.5 的 MoE 与 Agentic 能力、Kimi k1.5 的长上下文 RL、Kimi Linear 的 KDA/混合线性注意力，以及 Mooncake 的 KV-centric 解耦 Serving。
- vLLM/SGLang 是推理引擎；LMCache 是可接在引擎上的 KV Cache 层；Mooncake/Dynamo/llm-d 更偏集群级调度与解耦。面试中必须能说清这些层次，不能把它们当作同类产品横向罗列。
- LangGraph、Deep Agents、Claude Agent SDK、Hermes Agent、Kimi Agent SDK 属于不同抽象层的 Runtime/Harness 或产品化 Agent。学习目标是拆出状态、工具、子 Agent、上下文、技能、恢复和追踪机制，而不是背 API。
- Mem0 管长期记忆，Sandbox 管不可信执行，MCP 管工具/数据连接；三者解决的问题互不替代。

## 16 周路线总览

| 周次 | 主题 | 验收产出 |
|---|---|---|
| 1 | Transformer 与生成基础 | Attention 实现、复杂度与 KV Cache 推导、15 道原理题 |
| 2 | API 与最小 Agent | 无框架 Agent Loop、结构化工具调用、轨迹日志 |
| 3 | 工具、MCP 与委托授权 | 5 个可靠工具、只读 MCP Server、Scope 与 Token 撤销测试 |
| 4 | RAG、上下文工程与 Mem0 | 三种上下文策略、自建/Mem0 记忆及 ACL 隔离对比 |
| 5 | Agent 架构模式 | 三种架构实现及取舍报告 |
| 6 | 规划、恢复与编排 | 可恢复任务编排及单/多 Agent 消融 |
| 7 | Agent Runtime 与 Harness | LangGraph 主版本、Agents SDK 对照、主流 Harness 选型表 |
| 8 | Agent 评测与标准 Benchmark | 50+ 条业务评测集、一个标准 Benchmark 子集和 CI 阈值 |
| 9 | Agent 安全、身份与 Sandbox | 威胁模型、双租户隔离、隔离执行和 15+ 条攻击测试 |
| 10 | 生产可靠性与 LLMOps | OpenTelemetry、版本链、SLO、灰度回滚和故障注入 |
| 11 | 推理指标 | API Benchmark 和性能图表 |
| 12 | vLLM、SGLang、LMCache 与解耦 Serving | 同负载引擎实测、缓存复用实验和集群架构取舍 |
| 13 | DeepSeek/Kimi 技术与推理优化 | 前沿机制图谱、质量—延迟—成本决策表和模型路由实验 |
| 14 | 上下文/记忆优化与训练方法边界 | 端到端优化报告及 LoRA/SFT/DPO/RL 采用决策 |
| 15 | 资深综合项目 | 深度研究 Agent、业务基线、ADR、SLO、容量模型和自动测试 |
| 16 | 强化与面试 | 业务收益、发布闭环、作品集、Demo 和 120 道题复盘 |

---

## 第 1 周：Transformer 与生成基础

**计划日期：** ____ 至 ____

### 本周目标

- [ ] 无笔记画出 Decoder-only Transformer 单层结构，包含残差、归一化、Attention、MLP/SwiGLU 和 Logits 数据流。
- [ ] 推导 Scaled Dot-Product Attention，并解释除以 `sqrt(d_k)` 的原因。
- [ ] 区分训练、Prefill 和逐 Token Decode 三个过程。

### 知识与阅读

- [x] 阅读 [Attention Is All You Need](https://arxiv.org/abs/1706.03762) 的模型结构、Attention 和复杂度部分，写一页摘要。
- [x] 掌握 Q/K/V 张量形状、因果 Mask、Multi-head 拼接、输出投影、RMSNorm/LayerNorm 与 MLP/SwiGLU。
- [x] 比较 MHA、MQA、GQA 对质量、KV Cache 和带宽的影响。
- [x] 解释 RoPE、Tokenizer、Context Window、Temperature、Top-k 和 Top-p。
- [x] 写出 KV Cache 显存估算公式，并计算一个 7B 模型在 4K、16K 上下文下的示例。
- [x] 阅读 DeepSeek-V3 的 MLA/MoE 概览和 Kimi Linear 的 KDA 概览，画出 MHA/GQA、MLA、混合线性注意力的“状态大小—计算—表达能力”对比图；本周只求建立坐标，不深挖内核。

### 编码实验

- [x] 使用 PyTorch 手写单头和多头 Causal Attention，不调用封装的 Attention 层。
- [x] 用随机张量验证输出形状、Mask 和数值稳定性。
- [x] 编写至少 5 个测试，覆盖长度 1、不同 Batch、不同 Head 和无效形状。
- [x] 编写最小自回归生成循环，观察每次 Decode 只生成一个 Token。
- [x] 记录使用与不使用 KV Cache 时的计算差异和理论复杂度。

### 面试与验收

- [x] 完成 15 道 Transformer/推理基础题并记录答案。
- [x] 进行一次 10 分钟白板讲解：一个 Token 如何经过模型并生成下一个 Token。
- [x] 能回答“为什么 Decode 往往受显存带宽限制，而 Prefill 更偏计算密集”。
- [x] 代码可以独立运行，测试全部通过，并提交本周复盘。

**复盘：** 最大卡点：____；错误认识：____；需要补强：____

## 第 2 周：LLM API 与最小 Agent

**计划日期：** ____ 至 ____

### 本周目标

- [ ] 区分普通对话、结构化输出、工具调用和 Agent Loop。
- [ ] 能解释 Agent 的状态、动作、观察、终止和预算。
- [ ] 不依赖 Agent 框架完成一个可测试的最小 Agent。

### 知识与阅读

- [x] 梳理 system/developer/user/assistant/tool 消息的职责和信任边界。
- [x] 掌握 JSON Schema、结构化输出校验、Tool Call ID 和工具结果回传。
- [x] 理解流式响应、超时、取消、速率限制和非确定性输出。
- [x] 阅读 [ReAct](https://arxiv.org/abs/2210.03629) 的方法与实验，提炼“推理—行动—观察”循环。
- [x] 明确最大步数、Token/费用预算、超时和显式完成条件。

### 编码实验

- [x] 实现支持结构化输出和流式响应的模型调用适配层。
- [x] 实现 `model -> tool -> observation -> model` 最小循环。
- [x] 为每次模型调用和工具调用生成 Trace ID、开始时间、结束时间和结果状态。
- [x] 加入未知工具、参数校验失败、工具异常、超时和模型拒答处理。
- [x] 编写至少 8 个测试，覆盖正常完成、最大步数、错误恢复和提前终止。

### 面试与验收

- [x] 写出 Agent 与 Workflow、Chatbot、RAG Pipeline 的边界。
- [x] 回答至少 5 道 Agent Loop 设计题。
- [x] 完成 10 分钟演示，并能逐步解释一条完整轨迹。
- [x] 核心循环保持简单、可以独立运行，测试全部通过。

**复盘：** 最大卡点：____；错误认识：____；需要补强：____

## 第 3 周：可靠工具、MCP 与委托授权

**计划日期：** ____ 至 ____

### 本周目标

- [ ] 能把外部能力设计成边界清晰、可校验、可恢复的工具。
- [ ] 能解释 MCP Host、Client、Server 及协议生命周期。
- [ ] 能识别工具“功能过大、权限过大、身份混淆、输出不可信”的风险。

### 知识与阅读

- [x] 阅读 [MCP Architecture](https://modelcontextprotocol.io/docs/learn/architecture)，绘制连接和能力协商时序图。
- [x] 掌握 Tools、Resources、Prompts 的控制方和适用场景。
- [x] 理解 JSON-RPC、STDIO、Streamable HTTP、通知、取消和错误响应。
- [x] 整理工具描述、参数枚举、默认值、幂等键、超时、重试，以及成功、业务失败、系统失败和部分成功的返回结构。
- [x] 掌握 OAuth 2.0/OIDC、用户身份/Agent 身份/服务身份、Delegated Token、Scope、Token 过期与撤销，画出用户授权后 Agent 调用工具的时序图。
- [x] 阅读 [A2A Protocol](https://google-a2a.github.io/A2A/specification/) 概览，区分 MCP 的“Agent—工具/数据”连接与 A2A 的“独立 Agent—Agent”协作。

### 编码实验

- [x] 实现搜索、读取、计算、保存草稿和状态查询 5 个最小工具。
- [x] 为全部工具加入 Schema 校验、超时、结构化错误和日志。
- [x] 对具有副作用的工具加入幂等键和 dry-run/确认机制，并要求有效用户令牌与最小 Scope；测试缺失、过期、撤销和越权 Scope。
- [x] 实现一个只读 MCP Server，暴露至少 2 个 Tools 和 1 个 Resource，并透传 `tenant_id/user_id` 授权上下文而不把 Token 暴露给模型。
- [x] 使用 MCP Inspector 或客户端完成发现、调用、错误和取消测试。

### 面试与验收

- [x] 写出“函数调用与 MCP 的关系和差异”面试答案。
- [x] 完成至少 6 道工具/MCP 设计题。
- [x] 进行 10 分钟协议与委托授权时序讲解，说明身份、Token、模型和工具之间的安全边界。
- [x] 5 个工具和 MCP Server 均具备正常与异常测试。

**复盘：** 最大卡点：____；错误认识：____；需要补强：____

## 第 4 周：RAG、上下文工程与 Mem0

**计划日期：** ____ 至 ____

### 本周目标

- [ ] 能把上下文视为有限预算，而不是无条件追加历史。
- [ ] 能区分检索、工作记忆、Checkpoint、长期记忆、业务状态和模型 KV Cache。
- [ ] 用数据比较至少三种上下文策略。

### 知识与阅读

- [x] 掌握文档增量摄取、版本/去重、Chunking、Embedding、混合检索、Reranking、时效性、ACL 继承、删除传播和索引重建；结果必须同时满足相关性与访问权限。
- [x] 掌握 Recall@K、MRR、引用正确率、答案忠实度和无答案检测。
- [x] 阅读 [Anthropic Context Engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)，掌握选择、去重、摘要、压缩、隔离、溯源、优先级和过期策略。
- [x] 区分 Conversation History、Checkpoint、Semantic/Episodic/Procedural Memory、业务数据库、Prompt Cache 和 KV Cache。
- [x] 阅读 [LangGraph Memory](https://docs.langchain.com/oss/python/concepts/memory) 与 [Mem0 文档](https://docs.mem0.ai/introduction)，比较线程状态、命名空间记忆、热路径/后台写入、事实抽取及租户隔离。

### 编码实验

- [x] 构建一个带来源引用的最小文档检索链。
- [x] 准备至少 20 条检索问题和对应证据。
- [x] 实现全历史、摘要历史、检索式记忆三种上下文策略。
- [x] 对三种策略记录正确率、输入 Token、延迟和单位任务成本。
- [x] 加入无相关证据、冲突证据、超长/恶意文档，以及双租户越权、过期文档和删除后残留测试。
- [x] 用同一组多轮任务比较“自建向量记忆基线”和 Mem0，记录写入正确率、Recall@K、错误记忆率、租户隔离、更新/删除一致性、延迟与 Token；没有云 Key 时使用 Mem0 开源版。

### 面试与阶段关卡

- [x] 完成至少 8 道 RAG/上下文/记忆面试题。
- [x] 输出一份上下文/记忆决策表，明确何时不应使用长期记忆、何时必须回到权威业务数据源。
- [x] 无笔记讲清前四周的完整数据流：用户输入到工具、上下文和模型输出。
- [x] 阶段关卡通过：代码可运行、核心测试通过、有数据和失败案例。

**复盘：** 最大卡点：____；错误认识：____；需要补强：____

## 第 5 周：Agent 架构模式

**计划日期：** ____ 至 ____

### 本周目标

- [ ] 根据任务不确定性判断使用确定性 Workflow 还是 Agent。
- [ ] 掌握 Routing、Parallelization、ReAct、Plan-and-Execute 和 Evaluator-Optimizer。
- [ ] 用同一任务比较三种架构，而不是凭偏好选框架。

### 知识与阅读

- [x] 阅读 [Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents)，整理 Workflow 与 Agent 边界。
- [x] 分析 Prompt Chaining、Routing、并行、Orchestrator-Workers 和 Evaluator-Optimizer。
- [x] 解释 ReAct 的优势、轨迹膨胀和错误累积问题。
- [x] 比较 Plan-and-Execute 与逐步决策对长任务的影响。
- [x] 定义架构选择维度：成功率、可控性、延迟、成本、可观测性和恢复能力。

### 编码实验

- [x] 选择一个研究任务，分别实现固定 Workflow、ReAct 和 Plan-and-Execute。
- [x] 保持模型、工具和测试集一致，避免不可比实验。
- [x] 每种架构至少运行 3 次，记录成功率、步骤数、工具数、Token 和延迟。
- [x] 构造检索失败、工具失败和计划失效场景。
- [x] 输出架构对比表和推荐决策树。

### 面试与验收

- [x] 完成至少 8 道 Agent 架构设计题。
- [x] 回答“为什么不应该默认使用多 Agent”。
- [x] 进行一次 15 分钟架构评审陈述。
- [x] 三种实现和实验数据均可复现。

**复盘：** 最大卡点：____；错误认识：____；需要补强：____

## 第 6 周：规划、失败恢复与任务编排

**计划日期：** ____ 至 ____

### 本周目标

- [ ] 能设计可更新、可验证、可终止的计划。
- [ ] 能处理计划失效、工具部分成功和中断恢复。
- [ ] 先建立可恢复的单 Agent/Workflow 基线，再用消融判断是否需要多 Agent。

### 知识与阅读

- [x] 掌握任务分解、依赖图、子目标、完成条件和重规划触发器。
- [x] 区分自我反思、外部验证器和确定性检查器。
- [x] 比较状态机、任务队列、Manager、Handoff 和 Agent-as-Tool 的控制权边界。
- [x] 分析串行/并行编排、多 Agent 的上下文重复、错误传播、调度和成本问题。
- [x] 定义状态版本、幂等执行、断点和补偿动作。

### 编码实验

- [x] 为第 5 周任务加入结构化 Plan 和逐项完成状态。
- [x] 实现失败分类：可重试、需重规划、需人工介入、不可恢复。
- [x] 实现可恢复的单 Agent 基线；仅为消融增加 Manager + Specialist 对照版本。
- [x] 对至少 20 条任务运行消融实验并记录三次重复结果。
- [x] 测试进程中断、部分成功、状态冲突、重复委派和上下文丢失。

### 面试与验收

- [x] 完成至少 8 道规划、恢复与任务编排面试题。
- [x] 写出单 Agent、Workflow、多 Agent 的采用条件和反例清单。
- [x] 无笔记解释 Manager 与 Handoff 的控制权差异。
- [x] 消融报告同时包含收益、代价和统计波动。

**复盘：** 最大卡点：____；错误认识：____；需要补强：____

## 第 7 周：Agent Runtime 与 Harness——LangGraph、Agents SDK 与主流产品

**计划日期：** ____ 至 ____

### 本周目标

- [ ] 理解框架如何实现状态、调度、持久化和可观测性。
- [ ] 用 LangGraph 作为主实现，用 OpenAI Agents SDK 作为对照，保持“原生 Loop → LangGraph → Agents SDK”的学习顺序。
- [ ] 能从 Deep Agents、Claude、Hermes、Kimi 等产品中抽取可迁移的 Harness 机制，而不是背框架 API。

### 知识与阅读

- [x] 学习 LangGraph State、Node、Edge、Conditional Edge 和 Reducer。
- [x] 学习 Checkpoint、Thread、Interrupt、HITL、恢复和 Time Travel。
- [x] 阅读 [OpenAI Agents SDK Agents](https://openai.github.io/openai-agents-python/agents/) 与 [Tracing](https://openai.github.io/openai-agents-python/tracing/)，掌握 Runner、Tool、Handoff、Guardrail 和 Trace/Span。
- [x] 阅读 [Deep Agents](https://docs.langchain.com/oss/python/deepagents/overview)、[Claude Code Subagents](https://code.claude.com/docs/en/sub-agents) 与 [Claude Agent SDK Subagents](https://code.claude.com/docs/en/agent-sdk/subagents)，比较 Planning、Subagent、Filesystem Context、Skills、Memory 和 Agent Team。
- [x] 定位 [Hermes Agent](https://github.com/nousresearch/hermes-agent) 与 [Kimi Agent SDK](https://github.com/MoonshotAI/kimi-agent-sdk) 的核心入口，并建立“原生循环 → LangGraph → Agents SDK → 产品化 Harness”映射表。

### 编码实验

- [x] 用 LangGraph 重构第 6 周系统，并保留原生基线。
- [x] 加入持久化、人工审批、中断恢复和可重复执行。
- [x] 使用 Agents SDK 实现等价的工具、Handoff 和 Guardrail 流程。
- [x] 在 Deep Agents 或 Claude Agent SDK 中任选一个实现同任务最小版本，验证子 Agent 与上下文隔离；不重复实现全部框架。
- [x] 为原生、LangGraph、Agents SDK 及所选 Harness 版本采集相同任务的轨迹和指标。
- [x] 测试状态恢复、重复提交、审批拒绝和工具异常。

### 面试与验收

- [x] 完成至少 8 道 Agent 框架面试题。
- [x] 输出选型表：LangGraph、OpenAI Agents SDK、Deep Agents、Claude Agent SDK、Hermes Agent、Kimi Agent SDK，并补充 PydanticAI、AutoGen、Google ADK 的定位、抽象泄漏和锁定风险。
- [x] 进行 15 分钟源码入口与运行时流程讲解。
- [x] 主实现、对照实现和所选 Harness 最小版本均可运行，关键行为与原生基线一致。

**复盘：** 最大卡点：____；错误认识：____；需要补强：____

> 名称说明：本计划把“DeepAgent”映射为 LangChain 的 **Deep Agents**；把“Claude manage agent”映射为 Claude Code 的 Subagents/Agent Teams 与 Claude Agent SDK Subagents；把“Hermes”映射为 Nous Research 的 **Hermes Agent**。若你指的是其他同名项目，再按仓库地址替换。

## 第 8 周：Agent 评测体系与标准 Benchmark

**计划日期：** ____ 至 ____

### 本周目标

- [ ] 从“感觉好用”转向可复现、可回归的评测。
- [ ] 同时评估最终状态、轨迹、成本、延迟和安全副作用。
- [ ] 理解代码、模型和人工 Grader 的适用边界，以及标准 Benchmark 与业务评测集的互补关系。

### 知识与阅读

- [x] 阅读 [Demystifying Evals for AI Agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)，整理 Task/Trial/Grader/Trace/Outcome/Harness。
- [x] 区分 Capability Eval 与 Regression Eval。
- [x] 设计确定性检查、Rubric、Pairwise、LLM-as-Judge 和人工抽检。
- [x] 学习多次 Trial、置信区间、非确定性和评测污染问题。
- [x] 定义成功率、工具正确率、引用正确率、步骤数、Token、延迟和成本指标。
- [x] 阅读 BFCL、τ-bench、GAIA 的任务与评分设计；Coding Agent 方向再阅读 SWE-bench，选择与目标项目最相关的一个子集。

### 编码实验

- [x] 建立至少 50 条版本化任务，覆盖正常、边界、失败和对抗输入。
- [x] 为每条任务定义输入、环境、成功条件和至少一个 Grader。
- [x] 实现评测 Harness、并发运行、轨迹保存和结果聚合，并把所选标准 Benchmark 子集适配到统一 Task/Trial/Grader Schema 后实际运行。
- [x] 人工复核至少 20 条 LLM Judge 结果，记录误判类型。
- [x] 建立基线、回归阈值和 CI Eval Gate，运行一次 Prompt/架构变更并验证劣化版本会被阻断。

### 面试与阶段关卡

- [x] 完成至少 10 道 Agent 评测面试题。
- [x] 输出评测集数据卡，说明覆盖范围、缺陷和泄漏风险。
- [x] 无笔记讲清为什么最终答案正确不代表 Agent 行为安全。
- [x] 阶段关卡通过：50+ 任务可重复运行，结果可追溯到完整轨迹。

**复盘：** 最大卡点：____；错误认识：____；需要补强：____

## 第 9 周：Agent 安全、身份治理与 Sandbox

**计划日期：** ____ 至 ____

### 本周目标

- [ ] 把模型、用户、外部内容、工具输出和记忆视为不同信任域。
- [ ] 实施委托授权、多租户隔离、最小权限、显式审批、沙箱和审计。
- [ ] 用攻击测试验证防护，而不是只列安全原则。

### 知识与阅读

- [x] 阅读 [OWASP Top 10 for Agentic Applications 2026](https://genai.owasp.org/resource/owasp-top-10-for-agentic-applications-for-2026/)，建立风险映射。
- [x] 掌握直接/间接 Prompt Injection、Goal Hijack、Tool Misuse 和 Excessive Agency。
- [x] 分析记忆污染、敏感信息泄漏、身份与权限滥用、跨租户访问、供应链和 RCE 风险。
- [x] 设计 OAuth/OIDC 信任边界、用户委托授权、RBAC/ABAC、Scope、Secret 轮换、PII 脱敏、保留/删除、审计字段和人工确认策略。
- [x] 区分输入 Guardrail、工具 Guardrail、输出 Guardrail 和确定性策略执行。
- [x] 比较 [E2B](https://www.e2b.dev/docs)、[OpenSandbox](https://github.com/alibaba/OpenSandbox)、[Agent Sandbox](https://github.com/agent-sandbox/agent-sandbox) 与 [CubeSandbox](https://github.com/tencentcloud/CubeSandbox) 的托管/自建、容器/微虚机、E2B 兼容、冷启动、多租户和部署前提；记录 CubeSandbox 的 x86_64/KVM 要求。

### 编码实验

- [x] 为项目绘制数据流图、身份传播图和威胁模型，标记资产、租户、主体、入口、信任边界和影响。
- [x] 实现工具 Allowlist、参数校验、RBAC/ABAC、最小 Scope、Token 撤销和高风险动作审批；模型上下文与日志不得出现原始凭证。
- [x] 在 E2B 或 OpenSandbox 中任选一个接入代码/Shell 工具，实际限制网络出口、文件系统、Secret、CPU/内存、执行时间和生命周期；无法本地部署时使用托管 E2B。
- [x] 建立至少 15 条攻击测试，覆盖注入、越权、跨租户 RAG/记忆/缓存访问、数据外泄、恶意工具结果和 Sandbox 逃逸尝试。
- [x] 记录每条攻击的预期防护、实际结果和残余风险，并测试 PII 日志脱敏、租户数据导出与彻底删除。

### 面试与验收

- [x] 完成至少 10 道 Agent 安全面试题。
- [x] 回答“为什么仅靠 Prompt 不能构成安全边界”。
- [x] 进行一次 15 分钟威胁建模评审。
- [x] 关键高风险操作均有确定性策略或人工审批。

**复盘：** 最大卡点：____；错误认识：____；需要补强：____

## 第 10 周：生产可靠性、OpenTelemetry 与 LLMOps

**计划日期：** ____ 至 ____

### 本周目标

- [ ] 能处理限流、超时、部分成功、重试风暴和长任务中断。
- [ ] 使用 OpenTelemetry GenAI 语义约定建立端到端 Trace、结构化日志、指标和告警。
- [ ] 保证副作用操作在重试和恢复下不重复执行，并能安全灰度和回滚 Agent 版本。

### 知识与阅读

- [ ] 掌握异步并发、Backpressure、Rate Limit、Queue 和 Worker 模型。
- [ ] 掌握超时预算、指数退避、Jitter、熔断和降级。
- [ ] 区分 At-most-once、At-least-once、幂等和补偿事务。
- [ ] 设计符合 OpenTelemetry GenAI 语义约定的 Model/Agent/Workflow/Retrieval/Tool/Sandbox Trace/Span，并默认关闭或脱敏可能包含 Prompt、Tool Result 和 PII 的内容记录。
- [ ] 设计长任务 Checkpoint、Lease、Heartbeat、取消和恢复。

### 编码实验

- [ ] 为模型和工具设置独立超时、并发上限和全局预算。
- [ ] 对副作用工具加入 Idempotency Key 和结果查询接口。
- [ ] 实现可恢复队列或状态机，支持进程中断后续跑。
- [ ] 注入 429、网络超时、工具半成功、进程退出和重复消息。
- [ ] 使用 OpenTelemetry 导出 Trace/Metric/Log，建立 Dashboard，关联成功率、错误率、延迟、Token、成本、租户和版本。
- [ ] 版本化 Prompt、模型、Tool Schema、MCP Server、记忆策略和评测集；用第 8 周 Eval Gate 模拟一次 Shadow → Canary → 回滚，验证轨迹能定位到完整版本组合。

### 面试与验收

- [ ] 完成至少 10 道生产系统设计题。
- [ ] 定义可用性、P95、单位成功任务成本三项 SLO，完成容量估算、故障演练及一次灰度回滚报告。
- [ ] 无笔记解释“重试为什么可能扩大事故”。
- [ ] 故障注入测试通过，且没有重复副作用。

**复盘：** 最大卡点：____；错误认识：____；需要补强：____

## 第 11 周：推理性能指标与 API Benchmark

**计划日期：** ____ 至 ____

### 本周目标

- [ ] 正确区分 TTFT、TPOT/ITL、端到端延迟、吞吐和 Goodput。
- [ ] 设计可重复的 API Benchmark，而不是比较单次调用。
- [ ] 把性能指标关联到 Agent 用户体验和单位任务成本。

### 知识与阅读

- [ ] 定义 TTFT、TPOT、Tokens/s、Requests/s、P50/P95/P99 和 Goodput。
- [ ] 理解并发、输入长度、输出长度、流式响应和速率限制对结果的影响。
- [ ] 区分客户端排队、网络、服务端排队、Prefill、Decode 和工具时间。
- [ ] 学习 Warm-up、固定数据集、重复 Trial、异常值、置信区间和 Little's Law 的容量含义。
- [ ] 定义单位成功任务成本，而非仅计算单次 API Token 费用。

### 编码实验

- [ ] 编写异步 API Benchmark，记录请求级时间线和 Token 用量。
- [ ] 设计短/长输入、短/长输出以及并发 1/4/16 等工作负载。
- [ ] 每组配置至少重复 5 次，输出 P50/P95/P99。
- [ ] 绘制并发—吞吐、并发—尾延迟和质量—成本图表。
- [ ] 将模型调用指标与 Agent 端到端成功任务指标关联。

### 面试与验收

- [ ] 完成至少 10 道推理性能面试题。
- [ ] 回答“Tokens/s 更高为什么不一定意味着用户体验更好”。
- [ ] 进行 10 分钟性能报告讲解。
- [ ] Benchmark 配置、数据和结果可复现。

**复盘：** 最大卡点：____；错误认识：____；需要补强：____

## 第 12 周：vLLM、SGLang、LMCache 与解耦 Serving

**计划日期：** ____ 至 ____

### 本周目标

- [ ] 理解 Serving 系统如何在显存、吞吐和延迟之间权衡。
- [ ] 能解释 PagedAttention、Continuous Batching、Prefix Cache 和 Chunked Prefill。
- [ ] 能区分推理引擎、KV Cache 层和集群编排层，并比较 vLLM、SGLang、LMCache、TensorRT-LLM、Mooncake、Dynamo 与 llm-d 的定位。

### 知识与阅读

- [ ] 阅读 [PagedAttention](https://arxiv.org/abs/2309.06180)，整理 KV Cache 碎片与分页思想。
- [ ] 阅读 [vLLM 文档](https://docs.vllm.ai/en/latest/) 的 Serving、Benchmark 和缓存相关章节。
- [ ] 阅读 [SGLang 文档](https://docs.sglang.io/) 的 Runtime、Radix Cache 和 Serving 概览。
- [ ] 阅读 [LMCache Quickstart](https://docs.lmcache.ai/getting_started/quickstart.html) 与 [Integration](https://docs.lmcache.ai/developer_guide/integration.html)，说明它如何与 vLLM/SGLang/TensorRT-LLM 配合及如何度量命中与传输。
- [ ] 阅读 [Mooncake](https://arxiv.org/abs/2407.00079) 的 KV-centric 架构，并对照 [NVIDIA Dynamo Disaggregated Serving](https://docs.dynamo.nvidia.com/dynamo/design-docs/disaggregated-serving) 和 [llm-d P/D](https://github.com/llm-d/llm-d/blob/main/guides/pd-disaggregation/README.md)。
- [ ] 阅读 [TensorRT-LLM 文档](https://nvidia.github.io/TensorRT-LLM/)，掌握 Continuous Batching、Prefix Cache、Chunked Prefill、调度、抢占、KV-aware Routing 与 Prefill/Decode 解耦的收益条件。

### 编码与分析

- [ ] 准备一次 4–6 小时 NVIDIA GPU 实验，用相同模型、精度、输入集和并发分别启动 vLLM 与 SGLang；确实无法获得 GPU 时使用两者官方 Benchmark 数据复算，并在报告中明确“未完成实机验证”。
- [ ] 设计低延迟交互、高吞吐批处理、长上下文/多轮 Agent 三种负载，记录 TTFT、TPOT、吞吐、P95/P99、显存和错误率。
- [ ] 在 vLLM 或 SGLang 上接入 LMCache，对共享前缀/多轮会话比较冷缓存、热缓存和无复用基线，记录命中率、TTFT、传输字节与成本。
- [ ] 绘制聚合 Serving 与 Prefill/Decode 解耦时序图，计算不同并发、上下文长度和命中率下的 KV 容量/传输预算。
- [ ] 输出分层选型表：引擎层 vLLM/SGLang/TensorRT-LLM、缓存层 LMCache、编排层 Mooncake/Dynamo/llm-d，并为三种负载给出方案。

### 面试与阶段关卡

- [ ] 完成至少 12 道 Serving 面试题。
- [ ] 白板讲解 PagedAttention、Radix/Prefix Cache、分层 KV Cache 的关系和差异。
- [ ] 回答“何时 P/D 解耦或远端 KV Cache 反而更慢”，明确 KV 传输和网络条件。
- [ ] 阶段关卡通过：能从 Agent 请求特征推导 Serving 需求。

**复盘：** 最大卡点：____；错误认识：____；需要补强：____

## 第 13 周：DeepSeek/Kimi 技术、推理优化与模型路由

**计划日期：** ____ 至 ____

### 本周目标

- [ ] 理解优化技术影响的是计算、显存、带宽、延迟还是质量。
- [ ] 能说明量化和 Speculative Decoding 的收益条件与风险。
- [ ] 能把 DeepSeek/Kimi 的公开技术映射到 Agent 负载，并建立基于任务难度、上下文、风险和预算的模型路由策略。

### 知识与阅读

- [ ] 阅读 [FlashAttention](https://arxiv.org/abs/2205.14135)，解释 IO-aware Tiling。
- [ ] 阅读 [DeepSeek-V3](https://arxiv.org/abs/2412.19437)，解释 MLA、DeepSeekMoE、辅助损失自由负载均衡、MTP 与 FP8 分别解决什么问题。
- [ ] 阅读 [DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1) 与 [DeepSeek-V3.2-Exp](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp)，区分 GRPO/RL/蒸馏的训练机制与 DSA/FlashMLA 的推理机制。
- [ ] 阅读 [Kimi K2](https://github.com/MoonshotAI/Kimi-K2)、[Kimi K2.5](https://github.com/MoonshotAI/Kimi-K2.5)、[Kimi k1.5](https://arxiv.org/abs/2501.12599) 与 [Kimi Linear](https://github.com/MoonshotAI/Kimi-Linear)，理解 MoE/MuonClip、Agentic 能力、长上下文 RL/partial rollout、KDA 与混合线性注意力。
- [ ] 比较 FP16、BF16、FP8、INT8、INT4，以及权重、激活、KV Cache 量化的精度、显存和硬件要求。
- [ ] 掌握 Speculative Decoding 的 Draft/Verify/接受率，并区分 Prefix/Prompt Cache、语义缓存和结果缓存的正确性边界。

### 编码实验

- [ ] 制作“机制—瓶颈—收益—代价—代表实现”表，覆盖 MLA/MoE/MTP/DSA、KDA、FlashAttention、量化、推测解码；再定义快速、标准、高能力三档模型。
- [ ] 建立路由特征：任务类型、上下文长度、风险、预算和历史失败。
- [ ] 实现静态规则路由和模型分类路由两个版本。
- [ ] 使用第 8 周评测集比较固定大模型、固定小模型和动态路由。
- [ ] 输出成功率、P95、Token、费用和路由错误案例。

### 面试与验收

- [ ] 完成至少 12 道 DeepSeek/Kimi、量化、缓存和路由面试题，答案必须从产品名落到机制与约束。
- [ ] 制作质量—延迟—成本决策表。
- [ ] 回答“为什么量化后吞吐提升不等于所有任务成本都下降”。
- [ ] 路由实验可复现，并明确失败回退策略。

**复盘：** 最大卡点：____；错误认识：____；需要补强：____

## 第 14 周：上下文/记忆优化与训练方法边界

**计划日期：** ____ 至 ____

### 本周目标

- [ ] 从端到端任务而不是单次模型调用优化系统。
- [ ] 优先减少无效步骤、重复上下文和串行等待。
- [ ] 能判断问题应由上下文、记忆、缓存、Agent 架构、模型路由还是微调解决。

### 知识与分析

- [ ] 使用 Trace 找出模型、工具、检索、队列和串行依赖的时间占比。
- [ ] 分析上下文膨胀、重复检索、无效反思和多 Agent 重复调用。
- [ ] 评估上下文选择/压缩、记忆检索、Prompt Cache、KV Cache、并行工具和提前终止，明确它们位于不同层。
- [ ] 设计模型分级、失败升级、预算控制和降级响应，并识别何时需要模型能力升级。
- [ ] 比较 Prompt/RAG、SFT/LoRA、DPO、蒸馏和 Agentic RL 的数据、目标、成本、风险与评测要求。
- [ ] 为原始消息、运行状态、长期记忆、检索证据和缓存分别定义 Source of Truth、生命周期、隔离键、版本、删除和污染恢复策略。

### 编码实验

- [ ] 建立优化前基线：成功率、步骤数、Token、TTFT、P95 和单位成功任务成本。
- [ ] 实施上下文去重/压缩与 Mem0 检索阈值/写入策略优化，并分别运行质量、记忆污染、延迟和 Token 回归。
- [ ] 对无依赖工具调用实施受控并行，并加入循环检测或提前终止。
- [ ] 接入第 13 周模型路由、预算和失败升级策略。
- [ ] 针对一组持续失败案例写模型改造 ADR，明确“不训练/LoRA/SFT/DPO/RL”选择；16 周内不要求完成大规模训练。

### 面试与验收

- [ ] 完成至少 10 道 Agent 性能设计题。
- [ ] 输出优化前后对比报告、至少一个负面结果和一份模型改造决策记录。
- [ ] 进行 15 分钟性能评审，说明每项优化的证据。
- [ ] 所有质量与安全回归阈值通过。

**复盘：** 最大卡点：____；错误认识：____；需要补强：____

## 第 15 周：资深综合项目——可上线的深度研究 Agent

**计划日期：** ____ 至 ____

### 本周目标

- [ ] 把前 14 周知识整合成一个可评测、可恢复、可审计的系统。
- [ ] 保持核心架构简单，只有实验证明需要时才增加多 Agent。
- [ ] 形成技术、业务、治理和发布四类证据，而不只是功能演示。

### 设计与范围

- [ ] 明确用户角色、现有人工/非 Agent 流程和业务任务：检索多来源证据、核验冲突、生成带引用报告；定义耗时、一次通过率、人工升级率、单位成功任务成本和错误成本基线。
- [ ] 定义非目标：不自动发布、不执行高风险外部写操作、不追求开放域全自动自治。
- [ ] 绘制组件图、状态图、数据流图和信任边界图，并记录关键 ADR。
- [ ] 定义工具/状态 Schema、租户/用户/授权上下文、错误类型、预算、终止条件、SLO 和容量假设；用 ADR 说明选用/不选 LangGraph/Deep Agents、Mem0、Sandbox 与本地 Serving 的原因。
- [ ] 定义 50+ 评测任务、15+ 安全攻击和故障场景。

### 实现

- [ ] 实现查询分解、检索、去重、重排和证据集合。
- [ ] 实现来源引用、冲突检测、无证据拒答和报告生成。
- [ ] 实现 Checkpoint、中断恢复、人工审批和幂等执行；状态、RAG、长期记忆和缓存按 `tenant_id/user_id` 隔离，长期记忆在 Mem0 与自建基线中二选一。
- [ ] 实现 OpenTelemetry Trace、指标、费用统计和可查询运行记录；若使用本地模型，提供 vLLM/SGLang OpenAI-compatible Serving 适配与回退。
- [ ] 将不可信代码/Shell 操作路由到第 9 周选定的 Sandbox，并实现单元、集成和最小端到端测试。

### 验收

- [ ] 使用第 8 周 Harness 跑通完整评测并保存基线。
- [ ] 与人工/非 Agent 基线比较任务时间、成功率、人工升级率和单位成功任务成本，给出“上线/有限上线/不上线”结论。
- [ ] 使用第 9–10 周测试集执行安全和故障测试。
- [ ] 完成至少 3 份架构决策记录，解释容量、成本、风险及未采用的复杂方案。
- [ ] 系统可通过 Docker Compose 或等价容器方案从干净环境启动；以两个实例验证共享状态、横向扩展和滚动升级后完成演示任务。

**复盘：** 最大卡点：____；错误认识：____；需要补强：____

## 第 16 周：项目强化、作品集与面试

**计划日期：** ____ 至 ____

### 本周目标

- [ ] 将项目从“能跑”提升为“有证据证明可靠”。
- [ ] 把技术深度转化为清晰的面试表达。
- [ ] 完成作品集、演示和知识缺口清单。

### 强化与评测

- [ ] 对全部评测任务运行至少 3 个 Trial，并汇总均值与波动。
- [ ] 完成单/多 Agent、上下文策略、模型路由和关键优化的消融。
- [ ] 完成攻击集、故障集和中断恢复测试。
- [ ] 记录已知限制、失败案例和残余风险；定义用户反馈分类、人工升级原因、Online KPI、抽样复核和失败样本回流规则，用模拟反馈跑通闭环并检查 Offline Eval 与 Online KPI 是否同向。
- [ ] 确认日志和演示数据不含 API Key、个人信息或敏感内容。

### 作品集

- [ ] 完成 README：问题、架构、快速开始、评测、安全、性能和限制。
- [ ] 完成架构图、状态图、核心轨迹示例和指标图表。
- [ ] 完成一份技术与业务报告，包含基线、收益、决策、消融、失败、身份治理、发布回滚和改进。
- [ ] 准备 5–10 分钟可重复演示和备用录屏方案。
- [ ] 准备 30 分钟资深级系统设计评审：用户价值、自动化边界、目标、约束、SLO、容量、权衡、风险和演进。
- [ ] 完成一页“产品 → 核心机制 → 适用条件 → 证据”技术雷达，覆盖 DeepSeek/Kimi、vLLM/SGLang/LMCache、Agent Harness、Mem0 与 Sandbox，并记录下一次季度复核日期。

### 面试与最终验收

- [ ] 完成本文件 120 道题的第一轮口头回答。
- [ ] 对答不完整的问题建立“补知识—重答—验证”清单。
- [ ] 完成至少两次模拟面试：一次 Agent 架构、一次推理与性能。
- [ ] 最终验收通过：项目可运行、CI Eval Gate 可阻断回归、灰度回滚可演示、测试可复现、资料可公开、核心原理可无笔记讲解。

**复盘：** 最大卡点：____；错误认识：____；下一阶段：____

---

## 综合项目验收标准

- [ ] 任务输入、环境、成功条件和失败条件均有明确 Schema。
- [ ] 每次模型调用、工具调用、Handoff 和 Guardrail 均可追踪。
- [ ] 至少 50 条功能评测、15 条安全评测和 10 条故障测试。
- [ ] 至少运行一个 BFCL、τ-bench 或 GAIA 标准 Benchmark 子集，并解释其与业务评测集的覆盖差异。
- [ ] 评测包含最终状态、轨迹、质量、延迟、Token 和成本。
- [ ] 外部内容默认不可信，高风险动作采用最小权限和人工审批。
- [ ] OAuth/OIDC 委托授权、RBAC/ABAC、最小 Scope 和 Token 撤销可测试；原始凭证不进入模型上下文。
- [ ] 双租户 RAG、状态、长期记忆、缓存、日志和数据导出/删除测试通过，无跨租户泄漏。
- [ ] 不可信代码/Shell 在独立 Sandbox 中运行，网络、文件、Secret、资源、超时和销毁策略均有测试。
- [ ] Checkpoint、长期记忆、业务事实、Prompt Cache 与 KV Cache 边界清晰；记忆支持隔离、更新、删除和污染恢复。
- [ ] 支持超时、重试、幂等、中断、恢复和预算终止。
- [ ] OpenTelemetry Trace 能关联任务、租户、模型、Prompt、Tool Schema、MCP、记忆策略和评测集版本，敏感内容默认不采集。
- [ ] CI Eval Gate、Shadow/Canary 和自动/人工回滚演练通过，可复现一次劣化版本被阻断或回退。
- [ ] 与人工/非 Agent 基线比较成功率、耗时、人工升级率、错误成本和单位成功任务成本，并给出上线结论。
- [ ] 至少完成三个消融：架构、上下文、模型/路由。
- [ ] README、架构图、威胁模型、评测报告和性能报告完整。

## 120 道面试问题清单

### A. Transformer 与生成（1–15）

- [ ] 1. Scaled Dot-Product Attention 为什么要除以 `sqrt(d_k)`？
- [ ] 2. Q、K、V 分别承担什么作用，它们的形状如何变化？
- [ ] 3. Causal Mask 与 Padding Mask 有什么区别？
- [ ] 4. Multi-head Attention 为什么不等价于一个更宽的单头 Attention？
- [ ] 5. MHA、MQA、GQA 对 KV Cache 和质量有什么影响？
- [ ] 6. RoPE 如何编码相对位置信息，长上下文外推有什么问题？
- [ ] 7. Pre-Norm 与 Post-Norm 的差异是什么？
- [ ] 8. Prefill 和 Decode 的计算特征为什么不同？
- [ ] 9. KV Cache 保存什么，显存如何估算？
- [ ] 10. Temperature、Top-k、Top-p 如何共同影响采样？
- [ ] 11. Tokenizer 为什么会影响成本、延迟和多语言表现？
- [ ] 12. 为什么长上下文会同时影响 Attention 计算和 KV Cache，MLA、稀疏/线性注意力分别改变了什么？
- [ ] 13. EOS、Stop Sequence 和业务终止条件有什么区别？
- [ ] 14. Logits、概率和生成 Token 之间经过哪些步骤？
- [ ] 15. 为什么模型参数量不是推理速度的唯一决定因素？

### B. Agent 基础与工具（16–30）

- [ ] 16. 什么是 Agent，怎样与 Workflow、Chatbot、RAG 区分？
- [ ] 17. 一个最小 Agent Loop 需要哪些状态和终止条件？
- [ ] 18. ReAct 的核心思想、优势和典型失败是什么？
- [ ] 19. 结构化输出与工具调用有什么关系和差异？
- [ ] 20. 为什么 Tool Schema 越宽泛，可靠性和安全风险通常越高？
- [ ] 21. 如何设计工具错误，使模型能够恢复而不是反复重试？
- [ ] 22. 哪些工具可以安全重试，哪些必须使用幂等键？
- [ ] 23. 如何处理模型调用不存在的工具或非法参数？
- [ ] 24. 最大步数、Token 预算、费用预算和时间预算如何协作？
- [ ] 25. 工具结果为什么必须被视为不可信输入？
- [ ] 26. 如何设计需要人工审批的工具调用？
- [ ] 27. Function Calling 与 MCP 的关系是什么？
- [ ] 28. MCP Host、Client、Server 分别负责什么？
- [ ] 29. MCP Tools、Resources、Prompts 的控制方和场景有何不同？
- [ ] 30. OAuth/OIDC 委托授权中用户、Agent、客户端和工具服务分别是什么主体，Scope 与 Token 如何安全传递？

### C. 上下文、RAG 与记忆（31–45）

- [ ] 31. 为什么上下文工程不等于把更多信息塞进 Prompt？
- [ ] 32. Chunk Size 与 Overlap 如何影响召回和噪声？
- [ ] 33. 稀疏、稠密和混合检索的适用场景是什么？
- [ ] 34. Reranker 解决什么问题，会引入什么成本？
- [ ] 35. Recall@K、MRR、忠实度和引用正确率分别衡量什么？
- [ ] 36. 如何处理无答案、冲突/过期证据、文档 ACL、增量更新和删除后的索引残留？
- [ ] 37. Conversation History、Checkpoint、长期记忆、业务状态、Prompt Cache 和 KV Cache 为何不能混为一谈？
- [ ] 38. 什么信息应该进入长期记忆，什么信息不应该？
- [ ] 39. 摘要压缩会丢失什么，如何评测其影响？
- [ ] 40. Mem0 与自建向量记忆各有什么优缺点，如何评测抽取、召回、更新和删除？
- [ ] 41. 如何防止记忆污染和错误事实长期传播？
- [ ] 42. Context Window 足够大时为什么仍需要检索和压缩？
- [ ] 43. 如何进行 Context Budget 分配？
- [ ] 44. LangGraph Store、Mem0 与 Deep Agents 文件系统上下文分别适合保存什么？
- [ ] 45. 如何让答案中的引用可以被确定性验证？

### D. 架构、规划与多 Agent（46–60）

- [ ] 46. 什么情况下确定性 Workflow 优于 Agent？
- [ ] 47. Routing、Parallelization、Orchestrator-Workers 的差异是什么？
- [ ] 48. Plan-and-Execute 与 ReAct 的主要取舍是什么？
- [ ] 49. 如何判断计划已经失效并触发重规划？
- [ ] 50. 自我反思为什么可能增加成本却不提升正确率？
- [ ] 51. Evaluator-Optimizer 适合什么任务，终止条件怎么设计？
- [ ] 52. Manager、Handoff、Agent-as-Tool、Claude Subagent 和 Agent Team 的控制权有何不同？
- [ ] 53. 多 Agent 会引入哪些上下文、调度和错误传播成本？
- [ ] 54. 如何证明多 Agent 比单 Agent 更好？
- [ ] 55. 子任务间有依赖时如何安排并行？
- [ ] 56. 两个 Agent 结论冲突时应如何仲裁？
- [ ] 57. 如何避免循环委派和重复工作？
- [ ] 58. 如何给长任务定义可验证的阶段完成条件？
- [ ] 59. 状态机和事件驱动架构在 Agent 中各有什么价值？
- [ ] 60. A2A 与 MCP 分别解决什么互操作问题，为什么不能互相替代？

### E. 框架、评测与可观测性（61–75）

- [ ] 61. LangGraph 的 State、Node、Edge 和 Reducer 各负责什么？
- [ ] 62. Checkpoint 如何支持 HITL、恢复和 Time Travel？
- [ ] 63. Interrupt 与普通异常有什么区别？
- [ ] 64. OpenAI Agents SDK 的 Runner 如何组织 Agent、Tool、Handoff 和 Guardrail？
- [ ] 65. LangGraph、Deep Agents、Claude Agent SDK、Hermes Agent、Kimi Agent SDK 应按哪些维度选型？
- [ ] 66. OpenTelemetry GenAI 中 Model、Agent、Workflow、Retrieval 和 Tool Span 如何关联，哪些 Prompt/PII 内容默认不应采集？
- [ ] 67. Task、Trial、Grader、Trace、Outcome、Harness 如何区分？
- [ ] 68. Capability Eval 与 Regression Eval 有什么不同？
- [ ] 69. 为什么 Agent 评测通常要运行多次 Trial？
- [ ] 70. Code-based、Model-based、Human Grader 如何组合？
- [ ] 71. LLM-as-Judge 有哪些偏差，如何校准？
- [ ] 72. 最终答案正确时为什么轨迹仍可能判定失败？
- [ ] 73. 如何设计不会过度拟合实现细节的 Grader？
- [ ] 74. BFCL、τ-bench、GAIA 和业务评测集分别覆盖什么，如何防止数据泄漏、Benchmark 污染和“刷榜不解决业务”？
- [ ] 75. PydanticAI、AutoGen、Google ADK 与 LangGraph 的核心抽象分别适合什么团队和系统边界？

### F. 安全与可靠性（76–90）

- [ ] 76. 直接与间接 Prompt Injection 有什么区别？
- [ ] 77. 为什么 Prompt 不能作为权限边界？
- [ ] 78. Excessive Agency 的功能、权限和自主性风险分别是什么？
- [ ] 79. 如何组合 OAuth/OIDC、RBAC/ABAC、Scope、用户委托凭证和 Token 撤销实现 Agent 最小权限？
- [ ] 80. 工具输出中包含恶意指令时系统应如何处理？
- [ ] 81. 如何防止 Secret/PII 出现在上下文、日志和模型输出中，并验证 RAG、记忆、缓存和日志的多租户隔离？
- [ ] 82. 容器与微虚机 Sandbox 的隔离、冷启动、密度和运维取舍是什么？
- [ ] 83. E2B、OpenSandbox、Agent Sandbox、CubeSandbox 如何选型，E2B 兼容解决了什么、没有解决什么？
- [ ] 84. 如何测试记忆污染和 Goal Hijack？
- [ ] 85. 幂等、重试和补偿事务之间是什么关系？
- [ ] 86. 指数退避为什么需要 Jitter？
- [ ] 87. 重试风暴如何产生，怎样防止？
- [ ] 88. 如何处理中断时已经部分成功的工具调用？
- [ ] 89. Agent 的审计日志至少应包含哪些字段？
- [ ] 90. 如何设计故障注入测试验证恢复能力？

### G. 推理 Serving、优化与训练边界（91–105）

- [ ] 91. TTFT、TPOT、端到端延迟和吞吐分别反映什么？
- [ ] 92. Goodput 与吞吐有什么不同？
- [ ] 93. P50、P95、P99 为什么比平均延迟更重要？
- [ ] 94. Continuous Batching 如何提升 GPU 利用率，又为何可能恶化尾延迟？
- [ ] 95. PagedAttention 与 SGLang Radix/Prefix Cache 分别解决 KV Cache 的什么问题？
- [ ] 96. LMCache 与引擎原生 Prefix Cache 有何差异，命中率、传输和一致性如何度量？
- [ ] 97. Chunked Prefill 与 Prefill/Decode 解耦如何影响 Decode 干扰、TTFT、TPOT 和集群成本？
- [ ] 98. FlashAttention 与 FlashMLA 的 IO-aware 优化对象有何不同？
- [ ] 99. FP16、BF16、FP8、INT8、INT4 以及权重/激活/KV 量化的主要取舍是什么？
- [ ] 100. Speculative Decoding 为什么能保持目标分布，接受率和 Draft 成本如何影响收益？
- [ ] 101. DeepSeek-V3 的 MLA、DeepSeekMoE、MTP、FP8 与负载均衡分别解决什么问题？
- [ ] 102. DeepSeek-R1 的 GRPO/RL/蒸馏与 V3.2 的 DSA 分属训练和推理的哪一层？
- [ ] 103. Kimi K2/K2.5、Kimi k1.5、Kimi Linear 与 Mooncake 分别代表哪些模型、训练和 Serving 技术？
- [ ] 104. vLLM/SGLang、LMCache、TensorRT-LLM、Mooncake/Dynamo/llm-d 为什么属于不同系统层？
- [ ] 105. 如何判断问题应由 Prompt/RAG、上下文/记忆、Agent 架构、Serving/路由还是微调/RL 解决？

### H. 系统设计与项目追问（106–120）

- [ ] 106. 设计一个可核验引用的深度研究 Agent。
- [ ] 107. 如何让研究 Agent 在来源冲突时给出透明结论？
- [ ] 108. 如何用非 Agent 基线、成功率、耗时、人工升级率、错误成本和单位成功任务成本定义项目价值？
- [ ] 109. 如何将一次研究任务从分钟级扩展到小时级并支持恢复？
- [ ] 110. 如何控制长任务的 Token 和费用预算？
- [ ] 111. 如何设计模型路由并避免错误路由导致质量下降？
- [ ] 112. 如何让独立工具并行，同时保持状态一致？
- [ ] 113. 如何处理 API 供应商限流或不可用？
- [ ] 114. 如何版本化 Prompt/模型/Tool Schema/MCP/记忆策略/评测集，并用 CI Eval Gate、Shadow、Canary 和回滚安全发布？
- [ ] 115. 如何证明一次上下文压缩没有损害关键任务？
- [ ] 116. 如何衡量单位成功任务成本？
- [ ] 117. 项目中最重要的一次失败是什么，你如何定位和修复？
- [ ] 118. 哪个复杂架构你最终没有采用，为什么？
- [ ] 119. 如果流量扩大 100 倍，系统首先出现什么瓶颈？
- [ ] 120. 如何把用户反馈、人工升级原因、线上指标和失败轨迹回流为下一轮评测与改进，同时避免反馈污染？

## 推荐资料与源码入口

### 基础论文

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- [DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1)
- [DeepSeek-V3.2-Exp](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp)
- [DeepSeek FlashMLA](https://github.com/deepseek-ai/FlashMLA)
- [Kimi K2](https://github.com/MoonshotAI/Kimi-K2)
- [Kimi K2.5](https://github.com/MoonshotAI/Kimi-K2.5)
- [Kimi k1.5: Scaling Reinforcement Learning with LLMs](https://arxiv.org/abs/2501.12599)
- [Kimi Linear](https://github.com/MoonshotAI/Kimi-Linear)
- [Mooncake: KVCache-centric Disaggregated Serving](https://arxiv.org/abs/2407.00079)

### Agent、协议与评测

- [Anthropic: Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents)
- [Anthropic: Demystifying Evals for AI Agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)
- [Model Context Protocol: Architecture Overview](https://modelcontextprotocol.io/docs/learn/architecture)
- [LangGraph: Persistence](https://docs.langchain.com/oss/python/langgraph/persistence)
- [LangGraph: Human-in-the-loop](https://docs.langchain.com/oss/python/langchain/human-in-the-loop)
- [OpenAI Agents SDK: Agents](https://openai.github.io/openai-agents-python/agents/)
- [OpenAI Agents SDK: Tracing](https://openai.github.io/openai-agents-python/tracing/)
- [Deep Agents Overview](https://docs.langchain.com/oss/python/deepagents/overview)
- [Claude Code Subagents](https://code.claude.com/docs/en/sub-agents)
- [Claude Agent SDK Subagents](https://code.claude.com/docs/en/agent-sdk/subagents)
- [Hermes Agent](https://github.com/nousresearch/hermes-agent)
- [Kimi Agent SDK](https://github.com/MoonshotAI/kimi-agent-sdk)
- [PydanticAI](https://ai.pydantic.dev/)
- [Microsoft AutoGen](https://microsoft.github.io/autogen/stable/index.html)
- [Google Agent Development Kit](https://google.github.io/adk-docs/)
- [Agent2Agent Protocol Specification](https://google-a2a.github.io/A2A/specification/)
- [Anthropic: Effective Context Engineering for AI Agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
- [LangGraph Memory](https://docs.langchain.com/oss/python/concepts/memory)
- [Mem0 Documentation](https://docs.mem0.ai/introduction)

### 身份、标准评测与可观测性

- [OAuth 2.0 Authorization Framework — RFC 6749](https://www.rfc-editor.org/rfc/rfc6749)
- [OpenID Connect Core 1.0](https://openid.net/specs/openid-connect-core-1_0.html)
- [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html)
- [τ-bench](https://taubench.com/)
- [GAIA: A Benchmark for General AI Assistants](https://arxiv.org/abs/2311.12983)
- [SWE-bench](https://www.swebench.com/)
- [OpenTelemetry GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/registry/attributes/gen-ai/)

### Serving 与安全

- [vLLM Documentation](https://docs.vllm.ai/en/latest/)
- [SGLang Documentation](https://docs.sglang.io/)
- [LMCache Documentation](https://docs.lmcache.ai/getting_started/quickstart.html)
- [TensorRT-LLM Documentation](https://nvidia.github.io/TensorRT-LLM/)
- [NVIDIA Dynamo: Disaggregated Serving](https://docs.dynamo.nvidia.com/dynamo/design-docs/disaggregated-serving)
- [llm-d](https://github.com/llm-d/llm-d)
- [E2B Documentation](https://www.e2b.dev/docs)
- [Alibaba OpenSandbox](https://github.com/alibaba/OpenSandbox)
- [Agent Sandbox](https://github.com/agent-sandbox/agent-sandbox)
- [Tencent Cloud CubeSandbox](https://github.com/tencentcloud/CubeSandbox)
- [OWASP Top 10 for Agentic Applications 2026](https://genai.owasp.org/resource/owasp-top-10-for-agentic-applications-for-2026/)

## 总进度

| 阶段 | 周次 | 状态 | 核心关卡 |
|---|---:|---|---|
| 基础与工具 | 1–4 | [ ] 未完成 | Transformer、原生 Agent、MCP 委托授权、ACL-aware RAG 与记忆隔离 |
| 架构与评测 | 5–8 | [ ] 未完成 | 架构消融、框架对照、50+ 业务评测、标准 Benchmark 与 CI Gate |
| 安全、生产与推理 | 9–12 | [ ] 未完成 | 双租户安全、Sandbox、OpenTelemetry、灰度回滚、SLO 与 Serving |
| 优化与求职 | 13–16 | [ ] 未完成 | 路由、业务基线、线上反馈闭环、资深项目、作品集与模拟面试 |

**最终完成日期：** ____  
**最强能力：** ____  
**仍需补强：** ____  
**下一阶段目标：** ____
