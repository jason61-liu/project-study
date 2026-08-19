# Kimi K2 / K2.5 / k1.5 / Linear：MoE/MuonClip、Agentic 能力、长上下文 RL 与 KDA 混合线性注意力

> 目标：把 Moonshot 的四个模型各归到一条技术主线——K2 的 MoE+MuonClip 是「把 1T 模型训稳」、K2.5 的 PARL 是「把 agent 编排学进模型」、k1.5 的 partial rollout 是「把长上下文 RL 做便宜」、Linear 的 KDA 是「把注意力从二次降成线性」。四个模型解决四个不同问题。

---

## 1. 阅读前术语表

| 术语 | 说明 |
|---|---|
| MoE | Mixture-of-Experts：每个 token 只激活部分专家 |
| Muon | 一种基于矩阵正交化的优化器，比 AdamW 更省 token |
| MuonClip | Muon + 权重裁剪（QK-Clip）的组合 |
| QK-Clip | 裁剪 Q/K 投影权重以约束注意力 logit 的裁剪技巧 |
| Agentic | 模型能自主使用工具、规划、解决多步任务 |
| PARL | Parallel-Agent RL：训练编排器把任务拆成可并行子任务的 RL |
| 长上下文 RL | 在极长上下文（如 128k）上做强化学习 |
| Partial Rollout | 分段重放轨迹，复用历史片段、避免从头重算 |
| 线性注意力 | 用固定大小状态 + 递归更新替代 O(N²) 的 softmax 注意力 |
| KDA | Kimi Delta Attention：Kimi 的线性注意力机制 |
| 混合线性注意力 | 线性注意力层与全局注意力层交错，兼顾效率与召回 |

---

## 2. Kimi K2：MoE + MuonClip——把 1T 模型训稳

Kimi K2（2025-07）是一个 **1T 总参数、每 token 32B 激活**的 MoE 模型（384 专家 top-8 + 1 共享专家，61 层，128K 上下文，MLA 注意力），15.5T token 训练、**零训练不稳定**。两个关键点：

- **MoE（细粒度专家）**：和 DeepSeek-V3（文档 02）同路——总容量大、每 token 成本低。K2 的差异在**共享专家只有 1 个**，且专家规模更大（384 个）。
- **MuonClip 优化器**：大规模 MoE 训练长期被 AdamW 的高 token 消耗拖累。Muon 优化器靠矩阵正交化更新更省 token，但**直接放大到 1T 会不稳定**。MuonClip 的解法是加 **QK-Clip**：裁剪 Q/K 投影权重，把注意力 logit 的绝对值限制在一定范围，防止训练后期 logit 爆炸。这让 Muon 能稳定撑住 1T 规模。

**定位**：K2 是「基础设施能力」——先把 1T MoE 稳定训出来，为后续 agentic 能力（K2.5）和思考能力（k1.5 的方法论）打底。SWE-bench Verified 65.8%、LiveCodeBench v6 53.7%。

---

## 3. Kimi K2.5：PARL 与 Agentic 能力

Kimi K2.5（2026-01）在 K2 基础上加了三样：**视觉（MoonViT 400M 编码器）**、**Thinking 模式**、以及最关键的 **Agent Swarm**。仍是 1T / 32B 激活 / 384 专家，但上下文扩到 256K。

Agentic 能力的核心是 **PARL（Parallel-Agent Reinforcement Learning）**：训练一个**可训练的编排器（orchestrator）**，把用户任务动态拆成可并行的子任务，交给动态创建的子 agent（「研究员」「事实核查员」等角色）分头执行。模型可自主编排**多达 100 个子 agent、1500 次工具调用**，实测比单 agent 快约 4.5×。K2.5 还提供四种模式：Instant（快）、Thinking（深度推理）、Agent、Agent Swarm（beta）。

**定位**：K2.5 回答的是「怎么把『会拆任务、会并行调度工具』这件事**学进模型本身**」，而不是靠外部框架硬编码。SWE-bench Verified 76.8%、BrowseComp 74.9%、AIME 2025 96.1%。

---

## 4. Kimi k1.5：长上下文 RL 与 Partial Rollout

k1.5 的核心主张是：**把 RL 的上下文长度当作一个可扩展维度**，而 partial rollout 是让这种扩展「算得起」的关键工程。

### 4.1 长上下文 RL

把 RL 上下文拉到 **128K**，且性能随上下文变长持续提升——更长的上下文等于更多「思考步」，涌现出规划、反思、自我纠正，**无需 MCTS、价值函数或过程奖励模型**这些复杂手段。

### 4.2 Partial Rollout：长轨迹别从头重算

长轨迹 RL 的瓶颈是「每个新采样都要从头生成整条轨迹」。partial rollout 的解法：

- 把长轨迹**切成段**，存进 **replay buffer**；
- 新轨迹采样时**复用前几轮的旧片段**（iter n−m … n−1），只生成新的尾部，避免从头重算；
- 顺带做**重复检测**：识别重复片段、提前终止并施加惩罚，抑制冗余生成。

### 4.3 其他配套

策略优化用 **online mirror descent 的变体**（而非 PPO/GRPO），加采样策略、长度惩罚、数据配方优化；训练/推理用 **Megatron（训练）+ vLLM（推理）混合部署共享 GPU**，分时切换榨干利用率。

**定位**：k1.5 是「思考模型」的方法论源头——长上下文 RL + partial rollout 让长 CoT 训练可行。Long-CoT k1.5 对齐 o1（AIME 77.5、MATH-500 96.2、Codeforces 94 百分位）。

---

## 5. Kimi Linear：KDA 与混合线性注意力

Kimi Linear 解决的是一个**根本的注意力成本问题**：softmax 注意力是 O(N²)（算力）和 O(N)（KV Cache 随 N 线性涨），长上下文下又贵又占显存。

### 5.1 KDA：Kimi Delta Attention

KDA 是一种**线性注意力**，它在 Gated DeltaNet 基础上加了**细粒度、逐通道的门控**。核心变化：**用「固定大小的状态 + 可学习门控」的递归更新替代二次复杂度的 softmax 注意力**。KV Cache 不再是「随序列增长的张量」，而是压缩进一个固定大小的状态，因此 KV Cache 大幅缩小。

### 5.2 混合架构：3:1 交错

纯线性注意力对「精确召回长距离信息」能力弱（状态有容量上限）。Kimi Linear 用**混合（hybrid）**：KDA 层与全局注意力（MLA）层按 **3:1 交错**——大部分层用便宜的 KDA，少数层用精确的全局 MLA 兜底长程召回。

- **收益**：KV Cache 最多省 **75%**，1M token 上下文下解码吞吐最高 **6×**（TPOT 约 6.3× MLA）；
- **实现**：硬件感知设计——chunking、WY 表示、DPLR（对角 + 低秩）转移矩阵，最大化 Tensor Core 利用率；开源 KDA 内核（FLA）与 vLLM 集成；
- **模型**：`Kimi-Linear-48B-A3B-Instruct`（48B 总、3B 激活、1M 上下文）。

### 5.3 边界（重要）

线性注意力是**近似**——它压缩了 KV 状态，长程精确召回不如全注意力。值得注意的信号：Moonshot 在 1T 的 K2 Thinking 里**没有用 KDA**，MiniMax 的 M2 也回退到全注意力——**高效注意力在超大模型上的规模化仍存不确定性**。这跟文档 01 的 FlashAttention（精确、不省算力只省 IO）形成对照：KDA 是「近似 + 省算力省显存」，DSA（文档 03）是「稀疏近似 + 省算力」，三者的精确性/收益各不相同。

---

## 6. 四个模型的对照

| 模型 | 解决什么问题 | 核心机制 | 关键代价 |
|---|---|---|---|
| **K2** | 把 1T MoE 稳定训出来 | MoE（384 专家 top-8）+ MuonClip/QK-Clip | 优化器不稳定风险、工程复杂度 |
| **K2.5** | 把 agent 编排学进模型 | PARL + 视觉 + Thinking | RL 基础设施重写、子 agent 协调成本 |
| **k1.5** | 把长上下文 RL 做便宜 | 128K RL + partial rollout | 轨迹分段/重放的工程复杂度 |
| **Linear** | 把注意力从 O(N²) 降成 O(N) | KDA 线性注意力 + 3:1 混合 | 近似召回、超大模型规模化存疑 |

一条线串起来：**K2 训稳大模型 → k1.5 教会思考 → K2.5 教会编排 → Linear 让长上下文跑得动**。它们分别对应「规模」「智能」「agent」「效率」四个正交维度。

---

## 7. 本文结论

- **MuonClip** = Muon 优化器 + QK-Clip 权重裁剪，解决 1T MoE 训练的稳定性与 token 效率。
- **Agentic**：K2.5 用 PARL 训练编排器，把「拆任务 + 并行子 agent」变成模型内置能力，而非外部框架。
- **长上下文 RL + partial rollout**：k1.5 把上下文长度当扩展维度，用分段重放让长轨迹 RL 算得起。
- **KDA + 混合线性注意力**：Linear 用固定状态 + 门控的线性注意力替代 O(N²) softmax，3:1 混合保留长程召回，KV 省 75%、解码快 6×；但线性注意力是近似，超大模型上规模化仍有不确定性。
- 把这四个模型和 DeepSeek 系（文档 02/03）放到一起，可以形成一张「机制—瓶颈—收益—代价—代表实现」的大表——这正是第 13 周编码实验要做的第一件事。

---

## 参考资料

- [Kimi K2（GitHub）](https://github.com/MoonshotAI/Kimi-K2)
- [Kimi K2 技术报告（arXiv:2507.20534）](https://arxiv.org/abs/2507.20534)
- [Kimi K2.5（GitHub）](https://github.com/MoonshotAI/Kimi-K2.5)
- [Kimi k1.5（arXiv:2501.12599）](https://arxiv.org/abs/2501.12599)
- [Kimi Linear（GitHub）](https://github.com/MoonshotAI/Kimi-Linear)
- [KDA 内核（FLA）](https://github.com/fla-org/flash-linear-attention)
