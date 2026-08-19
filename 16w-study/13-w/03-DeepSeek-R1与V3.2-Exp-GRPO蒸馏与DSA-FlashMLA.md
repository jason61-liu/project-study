# DeepSeek-R1 与 DeepSeek-V3.2-Exp：GRPO/RL/蒸馏的训练机制 与 DSA/FlashMLA 的推理机制

> 目标：把「训练侧如何让模型学会推理」和「推理侧如何让推理更快」分开讲——前者是 R1 的 GRPO / 纯 RL / 蒸馏，后者是 V3.2-Exp 的 DSA 稀疏注意力 + FlashMLA 内核。两条线别混：一个改模型行为，一个改计算路径。

---

## 1. 阅读前术语表

| 术语 | 说明 |
|---|---|
| RL | Reinforcement Learning：用奖励信号调模型，而非监督标签 |
| GRPO | Group Relative Policy Optimization：无需 critic 的 PPO 变体 |
| PPO | Proximal Policy Optimization：需 value 网络估计 advantage 的 RL 算法 |
| R1-Zero | 纯 RL（无 SFT）训练出的推理模型，行为涌现但可读性差 |
| 蒸馏（Distillation） | 用大模型生成的数据去微调小模型 |
| DSA | DeepSeek Sparse Attention：token 级稀疏注意力 |
| FlashMLA | 针对 MLA 的优化 attention 内核（dense + sparse） |
| Lightning Indexer | DSA 里用来选「每个 query 关注哪些 top token」的索引器 |
| 冷启动 SFT | 用少量高质量 CoT 数据先 SFT，稳定后续 RL |

---

## 2. 两条线的总览

**训练线（R1）**回答：怎么让 base 模型（V3）涌现出推理、自我验证、反思？答案是用 RL 直接激励推理行为，再用蒸馏把它压进小模型。

**推理线（V3.2-Exp）**回答：长上下文推理时，注意力算力/带宽太贵怎么办？答案是把 MLA 的注意力做成**稀疏**（DSA，每个 query 只关注 top-k token），并用 FlashMLA 内核把稀疏/稠密 MLA 跑满硬件。

下面分开讲。

---

## 3. 训练机制：GRPO、纯 RL 与蒸馏

### 3.1 GRPO：为什么不用 PPO 的 critic

PPO 需要训练一个**价值网络（critic）**去估计每个 token 的 advantage，这个 critic 和策略网络同规模，**显存翻倍**。GRPO 的思路是**用「组内相对比较」取代 critic**：

1. 对每个问题采样一组 G 个回答（生产用 G≈64）；
2. 每个回答得到奖励 `r_i`；
3. **advantage = (r_i − 组内均值) / 组内标准差**——不需要单独的价值网络；
4. 更新时裁剪概率比（clip），并加 KL 惩罚把新策略拉在参考模型附近，防跑偏。

**代价**：advantage 是**序列级**的（没有逐 token 的功劳分配），组内全对或全错时梯度为零，且天然偏向更长的回答——这些都是 GRPO 的已知局限，后来的 DAPO 等用非对称裁剪、动态采样、token 级归一化、长度感知奖励来补。

### 3.2 R1-Zero：纯 RL 能涌现什么

R1-Zero 直接在 V3 base 上**只做 RL、不做任何 SFT**，用可验证的规则奖励（答案对不对 + 格式对不对），结果自发涌现了 chain-of-thought、自我验证、自我纠正、回溯，甚至「aha moment」。代价是**可读性差**：语言混杂、重复。

### 3.3 R1 的四阶段（修复 R1-Zero 的可读性）

1. **冷启动 SFT**：几千条人工校对的 long-CoT 数据，先把格式和语言稳定住；
2. **推理导向 RL**：GRPO，额外加「语言一致性奖励」（目标语言占比）减少中英混杂；
3. **拒绝采样 + SFT**：用 RL 后的 checkpoint 生成 60 万推理 + 20 万非推理 = 80 万样本，SFT 两轮；
4. **全场景 RL**：规则奖励（推理）+ 模型奖励（有用性、无害性）做对齐。

### 3.4 蒸馏：把推理压进小模型

用 R1 生成的 80 万样本，SFT 蒸馏到 Qwen（1.5B/7B/14B/32B）和 Llama（8B/70B）。**关键结论：蒸馏比直接对小模型做 RL 更有效**——小模型蒸馏版大幅超过同规模非蒸馏版。这解释了「为什么要先训大 R1 再压小」：RL 的涌现需要大模型容量，小模型直接 RL 学不会，但能靠大模型的数据「抄作业」。

---

## 4. 推理机制：DSA 与 FlashMLA

### 4.1 DSA：token 级稀疏注意力

标准 MLA 注意力是稠密的：每个 query 看所有 key。DSA（DeepSeek Sparse Attention）在 V3.2-Exp 里把它做成**细粒度、token 级稀疏**：先用一个轻量的 **lightning indexer** 选出「这个 query 最该关注的一批 token」（如 top-2048），注意力只在这批 token 上算。

- **收益**：长上下文的注意力算力/带宽大幅下降（稠密注意力是 O(N²) 算力，稀疏后按 top-k 缩放）；
- **代价**：引入了「选错 token」的风险——top-k 之外的重要 token 被漏掉；索引器本身也有开销；且需要专门的稀疏内核（FlashMLA sparse）才能跑出速度。

这跟文档 01 的 FlashAttention 是不同的省法：FlashAttention 是**精确**地省显存 IO（数学等价），DSA 是**近似**地省算力（主动不看部分 token）。

### 4.2 FlashMLA：把 MLA 跑满硬件的内核

FlashMLA 是 DeepSeek 开源的 MLA 优化内核，同时提供 **dense 和 sparse** 两种 kernel（sparse 即实现 DSA），覆盖 prefill 与 decode。实测（H800 SXM5）：

- 稀疏 MLA prefill：最高 **640 TFlops**（B200 上 1450 TFlops）；
- 稀疏 MLA decode：**410 TFlops**（FP8 KV Cache，matmul 用 BF16）；
- 稠密 MLA decode：**660 TFlops / 3000 GB/s**。

**门槛**：要求 SM90/SM100、CUDA 12.8+。这印证了文档 01 的结论——IO-aware 内核高度依赖硬件，MLA 的低秩结构（latent 上投影）也值得单独写 kernel 而不是复用 FlashAttention。vLLM/SGLang 都提供了 day-0 支持（用 FlashMLA 稀疏内核 + DeepGEMM 的 lightning indexer 内核）。

---

## 5. 两条线的对照（别混）

| | 训练线（R1） | 推理线（V3.2-Exp） |
|---|---|---|
| 改什么 | 模型**行为**（会推理、反思） | 推理**计算路径**（注意力稀疏 + 内核） |
| 解决什么 | 怎么让模型学会推理 | 怎么让推理更快更省 |
| 关键机制 | GRPO / 纯 RL / 蒸馏 | DSA 稀疏注意力 / FlashMLA |
| 精确性 | 改变输出质量 | DSA 是近似（漏 token 风险），FlashMLA 是精确内核 |
| 代价 | RL 训练算力、可读性、奖励设计 | 索引器开销、稀疏漏检、硬件门槛 |

一句话：**R1 教你「怎么想」，V3.2-Exp 让你「想得更快」**。两者可以叠加——一个会用 DSA 稀疏注意力做推理的 R1 类模型，既会推理、又跑得快。

---

## 6. 本文结论

- **GRPO** 用组内相对比较替代 PPO 的 critic，省一半显存，代价是序列级 advantage 与长度偏置。
- **纯 RL（R1-Zero）** 证明不靠 SFT 也能涌现推理，但可读性差；**R1 四阶段**用冷启动 SFT + 推理 RL + 拒绝采样 + 对齐 RL 修复之。
- **蒸馏** 比直接小模型 RL 更有效——涌现推理需要大容量，小模型靠大模型数据「抄」。
- **DSA** 是 token 级稀疏注意力，用 lightning indexer 选 top-k，近似地省长上下文算力；**FlashMLA** 是把它和稠密 MLA 跑满 Hopper/Blackwell 的内核。
- 训练线与推理线是两个正交维度，别把「GRPO」和「DSA」混为一谈。

---

## 参考资料

- [DeepSeek-R1（arXiv:2501.12948 / Nature 2025）](https://arxiv.org/abs/2501.12948)
- [DeepSeek-R1 官方仓库](https://github.com/deepseek-ai/DeepSeek-R1)
- [DeepSeek-V3.2-Exp 官方仓库](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp)
- [FlashMLA 内核](https://github.com/deepseek-ai/FlashMLA)
- [vLLM 对 V3.2-Exp 的稀疏注意力支持](https://vllm.ai/blog/2025-09-29-deepseek-v3-2)
