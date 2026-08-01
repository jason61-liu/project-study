# 第 1 周复盘：Transformer 与推理基础

> 复盘日期：2026-07-24  
> 本周主题：Decoder-only Transformer、Attention、KV Cache 与自回归生成

## 1. 验收结论

| 验收项                               | 状态             | 证据                                                              |
| --------------------------------- | -------------- | --------------------------------------------------------------- |
| 完成 15 道 Transformer/推理基础题并记录答案    | 已完成            | 本文第 3 节                                                         |
| 准备一次 10 分钟白板讲解                    | 讲解稿已完成，待本人计时演练 | 本文第 4 节                                                         |
| 回答 Decode 为何偏带宽受限、Prefill 为何偏计算密集 | 已完成            | 本文第 5 节                                                         |
| 代码可以独立运行                          | 已验证            | `source/validate_random_tensors.py`、`source/generation_demo.py` |
| 测试全部通过                            | 已验证            | 9 passed                                                        |
| 提交本周复盘                            | 已完成            | 本文                                                              |

说明：文档和讲解稿已经准备完成，但“进行一次白板讲解”需要本人实际计时演练，因此不虚构为已完成。

## 2. 本周知识地图

```text
文本
  ↓ Tokenizer
Token ID
  ↓ Embedding
隐藏状态 [B, T, D]
  ↓ N 个 Decoder Block
  ├─ Norm → Causal Self-Attention → 残差
  └─ Norm → MLP/SwiGLU → 残差
  ↓ Final Norm
最后位置的隐藏状态
  ↓ LM Head
Logits [B, V]
  ↓ Temperature / Top-k / Top-p / Sampling
下一个 Token ID
  ↓ 追加到上下文并继续 Decode
```

贯穿这条数据流的四个关键问题是：

1. Attention 如何在不看到未来 token 的前提下聚合历史信息；
2. KV Cache 如何避免 Decode 时重复计算历史 K、V；
3. MHA、GQA、MQA 如何改变 Cache 大小与显存带宽；
4. Prefill 和 Decode 为什么呈现不同的硬件瓶颈。

## 3. Transformer/推理基础 15 题

### 1. 一个 Decoder-only Transformer 如何预测下一个 token？

输入文本先被 Tokenizer 转成 Token ID，再经 Embedding 变成隐藏向量。隐藏向量依次通过多层因果自注意力和 MLP，最后经过 Final Norm 与 LM Head 得到词表上每个 token 的 Logit。采样策略把 Logit 转换成下一个 Token ID，将其追加到上下文后继续生成。

模型输出的是条件分布：

$$
P(x_{t+1}\mid x_1,\ldots,x_t)
$$

它不是一次生成整段答案，而是不断重复“计算分布—选择一个 token—追加上下文”。

### 2. Q、K、V 分别表示什么？常见张量形状是什么？

输入隐藏状态为：

$$
X\in\mathbb{R}^{B\times T\times D}
$$

经过三个线性投影得到 Q、K、V。拆分多头后，常见形状为：

$$
Q,K,V\in\mathbb{R}^{B\times H\times T\times D_h}
$$

其中 Q 表示当前位置要查找什么，K 表示各位置可被匹配的特征，V 表示匹配后实际汇总的内容。Attention 使用 Q 与 K 计算权重，再对 V 加权求和。

### 3. Scaled Dot-Product Attention 的公式是什么？为什么除以 $\sqrt{D_h}$？

$$
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{D_h}}+M
\right)V
$$

如果 Q、K 各维分量的方差近似为 1，点积的方差会随维度 $D_h$ 增长。除以 $\sqrt{D_h}$ 可以把分数尺度拉回相对稳定的范围，避免 Softmax 过早饱和、梯度过小或低精度计算不稳定。

### 4. Causal Mask 的作用是什么？

Causal Mask 禁止位置 $t$ 关注未来位置 $t+1,t+2,\ldots$，从而保证训练和推理遵循自回归约束。实现时通常把未来位置的 Attention Score 置为 $-\infty$，这样 Softmax 后对应权重为 0。

使用 KV Cache 进行单 token Decode 时，新 Query 对应序列末尾，可以关注 Cache 中的全部历史 K/V 和当前 token 的 K/V。

### 5. Multi-Head Attention 为什么需要多个 Head？

多个 Head 让模型在不同表示子空间中并行学习不同关系，例如局部依赖、长距离引用、语法关系或实体关联。每个 Head 独立计算 Attention，结果拼接后再通过输出投影混合：

$$
\operatorname{MHA}(X)
=\operatorname{Concat}(head_1,\ldots,head_H)W_O
$$

多头的价值主要是表达能力，而不是把总计算量降低为单头的若干分之一。

### 6. 残差连接和 Norm 分别解决什么问题？

残差连接为信息和梯度提供直接通路，使深层网络更容易优化，并允许每个子层学习对当前表示的增量修改。Norm 控制激活尺度，提高训练稳定性。

现代 Decoder 模型常使用 Pre-Norm：

$$
X'=X+\operatorname{Attention}(\operatorname{Norm}(X))
$$

$$
Y=X'+\operatorname{MLP}(\operatorname{Norm}(X'))
$$

LayerNorm 会中心化并缩放，RMSNorm 主要依据均方根缩放，计算更简单。

### 7. MLP/SwiGLU 在 Transformer 中承担什么作用？

Attention 负责跨 token 聚合信息，MLP 则对每个 token 的表示独立进行非线性变换和通道混合。SwiGLU 常写为：

$$
\operatorname{SwiGLU}(x)
=\left(\operatorname{SiLU}(xW_g)\odot xW_u\right)W_d
$$

其中门控分支决定哪些特征通过。实际模型中，MLP 往往包含大量参数和计算，不能把 Transformer 简化为只有 Attention。

### 8. 训练、Prefill 和 Decode 有什么区别？

| 阶段 | 输入形态 | 是否并行处理多个位置 | KV Cache |
|---|---|---|---|
| 训练 | 完整训练序列 | 是 | 通常不以推理 Cache 方式长期保存 |
| Prefill | 完整 Prompt | 是 | 计算并写入 Prompt 的 K/V |
| Decode | 通常每步 1 个 token | 否，生成具有时序依赖 | 读取历史 K/V，并追加新 K/V |

训练使用 Teacher Forcing，可以一次计算所有目标位置；线上生成必须先得到上一个 token，才能确定下一个 token 的输入。

### 9. KV Cache 保存什么？它优化了什么，没有优化什么？

KV Cache 为每层保存历史 token 投影后的 K 和 V。Decode 时只需为新 token 计算新的 Q/K/V，不再对整个历史前缀重复执行 K/V 投影，也不再重算历史 Query 之间的 Attention。

但 KV Cache 不会让 Attention 对上下文长度变成 $O(1)$。新 Query 仍要读取并匹配全部历史 Key，再根据权重汇总全部历史 Value，因此单步 Attention 仍随历史长度线性增长。

### 10. KV Cache 显存如何估算？

理论容量为：

$$
\boxed{
\text{KV bytes}=2BLTH_{kv}D_hs
}
$$

其中 2 表示 K 和 V，$B$ 是 Batch Size，$L$ 是层数，$T$ 是缓存长度，$H_{kv}$ 是 KV Head 数，$D_h$ 是 Head Dim，$s$ 是每元素字节数。

以 32 层、32 个 KV Head、Head Dim 128、BF16、Batch 1 为例：

- 每 token、全部层：512 KiB；
- 4K 上下文：2 GiB；
- 16K 上下文：8 GiB。

### 11. MHA、GQA、MQA 对质量与 KV Cache 有什么影响？

| 类型 | Query Head | KV Head | 主要特点 |
|---|---:|---:|---|
| MHA | 多个 | 与 Query Head 数相同 | 表达能力强，Cache 和读取量最大 |
| GQA | 多个 | 多个 Query Head 共享一组 K/V | 质量和效率的常用折中 |
| MQA | 多个 | 只有一组 K/V | Cache 最小，共享最强 |

其他结构相同时，GQA/MQA 相对 MHA 的 KV Cache 比例为：

$$
\frac{H_{kv}}{H_q}
$$

减少 KV Head 不只节省容量，也减少 Decode 时需要从显存读取的 K/V 数据。

### 12. RoPE 的作用是什么？

RoPE 通过与位置相关的旋转变换作用于 Q 和 K，把位置信息编码进它们的相对相位。其重要性质是，旋转后 Q、K 的点积能够体现相对位置差异。

RoPE 不改变 Q/K 的形状，也不直接降低 Attention 复杂度。超出训练长度使用时仍要考虑频率外推和模型是否接受过长上下文训练。

### 13. Tokenizer、Token 与 Context Window 有什么关系？

Tokenizer 把文本切分为模型词表中的 Token ID。Token 不是固定的“一个字”或“一个单词”，不同语言、词表和文本内容的 token 数可能明显不同。

Context Window 限制一次请求中模型能够处理的 token 总数，通常需要同时容纳：

$$
\text{System/历史消息/检索内容/Prompt}+\text{生成 token}
$$

因此字符数不能直接等同于 token 数，Agent 系统还需要为输出和工具结果预留预算。

### 14. Temperature、Top-k 和 Top-p 如何影响采样？

Temperature 对 Logit 做缩放：

$$
p_i=\operatorname{softmax}\left(\frac{z_i}{\tau}\right)
$$

$\tau$ 越低，分布越尖锐；越高，随机性越强。Top-k 只保留概率最高的 $k$ 个候选，Top-p 则保留累计概率达到阈值 $p$ 的最小候选集合。

贪心解码直接选择最大 Logit，结果稳定但可能重复或缺少多样性。Temperature 为 0 通常是 API 对贪心或近似确定性行为的约定，不应机械代入除法公式。

### 15. 为什么 Decode 往往受显存带宽限制，而 Prefill 更偏计算密集？

Prefill 一次处理多个 token，大矩阵乘法可以让同一份权重被多个 token 复用，GPU Tensor Core 利用率和算术强度较高，因此更容易接近计算吞吐上限。

Decode 每步通常只处理一个 token。为了生成这个 token，各层仍要读取大量模型权重，还要读取随上下文增长的 KV Cache，但能执行的有效计算相对较少。数据搬运时间可能超过计算时间，因此更容易受显存带宽限制。

简化判断是：

$$
\text{算术强度}
=\frac{\text{FLOPs}}{\text{从显存搬运的字节数}}
$$

Prefill 通常具有更高算术强度，Decode 通常具有更低算术强度。Continuous Batching 可以提高 Decode 的批量和权重复用，但不会自动消除长上下文 KV 读取压力。

## 4. 10 分钟白板讲解稿：一个 Token 如何经过模型并生成下一个 Token

### 0:00–1:00：先画全局闭环

在白板上写：

```text
文本 → Tokenizer → Token ID → Embedding → N×Decoder Block
    → Final Norm → LM Head → Logits → Sampling → 下一个 Token
                                             ↑             ↓
                                             └── 追加上下文 ┘
```

开场表述：

> Decoder-only 模型做的事情可以概括为：给定已有 token，计算下一个 token 的条件概率分布。生成不是一次完成，而是循环执行这个过程。

### 1:00–2:00：Tokenizer 与 Embedding

举例把一句文本表示为：

$$
[x_1,x_2,\ldots,x_T]
$$

每个 Token ID 查 Embedding 表后得到：

$$
X\in\mathbb{R}^{B\times T\times D}
$$

强调 Tokenizer 决定文本如何离散化，Embedding 把离散 ID 映射到连续向量空间。

### 2:00–5:00：放大一个 Decoder Block

画出 Pre-Norm 结构：

```text
X ────────────────┐
│                 │
└→ Norm → Attention → + → X'
                         │
X' ─────────────────────┤
│                       │
└→ Norm → MLP/SwiGLU ─→ + → Y
```

Attention 内部继续展开：

1. $XW_Q,XW_K,XW_V$ 得到 Q、K、V；
2. 拆成多个 Head；
3. 计算 $QK^\top/\sqrt{D_h}$；
4. 加 Causal Mask，屏蔽未来位置；
5. Softmax 得到权重；
6. 权重乘 V，合并多个 Head；
7. 输出投影后通过残差连接。

补充一句：

> Attention 负责 token 之间的信息交换，MLP 负责每个 token 内部的通道变换；残差和 Norm 让深层模型可以稳定训练。

### 5:00–6:30：从最后隐藏状态到下一个 Token

经过全部 Decoder Block 后，对最后位置的隐藏状态执行 Final Norm 和 LM Head：

$$
h_T\in\mathbb{R}^{D}
\quad\rightarrow\quad
z=h_TW_{\text{vocab}}\in\mathbb{R}^{V}
$$

$V$ 是词表大小。Logit 经过 Temperature、Top-k/Top-p 和 Softmax 后形成候选分布，最后选择或采样出 $x_{T+1}$。

强调：

> 模型直接输出的是 Logit，不是已经确定的文字；采样策略也是生成行为的一部分。

### 6:30–8:00：区分 Prefill 与 Decode

画两条路径：

```text
Prefill：Prompt 的 T 个 token → 并行计算 → 建立各层 KV Cache
Decode ：新 token             → 单步计算 → 读取历史 Cache → 追加新 K/V
```

解释：

- Prefill 处理完整 Prompt，各位置可以通过 Causal Mask 并行计算；
- 第一次输出 token 后，生成过程存在依赖，只能逐 token Decode；
- KV Cache 避免每一步重新投影全部历史 K/V；
- 新 Query 仍然需要扫描历史 Key 和 Value。

### 8:00–9:15：解释性能瓶颈

写下：

$$
\text{算术强度}=\frac{\text{FLOPs}}{\text{Memory Bytes}}
$$

表述：

> Prefill 中同一份权重服务于很多 token，大矩阵乘法计算密集；Decode 中每步只有很少的新 token，却仍要读取大量权重和越来越长的 KV Cache，因此数据搬运相对计算更多，常见瓶颈从算力转向显存带宽。

同时说明这不是绝对规则：模型结构、Batch Size、Prompt 长度、硬件和推理引擎都会改变瓶颈。

### 9:15–10:00：总结并接受追问

最后总结三句话：

1. 模型逐层把已有 token 转成最后位置的隐藏状态，再映射为词表分布；
2. Prefill 并行处理 Prompt，Decode 逐 token 生成并利用 KV Cache；
3. KV Cache 用显存换计算，但长上下文仍带来容量和带宽成本。

建议计时演练时让主体讲解控制在 9 分 15 秒，留下约 45 秒回答“为什么要缩放”“Cache 保存什么”“为什么 Decode 不是 $O(1)$”等追问。

## 5. Prefill 与 Decode 的性能回答

### 30 秒面试版

> Prefill 一次处理整段 Prompt，权重加载后可以被多个 token 复用，主要计算表现为较大的矩阵乘法，算术强度和 GPU 并行度都比较高，所以通常更偏计算密集。Decode 每步一般只有一个新 token，各层仍要读取模型权重，并读取随上下文增长的 KV Cache，但执行的有效 FLOPs 相对较少，算术强度低，所以往往受显存带宽限制。Continuous Batching 能提升 Decode 的批量和权重复用，但长上下文下 KV Cache 的读取压力仍然存在。

### 深入解释

性能上限可以用 Roofline 思路理解：

$$
\text{可达性能}
\le
\min(
\text{峰值计算吞吐},
\text{显存带宽}\times\text{算术强度}
)
$$

#### Prefill

- Query 长度较大，可以把许多 token 合并进 GEMM；
- 模型权重从显存读入后可服务多个 token；
- GPU 有更多并行工作，Tensor Core 更容易被充分利用；
- Attention 需要构造较大的 Score 矩阵，长 Prompt 下计算量增长明显。

#### Decode

- 每条序列每步通常只有一个 Query；
- 权重矩阵很大，但单步可复用权重的 token 数少；
- 每层都需要读取历史 KV Cache，读取量随上下文长度增长；
- 小矩阵或矩阵—向量计算较难达到峰值计算吞吐；
- 因此 Token/s 经常更依赖 HBM 带宽，而不是理论 FLOPS。

#### 需要避免的绝对化

“Prefill 一定计算受限、Decode 一定带宽受限”只是常见工作区间，不是定律。以下因素会移动瓶颈：

- Batch Size 与 Continuous Batching；
- Prompt 和生成长度；
- MHA、GQA、MQA、MLA 等模型结构；
- 权重与 KV Cache 量化；
- FlashAttention、PagedAttention、算子融合；
- GPU 算力/带宽比及并行策略。

## 6. 代码与测试验证记录

### 6.1 运行环境

```bash
source /Users/shiyiliu/workspace/pyproject/.venv/bin/activate
cd /Users/shiyiliu/workspace/pyproject/16w-study/1-w/source
```

### 6.2 独立运行

```bash
python validate_random_tensors.py
python generation_demo.py
```

2026-07-24 实际验证结果：

- 单头、多头 Attention 输出形状正确；
- Causal Mask 后未来位置权重全部为 0；
- 大数输入下输出和 Attention 权重均为有限值；
- KV Cache 单 token 输出与完整前缀重算的最后位置一致；
- Cache 与非 Cache 模式生成结果一致；
- 非 Cache 模式各步输入长度为 `[4, 5, 6, 7, 8]`；
- Cache 模式各步输入长度为 `[4, 1, 1, 1, 1]`；
- Attention Score 对数从 190 降为 42。

这里的 190 和 42 是该最小示例中每个 Head 的 Query-Key 位置对累计数，用于展示计算差异，不应直接解释为真实模型延迟或 FLOPs。

### 6.3 自动测试

```bash
python -m pytest -q
```

结果：

```text
......... [100%]
9 passed, 1 warning in 0.42s
```

9 个测试覆盖：

1. 单头输出形状与因果 Mask；
2. 序列长度为 1；
3. 不同 Batch Size；
4. 不同 Head 数；
5. 无效输入形状；
6. 无效 KV Cache 形状；
7. 大数输入的数值稳定性；
8. Cache Decode 与完整重算等价；
9. Prefill 后逐 token Decode。

警告来自受限环境无法在项目上级目录创建 `.pytest_cache`，不影响测试执行和结果。

## 7. 本周复盘

### 已完成的产出

- Transformer 结构与 Attention 论文摘要；
- Q/K/V、Norm、MLP/SwiGLU 等核心模块笔记；
- RoPE、Tokenizer、上下文窗口和采样参数说明；
- MHA、MQA、GQA 的质量、Cache 和带宽对比；
- 7B 模型在 4K、16K 上下文下的 KV Cache 推导；
- MHA/GQA、MLA 与混合线性注意力对比图；
- 手写单头、多头 Causal Attention；
- KV Cache 和最小自回归生成示例；
- 9 个自动测试；
- 15 道基础题和 10 分钟白板讲解稿。

### 最大卡点

KV Cache 场景下的因果位置不再只是普通的 $T\times T$ 下三角 Mask。Query 可能只包含新 token，而 Key 包含历史 Cache 与当前 token，因此必须根据 `key_length - query_length` 恢复 Query 的绝对位置。

### 修正的错误认识

1. KV Cache 并不会让 Decode Attention 变成 $O(1)$，新 Query 仍需扫描历史 K/V；
2. “7B”不能唯一决定 KV Cache，真正决定容量的是层数、KV Head、Head Dim、长度、Batch 和数据类型；
3. Attention 不是 Transformer 的全部，MLP 通常同样占据大量参数和计算；
4. 总显存不等于模型权重加一个理想 KV Cache 数字，还包含激活、工作区、碎片和框架开销；
5. Prefill/Decode 的计算密集与带宽密集是基于算术强度的常见判断，不应表述为无条件定律。

### 当前不足

- 代码是最小 Attention/生成实验，不是完整 Decoder Block，尚未实现 Norm、RoPE 和 SwiGLU；
- 目前对复杂度的验证以理论推导和元素计数为主，还没有 GPU 延迟、吞吐和显存数据；
- 白板讲解稿已完成，但仍需脱稿、计时并接受追问；
- MLA 与混合线性注意力当前只建立了概念坐标，尚未深入实现。

### 下一步

1. 完成一次 10 分钟计时白板演练，记录超时点和无法顺畅回答的问题；
2. 在后续推理性能周使用真实模型测量 TTFT、TPOT、吞吐和峰值显存；
3. 保留“张量形状—复杂度—硬件瓶颈—工程权衡”的回答框架；
4. 进入第 2 周最小 Agent 前，确保能够脱稿回答本文 15 题。

## 8. 相关资料

- [Transformer 核心张量与模块原理](./Transformer核心张量与模块原理.md)
- [《Attention Is All You Need》结构与技术摘要](./《Attention%20Is%20All%20You%20Need》结构与技术摘要.md)
- [RoPE、Tokenizer、上下文窗口与采样参数原理](./RoPE、Tokenizer、上下文窗口与采样参数原理.md)
- [MHA、MQA 与 GQA：质量、KV Cache 和带宽对比](./MHA、MQA与GQA：质量、KV%20Cache和带宽对比.md)
- [KV Cache 显存估算与 7B 模型示例](./KV%20Cache显存估算与7B模型示例.md)
- [MHA、GQA、MLA 与混合线性注意力对比图](./MHA、GQA、MLA与混合线性注意力对比图.md)
- [代码运行说明](./source/README.md)
