# MHA、MQA 与 GQA：质量、KV Cache 和带宽对比

## 1. 为什么要比较这三种注意力

MHA、MQA 和 GQA 的核心差异，不在 Query Head，而在于 **多少个 Query Head 共享一组 Key、Value Head**。

- **MHA（Multi-Head Attention，多头注意力）**：每个 Query Head 都有自己对应的 Key、Value Head。
- **MQA（Multi-Query Attention，多查询注意力）**：所有 Query Head 共享唯一一组 Key、Value Head。
- **GQA（Grouped-Query Attention，分组查询注意力）**：将 Query Head 分组，每组共享一组 Key、Value Head。

这种共享主要影响三件事：

1. 模型表达能力和最终质量；
2. 自回归推理时 KV Cache 的容量；
3. Decode 阶段读取 KV Cache 所需的显存带宽。

先给出结论：

| 方案 | KV Head 数 | 质量倾向 | KV Cache | Decode KV 读带宽 | 典型定位 |
|---|---:|---|---:|---:|---|
| MHA | $H_{kv}=H_q$ | 通常最有表达自由度 | 最大 | 最大 | 质量优先、较短上下文 |
| MQA | $H_{kv}=1$ | 共享最强，质量风险相对最高 | 最小 | 最小 | 极致压缩 KV、吞吐优先 |
| GQA | $1<H_{kv}<H_q$ | 通常接近 MHA | 居中 | 居中 | 现代 LLM 常用折中 |

“通常”非常重要：架构名称本身不能决定最终质量，训练数据、参数量、训练方式和推理实现都会影响结果。

## 2. 统一符号

| 符号 | 含义 |
|---|---|
| $B$ | Batch Size，即同时解码的序列数 |
| $T$ | 已缓存的上下文 token 数 |
| $L$ | Transformer 层数 |
| $D$ | 模型隐藏维度 |
| $H_q$ | Query Head 数 |
| $H_{kv}$ | Key、Value Head 数 |
| $D_h$ | 每个 Head 的维度 |
| $s$ | 每个 KV 元素占用的字节数 |
| $G$ | 每组共享一组 KV 的 Query Head 数，$G=H_q/H_{kv}$ |

通常有：

$$
D=H_qD_h
$$

为便于理解，后文统一使用：

$$
L=32,\quad D=4096,\quad H_q=32,\quad D_h=128
$$

并比较：

- MHA：$H_{kv}=32$；
- GQA：$H_{kv}=8$，每 4 个 Query Head 共享一组 KV；
- MQA：$H_{kv}=1$，32 个 Query Head 共享一组 KV。

## 3. 三种结构如何工作

### 3.1 MHA：每个 Query Head 使用独立 KV

MHA 中：

$$
H_{kv}=H_q
$$

每个 Head 分别计算：

$$
O_h=\operatorname{softmax}\left(\frac{Q_hK_h^\top}{\sqrt{D_h}}\right)V_h
$$

结构可以表示为：

```text
Q0 ↔ K0,V0
Q1 ↔ K1,V1
Q2 ↔ K2,V2
Q3 ↔ K3,V3
```

不同 Query Head 不仅能学习不同的查询方式，也能拥有不同的 Key、Value 表示，表达自由度最高。但推理时必须为每一层、每个历史 token 保存全部 KV Head。

### 3.2 MQA：所有 Query Head 共享一组 KV

MQA 中：

$$
H_{kv}=1
$$

计算变为：

$$
O_h=\operatorname{softmax}\left(\frac{Q_hK^\top}{\sqrt{D_h}}\right)V
$$

```text
Q0 ─┐
Q1 ─┼─→ 共享 K0,V0
Q2 ─┤
Q3 ─┘
```

Query Head 仍然彼此不同，因此它们可以提出不同的“问题”；但所有 Head 只能从同一套 Key、Value 表示中检索和汇总信息。这样显著减少 KV Cache，也减少生成每个 token 时必须从显存读取的数据。

### 3.3 GQA：一组 Query Head 共享一组 KV

GQA 介于二者之间：

$$
1<H_{kv}<H_q,\qquad G=\frac{H_q}{H_{kv}}
$$

例如 $H_q=8,H_{kv}=2$：

```text
Q0 ─┐
Q1 ─┼─→ K0,V0
Q2 ─┤
Q3 ─┘

Q4 ─┐
Q5 ─┼─→ K1,V1
Q6 ─┤
Q7 ─┘
```

GQA 保留多组不同的 Key、Value 表示，同时允许组内复用。它试图获得接近 MHA 的质量和接近 MQA 的推理效率。

## 4. 张量形状差异

对一层输入：

$$
X:[B,T,D]
$$

三种方案的 Query 形状相同：

$$
Q:[B,H_q,T,D_h]
$$

Key、Value 形状为：

$$
K,V:[B,H_{kv},T,D_h]
$$

因此区别可以集中写成：

| 方案 | $Q$ | $K,V$ |
|---|---|---|
| MHA | $[B,H_q,T,D_h]$ | $[B,H_q,T,D_h]$ |
| GQA | $[B,H_q,T,D_h]$ | $[B,H_{kv},T,D_h]$ |
| MQA | $[B,H_q,T,D_h]$ | $[B,1,T,D_h]$ |

实现 GQA/MQA 时，逻辑上可以把 KV 扩展到 Query Head 数再计算，但不应真的复制并长期保存这些数据。高效内核会在计算中广播或复用共享 KV，否则会抵消带宽收益。

## 5. 对模型质量的影响

### 5.1 为什么 MHA 的表达自由度最高

MHA 允许每个 Head 分别学习：

$$
W_Q^{(h)},\quad W_K^{(h)},\quad W_V^{(h)}
$$

某些 Head 可以偏重局部关系，另一些 Head 可以偏重长距离依赖、语法关系或特定特征。这里不能简单地把单个 Head 对应为一个稳定的人类概念，但独立 KV 确实提供了更多表示自由度。

### 5.2 为什么 MQA 可能损失质量

MQA 仍保留多个独立 Query Head，但将 Key、Value 压缩成一组。所有 Head 必须共享：

- 历史 token 如何被索引，即 Key 表示；
- 被检索后提供什么内容，即 Value 表示。

这种强共享形成信息瓶颈。模型规模、任务或训练方法不合适时，它可能比 MHA 更容易出现质量下降。不过，这不意味着 MQA 一定明显变差；充分训练的模型可能学会利用共享 KV，并用大幅推理收益换取很小的质量差异。

### 5.3 为什么 GQA 通常是更稳妥的折中

GQA 不要求全部 Query Head 共享同一套 KV，而是保留若干 KV 组。相较 MQA，它降低了共享造成的信息瓶颈；相较 MHA，它又按比例减少 Cache 和带宽需求。

GQA 论文还讨论了从已有 MHA Checkpoint 转换并继续训练的 **uptraining**：把多个 MHA KV Head 聚合成较少的 GQA Head，再使用一小部分原训练计算继续适配。论文实验表明，合适的 GQA 配置可以达到接近 MHA 的质量，同时获得接近 MQA 的速度。

### 5.4 比较质量时必须控制变量

不能只看“MHA/MQA/GQA”标签就下结论。严谨比较至少要控制：

1. 总训练 token 数和数据质量；
2. 总参数量或 Attention 参数量；
3. 模型宽度、层数和 FFN 配置；
4. 从头训练还是由 MHA 转换；
5. 继续训练的步数和学习率；
6. 评测任务，尤其是长上下文检索与生成任务。

因此更准确的表述是：

> 在其他条件接近时，减少 KV Head 会增强参数共享并降低 KV 表达自由度；GQA 通常能以较小质量代价换取显著推理收益，但实际差距必须通过目标模型和任务评测确定。

## 6. 对 KV Cache 容量的影响

### 6.1 为什么要缓存 K 和 V

自回归生成第 $T+1$ 个 token 时，需要让新 Query 关注前 $T$ 个 token。历史 token 的 Key、Value 不会改变，因此可以缓存起来，避免每一步重新投影全部历史 token。

每一层都要缓存 K 和 V，所以一个 Batch 的 KV Cache 字节数为：

$$
\boxed{
\text{KV bytes}=2LBT H_{kv}D_hs
}
$$

其中开头的 2 分别代表 K 和 V。

如果 $D=H_qD_h$，还可以写成：

$$
\text{KV bytes}
=2LBTDs\frac{H_{kv}}{H_q}
$$

因此相对于 MHA：

$$
\boxed{
\text{Cache 比例}=\frac{H_{kv}}{H_q}
}
$$

于是：

- MHA：比例为 $1$；
- GQA：比例为 $H_{kv}/H_q$；
- MQA：比例为 $1/H_q$。

### 6.2 4K 上下文计算示例

使用前述 32 层示例，令：

$$
B=1,\quad T=4096,\quad s=2\text{ bytes（FP16/BF16）}
$$

MHA：

$$
2\times32\times1\times4096\times32\times128\times2
=2\text{ GiB}
$$

GQA（8 个 KV Head）：

$$
2\times32\times1\times4096\times8\times128\times2
=512\text{ MiB}
$$

MQA：

$$
2\times32\times1\times4096\times1\times128\times2
=64\text{ MiB}
$$

| 方案 | $H_{kv}$ | 每 token、每层 KV | 4K Cache | 相对 MHA |
|---|---:|---:|---:|---:|
| MHA | 32 | 16 KiB | 2 GiB | $1\times$ |
| GQA | 8 | 4 KiB | 512 MiB | $1/4$ |
| MQA | 1 | 512 B | 64 MiB | $1/32$ |

### 6.3 16K 上下文计算示例

KV Cache 与 $T$ 线性增长。将上下文从 4K 增加到 16K，Cache 变为 4 倍：

| 方案 | 4K Cache | 16K Cache |
|---|---:|---:|
| MHA | 2 GiB | 8 GiB |
| GQA，$H_{kv}=8$ | 512 MiB | 2 GiB |
| MQA | 64 MiB | 256 MiB |

若 Batch Size 从 1 增加到 16，上表还要再乘 16。实际 Serving 系统还可能存在分页块未填满、对齐、元数据、临时工作区和 Cache 量化等因素，所以公式给出的是 KV 张量本体的理论容量，不等于进程的全部显存占用。

## 7. 对显存带宽和 Decode 速度的影响

### 7.1 Decode 为什么经常受带宽限制

Decode 阶段通常一次只为每个序列计算一个新 token。对每一层，新 Query 都要读取历史的 K、V：

$$
Q_{new}K_{cache}^{\top}
$$

$$
\operatorname{softmax}(\cdot)V_{cache}
$$

单步计算规模较小，却要从显存读取随上下文增长的 KV Cache，因此长上下文、小 Batch 的 Decode 经常呈现较低算术强度，容易受内存带宽限制。

### 7.2 理想情况下的 KV 读取量

忽略新 KV 写入、元数据和其他中间结果，一次生成一个 token 时，读取历史 KV 的理论下界近似为：

$$
\boxed{
\text{KV read bytes/token}\approx2LBT H_{kv}D_hs
}
$$

它与 KV Cache 本体大小具有相同形式，因为每个新 Query 需要扫描当前历史 Cache。于是，在能够真正复用共享 KV 的高效实现中：

$$
\text{MHA:GQA:MQA 的 KV 读带宽需求}
=H_q:H_{kv}:1
$$

对于 $H_q=32,H_{kv}=8$ 的例子，理论比例是：

$$
32:8:1
$$

即 GQA 的 KV 数据量是 MHA 的 $1/4$，MQA 是 MHA 的 $1/32$。

### 7.3 用带宽估算单 token 时间下界

若设备可用显存带宽为 $BW$，只考虑读取 KV Cache，则时间下界近似为：

$$
t_{kv}\ge\frac{\text{KV read bytes/token}}{BW}
$$

以 4K 上下文、Batch 1 和假设可持续带宽 $1\text{ TB/s}$ 为例：

| 方案 | 理论 KV 读取量/token | 仅 KV 读取的时间下界 |
|---|---:|---:|
| MHA | 2 GiB | 约 2.15 ms |
| GQA，$H_{kv}=8$ | 512 MiB | 约 0.54 ms |
| MQA | 64 MiB | 约 0.067 ms |

这里采用 $1\text{ TB/s}=10^{12}\text{ B/s}$，而 GiB 是二进制单位。该表只是便于建立数量级直觉，并不是端到端延迟预测。

### 7.4 为什么实际加速不会等于 Cache 缩小倍数

将 KV Cache 缩小 32 倍，不代表端到端 Decode 一定加速 32 倍，因为还存在：

- Query 和输出投影；
- FFN、归一化、残差连接和采样；
- Attention Score 与 Value 聚合的算术操作；
- Kernel Launch、同步、调度和通信；
- KV 分页、数据布局和非连续访问；
- Tensor Parallel 下的切分与跨卡通信；
- 实际带宽利用率不足。

此外，MHA、GQA 和 MQA 都保留 $H_q$ 个 Query Head。对每个 Query Head 计算 Attention 输出所需的主要算术量并不会简单地随 $H_{kv}$ 等比例减少；减少最明显的是 KV 投影、KV 存储和共享 KV 的显存读取量。

### 7.5 对并发量和吞吐的间接影响

KV Cache 变小不仅降低单步读取量，还允许同一张 GPU 容纳更多并发序列：

$$
\text{可容纳序列数}\approx
\frac{\text{可用于 KV 的显存}}{\text{单序列 KV Cache}}
$$

更高并发可能提升吞吐，但也会增加调度、Batch Attention 和延迟方面的权衡。因此应同时观察：

- TTFT（Time To First Token）；
- TPOT（Time Per Output Token）；
- 每秒输出 token 数；
- 给定延迟 SLO 下的最大并发量；
- KV Cache 实际占用和有效带宽。

## 8. 对参数量和 Prefill 的影响

虽然重点是 KV Cache 和带宽，但 KV Head 数也会影响投影参数量。

忽略 Bias：

$$
W_Q:[D,H_qD_h]
$$

$$
W_K,W_V:[D,H_{kv}D_h]
$$

$$
W_O:[H_qD_h,D]
$$

Attention 投影参数量为：

$$
P_{attn}=2D^2+2DH_{kv}D_h
$$

因为 $H_qD_h=D$：

- MHA：$P_{attn}=4D^2$；
- MQA/GQA：K、V 投影参数随 $H_{kv}$ 减少；
- Q 和输出投影参数保持不变。

在 Prefill 阶段，多个 token 可以并行计算，矩阵乘法更容易充分利用 GPU 算力。此时 KV Cache 写入量仍会随 $H_{kv}$ 减少，但性能收益通常不像长上下文 Decode 那样直接受 KV 读带宽主导。不能把 Decode 的理论带宽比例直接套到 TTFT 上。

## 9. 选型建议

### 优先考虑 MHA 的情况

- 质量和表达自由度优先；
- 上下文较短，KV Cache 不是主要瓶颈；
- 并发量较低，有足够显存和带宽；
- 已有 MHA Checkpoint，不希望承担架构转换与继续训练成本。

### 优先考虑 MQA 的情况

- KV Cache 容量或 Decode 带宽是首要瓶颈；
- 需要极高并发或很长上下文；
- 可以从头训练并充分验证质量；
- 推理内核能真正复用共享 KV。

### 优先考虑 GQA 的情况

- 希望显著降低 KV Cache，同时控制质量风险；
- 面向通用 LLM Serving，需要兼顾吞吐、延迟与质量；
- 可以通过实验选择合适的 $H_{kv}$；
- Tensor Parallel 或硬件实现更适合多个可切分的 KV Head。

实际选型最好对多个 $H_{kv}$ 做消融实验。常见目标不是寻找“理论上最好”的结构，而是在目标硬件和质量门槛下找到最少的 KV Head。

## 10. 常见误区

1. **MQA 只有一个 Attention Head**：错误。MQA 仍有多个 Query Head，只是共享一组 K、V。
2. **GQA 会减少 Query Head 数**：错误。通常减少的是 KV Head 数，Query Head 数保持不变。
3. **Cache 缩小几倍，端到端速度就提高几倍**：错误。FFN、投影、算术计算、通信和调度不会同比消失。
4. **KV Cache 让每步 Attention 变成常数复杂度**：错误。它避免重新计算历史 K、V，但新 Query 仍需扫描历史 Cache，单 token 工作量仍随 $T$ 近似线性增长。
5. **GQA 一定与 MHA 质量相同**：错误。GQA 通常是较好折中，但结果依赖训练与任务。
6. **把共享 KV 物理复制到所有 Query Head 也一样快**：错误。逻辑结果可能相同，但物理复制会增加内存流量，损害 MQA/GQA 的核心收益。
7. **KV Cache 只与上下文长度有关**：错误。它还正比于层数、Batch Size、KV Head 数、Head 维度和数据类型字节数。

## 11. 面试版总结

> MHA 为每个 Query Head 保留独立 K、V，表达自由度高，但 KV Cache 和 Decode 带宽开销最大。MQA 让所有 Query Head 共享一组 K、V，把理论 KV Cache 和 KV 读带宽降到 MHA 的 $1/H_q$，但共享最强，存在更高的质量风险。GQA 使用若干 KV Head，让一组 Query Head 共享 K、V，其 Cache 和带宽相对 MHA 按 $H_{kv}/H_q$ 缩小，通常能以较小质量代价获得显著推理收益。三者都保留多个 Query Head，而且 KV Cache 并未消除新 Query 对全部历史 token 的扫描。

## 12. 自测题

### 12.1 题目

1. MQA 为什么仍然属于多头注意力？
2. 当 $H_q=32,H_{kv}=4$ 时，每组有多少个 Query Head？KV Cache 是 MHA 的几分之一？
3. 为什么 GQA 的端到端加速通常小于 KV Cache 的缩小倍数？
4. 为什么长上下文 Decode 比 Prefill 更容易受 KV 读带宽限制？
5. 若层数、Batch Size 或上下文长度翻倍，KV Cache 分别如何变化？
6. 从 MHA Checkpoint 转成 GQA 后，为什么还需要继续训练？

### 12.2 参考答案

1. **MQA 为什么仍然属于多头注意力？**

   MQA 只把 Key、Value Head 数降为 1，并没有把 Query Head 数降为 1：

   $$
   Q:[B,H_q,T,D_h],\qquad K,V:[B,1,T,D_h]
   $$

   每个 Query Head 仍有独立的 Query 投影，可以产生不同的注意力分数和输出。多个 Query Head 只是共同读取同一组 K、V，所以 MQA 仍然是多头注意力。

2. **当 $H_q=32,H_{kv}=4$ 时，每组有多少个 Query Head？KV Cache 是 MHA 的几分之一？**

   每组 Query Head 数为：

   $$
   G=\frac{H_q}{H_{kv}}=\frac{32}{4}=8
   $$

   即每 8 个 Query Head 共享一组 K、V。相对于 $H_{kv}=H_q=32$ 的 MHA，KV Cache 比例为：

   $$
   \frac{H_{kv}}{H_q}=\frac{4}{32}=\frac{1}{8}
   $$

   因此，其他条件相同时，该 GQA 配置的 KV Cache 是 MHA 的 $1/8$，理论 KV 读取量也约为 MHA 的 $1/8$。

3. **为什么 GQA 的端到端加速通常小于 KV Cache 的缩小倍数？**

   GQA 主要减少 K、V 投影、KV Cache 容量以及 Decode 时读取共享 KV 的数据量，但一次完整推理还包括：

   - Query 投影和输出投影；
   - Attention Score 计算与 Value 聚合；
   - FFN、归一化、残差连接和采样；
   - Kernel Launch、分页管理、跨卡通信和调度。

   这些工作不会随 $H_{kv}$ 同比例减少。若 KV 读带宽原本也不是唯一瓶颈，那么 Cache 缩小 8 倍，端到端速度就更不可能直接提高 8 倍。

4. **为什么长上下文 Decode 比 Prefill 更容易受 KV 读带宽限制？**

   Prefill 会同时处理多个输入 token，主要计算可以组织成较大的矩阵乘法，GPU 更容易获得较高的并行度和算术强度。

   Decode 通常每个序列一次只生成一个新 token，但每一层的新 Query 都要扫描长度为 $T$ 的历史 K、V。单步需要读取大量 Cache，能对这些数据执行的运算却相对有限。随着 $T$ 增长，KV 读取量线性增加，因此长上下文 Decode 更容易成为显存带宽受限任务。

5. **若层数、Batch Size 或上下文长度翻倍，KV Cache 分别如何变化？**

   根据：

   $$
   \text{KV bytes}=2LBT H_{kv}D_hs
   $$

   KV Cache 分别与 $L$、$B$、$T$ 成正比：

   - 仅层数 $L$ 翻倍：Cache 变为 2 倍；
   - 仅 Batch Size $B$ 翻倍：Cache 变为 2 倍；
   - 仅上下文长度 $T$ 翻倍：Cache 变为 2 倍；
   - 三者同时翻倍：Cache 变为 $2\times2\times2=8$ 倍。

6. **从 MHA Checkpoint 转成 GQA 后，为什么还需要继续训练？**

   MHA 的每个 Query Head 原本对应独立的 K、V 投影。转换为 GQA 时，多个 KV Head 会被合并或聚合为较少的共享 KV Head，这会立即改变模型原有的注意力表示和输出分布。

   继续训练可以让共享后的 K、V 投影、各 Query Head、输出投影以及后续层共同适应新的分组关系，从而恢复转换造成的质量损失。只做结构转换而不继续训练，通常无法保证达到原 MHA Checkpoint 的质量。

## 13. 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)：MHA 与原始 Transformer。
- [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150)：MQA 及其解码带宽动机。
- [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)：GQA、uptraining 以及质量与速度折中。
- [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288)：在较大模型配置中采用 GQA 的公开案例。
