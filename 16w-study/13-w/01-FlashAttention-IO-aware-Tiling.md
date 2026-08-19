# FlashAttention：IO-aware Tiling

> 目标：说清标准 Attention 的瓶颈不在算力而在显存读写，FlashAttention 如何用「IO-aware Tiling + Online Softmax + 核融合」把 O(N²) 的 HBM 访问降到 O(N)，以及它的收益边界（序列越长越明显）。

---

## 1. 阅读前术语表

| 术语 | 说明 |
|---|---|
| HBM | High Bandwidth Memory，显存，容量大但带宽相对低（~1.5–3 TB/s） |
| SRAM | 片上共享内存，容量小（~20MB/H100）但带宽极高（~19 TB/s） |
| IO-aware | 以「最小化 HBM 读写次数」而非「最小化 FLOPs」为目标来设计算法 |
| Tiling | 把 Q/K/V 分块，一块一块搬进 SRAM 计算，避免整块驻留 HBM |
| Online Softmax | 流式 softmax：维护 running max 和 running sum，边算边归一化 |
| 核融合（Kernel Fusion） | 把多步（matmul→softmax→matmul）合成一个 CUDA kernel，中间结果不落地 |
| FlashAttention-2/3 | 后续版本：-2 改进工作划分与并行，-3 用 Hopper 的 TMA/WGMMA 与 FP8 |

---

## 2. 标准 Attention 的瓶颈：不是算力，是显存

标准的 scaled dot-product attention 三步：

```text
S = Q @ K^T        # [N, N]，读 Q/K，写 S —— O(N²) 显存
P = softmax(S)     # [N, N]，读 S，写 P —— 又 O(N²) 显存
O = P @ V          # [N, N]，读 P/V，写 O
```

问题在于：中间矩阵 S 和 P 都是 N×N，N 是序列长度。softmax 的**行归一化**（每行要求和、做除法）迫使实现必须把整行读完、归一化完，才能做下一步 matmul。朴素实现于是把 S、P 都物化到 HBM——对长序列（N=8k、16k）这就是 GB 级的读写。

所以标准 Attention 是 **memory-bound**：GPU 大部分时间在等 HBM 往返，算力（Tensor Core）反而闲着。FlashAttention 的洞察就是**换一个优化目标**——既然瓶颈在 IO，就把「减少 HBM 访问次数」当作第一目标，FLOPs 不变也没关系。

---

## 3. 三个核心技巧

### 3.1 Tiling：分块搬进 SRAM

不一次性算整张 N×N。把 Q 按行切成多个 block，每个 Q-block 先搬进 SRAM；然后遍历所有 K/V block，逐块算 `S_ij = Q_i @ K_j^T`、局部 softmax、`O_i += P_ij @ V_j`。关键：**S_ij 和 P_ij 永远停留在 SRAM，算完即弃，从不写回 HBM**。

### 3.2 Online Softmax：流式归一化

softmax 的分母是「整行 exp 之和」，朴素的 tiling 要么先扫一遍算 max/和、再扫一遍做除法（2-pass），要么保留整行。Online softmax 用**重缩放（rescaling）** trick 单遍解决：维护一个 running max `m` 和 running sum `l`，每来一个新 block 就更新 `m_new = max(m, m_block)`，把已累计的输出乘以 `exp(m - m_new)` 修正，再累加新块。这样一行只扫一遍，且不用保留整行。

### 3.3 核融合 + 反向重算

三步（matmul→softmax→matmul）合成**一个** kernel 后，S/P 根本不需要写回 HBM。反向传播时也不存 S/P，只存每行的 `m` 和 `l`（O(N) 而不是 O(N²)），在 backward 里**重算**一遍 attention（FLOPs 换显存）。

**净效果**：FLOPs 仍是 O(N²)（注意力本质如此），但 HBM 访问从 O(N²) 降到 O(N)。实现上带来 2–4× 加速（长序列更高），显存从 O(N²) 降到 O(N)——**这直接让长上下文的训练/推理成为可能**。

---

## 4. 后续版本：把 IO-aware 推到极致

- **FlashAttention-2**：更好的工作划分（按序列长度并行而非 batch）、减少非 matmul 的 FLOPs、单个 thread block 内更少的同步，比 v1 再快约 2×。
- **FlashAttention-3**：针对 Hopper（H100）用异步的 TMA（张量内存加速器）搬运、WGMMA（warp 组矩阵乘）、warp 特化，支持 FP8，进一步压榨 Tensor Core 与内存流水。

这条演进线说明 IO-aware 的核心思想没变——**每一代都在想尽办法让数据在 SRAM 里多算几轮、少回一次 HBM**。

---

## 5. 收益边界与代价

- **序列越长越赚**：N 小时（如 N=128），分块与 SRAM 管理的固定开销可能抵消收益，甚至不如朴素实现；N 一大（≥512，尤其 2k 以上），O(N²)→O(N) 的 IO 差立刻拉开。
- **不改数学、只改实现**：FlashAttention 是**精确**算法（在线 softmax 数学等价），不是近似注意力，因此精度无损，只是换了内存访问模式——这一点常被和稀疏/线性注意力（文档 04 的 KDA）混淆，后者才是近似。
- **代价**：kernel 高度依赖硬件特性（各代 CUDA、Tensor Core 布局），可移植性差、要随 GPU 代际重写；这解释了 FlashMLA（文档 03）为何要为 MLA 单独写 kernel。

---

## 6. 本文结论

FlashAttention 把注意力的优化目标从「减少 FLOPs」改成「减少 HBM 读写」（IO-aware），用三个技巧落地：**Tiling** 分块进 SRAM、**Online Softmax** 流式归一化、**核融合 + 反向重算** 让 N×N 中间矩阵不落地。结果是 FLOPs 不变、但显存从 O(N²) 降到 O(N)、HBM 访问从 O(N²) 降到 O(N)，换来 2–4× 加速。要记住的两点：**它是精确算法而非近似**，且**收益随序列长度单调上升**——这是理解后面 MLA（KV 压缩）、DSA（稀疏注意力）、KDA（线性注意力）这些「进一步省显存/算力但引入近似或结构约束」方案的基线。

---

## 参考资料

- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness（arXiv:2205.14135）](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2（arXiv:2307.08691）](https://arxiv.org/abs/2307.08691)
- [FlashAttention-3（arXiv:2407.08608）](https://arxiv.org/abs/2407.08608)
- [Dao et al., FlashAttention 官方仓库](https://github.com/Dao-AILab/flash-attention)
