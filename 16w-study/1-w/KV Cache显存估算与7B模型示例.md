# KV Cache 显存估算与 7B 模型示例

## 1. 目标

本文解决两个问题：

1. 如何根据模型结构、上下文长度、Batch Size 和数据类型估算 KV Cache；
2. 一个代表性的 7B Decoder 模型，在 4K、16K 上下文下需要多少 KV Cache 显存。

先给出结论。在本文的默认假设下：

| 上下文长度 | KV Cache |
|---:|---:|
| 4K，即 4096 token | 2 GiB |
| 16K，即 16384 token | 8 GiB |

这个结果只表示 KV Cache 张量本体，不是运行模型所需的全部显存。

## 2. 为什么“7B”不能唯一决定 KV Cache

7B 表示模型大约有 70 亿参数，但 KV Cache 大小不直接由总参数量决定。两个参数量相近的模型，如果层数、KV Head 数或 Head Dim 不同，KV Cache 可能相差数倍。

KV Cache 主要由以下配置决定：

| 符号 | 含义 |
|---|---|
| $B$ | 同时缓存的序列数，即 Batch Size |
| $T$ | 每条序列已经缓存的 token 数 |
| $L$ | Transformer 层数 |
| $H_{kv}$ | 每层 Key、Value Head 数 |
| $D_h$ | 每个 KV Head 的维度 |
| $s$ | 每个 Cache 元素占用的字节数 |

因此，计算前必须先明确模型结构和 Cache 数据类型。

## 3. KV Cache 到底保存什么

自回归生成时，每一层都会为历史 token 保存投影后的 Key 和 Value：

$$
K_{\text{cache}},V_{\text{cache}}
$$

它们的形状通常可以写成：

$$
K,V:[B,T,H_{kv},D_h]
$$

实现也可能采用 §[B,H_{kv},T,D_h]§ 等不同维度顺序，但元素总数相同。

对每一层，每个 token 需要保存：

$$
H_{kv}D_h
$$

个 Key 元素，以及相同数量的 Value 元素。因此公式中会出现一个系数 2：

$$
2=\text{一份 K}+\text{一份 V}
$$

KV Cache 避免了生成新 token 时反复计算全部历史 K、V，但新 Query 仍要读取并匹配历史 Key，再用权重汇总历史 Value。

## 4. 通用估算公式

### 4.1 元素数量

整个模型的 KV Cache 元素数量为：

$$
\boxed{
N_{\text{KV}}=2BLTH_{kv}D_h
}
$$

### 4.2 字节数

乘以每个元素占用的字节数：

$$
\boxed{
\text{KV bytes}=2BLTH_{kv}D_hs
}
$$

换算成二进制显存单位：

$$
1\text{ KiB}=1024\text{ B}
$$

$$
1\text{ MiB}=1024^2\text{ B}
$$

$$
1\text{ GiB}=1024^3\text{ B}
$$

### 4.3 使用隐藏维度改写

如果：

$$
D=H_qD_h
$$

其中 $H_q$ 是 Query Head 数，则：

$$
D_h=\frac{D}{H_q}
$$

代入后：

$$
\boxed{
\text{KV bytes}
=2BLTDs\frac{H_{kv}}{H_q}
}
$$

对标准 MHA：

$$
H_{kv}=H_q
$$

所以：

$$
\text{KV bytes}_{\text{MHA}}=2BLTDs
$$

对 GQA/MQA，Cache 相对 MHA 的比例为：

$$
\boxed{
\frac{\text{GQA/MQA Cache}}{\text{MHA Cache}}
=\frac{H_{kv}}{H_q}
}
$$

## 5. 代表性 7B 模型的假设

本文采用一个 LLaMA-2-7B-like 的 Decoder 配置做估算。这里的 “-like” 表示使用其常见的 7B 结构数量级，不把计算结果推广到所有 7B 模型：

| 配置 | 取值 |
|---|---:|
| 层数 $L$ | 32 |
| 隐藏维度 $D$ | 4096 |
| Query Head 数 $H_q$ | 32 |
| KV Head 数 $H_{kv}$ | 32，即 MHA |
| Head Dim $D_h$ | $4096/32=128$ |
| Batch Size $B$ | 1 |
| KV Cache 类型 | BF16/FP16 |
| 每元素字节数 $s$ | 2 |

注意：BF16 和 FP16 都是每个元素 2 字节，所以仅从容量公式看结果相同。

## 6. 先计算每 token 的 KV Cache

### 6.1 每层、每 token

$$
\text{bytes/token/layer}
=2H_{kv}D_hs
$$

代入：

$$
2\times32\times128\times2
=16384\text{ B}
=16\text{ KiB}
$$

因此，每层为一个 token 保存 K、V 需要 16 KiB。

### 6.2 全部 32 层、每 token

$$
16\text{ KiB}\times32
=512\text{ KiB}
$$

可以记住这个中间结论：

> 在本文配置中，每增加一个序列 token，Batch 1 的整个模型需要增加 512 KiB KV Cache。

## 7. 4K 上下文计算

令：

$$
T=4096
$$

直接代入通用公式：

$$
\text{KV bytes}
=2\times1\times32\times4096\times32\times128\times2
$$

$$
=2,147,483,648\text{ B}
$$

换算为 GiB：

$$
\frac{2,147,483,648}{1024^3}
=2\text{ GiB}
$$

也可以使用每 token 的结果快速计算：

$$
512\text{ KiB/token}\times4096\text{ tokens}
=2\text{ GiB}
$$

因此：

$$
\boxed{
\text{4K Context KV Cache}=2\text{ GiB}
}
$$

## 8. 16K 上下文计算

令：

$$
T=16384
$$

16K 是 4K 的 4 倍，而 KV Cache 与 $T$ 线性增长：

$$
\text{KV Cache}_{16K}
=4\times\text{KV Cache}_{4K}
$$

$$
=4\times2\text{ GiB}
=8\text{ GiB}
$$

完整代入为：

$$
2\times1\times32\times16384\times32\times128\times2
=8,589,934,592\text{ B}
$$

所以：

$$
\boxed{
\text{16K Context KV Cache}=8\text{ GiB}
}
$$

## 9. 结果汇总

### 9.1 默认 MHA、Batch 1、BF16/FP16

| 项目 | 结果 |
|---|---:|
| 每 token、每层 | 16 KiB |
| 每 token、全部 32 层 | 512 KiB |
| 4K 上下文 | 2 GiB |
| 16K 上下文 | 8 GiB |

### 9.2 随上下文长度变化

| 上下文 | token 数 | KV Cache |
|---:|---:|---:|
| 1K | 1024 | 512 MiB |
| 2K | 2048 | 1 GiB |
| 4K | 4096 | 2 GiB |
| 8K | 8192 | 4 GiB |
| 16K | 16384 | 8 GiB |
| 32K | 32768 | 16 GiB |

上下文每翻倍，KV Cache 也翻倍。

## 10. GQA 和 MQA 会减少多少

保持其他配置不变，只改变 KV Head 数：

| Attention | $H_q$ | $H_{kv}$ | 相对 MHA | 4K | 16K |
|---|---:|---:|---:|---:|---:|
| MHA | 32 | 32 | $1$ | 2 GiB | 8 GiB |
| GQA | 32 | 8 | $1/4$ | 512 MiB | 2 GiB |
| MQA | 32 | 1 | $1/32$ | 64 MiB | 256 MiB |

例如 GQA 的 $H_{kv}=8$：

$$
\frac{H_{kv}}{H_q}
=\frac{8}{32}
=\frac14
$$

因此相同上下文下，它的理论 KV Cache 是 MHA 的四分之一。

## 11. Batch Size 的影响

KV Cache 与 Batch Size $B$ 线性增长。对默认 MHA、BF16/FP16 配置：

| Batch Size | 4K | 16K |
|---:|---:|---:|
| 1 | 2 GiB | 8 GiB |
| 4 | 8 GiB | 32 GiB |
| 8 | 16 GiB | 64 GiB |
| 16 | 32 GiB | 128 GiB |

这里假设每个序列都已经达到表中的完整上下文长度。实际连续批处理系统中，不同序列长度可能不同。

## 12. Cache 数据类型的影响

保持 MHA、Batch 1 不变：

| Cache 类型 | 理论字节/元素 | 4K | 16K |
|---|---:|---:|---:|
| FP32 | 4 | 4 GiB | 16 GiB |
| BF16/FP16 | 2 | 2 GiB | 8 GiB |
| FP8/INT8 | 1 | 1 GiB | 4 GiB |
| INT4，理想打包 | 0.5 | 512 MiB | 2 GiB |

量化 Cache 在实际实现中还可能保存 Scale、Zero Point、对齐填充等元数据，所以实际容量可能略高于表中的理想结果；量化也可能影响质量和内核性能。

## 13. 为什么公式结果不等于进程总显存

模型实际运行显存通常近似为：

$$
\text{总显存}
\approx
\text{模型权重}
+\text{KV Cache}
+\text{激活与临时工作区}
+\text{框架和通信开销}
$$

对于约 70 亿参数：

$$
7\times10^9\times2\text{ B}
\approx14\text{ GB}
\approx13.0\text{ GiB}
$$

这是 BF16/FP16 权重本体的大致容量，不包含量化元数据、内存对齐或其他运行状态。

实际 KV Cache 占用还可能受到以下因素影响：

1. **Paged Attention 块分配**：最后一个块未填满会产生内部碎片；
2. **Padding**：静态 Batch 可能按最长序列分配；
3. **预分配策略**：推理引擎可能预先保留 Cache 池；
4. **Tensor Parallel**：KV Head 理想情况下可跨卡切分，但 MQA 等配置可能需要复制；
5. **Cache 量化元数据**：Scale 和分组信息占用额外空间；
6. **滑动窗口 Attention**：只缓存有限历史时，容量不再随完整会话无限增长；
7. **生成长度**：Prompt 与已经生成的 token 都会进入 Cache。

因此，部署前应同时看理论公式、引擎实际分配量和峰值显存。

## 14. 可复算代码

```python
def kv_cache_gib(
    batch_size: int,
    seq_len: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    bytes_per_element: float = 2,
) -> float:
    total_bytes = (
        2
        * batch_size
        * seq_len
        * num_layers
        * num_kv_heads
        * head_dim
        * bytes_per_element
    )
    return total_bytes / (1024**3)


config = {
    "batch_size": 1,
    "num_layers": 32,
    "num_kv_heads": 32,
    "head_dim": 128,
    "bytes_per_element": 2,
}

for seq_len in (4096, 16384):
    size = kv_cache_gib(seq_len=seq_len, **config)
    print(f"{seq_len:5d} tokens: {size:.2f} GiB")
```

输出：

```text
 4096 tokens: 2.00 GiB
16384 tokens: 8.00 GiB
```

## 15. 常见误区

1. **用 7B 参数量直接推导 KV Cache**：错误。还需要层数、KV Head 数和 Head Dim。
2. **忘记同时缓存 K 和 V**：公式最前面的 2 不能遗漏。
3. **把 Query Head 数直接当作 KV Head 数**：GQA/MQA 中二者不同。
4. **只计算一层**：每个 Transformer 层都有独立的 KV Cache。
5. **只计算 Prompt**：生成出的 token 也会持续写入 Cache。
6. **混用 GB 和 GiB**：$1\text{ GB}=10^9$ B，$1\text{ GiB}=2^{30}$ B。
7. **把 KV Cache 当成全部显存**：权重、激活、工作区和框架开销仍然存在。
8. **认为 KV Cache 消除了所有历史 Attention 计算**：它只避免重复投影历史 K、V，新 Query 仍需扫描历史 Cache。

## 16. 面试版回答

> KV Cache 为每个 Transformer 层、每个历史 token 保存 Key 和 Value。其理论容量是 $2BLTH_{kv}D_hs$ 字节，其中 2 代表 K 和 V。以 32 层、32 个 KV Head、Head Dim 128、BF16、Batch 1 的代表性 7B MHA 模型为例，每 token、每层需要 16 KiB，全部层需要 512 KiB。因此 4K 上下文需要 2 GiB，16K 需要 8 GiB。GQA/MQA 相对 MHA 按 $H_{kv}/H_q$ 缩小，但实际部署还要考虑分页碎片、预分配、量化元数据和并行切分。

## 17. 自测题

1. 为什么 KV Cache 公式前面有一个 2？
2. 上下文从 4K 增长到 16K，为什么 Cache 正好变成 4 倍？
3. 在 $H_q=32,H_{kv}=4$ 的 GQA 中，Cache 是 MHA 的几分之一？
4. 默认示例的 Batch Size 从 1 增加到 8，4K Cache 变成多少？
5. 为什么两个同为 7B 的模型可能具有不同的 KV Cache 大小？

### 参考答案

1. 因为每层、每个历史 token 都要同时缓存一份 Key 和一份 Value。
2. KV Cache 与 $T$ 成正比，而 $16384/4096=4$。
3. 比例为 $H_{kv}/H_q=4/32=1/8$。
4. $2\text{ GiB}\times8=16\text{ GiB}$。
5. 7B 只说明总参数量；层数、KV Head 数、Head Dim、Cache 类型等结构仍可能不同。

## 18. 延伸阅读

- [MHA、MQA 与 GQA：质量、KV Cache 和带宽对比](./MHA、MQA与GQA：质量、KV%20Cache和带宽对比.md)
- [Transformer 核心张量与模块原理](./Transformer核心张量与模块原理.md)
