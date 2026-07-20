# 手写 Causal Attention 与 KV Cache

本目录使用 PyTorch 基础张量运算手写单头和多头 Causal Self-Attention，没有调用 `nn.MultiheadAttention`、`torch.nn.functional.scaled_dot_product_attention` 或其他封装 Attention 层。

## 文件

- `causal_attention.py`：单头、多头注意力、因果 Mask 和 KV Cache。
- `validate_random_tensors.py`：随机张量形状、Mask、数值稳定性及 Cache 等价性验证。
- `generation_demo.py`：最小自回归生成循环，对比使用与不使用 KV Cache。
- `tests/test_causal_attention.py`：9 个测试，覆盖长度 1、不同 Batch、不同 Head、无效输入、数值稳定性、Cache 等价性和单 token Decode。

## 运行

```bash
source /Users/shiyiliu/workspace/pyproject/.venv/bin/activate
cd /Users/shiyiliu/workspace/pyproject/16w-study/1-w/source

python validate_random_tensors.py
python generation_demo.py
pytest -q
```

## 手写计算流程

给定输入 `x: [B, T, D]`：

1. 分别线性投影得到 Q、K、V；
2. 多头版本把它们变成 `[B, H, T, D_h]`；
3. 计算缩放点积 `QK^T / sqrt(D_h)`；
4. 将未来位置填成 `-inf`；
5. 在 float32 中执行 Softmax；
6. 计算 `AttentionWeights @ V`；
7. 合并多个 Head 并执行输出投影。

当存在 KV Cache 时，新 token 的 K/V 会追加到历史 K/V 后面。因果 Mask 会根据 `key_length - query_length` 自动计算当前 query 的绝对位置。

## 不使用与使用 KV Cache 的差异

设当前上下文长度为 `T`，隐藏维度为 `D`，生成 token 数为 `G`。

### 不使用 KV Cache

每一步都重新输入完整前缀：

- Q/K/V 投影：每步约 `O(TD²)`；
- 全局注意力：每步约 `O(T²D)`；
- 生成 `G` 个 token 时会反复计算旧 token 的 K/V 和旧位置之间的注意力。

若只观察序列长度，逐 token 生成全过程的注意力计算近似为：

```text
Σ(T + i)², i = 0 ... G-1
```

### 使用 KV Cache

第一次对 prompt 做 Prefill，之后每次只输入新生成的一个 token：

- 新 token 的 Q/K/V 投影：每步约 `O(D²)`；
- 新 query 与全部历史 key 的注意力：每步约 `O(TD)`；
- 历史 K/V 不再重复投影；
- 每层 Cache 空间约为 `O(TD)`，即用显存换取更少计算。

生成全过程的 attention score 对数近似为：

```text
Prefill: T²
Decode:  Σ(T + i), i = 1 ... G-1
```

因此 KV Cache **不会让单步 Attention 对上下文长度变成 O(1)**：新 query 仍需读取并匹配全部历史 key。它消除的是历史 token 的重复投影和历史 query 的重复注意力计算。

## 数值稳定性

- 点积先除以 `sqrt(D_h)`，避免维度增大时分数尺度过大；
- Attention score 与 Softmax 使用 float32；
- Mask 使用 `-inf`，Softmax 后未来位置权重严格为 0；
- 每个 query 至少允许关注一个位置，所以不会出现“整行都是 `-inf`”造成的 `NaN`。
