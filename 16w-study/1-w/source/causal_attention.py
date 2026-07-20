"""不依赖封装 Attention 层的单头/多头 Causal Self-Attention。"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class KVCache:
    """每层注意力缓存，形状均为 ``[batch, heads, sequence, head_dim]``。"""

    key: Tensor
    value: Tensor

    @property
    def sequence_length(self) -> int:
        return self.key.size(2)


def make_causal_mask(query_length: int, key_length: int, device: torch.device) -> Tensor:
    """创建 ``[query_length, key_length]`` 的布尔 Mask，True 表示允许关注。

    ``key_length`` 可以大于 ``query_length``，多出的前缀视为 KV Cache 中的历史 token。
    """

    if query_length < 1 or key_length < query_length:
        raise ValueError("需要满足 query_length >= 1 且 key_length >= query_length")

    past_length = key_length - query_length
    query_positions = torch.arange(query_length, device=device) + past_length
    key_positions = torch.arange(key_length, device=device)
    return key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)


def _validate_input(x: Tensor, embed_dim: int) -> None:
    if x.ndim != 3:
        raise ValueError(f"输入必须是 [batch, sequence, embed_dim]，实际为 {tuple(x.shape)}")
    if x.size(1) < 1:
        raise ValueError("sequence 长度必须至少为 1")
    if x.size(2) != embed_dim:
        raise ValueError(f"最后一维必须为 embed_dim={embed_dim}，实际为 {x.size(2)}")


def _validate_cache(cache: KVCache, batch: int, heads: int, head_dim: int) -> None:
    expected_prefix = (batch, heads)
    for name, tensor in (("key", cache.key), ("value", cache.value)):
        if tensor.ndim != 4:
            raise ValueError(f"cache.{name} 必须是四维张量")
        if tensor.shape[:2] != expected_prefix or tensor.size(3) != head_dim:
            raise ValueError(
                f"cache.{name} 应为 [batch={batch}, heads={heads}, past, head_dim={head_dim}]，"
                f"实际为 {tuple(tensor.shape)}"
            )
    if cache.key.shape != cache.value.shape:
        raise ValueError("cache.key 与 cache.value 的形状必须相同")


def _causal_attention(q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
    """计算注意力；输入形状均使用 ``[B, H, T, D]``。"""

    query_length, key_length = q.size(2), k.size(2)
    mask = make_causal_mask(query_length, key_length, q.device)

    # 在 float32 中计算分数和 softmax，避免低精度输入放大后溢出。
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) / math.sqrt(q.size(-1))
    scores = scores.masked_fill(~mask.view(1, 1, query_length, key_length), float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    context = torch.matmul(weights, v.float()).to(v.dtype)
    return context, weights


class SingleHeadCausalAttention(nn.Module):
    """单头因果自注意力，未调用 ``nn.MultiheadAttention`` 等封装层。"""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        if embed_dim < 1:
            raise ValueError("embed_dim 必须大于 0")
        self.embed_dim = embed_dim
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(
        self,
        x: Tensor,
        cache: KVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, Tensor, KVCache | None]:
        _validate_input(x, self.embed_dim)
        batch = x.size(0)

        q = self.q_proj(x).unsqueeze(1)
        new_k = self.k_proj(x).unsqueeze(1)
        new_v = self.v_proj(x).unsqueeze(1)

        if cache is not None:
            _validate_cache(cache, batch, heads=1, head_dim=self.embed_dim)
            if cache.key.device != x.device or cache.key.dtype != new_k.dtype:
                raise ValueError("KV Cache 必须与输入位于相同设备并使用相同 dtype")
            k = torch.cat((cache.key, new_k), dim=2)
            v = torch.cat((cache.value, new_v), dim=2)
        else:
            k, v = new_k, new_v

        context, weights = _causal_attention(q, k, v)
        output = self.out_proj(context.squeeze(1))
        next_cache = KVCache(k, v) if use_cache else None
        return output, weights.squeeze(1), next_cache


class MultiHeadCausalAttention(nn.Module):
    """多头因果自注意力，显式实现投影、拆头、Mask、Softmax 与合并。"""

    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        if embed_dim < 1 or num_heads < 1:
            raise ValueError("embed_dim 和 num_heads 必须大于 0")
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim 必须能被 num_heads 整除")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def _split_heads(self, x: Tensor) -> Tensor:
        batch, sequence, _ = x.shape
        return x.view(batch, sequence, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: Tensor) -> Tensor:
        batch, _, sequence, _ = x.shape
        return x.transpose(1, 2).contiguous().view(batch, sequence, self.embed_dim)

    def forward(
        self,
        x: Tensor,
        cache: KVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, Tensor, KVCache | None]:
        _validate_input(x, self.embed_dim)
        batch = x.size(0)

        q = self._split_heads(self.q_proj(x))
        new_k = self._split_heads(self.k_proj(x))
        new_v = self._split_heads(self.v_proj(x))

        if cache is not None:
            _validate_cache(cache, batch, self.num_heads, self.head_dim)
            if cache.key.device != x.device or cache.key.dtype != new_k.dtype:
                raise ValueError("KV Cache 必须与输入位于相同设备并使用相同 dtype")
            k = torch.cat((cache.key, new_k), dim=2)
            v = torch.cat((cache.value, new_v), dim=2)
        else:
            k, v = new_k, new_v

        context, weights = _causal_attention(q, k, v)
        output = self.out_proj(self._merge_heads(context))
        next_cache = KVCache(k, v) if use_cache else None
        return output, weights, next_cache
