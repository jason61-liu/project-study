"""最小自回归生成：对比每步重算全部前缀与 KV Cache 单 token Decode。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from causal_attention import KVCache, MultiHeadCausalAttention


@dataclass
class GenerationStats:
    mode: str
    input_lengths: list[int]
    score_pairs: int


class TinyCausalLM(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.attention = MultiHeadCausalAttention(embed_dim, num_heads)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(
        self,
        token_ids: Tensor,
        cache: KVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, KVCache | None, int]:
        hidden = self.embedding(token_ids)
        output, weights, next_cache = self.attention(hidden, cache=cache, use_cache=use_cache)
        return self.lm_head(output), next_cache, weights.size(-2) * weights.size(-1)


@torch.inference_mode()
def generate_without_cache(model: TinyCausalLM, prompt: Tensor, new_tokens: int) -> tuple[Tensor, GenerationStats]:
    """每一步都把完整前缀重新送入模型。"""

    generated = prompt.clone()
    input_lengths: list[int] = []
    score_pairs = 0

    for _ in range(new_tokens):
        input_lengths.append(generated.size(1))
        logits, _, pairs = model(generated)
        score_pairs += pairs
        next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
        generated = torch.cat((generated, next_token), dim=1)

    return generated, GenerationStats("不使用 KV Cache", input_lengths, score_pairs)


@torch.inference_mode()
def generate_with_cache(model: TinyCausalLM, prompt: Tensor, new_tokens: int) -> tuple[Tensor, GenerationStats]:
    """首次 Prefill 整个 prompt，之后每次只输入上一步生成的一个 token。"""

    generated = prompt.clone()
    current_input = prompt
    cache: KVCache | None = None
    input_lengths: list[int] = []
    score_pairs = 0

    for _ in range(new_tokens):
        input_lengths.append(current_input.size(1))
        logits, cache, pairs = model(current_input, cache=cache, use_cache=True)
        score_pairs += pairs
        next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
        generated = torch.cat((generated, next_token), dim=1)
        current_input = next_token

    return generated, GenerationStats("使用 KV Cache", input_lengths, score_pairs)


def main() -> None:
    torch.manual_seed(7)
    model = TinyCausalLM(vocab_size=32, embed_dim=16, num_heads=4).eval()
    prompt = torch.tensor([[1, 5, 9, 2]])
    new_tokens = 5

    uncached_tokens, uncached_stats = generate_without_cache(model, prompt, new_tokens)
    cached_tokens, cached_stats = generate_with_cache(model, prompt, new_tokens)

    print(f"prompt: {prompt.tolist()}")
    for stats in (uncached_stats, cached_stats):
        print(f"{stats.mode}: 每步输入长度={stats.input_lengths}, attention score 对数={stats.score_pairs}")
    print(f"生成结果一致: {torch.equal(uncached_tokens, cached_tokens)}")
    print(f"tokens: {cached_tokens.tolist()}")
    print("观察：KV Cache 完成 Prefill 后，每次 Decode 的输入长度恒为 1，即每步只生成并处理一个 token。")


if __name__ == "__main__":
    main()
