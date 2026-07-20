"""用随机张量验证形状、因果 Mask、数值稳定性和 Cache 等价性。"""

import torch

from causal_attention import MultiHeadCausalAttention, SingleHeadCausalAttention


def main() -> None:
    torch.manual_seed(42)

    single = SingleHeadCausalAttention(embed_dim=16).eval()
    single_input = torch.randn(2, 5, 16)
    single_output, single_weights, _ = single(single_input)

    multi = MultiHeadCausalAttention(embed_dim=24, num_heads=4).eval()
    multi_input = torch.randn(3, 6, 24)
    multi_output, multi_weights, _ = multi(multi_input)

    future_positions = torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1)
    masked_weights = single_weights.masked_select(future_positions.unsqueeze(0))

    large_input = torch.randn(2, 4, 24) * 1e4
    stable_output, stable_weights, _ = multi(large_input)

    full_output, _, _ = multi(multi_input)
    prefix_output, _, cache = multi(multi_input[:, :-1], use_cache=True)
    last_output, cached_weights, _ = multi(multi_input[:, -1:], cache=cache, use_cache=True)

    print(f"单头 output={tuple(single_output.shape)}, weights={tuple(single_weights.shape)}")
    print(f"多头 output={tuple(multi_output.shape)}, weights={tuple(multi_weights.shape)}")
    print(f"未来位置权重全为 0: {torch.count_nonzero(masked_weights).item() == 0}")
    print(f"大数输入输出有限: {torch.isfinite(stable_output).all().item()}")
    print(f"大数输入权重有限: {torch.isfinite(stable_weights).all().item()}")
    print(f"Cache 后单 token 权重形状: {tuple(cached_weights.shape)}")
    print(f"Cache 与完整计算最后位置一致: {torch.allclose(last_output, full_output[:, -1:], atol=1e-5)}")
    assert prefix_output.shape == (3, 5, 24)


if __name__ == "__main__":
    main()
