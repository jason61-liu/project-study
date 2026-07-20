import pytest
import torch

from causal_attention import KVCache, MultiHeadCausalAttention, SingleHeadCausalAttention
from generation_demo import TinyCausalLM, generate_with_cache, generate_without_cache


def test_single_head_output_shape_and_causal_mask() -> None:
    torch.manual_seed(0)
    layer = SingleHeadCausalAttention(embed_dim=8)
    output, weights, _ = layer(torch.randn(2, 4, 8))

    assert output.shape == (2, 4, 8)
    assert weights.shape == (2, 4, 4)
    future = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)
    assert torch.count_nonzero(weights.masked_select(future.unsqueeze(0))) == 0
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(2, 4))


def test_sequence_length_one() -> None:
    layer = SingleHeadCausalAttention(embed_dim=6)
    output, weights, _ = layer(torch.randn(3, 1, 6))

    assert output.shape == (3, 1, 6)
    torch.testing.assert_close(weights, torch.ones(3, 1, 1))


def test_different_batch_sizes() -> None:
    layer = MultiHeadCausalAttention(embed_dim=12, num_heads=3)

    for batch in (1, 4, 7):
        output, weights, _ = layer(torch.randn(batch, 5, 12))
        assert output.shape == (batch, 5, 12)
        assert weights.shape == (batch, 3, 5, 5)


def test_different_head_counts() -> None:
    x = torch.randn(2, 3, 16)

    for heads in (1, 2, 4, 8):
        output, weights, _ = MultiHeadCausalAttention(16, heads)(x)
        assert output.shape == (2, 3, 16)
        assert weights.shape == (2, heads, 3, 3)


def test_invalid_input_shapes() -> None:
    layer = MultiHeadCausalAttention(embed_dim=12, num_heads=3)

    with pytest.raises(ValueError, match="输入必须"):
        layer(torch.randn(2, 12))
    with pytest.raises(ValueError, match="最后一维"):
        layer(torch.randn(2, 4, 11))
    with pytest.raises(ValueError, match="整除"):
        MultiHeadCausalAttention(embed_dim=10, num_heads=3)


def test_invalid_cache_shape() -> None:
    layer = MultiHeadCausalAttention(embed_dim=12, num_heads=3)
    bad_cache = KVCache(torch.randn(2, 2, 4, 4), torch.randn(2, 2, 4, 4))

    with pytest.raises(ValueError, match="cache.key"):
        layer(torch.randn(2, 1, 12), cache=bad_cache)


def test_large_values_are_numerically_stable() -> None:
    torch.manual_seed(1)
    layer = MultiHeadCausalAttention(embed_dim=16, num_heads=4)
    output, weights, _ = layer(torch.randn(2, 6, 16) * 1e4)

    assert torch.isfinite(output).all()
    assert torch.isfinite(weights).all()
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(2, 4, 6))


def test_cached_decode_matches_full_recomputation() -> None:
    torch.manual_seed(2)
    layer = MultiHeadCausalAttention(embed_dim=16, num_heads=4).eval()
    x = torch.randn(2, 5, 16)

    full_output, _, _ = layer(x)
    _, _, cache = layer(x[:, :4], use_cache=True)
    cached_output, cached_weights, next_cache = layer(x[:, 4:], cache=cache, use_cache=True)

    torch.testing.assert_close(cached_output, full_output[:, 4:], atol=1e-5, rtol=1e-5)
    assert cached_weights.shape == (2, 4, 1, 5)
    assert next_cache is not None and next_cache.sequence_length == 5


def test_generation_decodes_one_token_after_prefill() -> None:
    torch.manual_seed(3)
    model = TinyCausalLM(vocab_size=20, embed_dim=12, num_heads=3).eval()
    prompt = torch.tensor([[1, 2, 3, 4]])

    uncached, uncached_stats = generate_without_cache(model, prompt, new_tokens=4)
    cached, cached_stats = generate_with_cache(model, prompt, new_tokens=4)

    assert torch.equal(cached, uncached)
    assert uncached_stats.input_lengths == [4, 5, 6, 7]
    assert cached_stats.input_lengths == [4, 1, 1, 1]
    assert cached_stats.score_pairs < uncached_stats.score_pairs
