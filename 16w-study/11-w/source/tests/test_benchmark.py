from __future__ import annotations

import asyncio

from benchmark import run_config
from model_sim import LLMServer, ModelProfile
from workload import WorkloadConfig


def make_server(*, server_concurrency=8, prefill_concurrency=4, prefill_ms=0.05, decode_ms=0.5):
    return LLMServer(ModelProfile(
        prefill_ms_per_token=prefill_ms,
        decode_ms_per_token=decode_ms,
        prefill_concurrency=prefill_concurrency,
        server_concurrency=server_concurrency,
        jitter=0.0,
        seed=1,
    ))


def test_decode_peak_respects_server_concurrency():
    async def scenario():
        server = make_server(server_concurrency=8)
        await run_config(server, WorkloadConfig("c16", 64, 32, concurrency=16, trials=1))
        assert server.decode_peak == 8
        assert server.prefill_peak <= 4
    asyncio.run(scenario())


def test_throughput_scales_then_saturates():
    async def scenario():
        throughputs: dict[int, float] = {}
        for c in (1, 4, 16):
            server = make_server(server_concurrency=8)
            result = await run_config(server, WorkloadConfig(f"c{c}", 64, 64, concurrency=c, trials=3))
            throughputs[c] = result.aggregate(server)["throughput_tokens_s"]
        assert throughputs[4] > throughputs[1]
        assert throughputs[16] > throughputs[1]
        # Saturation: throughput does not keep scaling linearly with concurrency.
        assert throughputs[16] < throughputs[1] * 16
    asyncio.run(scenario())


def test_tail_latency_grows_with_concurrency():
    async def scenario():
        p95: dict[int, float] = {}
        for c in (1, 16):
            server = make_server(server_concurrency=8)
            result = await run_config(server, WorkloadConfig(f"c{c}", 64, 64, concurrency=c, trials=3))
            p95[c] = result.aggregate(server)["e2e_ms"]["p95"]
        assert p95[16] > p95[1]
    asyncio.run(scenario())


def test_ttft_scales_with_input_length():
    async def scenario():
        s1 = make_server()
        short = await run_config(s1, WorkloadConfig("si", 256, 64, concurrency=1, trials=2))
        short_agg = short.aggregate(s1)

        s2 = make_server()
        long = await run_config(s2, WorkloadConfig("li", 4096, 64, concurrency=1, trials=2))
        long_agg = long.aggregate(s2)

        # Prefill is linear in input -> long input should dominate TTFT by far.
        assert long_agg["ttft_ms"]["mean"] > short_agg["ttft_ms"]["mean"] * 5
    asyncio.run(scenario())


def test_tpot_is_roughly_constant_regardless_of_output_length():
    async def scenario():
        s1 = make_server()
        short_out = await run_config(s1, WorkloadConfig("so", 256, 64, concurrency=1, trials=2))
        short_tpot = short_out.aggregate(s1)["tpot_ms"]["mean"]

        s2 = make_server()
        long_out = await run_config(s2, WorkloadConfig("lo", 256, 512, concurrency=1, trials=2))
        long_tpot = long_out.aggregate(s2)["tpot_ms"]["mean"]

        # TPOT is per-token decode time; it should not scale with output length.
        assert abs(long_tpot - short_tpot) < short_tpot * 0.5 + 1.0
    asyncio.run(scenario())
