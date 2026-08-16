"""Workload definitions: the independent variables a benchmark must control."""

from __future__ import annotations

from dataclasses import dataclass

# Token budgets for the short / long dimensions.
SHORT_INPUT = 256
LONG_INPUT = 4096
SHORT_OUTPUT = 64
LONG_OUTPUT = 512

CONCURRENCIES = (1, 4, 16)
DEFAULT_TRIALS = 5


@dataclass(frozen=True)
class WorkloadConfig:
    """One cell of the benchmark matrix.

    ``concurrency`` is the number of in-flight requests launched together in a
    single trial; ``trials`` is how many times that trial is repeated so a
    distribution (P50/P95/P99) can be computed.
    """

    name: str
    input_tokens: int
    output_tokens: int
    concurrency: int
    streaming: bool = True
    trials: int = DEFAULT_TRIALS

    def key(self) -> str:
        return (
            f"in{self.input_tokens}-out{self.output_tokens}-"
            f"c{self.concurrency}-{'stream' if self.streaming else 'batch'}"
        )


def concurrency_sweep(input_tokens: int = 1024, output_tokens: int = 256) -> list[WorkloadConfig]:
    """Fixed input/output length, sweep concurrency 1/4/16.

    Drives the concurrency -> throughput and concurrency -> tail-latency curves.
    """
    return [
        WorkloadConfig(name=f"concurrency-{c}", input_tokens=input_tokens, output_tokens=output_tokens, concurrency=c)
        for c in CONCURRENCIES
    ]


def length_sweep(concurrency: int = 4) -> list[WorkloadConfig]:
    """Short/long input x short/long output at a fixed concurrency.

    Isolates how input length moves TTFT and output length moves end-to-end
    latency and cost (see document 02).
    """
    return [
        WorkloadConfig(name="short-short", input_tokens=SHORT_INPUT, output_tokens=SHORT_OUTPUT, concurrency=concurrency),
        WorkloadConfig(name="short-long", input_tokens=SHORT_INPUT, output_tokens=LONG_OUTPUT, concurrency=concurrency),
        WorkloadConfig(name="long-short", input_tokens=LONG_INPUT, output_tokens=SHORT_OUTPUT, concurrency=concurrency),
        WorkloadConfig(name="long-long", input_tokens=LONG_INPUT, output_tokens=LONG_OUTPUT, concurrency=concurrency),
    ]


def all_workloads() -> list[WorkloadConfig]:
    return concurrency_sweep() + length_sweep()
