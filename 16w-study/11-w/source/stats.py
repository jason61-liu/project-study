"""Deterministic statistics: percentiles, summary and confidence intervals."""

from __future__ import annotations

import math
from typing import Any

JSON = dict[str, Any]

Z_95 = 1.96


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolation percentile (matches numpy.percentile default).

    ``q`` is in [0, 1]. Handles the single-element and empty edge cases.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def sample_stddev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def summarize(values: list[float]) -> JSON:
    """Aggregate a sample into mean + P50/P95/P99 + min/max."""
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": round(mean(values), 3),
        "p50": round(percentile(values, 0.50), 3),
        "p95": round(percentile(values, 0.95), 3),
        "p99": round(percentile(values, 0.99), 3),
        "min": round(min(values), 3),
        "max": round(max(values), 3),
    }


def confidence_interval(values: list[float], z: float = Z_95) -> tuple[float, float]:
    """Normal-approximation CI for the mean: mean +/- z * s / sqrt(n)."""
    if len(values) < 2:
        m = mean(values)
        return m, m
    m = mean(values)
    margin = z * sample_stddev(values) / math.sqrt(len(values))
    return m - margin, m + margin


def error_ratio(values: list[float]) -> float:
    """Relative half-width of the 95% CI, as a fraction of the mean.

    Small value -> the mean is a reliable estimate; large value -> need more
    trials (see document 04: interval width shrinks with sqrt(n)).
    """
    if not values:
        return float("inf")
    lo, hi = confidence_interval(values)
    m = mean(values)
    if m == 0.0:
        return 0.0
    return (hi - lo) / 2.0 / m
