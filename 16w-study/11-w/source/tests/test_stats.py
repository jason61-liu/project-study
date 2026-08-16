from __future__ import annotations

import numpy as np
import pytest

from stats import confidence_interval, mean, percentile, sample_stddev, summarize


def test_percentile_matches_numpy_linear():
    values = [3.0, 1.0, 2.0, 5.0, 4.0]
    for q in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0):
        assert percentile(values, q) == pytest.approx(np.percentile(values, q * 100), abs=1e-9)


def test_percentile_edge_cases():
    assert percentile([], 0.5) == 0.0
    assert percentile([42.0], 0.5) == 42.0
    assert percentile([42.0], 0.0) == 42.0
    assert percentile([42.0], 1.0) == 42.0


def test_summarize_reports_p50_p95_p99():
    summary = summarize([1.0, 2.0, 3.0, 4.0, 5.0])
    assert summary["n"] == 5
    assert summary["mean"] == pytest.approx(3.0)
    assert summary["p50"] == pytest.approx(3.0)
    assert summary["p95"] == pytest.approx(4.8)
    assert summary["p99"] == pytest.approx(4.96)


def test_confidence_interval_matches_formula():
    # values=[1,3]: mean=2, s=sqrt(2), n=2 -> margin = 1.96*sqrt(2)/sqrt(2) = 1.96
    lo, hi = confidence_interval([1.0, 3.0])
    assert lo == pytest.approx(2.0 - 1.96)
    assert hi == pytest.approx(2.0 + 1.96)


def test_sample_stddev():
    assert sample_stddev([1.0, 2.0, 3.0]) == pytest.approx(1.0)
    assert sample_stddev([5.0]) == 0.0


def test_mean_empty():
    assert mean([]) == 0.0
