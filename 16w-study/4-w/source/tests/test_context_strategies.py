"""三种上下文策略的统一指标测试。"""

from context_eval import FullHistory, RetrievalMemoryContext, SummaryHistory, build_context_cases, evaluate_context_strategies


def test_three_strategies_use_the_exact_same_multiturn_cases() -> None:
    """比较必须控制任务变量，而不是为每种策略换一套有利问题。"""

    cases = build_context_cases()
    assert len(cases) == 8
    assert all(len(case.history) >= 5 for case in cases)
    for strategy in (FullHistory(), SummaryHistory(), RetrievalMemoryContext()):
        assert all(case.query in strategy.compose(case) for case in cases)


def test_context_metrics_cover_accuracy_tokens_latency_and_unit_cost() -> None:
    """每种策略都记录用户要求的四类指标及逐任务明细。"""

    report = evaluate_context_strategies(input_usd_per_million=1.0)

    assert set(report) == {"full_history", "summary_history", "retrieval_memory"}
    for metrics in report.values():
        assert {"accuracy", "average_input_tokens", "average_latency_ms", "unit_task_cost_usd", "rows"} <= metrics.keys()
        assert len(metrics["rows"]) == 8
        assert metrics["average_latency_ms"] >= 0
        assert "not a provider invoice" in metrics["cost_basis"]


def test_retrieval_memory_reduces_tokens_without_losing_gold_facts() -> None:
    """本数据集上检索式记忆应比全历史更短，并保持全部关键事实。"""

    report = evaluate_context_strategies()

    assert report["retrieval_memory"]["accuracy"] == 1.0
    assert report["retrieval_memory"]["average_input_tokens"] < report["full_history"]["average_input_tokens"]
    assert report["summary_history"]["average_input_tokens"] < report["full_history"]["average_input_tokens"]

