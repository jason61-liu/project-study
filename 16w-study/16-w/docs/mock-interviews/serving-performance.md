# 模拟面试二：推理 Serving 与性能

时长：40 分钟；主题：把研究 Agent 扩展到 100 倍流量；结果：38/50。

## 题目与追问

1. 指标分解：正确区分端到端、TTFT、TPOT、吞吐、Goodput 和 p95/p99，并提出模型/工具/检索/队列分段 Trace。得分 9/10。
2. 容量估算：从现有 0.158 task/s 峰值推到 100 倍约 15.8 task/s，指出 SQLite 写锁先出问题，提出 PostgreSQL、队列与 Worker。得分 8/10；缺真实服务时间分布和峰值持续时间。
3. vLLM 与 SGLang：能解释 PagedAttention 和 Radix Cache，提出同硬件 A/B。得分 8/10；没有真实 benchmark 数据。
4. 长上下文：说明 Chunked Prefill、PD 解耦、KV 传输和 LMCache 的收益条件。得分 7/10；需要把 KV 字节量和网络带宽算到具体交叉点。
5. 量化与路由：能按权重/激活/KV 区分质量风险，提出分桶路由和失败升级。得分 6/10；缺 DeepSeek/Kimi 固定快照上的准确率与成本实测。

## 面试官式反馈

概念边界清楚，能避免“吞吐提高等于用户体验提高”的误区。但性能证据仅有毫秒级本地 Python 基线，不含 GPU、网络和模型；应完成一次 vLLM/SGLang 同模型实验，并用 Little's Law、KV 容量和 Goodput 给出可复核数字。

