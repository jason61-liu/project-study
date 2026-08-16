# 第 11 周编码实验：异步 API Benchmark

本目录把第 11 周的推理性能知识落成一个可执行的异步 Benchmark。它不依赖真实模型或 GPU，而是用一个**结构上真实**的模拟 LLM 服务（Prefill 线性于输入、Decode 线性于输出、有限并发槽产生排队）复现 TTFT/TPOT/吞吐/尾延迟的曲线；生产接入时只需把 `LLMServer.stream` 替换成真实 Provider 的流式接口，其余统计与图表逻辑不变。

## 1. 交付结构

```text
source/
├── model_sim.py     # 模拟 LLM 服务：Prefill/Decode 时序、并发槽、计费
├── workload.py      # 工作负载矩阵：短/长输入 × 短/长输出 × 并发 1/4/16
├── stats.py         # P50/P95/P99、均值、置信区间
├── benchmark.py     # 异步 harness：请求级时间线 + Token 用量 + 吞吐
├── agent_link.py    # 把模型调用指标关联到 Agent 端到端成功/成本
├── charts.py        # 并发—吞吐、并发—尾延迟、质量—成本 三张图
├── run_bench.py     # 端到端运行入口
├── tests/           # 确定性测试
└── artifacts/       # benchmark.json / timelines.jsonl / summary.md / charts/
```

## 2. 执行模型：模拟服务怎么「像真的」

`model_sim.py` 复现了两个决定基准结果的时间行为：

- **Prefill**：时间 = `prefill_ms_per_token × input_tokens`，计算受限、随输入线性增长，决定 TTFT；
- **Decode**：时间 = `decode_ms_per_token × output_tokens`，内存受限、每 token 一步，决定 TPOT 与总生成时间；
- **有限并发槽**：`prefill_concurrency` 与 `server_concurrency` 两个 Semaphore，超过后请求排队——这正是「并发—吞吐」饱和曲线和「并发—尾延迟」上升的来源。

一次流式调用记录完整时间线（`t_sent → 首 token → 每个 token → 末 token`），派生出 `ttft_ms / tpot_ms / e2e_ms / queue_wait_ms` 与 Token 成本：

```text
cost = input_tokens × input_price + output_tokens × output_price   (microUSD)
```

## 3. 工作负载矩阵

`workload.py` 提供两组负载：

| 组 | 维度 | 用途 |
|---|---|---|
| 并发扫描 | 固定 input=1024 / output=256，并发 1/4/16 | 画并发—吞吐、并发—尾延迟 |
| 长度扫描 | 短/长输入 × 短/长输出（256/4096 × 64/512，并发 4） | 隔离输入→TTFT、输出→E2E/成本 |

每组配置重复 `trials=5` 次，输出 P50/P95/P99 与 95% 置信区间，而不是单次调用的数字。

## 4. 从模型指标到 Agent 端到端

`agent_link.py` 回答第 11 周最后一个要求「把模型调用指标关联到 Agent 端到端成功任务指标」。一个 Agent 任务按 `output_budget` 生成，每次尝试以 `min(1, output_budget / difficulty)` 的概率成功，失败则重试（最多 `max_attempts` 次）。于是：

- **成功率**随 output budget 单调上升（质量维度）；
- **重试放大**只抬高成本分子、不增加成功分母；
- **单位成功任务成本** = 总成本 ÷ 成功任务数，与单次调用成本在低成功率下显著背离。

`quality_sweep()` 扫 output budget ∈ {100,200,300,400,500}，画出「质量 → 单位成功任务成本」曲线。

## 5. 运行与测试

```bash
/Users/shiyiliu/workspace/pyproject/.venv/bin/python \
  -m pytest -q -o cache_dir=/tmp/week11-pytest-cache \
  11-w/source/tests

/Users/shiyiliu/workspace/pyproject/.venv/bin/python \
  11-w/source/run_bench.py \
  --output 11-w/source/artifacts
```

关键产物：

- [benchmark.json](./artifacts/benchmark.json) — 并发扫描 + 长度扫描 + Agent 关联的完整聚合；
- [timelines.jsonl](./artifacts/timelines.jsonl) — 每次请求的 TTFT/TPOT/E2E/排队/Token 时间线；
- [summary.md](./artifacts/summary.md) — 人类可读的结果表；
- [charts/](./artifacts/charts/) — 三张 PNG 图。

## 6. 生产化边界

该实现验证统计口径与曲线形状，不声称模拟器的绝对延迟是生产值。生产替换时仍须保持：

1. `LLMServer.stream` 换成真实 Provider 流式接口，保留相同的 `StreamSample` 时间线记录；
2. 每个延迟指标的起点/终点口径要写死（请求发出 vs 服务端收到；末 token vs 副作用提交）；
3. 速率限制（RPM/TPM）状态要单独标注，避免把「配额」误当成「模型能力」；
4. 报告分布（P50/P95/P99 + 置信区间）而不是单点；样本量足够时才谈 P99；
5. Agent 关联里「成功」必须落到权威 Outcome（正确、限时、限预算、无重复副作用），而不是「模型返回了文本」。
