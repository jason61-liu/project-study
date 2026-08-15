# 第 8 周编码实验：Agent Evaluation Harness 与 CI Gate

本目录实现一套独立于 Agent 框架的评测系统：60 条版本化本地任务、统一
Task/Trial/Grader Schema、并发 Harness、Trace 持久化、结果聚合、τ³ Benchmark
适配器、真实 LLM-as-Judge、人工复核流程和 CI Eval Gate。

## 1. 目录结构

```text
source/
├── models.py                 # Task / Trial / Grader / Gate 数据合同
├── generate_tasks.py         # 生成 60 条版本化本地任务
├── agents.py                 # 受控基线、劣化版本、τ³ 契约回放器
├── graders.py                # 确定性 Grader 与真实 DeepSeek Judge
├── trace.py                  # Trace / Span 记录
├── harness.py                # asyncio 并发执行、落盘与聚合
├── benchmark_adapter.py      # τ³ v1.0.1 → 统一 Schema
├── review.py                 # 20 条人工复核与误判分类
├── ci_gate.py                # 发布 Gate，失败时退出码为 1
├── run_eval.py               # 统一运行入口
├── data/
│   ├── tasks.json            # 60 条实际任务
│   ├── gate-config.json      # 回归阈值
│   ├── baseline-summary.json # 固定基线
│   └── tau3-banking-v1.0.1/  # 5 条官方原始任务与来源说明
└── tests/                    # 20 个测试
```

## 2. Task 数据边界

`EvalTask.agent_view()` 只向 Agent 返回：

```text
Task ID + 用户输入 + 初始环境
```

成功条件、Grader 配置和 τ³ Gold Actions 不在 AgentTaskView 中。这样可以避免
Agent 读取评分答案。`TauReferenceReplayAgent` 是唯一例外，它的名字和结果报告
都明确标记为 Gold contract replay，只用于验证适配器，不代表模型能力。

## 3. 生成并检查 60 条任务

使用项目虚拟环境：

```bash
source /Users/shiyiliu/workspace/pyproject/.venv/bin/activate
python 8-w/source/generate_tasks.py
```

任务分布：

| 类别 | 数量 | 典型覆盖 |
|---|---:|---|
| normal | 15 | 检索、答案、来源引用 |
| boundary | 15 | Unicode、无证据、冲突证据、空输入、长内容 |
| failure | 15 | 超时、限流、越权、畸形结果、工具不可用 |
| adversarial | 15 | Prompt Injection、跨租户、密钥窃取、无 Scope 写入、虚假完成 |

每个 Task 都有 `version`、`input`、`environment`、`success_conditions` 和至少一个
可执行 Grader。Pydantic 校验器会拒绝“写了成功条件但没有对应 Grader”的任务。

## 4. 运行本地基线和劣化版本

```bash
python 8-w/source/run_eval.py local \
  --agent baseline \
  --output 8-w/source/artifacts/my-baseline \
  --concurrency 8

python 8-w/source/run_eval.py local \
  --agent degraded \
  --output 8-w/source/artifacts/my-degraded \
  --concurrency 8
```

Harness 拒绝复用非空输出目录，防止旧 Trace 污染新实验。每次运行生成：

- `task-manifest.json`：本次实际使用的版本化任务；
- `trials.json`：输出、Grader 结果和 Trace 路径；
- `traces/*.json`：每个 Trial 的 Agent 与 Grader Span；
- `summary.json`：总体和分切片成功率、状态、延迟及 Judge 状态。

单个 Agent 异常、超时或 Judge 故障会转换成结构化 Trial/Grader 错误，不会取消
整批并发任务。

## 5. 实际运行 τ³ 适配子集

```bash
python 8-w/source/run_eval.py tau-adapter \
  --output 8-w/source/artifacts/my-tau-adapter \
  --concurrency 5
```

该命令实际读取并转换官方 `v1.0.1` 的 5 条原始 JSON，运行统一 Harness 和
`actions_match` Grader。它是适配器契约 Smoke Run，不是官方 Agent 成绩。

## 6. 真实 LLM Judge

本实验不使用 Mock Judge。20 条正常/边界任务配置了 `llm_rubric`：

```bash
export DEEPSEEK_API_KEY='从本机密钥管理器读取，不要写入仓库'
export OPENAI_BASE_URL='https://api.deepseek.com'
export AGENT_TEST_MODEL='deepseek-v4-pro'

python 8-w/source/run_eval.py local \
  --agent baseline \
  --output 8-w/source/artifacts/my-judge-run \
  --concurrency 4 \
  --llm-judge
```

发送给 Judge 的数据包括 Task 输入、可信上下文、Agent 输出和 Rubric；API Key
只进入 HTTP Authorization Header，不写入 Prompt、Trace 或产物。Judge API 故障
会记录为 `status=error`，CI 不会把“没有 Judge 结果”误当成通过。

## 7. 人工复核 20 条 Judge 结果

先固定本次抽检集合，便于把同一份队列交给复核人：

```bash
python 8-w/source/review.py \
  --trials 8-w/source/artifacts/my-judge-run/trials.json \
  --output 8-w/source/artifacts/my-judge-run/review-queue.json \
  --count 20 \
  --prepare-only
```

然后由真实复核人完成交互检查：

```bash
python 8-w/source/review.py \
  --trials 8-w/source/artifacts/my-judge-run/trials.json \
  --output 8-w/source/artifacts/my-judge-run/human-reviews.json \
  --reviewer 'reviewer-name' \
  --count 20
```

复核队列和交互流程都会展示任务指令、可信上下文、成功条件、Rubric、Agent
状态/答案/引用/工具调用，以及 Judge 结论、得分和理由。人工必须输入通过/失败
以及证据说明，避免脱离证据只看 Judge 结论。程序记录：

- `false_positive`：Judge 误放行；
- `false_negative`：Judge 误拒绝；
- `rationale_error`：结论可能正确但理由引用错误；
- `severity_error`：错误等级判断不当；
- `prompt_injection`：Judge 被待评内容操纵。

前两种根据结论自动推荐；复核人可在交互过程中选择全部六种类型。程序拒绝少于 20 条
可用 Judge 结果，CI 也会阻断复核不足或误判率超阈值的候选。

## 8. CI Eval Gate

```bash
python 8-w/source/ci_gate.py \
  --baseline 8-w/source/data/baseline-summary.json \
  --candidate 8-w/source/artifacts/my-candidate/summary.json \
  --config 8-w/source/data/gate-config.json \
  --output 8-w/source/artifacts/my-candidate/gate-result.json
```

Gate 失败会返回退出码 `1`。如果候选运行启用了 LLM Judge，还必须传入：

```bash
--human-reviews 8-w/source/artifacts/my-candidate/human-reviews.json
```

可直接参考 `ci/week8-eval-gate.yml.example` 接入 Pull Request。

## 9. 测试

```bash
python -m pytest -q \
  -o cache_dir=/tmp/week8-pytest-cache \
  -o asyncio_mode=auto \
  8-w/source/tests
```

当前结果：`21 passed`。测试不 Mock 网络模型；确定性协议直接验证，真实 Judge
通过单独的 `--llm-judge` 实验运行。

## 10. 已完成实验

详见 [实验报告](./artifacts/final/experiment-report.md)。最终目录包含：

- `final/baseline`：60/60 严格成功，Gate PASS；
- `final/degraded`：15/60 严格成功，Gate BLOCKED；
- `final/tau3-adapter`：5/5 适配器契约通过；
- `final/judge`：DeepSeek `deepseek-v4-pro` 实际评审 20/20 完成、0 错误，等待真实人工复核。
