# 第五周：三种 Agent 架构可比实验

本实验选择“为受监管客服形成带证据的 Agent 架构研究结论”作为统一研究任务，对比：

- `FixedWorkflow`：代码固定执行检索、读取和综合；
- `ReAct`：模型根据 Observation 逐步选择工具或完成；
- `PlanAndExecute`：模型先规划，执行器按计划运行，语料版本改变时重规划。

## 可比性约束

三种架构共享：

- 同一个 `DeepSeekResearchModel` 实例及模型参数；
- 同一份本地研究语料；
- 同一套 `search_documents`、`read_document` Tool Schema；
- 同一组正常、检索失败、工具失败和计划失效场景；
- 同一个完成评分器和最大步骤预算；
- 同样的 Trace、Token 和墙钟延迟统计方式。

唯一允许变化的是控制策略。每次 repetition 会轮换三种架构的执行顺序，减少服务时间漂移和缓存顺序造成的偏差。

## 目录

```text
source/
├── architectures.py        # 三种控制策略
├── model_client.py         # 真实 DeepSeek JSON 客户端
├── models.py               # 共享契约、Trace 和指标
├── tools.py                # 统一工具及结构化故障注入
├── run_experiment.py       # 3+ 次运行、聚合、表格和决策树
├── data/
│   ├── corpus.json         # 统一研究资料
│   └── scenarios.json      # 统一测试集与 Gold 条件
├── tests/                  # 12 个控制流和恢复测试
└── artifacts/              # 正式实验输出
```

## 运行测试

```bash
source /Users/shiyiliu/workspace/pyproject/.venv/bin/activate
cd /Users/shiyiliu/workspace/pyproject/16w-study/5-w/source
python -m pytest -q -p no:cacheprovider
```

当前结果：`12 passed`。

测试中的 `TestResearchModel` 只用于验证状态机、故障恢复、指标聚合和报告格式。它明确标记为 `test-only-not-for-benchmark`，不用于正式架构结论。

## 运行真实实验

先在当前 shell 注入凭证，不要把 Key 写入源码或报告：

```bash
source /Users/shiyiliu/workspace/pyproject/.venv/bin/activate
export DEEPSEEK_API_KEY="你的 Key"
export OPENAI_BASE_URL="https://api.deepseek.com"
export AGENT_TEST_MODEL="deepseek-v4-pro"

cd /Users/shiyiliu/workspace/pyproject/16w-study/5-w/source
python run_experiment.py --repeats 3
```

默认执行：

```text
4 scenarios × 3 repetitions × 3 architectures = 36 trajectories
```

输出：

- `artifacts/architecture-comparison.json`：逐轨迹 Trace、工具错误、Token 和延迟；
- `artifacts/architecture-comparison.md`：汇总对比表、分场景成功率和推荐决策树。

少于 3 次会直接失败，避免拿单次非确定性结果下结论。

## 指标口径

| 指标 | 定义 |
|---|---|
| 成功率 | 必需答案术语和 Gold 来源同时满足的轨迹比例 |
| 步骤数 | 架构控制状态推进次数，包含确定性恢复步骤 |
| 工具数 | Tool Runtime 实际执行次数，包括失败与重试 |
| 模型数 | 模型 API 调用次数 |
| Token | 服务端返回的输入、输出及两者总和 |
| 延迟 | 从架构开始到结束的真实墙钟时间 |
| p95 | 同一架构全部场景和重复运行的长尾延迟 |

失败也计入平均 Token 和延迟，防止只统计成功轨迹造成幸存者偏差。

## 故障场景

| 场景 | 注入位置 | 结构化错误 | 预期恢复 |
|---|---|---|---|
| `retrieval_failure` | 第一次搜索 | `retriever_unavailable` | 重试搜索 |
| `tool_failure` | 第一次读取 | `tool_unavailable` | 有界重试/部分降级 |
| `plan_invalidated` | 首次按旧版本读取 | `plan_invalidated` | 获取新版本并重规划/重读 |

故障由同一个 `ResearchToolRuntime` 注入，三种架构面对相同错误语义。模型回复没有被替换或预编排。

## 推荐决策树

```text
任务路径是否可预定义？
├─ 是
│  ├─ 单次调用达到质量 SLO？ ─ 是 → Single Call
│  └─ 否 → Fixed Workflow
└─ 否
   ├─ 是否有可靠且频繁的环境反馈？
   │  ├─ 否 → 先补证据工具，不使用自治 Agent
   │  └─ 是
   │     ├─ 是否需要全局依赖和里程碑？
   │     │  ├─ 是 → Plan-and-Execute / Rolling Horizon
   │     │  └─ 否 → Bounded ReAct
   │     └─ 不可逆动作 → Runtime 确认、幂等、Checkpoint
   └─ 若成功率增益不能覆盖 p95 延迟和 Token → 降级为 Workflow
```

正式报告还会在树末尾加入本次实验的数据优选。该推荐只适用于本任务、模型、语料和预算，不能外推为通用架构排名。

