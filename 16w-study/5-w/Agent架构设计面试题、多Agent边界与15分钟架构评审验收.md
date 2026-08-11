# Agent 架构设计面试题、多 Agent 边界与 15 分钟架构评审验收

> 本文对应第五周阶段验收，基于 `source/` 中固定 Workflow、ReAct、Plan-and-Execute 三种实现。重点不是背诵模式名称，而是能从任务结构、控制权、失败传播、指标和恢复条件推导架构。

## 一、验收结论先行

| 验收项 | 当前状态 | 证据或缺口 |
|---|---|---|
| 至少 8 道架构设计题 | 通过 | 本文包含 12 道题及参考答案 |
| 回答为什么不默认多 Agent | 通过 | 第三部分从正确性、协调、安全、成本和恢复分析 |
| 15 分钟架构评审陈述 | 通过：讲稿就绪 | 第四部分按分钟组织，可无笔记演示 |
| 三种实现可运行 | 通过 | `FixedWorkflow`、`ReAct`、`PlanAndExecute` |
| 核心测试 | 通过 | 当前执行结果：`12 passed in 0.16s` |
| 故障场景 | 通过 | 检索失败、工具失败、计划失效均有测试 |
| 实验协议可复现 | 通过 | 统一模型/工具/测试集，默认 3 次，轮换顺序 |
| 真实实验数据已生成 | **待完成** | 当前进程无模型凭证，正式 JSON/Markdown 尚未生成 |

这里必须区分：**代码和实验协议可复现，不等于真实实验已经执行。** 在 `architecture-comparison.json` 生成并通过校验前，不能宣称“三种架构的真实成功率、Token 和延迟已验收”。

## 二、12 道 Agent 架构设计题

### 问题 1：拿到一个新任务时，如何判断使用 Single Call、Workflow 还是 Agent？

参考答案：

先定义任务级完成谓词，再建立最低复杂度基线。

1. Retrieval、few-shot 和结构化输出增强后的单次调用能满足成功率与安全 SLO，就停在 Single Call；
2. 如果步骤、分支和顺序可以事前枚举，使用 Workflow；
3. 如果所需步骤、工具或顺序依赖运行时 Observation，才考虑 Agent；
4. 如果没有可靠环境反馈、完成条件或可逆动作，不应该升级自治；
5. Agent 必须用评测证明成功率增益覆盖延迟、成本和新增故障面。

关键判断不是“是否用了多个模型”，而是下一步控制策略由代码还是模型决定。

```text
Workflow: next = transition_table[state, event]
Agent:    action ~ model(goal, state, observation, authorized_tools)
```

权限、预算、Schema、副作用确认和硬终止条件始终留在 Runtime。

### 问题 2：如何为受监管客服设计固定 Workflow？

参考答案：

适合把流程做成显式状态机：

```text
authenticate
  → classify_intent
  → retrieve_policy_with_ACL
  → fetch_authoritative_order_state
  → generate_proposal
  → policy_gate
  → dry_run
  → user_confirmation
  → idempotent_commit
  → reconciliation
```

每个节点定义输入输出 Schema、超时、可重试错误、不可重试错误和补偿动作。LLM 可以负责意图分类、政策解释和回复生成，但不能跳过身份验证、授权、确认或对账。

完成条件必须是订单/工单等业务状态满足谓词，而不是模型说“处理完成”。写操作使用幂等键；超时后状态未知时先查询提交状态，不盲目重放。

### 问题 3：ReAct 为什么适合开放检索？生产化时需要补什么？

参考答案：

开放检索无法预知需要几轮查询和哪些资料。ReAct 使用：

```text
State → Action(search/read) → Observation → State update
```

早期证据可以改变下一次查询，失败 Observation 也能触发改写、重试或停止。

生产化需要补充：

- Typed State，区分假设、已验证事实、待办和已提交动作；
- 最大步骤、Token、费用和全局 deadline；
- Tool Schema、allowlist、Scope 和超时；
- Observation 的来源、版本、时效和完整性；
- Artifact 外置和 Context Compaction；
- 基于证据的完成谓词；
- Checkpoint、取消传播和结构化恢复。

ReAct 的价值来自外部校正，不是来自更长的自然语言 Thought。

### 问题 4：Plan-and-Execute 遇到计划失效应该怎么办？

参考答案：

计划必须携带假设、输入 Artifact 版本和重规划触发器。执行器发现 `expected_version != current_version` 时：

1. 停止消费旧计划的后续步骤；
2. 标记 `plan_invalidated`，记录旧计划版本和触发证据；
3. 废弃或重新验证旧版本证据；
4. 用新版本状态调用 Planner；
5. 生成新 `plan_version`；
6. 从安全 Checkpoint 恢复，而不是简单修改版本号继续旧步骤。

本项目的 Plan-and-Execute 会在语料版本变化后清空旧证据、重新规划并重新检索，避免“新版本号执行旧计划”的伪恢复。

### 问题 5：如何保证三种架构实验可比？

参考答案：

控制变量必须包括：

- 同一模型和参数；
- 同一 Tool Schema 和实现；
- 同一语料及版本；
- 同一测试集和 Gold；
- 同一身份、权限和预算；
- 同一成功评分器；
- 同一超时和错误语义；
- 同一指标采集方式。

只允许控制策略不同。每种架构至少运行 3 次，以暴露非确定性；本项目还按 repetition 轮换执行顺序，减少时间段、缓存和服务负载偏差。

报告失败轨迹的 Token 和延迟，避免只统计成功样本产生幸存者偏差。模型版本、运行时间、依赖版本、数据校验和也必须随报告保存。

### 问题 6：检索失败、工具失败和计划失效如何统一建模？

参考答案：

不要让工具抛出无法判断的大段异常，统一返回结构化错误：

```json
{
  "status": "error",
  "tool": "search_documents",
  "call_id": "call_...",
  "error_type": "retriever_unavailable",
  "retryable": true,
  "corpus_version": 1
}
```

三类错误的恢复语义不同：

| 错误 | 能否直接重试 | 正确动作 |
|---|---:|---|
| `retriever_unavailable` | 是，有界 | 退避重试或降级检索 |
| `tool_unavailable` | 视幂等性 | 读操作重试；写操作先查状态 |
| `plan_invalidated` | 否 | 取得新状态并重规划 |
| `invalid_arguments` | 否，原参数 | 修正 Schema 字段后再调用 |
| `permission_denied` | 否 | 请求授权或终止 |

错误类型决定恢复状态转移，而不是统一 `retry=3`。

### 问题 7：Agent 如何判断真正完成？

参考答案：

区分模型停止、循环终止和业务完成：

- 模型停止：一次生成结束；
- 循环终止：完成、拒绝、预算耗尽、超时、取消或失败；
- 业务完成：外部世界满足完成谓词。

研究任务的完成谓词可以是：必需结论存在、Gold 证据覆盖、引用可追溯、冲突证据已处理、输出 Schema 合法。代码任务则要求测试通过和静态检查满足约束。

模型返回 `final` 只是完成候选，Runtime 必须再次校验。如果不满足，应继续、降级或以 `failed/pending` 结束，不能把自然语言信心当证据。

### 问题 8：怎样设计 Agent 的预算和提前终止？

参考答案：

预算至少包括：

```text
max_steps
max_input_tokens / max_output_tokens
max_cost
global_deadline
per_tool_timeout
max_retries_by_error_type
max_parallel_workers
```

调用前预留预算，调用后按 usage 结算。接近软阈值时减少候选、压缩上下文或切换低成本路径；达到硬阈值时停止新动作并返回结构化状态。

提前终止条件包括：完成谓词满足、证据不足且继续搜索边际收益低、连续无进展、同一状态循环、不可恢复权限错误、用户取消。还要检测 `A→B→A` 轨迹振荡和重复工具参数哈希。

### 问题 9：什么时候值得使用多 Agent？

参考答案：

只有当任务具备真实并行或隔离价值：

- 子任务相互独立，能并行缩短关键路径；
- 需要不同工具、权限、Context 或专业模型；
- 单个 Context 无法容纳全部资料，需要隔离信息域；
- 需要独立生成与评价，且误差相关性足够低；
- 动态子任务事前未知，Orchestrator 分解有可测收益。

同时必须定义 Worker Contract、作用域、预算、依赖图、结果 Schema、冲突处理和取消传播。如果多个“Agent”只是使用同一模型、同一 Prompt、同一数据做相似工作，它们通常只是昂贵且高度相关的重复采样。

### 问题 10：如何设计多 Agent 的 Orchestrator-Worker Contract？

参考答案：

任务包至少包含：

```text
task_id / parent_trace_id
objective / non_goals
allowed_scope / tenant / scopes
input_artifact_ids + versions
dependencies
deadline / token / cost budget
expected_output_schema
completion_predicate
conflict_policy
```

Orchestrator 负责分解覆盖率、依赖和预算；Worker 只在最小作用域内执行；Aggregator 验证来源、去重和冲突。不能把 Worker 的自然语言答案直接拼接成最终事实。

动态 Fan-out 必须有全局上限。父任务取消时，要停止未开始 Worker、向在途 Worker 传播取消，并处理已发生的副作用。

### 问题 11：Agent 的 Checkpoint 应保存什么？

参考答案：

Checkpoint 不是完整 Conversation History 的复制。它应保存恢复所需的最小结构化状态：

```text
goal / constraints
plan_version / current_milestone
verified_facts + evidence_versions
pending_tasks / blockers
budget_remaining
committed_side_effects
idempotency_keys
failed_attempts_not_to_repeat
authorization_context_reference
```

恢复时重新验证身份、权限、证据时效和外部提交状态。旧 Checkpoint 中的 Access Token 不应持久化，也不能假设外部世界仍与暂停时一致。

### 问题 12：如何从高自治 Agent 安全降级？

参考答案：

在设计阶段就保留降级路径：

```text
Autonomous Agent
  → bounded ReAct
  → Plan-and-Execute with fixed gates
  → deterministic Workflow
  → human handoff
```

触发条件可以是成功率下降、p95 延迟或 Token 越界、未知失败率上升、恢复率下降、权限异常、模型/工具版本变化。

降级不是简单换 Prompt，需要保持统一输入输出契约、Artifact 格式和业务完成谓词，使上层调用方不依赖某种内部控制策略。高风险写操作无论在哪一级都不应取消 Runtime 门禁。

## 三、为什么不应该默认使用多 Agent

### 3.1 多 Agent 增加的是系统边界，不只是“更多智能”

每增加一个 Agent，都会新增：

- 一个概率决策点；
- 一份可能不完整的局部 Context；
- 一组身份和权限边界；
- 一条消息序列化与反序列化链路；
- 一份预算、超时和取消状态；
- 一个需要 Trace、Checkpoint 和恢复的执行单元。

因此多 Agent 是分布式系统设计问题，而不是把同一个 Prompt 复制 N 次。

### 3.2 正确率不会自然随 Agent 数增加

粗略写成：

```text
P(success)
≈ P(decomposition correct)
 × Π P(worker_i correct | assignment_i)
 × P(aggregation correct | worker outputs)
```

该式不是严格独立模型，但揭示了新增失败面：任务可能分错、Worker 可能失败、Aggregator 可能误合并。Worker 共享模型和数据时错误高度相关，多数投票的独立误差假设不成立。

如果单 Agent 成功率已经足够，多 Agent 可能因为协调错误让端到端成功率下降。

### 3.3 协调开销可能超过并行收益

```text
C_multi = C_orchestrator + ΣC_worker + C_aggregate + C_retry
L_multi ≈ L_orchestrator + max(L_worker) + L_aggregate
```

只有独立 Worker 的并发节省超过分解与聚合成本，墙钟时间才下降。最慢 Worker 决定尾延迟；总 Token 和工具费用通常增加。

有数据依赖的任务不能通过并行化消除关键路径。强行拆分会让 Worker 猜测上游结果，后续再返工。

### 3.4 Context 被切碎后会产生信息不对称

Worker 只看到局部信息有利于隔离，也可能遗漏全局约束。常见问题：

- 两个 Worker 使用不同版本的需求；
- 一个 Worker 不知道另一个已提交副作用；
- 局部优化互相冲突；
- 摘要传递时把假设变成事实；
- Aggregator 无法判断哪个来源更新或更权威。

必须用版本化 Artifact 和明确依赖传递状态，而不是靠自然语言转述。

### 3.5 权限与数据泄露面被放大

默认把父 Agent 的全部 Token、工具和 Context 复制给 Worker，违反最小权限。正确做法是每个 Worker 获得完成子任务所需的最小 Scope 和数据分区，Token 不进入模型上下文。

Agent 越多，越需要处理跨租户隔离、Delegated Token、撤销、审计和 Prompt Injection 传播。没有成熟授权代理时，多 Agent 会扩大而不是降低风险。

### 3.6 可观测性和恢复更困难

单循环可以沿一条 Trace 定位状态转移；多 Agent 需要因果关联：

```text
root_trace
  → orchestration_span
      → worker_trace_A
      → worker_trace_B
  → aggregation_span
```

还要回答：部分 Worker 成功时是否提交、父任务取消如何传播、Worker 重试是否重复副作用、Aggregator 失败能否复用已有结果。

### 3.7 默认策略

推荐复杂度阶梯：

```text
Single Call
  → Fixed Workflow
  → Single Agent with bounded loop
  → Parallel workers for proven independent tasks
  → Multi-Agent only with measured coordination benefit
```

默认不使用多 Agent，不是因为它永远无效，而是因为复杂度必须由可测的任务分解、并行、隔离或专业化收益来支付。

## 四、15 分钟架构评审陈述

下面是一份可以直接口述的评审稿。记忆主线：

```text
Problem → Control → Three Architectures → Failures → Fair Eval → Decision → Reproduce
```

### 0:00—1:30：问题和成功标准

“本阶段研究的问题不是哪种 Agent 架构最先进，而是在同一个研究任务上，控制策略如何影响成功率、步骤、工具调用、Token、延迟和失败恢复。

任务是为受监管客服形成带证据的架构建议。完成必须同时满足必需技术结论和 Gold 来源引用。越权、无证据完成或缺失必需来源均计为失败。”

展示：`data/scenarios.json` 的问题、`required_terms` 和 `required_sources`。

### 1:30—3:00：控制变量和边界

“为了避免不可比，三种实现共享同一个 DeepSeek 模型实例、Temperature、工具 Schema、语料、测试集、最大步骤和评分器。唯一变化是控制流。

工具只有搜索和读取。模型看见 Tool Schema，但执行由 Runtime 负责。Runtime 返回 Tool Call ID、结构化错误和语料版本。模型无法通过文本改变权限、预算或错误状态。”

强调：同模型不等于确定性，所以每个场景至少运行 3 次，并轮换架构执行顺序。

### 3:00—5:30：三种实现

“固定 Workflow 由代码执行 `search → read → synthesize`。检索失败重试、读取失败重试、版本变化重新检索，全部是预定义状态转移。它的优势是步骤和恢复可预测。

ReAct 每一轮让模型根据 Observation 选择搜索、读取或完成。Runtime 强制最大步骤。它能动态恢复，但模型调用和上下文增长更多，也可能提前完成或重复动作。

Plan-and-Execute 先生成查询计划，再执行。发现 corpus version 改变时，执行器停止旧计划、清除旧证据、调用 Replanner 并重新检索。它在全局覆盖和适应变化之间折中。”

现场打开：[architectures.py](./source/architectures.py)。

### 5:30—7:30：三类失败为什么不同

“我构造三类失败：第一次检索返回 `retriever_unavailable`；第一次读取返回 `tool_unavailable`；首次读取时 corpus version 从 1 变为 2，返回 `plan_invalidated`。

前两类是暂时可重试错误；计划失效不是把旧请求重试一次，而是旧假设不再成立，必须重新获取状态和规划。所有架构面对相同错误结构，所以恢复差异来自控制策略，而不是工具偷偷给某个架构更容易的输入。”

展示：[tools.py](./source/tools.py)。

### 7:30—9:30：指标和 Trace

“每条轨迹记录 success、steps、tool_calls、model_calls、input/output tokens、wall-clock latency、failure_types 和完整 Span。

成功率是任务级 Gold 条件，不是 LLM Judge 自评。Token 使用 API usage；失败轨迹也进入平均值。延迟同时报告平均值和 p95。这样能看到 ReAct 是否用更多步骤换来恢复，Plan-and-Execute 是否因 Planner 和 Replanner 增加 Token，以及 Workflow 是否在复杂失败中失去适应性。”

补充：正式比较还应报告置信区间；3 次是最低验收，不是严谨统计上限。

### 9:30—11:00：为什么不默认多 Agent

“这个实验故意先比较单控制器架构，而不默认多 Agent。多 Agent 新增分解、Worker、聚合和跨 Agent 状态一致性，每层都可能失败。

它只有在子任务真正独立、能并行，或需要不同权限、Context 和专业工具时才值得。否则相同模型的多个 Worker 错误高度相关，既不能获得可靠投票增益，又增加 Token、尾延迟、权限面和恢复难度。”

给出一句结论：**先证明任务需要分布式控制，再引入多 Agent。**

### 11:00—12:30：如何读架构对比表

“排序先看硬约束，再看 Pareto，而不是简单平均。权限泄露或未确认副作用不能被更高成功率抵消。

在满足安全约束的方案中，先按成功率，再看 p95 和平均 Token。报告还按 normal、retrieval failure、tool failure、plan invalidated 拆分，防止平均值掩盖某一类失败。”

当前状态必须如实说明：“本地控制流测试为 12 passed；当前进程尚未运行真实 DeepSeek 36 轨迹，因此此时只评审实验设计，不能展示伪造的架构赢家。”

### 12:30—14:00：推荐决策树

“如果路径可预定义，优先 Single Call 或固定 Workflow。如果路径未知，但有可靠环境反馈，才选择有界 ReAct；如果任务还需要全局依赖和里程碑，选择 Plan-and-Execute 或 Rolling Horizon。

任何不可逆动作都经过 Runtime 确认、幂等和 Checkpoint。如果 Agent 的成功率增益不能覆盖 p95、Token 和恢复复杂度，就降级到 Workflow。”

### 14:00—15:00：复现与结束

“复现分三层：先跑 12 个核心测试；再锁定模型、依赖和数据校验和；最后运行 36 条真实轨迹，保存 JSON 原始 Trace 和 Markdown 汇总。

我的架构结论不是永恒排名，而是绑定任务集、模型版本、工具版本和预算的实验结果。任何一项变化都应重新运行，而不是沿用旧赢家。”

结束语：**选择最简单且满足硬约束的架构，用真实失败和任务级数据证明复杂度。**

## 五、评审现场可能被追问的问题

### 为什么只运行 3 次？

3 次是本阶段最低验收，用于暴露非确定性，不足以形成窄置信区间。生产决策应扩大到 20—100 次或根据效应量做样本量估计，并按场景分层报告置信区间。

### 为什么固定 Workflow 也会调用模型？

Workflow 与 Agent 的边界是控制流归属，不是是否调用模型。固定 Workflow 的模型负责综合内容，代码决定检索、读取、恢复和终止路径。

### ReAct 的步骤和 Workflow 的步骤可直接比较吗？

必须统一口径。本项目把步骤定义为控制状态推进次数，包括恢复步骤；同时单独报告模型调用和工具调用，避免一个混合指标掩盖结构差异。

### Temperature=0 是否代表确定性？

不代表。服务端实现、并行、模型版本和浮点计算仍可能造成差异，所以需要重复运行和保存原始输出。

### 为什么不用 LLM Judge？

当前任务有可声明的必需术语和 Gold 来源，优先确定性评分以避免 Judge 偏差。若评估论证质量，可增加经过人工标注校准的 Judge，但不能替换权限、引用和完成谓词。

## 六、三种实现与实验数据复现手册

### 6.1 环境

```bash
source /Users/shiyiliu/workspace/pyproject/.venv/bin/activate
cd /Users/shiyiliu/workspace/pyproject/16w-study/5-w/source

python --version
python -m pip show openai pytest
```

正式归档时应保存完整版本快照：

```bash
python -m pip freeze > artifacts/environment.lock.txt
```

### 6.2 数据完整性

```bash
shasum -a 256 data/corpus.json data/scenarios.json > artifacts/data.sha256
shasum -a 256 architectures.py tools.py model_client.py run_experiment.py > artifacts/code.sha256
```

报告必须绑定这些校验和。否则两次运行即使文件名相同，也可能使用了不同语料或控制逻辑。

### 6.3 核心测试

```bash
python -m pytest -q -p no:cacheprovider
```

当前已验证：

```text
............
12 passed in 0.16s
```

测试覆盖：

- 正常检索和版本返回；
- 检索失败后恢复；
- 工具失败的结构化错误和恢复；
- 计划版本失效；
- 三种架构正常完成；
- ReAct 检索失败恢复；
- Plan-and-Execute 重规划；
- 少于 3 次被拒绝；
- 三种架构运行数一致；
- 报告包含对比表和决策树。

### 6.4 真实模型实验

凭证只放进进程环境，不能写入源码、Trace 或 Markdown：

```bash
export DEEPSEEK_API_KEY="你的 Key"
export OPENAI_BASE_URL="https://api.deepseek.com"
export AGENT_TEST_MODEL="deepseek-v4-pro"

python run_experiment.py --repeats 3
```

预期规模：

```text
4 个场景 × 3 次重复 × 3 种架构 = 36 条轨迹
```

### 6.5 输出检查

```bash
test -s artifacts/architecture-comparison.json
test -s artifacts/architecture-comparison.md

python - <<'PY'
import json
from pathlib import Path

report = json.loads(Path("artifacts/architecture-comparison.json").read_text())
assert report["model"] == "deepseek-v4-pro"
assert report["repeats_per_scenario"] >= 3
assert report["scenario_count"] == 4
assert len(report["runs"]) == report["scenario_count"] * report["repeats_per_scenario"] * 3
assert set(report["summary"]) == {"fixed_workflow", "react", "plan_and_execute"}
assert all(run["input_tokens"] >= 0 and run["latency_ms"] > 0 for run in report["runs"])
print("report validation passed")
PY
```

### 6.6 架构对比表的正式字段

真实运行后，Markdown 自动生成：

| 架构 | 运行数 | 成功率 | 平均步骤 | 平均工具数 | 平均模型数 | 平均 Token | 平均延迟 | p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_workflow | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 |
| react | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 |
| plan_and_execute | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 |

这里保留“待运行”是刻意的：当前没有真实模型 Artifact，不能用单元测试替身的固定 Token 和 2 ms 延迟填表。

### 6.7 复现合格标准

| Gate | 通过条件 | 当前状态 |
|---|---|---|
| 代码 | 三种架构从相同接口运行 | 通过 |
| 工具 | Schema、语料、故障注入一致 | 通过 |
| 测试 | 核心测试无失败 | 通过，12 passed |
| 重复 | 每场景每架构至少 3 次 | 代码强制；真实运行待执行 |
| 原始数据 | 36 条真实轨迹 JSON | 待执行 |
| 汇总 | 对比表、分场景成功率、决策树 | 生成器通过测试；真实报告待执行 |
| 环境 | 模型、依赖、代码和数据版本可追溯 | 命令已定义；正式归档待执行 |
| 安全 | 报告不包含 API Key | 通过设计约束，运行后仍需扫描 |

## 七、最终记忆框架

面试或评审时只需记住八个词：

```text
目标 → 控制权 → 证据 → 失败 → 预算 → 指标 → 恢复 → 复现
```

任何架构问题都按这八项回答：

1. 业务完成条件是什么？
2. 谁决定下一步？
3. 事实从哪里来？
4. 错误怎样传播？
5. 最多消耗多少资源？
6. 用什么数据证明更好？
7. 失败后怎样安全继续或降级？
8. 别人能否在同一版本上重现结论？

能回答这八项，才是在做 Agent 架构设计，而不是在列框架名称。

