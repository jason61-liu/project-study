# 第 7 周编码实验：四种 Agent 执行架构对比

本目录把第 6 周的“结构化计划、失败恢复、幂等执行和 Manager/Specialist 消融”继续推进为四种可比较实现：

| 实现 | 文件 | 本实验关注点 |
|---|---|---|
| 原生基线 | `native_baseline.py` | 手写状态机、JSON 状态恢复和幂等提交 |
| LangGraph | `langgraph_workflow.py` | StateGraph、SQLite Checkpoint、Interrupt、Resume |
| OpenAI Agents SDK | `agents_sdk_workflow.py` | Tool、Handoff、Guardrail、原生工具审批中断 |
| Deep Agents | `deep_agents_workflow.py` | 一个有界研究 Subagent 和上下文隔离 |

## 1. 统一业务任务

四种版本处理同一任务：检索四份本地证据，解释 Runtime、Checkpoint、幂等审批、Subagent 与上下文隔离，然后保存并发布报告。

所有版本共享：

- `data/corpus.json` 中的相同语料；
- `common.ResearchToolRuntime` 中相同的搜索、读取、保存和发布工具；
- 相同 `ResearchTask` 输入；
- 相同完成谓词 `score()`；
- 相同 `TraceEvent` 与 `RunReport` 指标结构。

这样实验变量是编排方式，而不是工具质量或评分器差异。

## 2. 状态恢复与幂等为何分开

LangGraph 使用两个 SQLite 文件：

```text
checkpoints.sqlite
  保存 Graph State、下一节点和 Interrupt

business.sqlite
  保存草稿、审批 submission_id 和发布幂等键
```

Checkpoint 让程序知道“恢复后从哪里继续”，但不能保证外部发布只发生一次。`ArtifactLedger` 使用数据库唯一约束保证：

```text
同一个 submission_id 只能消费一次审批
同一个 run_id:publish:v1 只能生成一个 publication_id
```

## 3. 人工审批流程

LangGraph 路径：

```text
search → read → draft → interrupt
                         │
          Command(resume={decision})
                         │
                 approve → publish → END
                 reject  → END
```

Agents SDK 路径使用 `@function_tool(needs_approval=True)`。Runner 在调用发布工具前返回 `interruptions`；应用将 `RunState.to_json()` 持久化，审批后调用 `state.approve()` 或 `state.reject()`，再把 `RunState` 交回 `Runner.run()`。

正式实验会在审批前丢弃原 `AgentsSDKWorkflow` 对象，再从磁盘上的 RunState、元数据和
SQLite 账本重建 Agent 与工具运行时，因此覆盖的不是“同一进程暂停”，而是可复现的
“进程内对象全部丢失后恢复”路径。

## 4. Deep Agents 的实验边界

Deep Agents 版本只验证一个变量：主 Agent 将检索交给 `evidence-researcher` 子 Agent，子 Agent 使用独立上下文并只返回证据结果。

父上下文放入 canary `PARENT_ONLY_7W`，委派任务禁止携带它。结果满足以下条件才认为隔离验证成功：

- 实际工具事件的 actor 是 `research_subagent`；
- 最终结果不含 canary；
- 最终答案仍包含所需证据 ID。

它不重复实现 LangGraph 已经验证的审批和恢复，避免把 Harness Demo 扩成另一套完整框架。

## 5. 环境与依赖

必须使用项目虚拟环境：

```bash
source /Users/shiyiliu/workspace/pyproject/.venv/bin/activate
python -m pip install -r 7-w/source/requirements.txt
```

真实模型通过 OpenAI-compatible 客户端调用 DeepSeek：

```bash
export DEEPSEEK_API_KEY='由本机密钥管理器提供，不要写入仓库'
export OPENAI_BASE_URL='https://api.deepseek.com'
export AGENT_TEST_MODEL='deepseek-v4-pro'
```

代码不会把 Key 传给模型、Trace 或工具参数。

## 6. 运行测试

```bash
source /Users/shiyiliu/workspace/pyproject/.venv/bin/activate
python -m pytest -q -o cache_dir=/tmp/week7-pytest-cache 7-w/source/tests
```

测试覆盖：

- 在发布前正确中断；
- LangGraph 新进程对象从 SQLite Checkpoint 恢复；
- 重复审批 submission 不重复发布；
- 审批拒绝后不发布；
- 读取工具异常后有界重试；
- 未批准时工具 Runtime 拒绝发布；
- 未知工具返回结构化错误；
- Agents SDK 和 Deep Agents 使用真实框架入口。

测试不 Mock 网络模型。恢复、审批和幂等属于确定性协议，直接运行真实 Runtime、SQLite 和工具；真实模型路径由下面的实验命令执行。

## 7. 运行真实四架构实验

单次冒烟：

```bash
source /Users/shiyiliu/workspace/pyproject/.venv/bin/activate
python 7-w/source/run_comparison.py --repeats 1
```

三次重复实验：

```bash
python 7-w/source/run_comparison.py \
  --output 7-w/source/artifacts/final \
  --repeats 3
```

产物写入 `--output` 指定的目录：

- `architecture-comparison.json`：完整轨迹和逐次指标；
- `architecture-comparison.md`：成功率、步骤、工具数、模型数、Token 和延迟对比；
- `runs/`：Checkpoint 和业务幂等账本。

运行器拒绝复用已经存在的 `runs/` 目录，防止旧 Checkpoint、幂等缓存或发布收据
污染新一轮数据。复现实验时应指定一个全新的 `--output` 路径。

## 8. 指标解释

| 指标 | 口径 |
|---|---|
| 成功率 | 完成状态、必需术语和必需引用同时满足 |
| 步骤数 | 统一 Trace 中记录的 Model/Tool/Control/Approval/Subagent Span 数 |
| 工具数 | `kind=tool` 的 Span 数，含失败尝试 |
| 模型数 | `kind=model` 的 Span 数 |
| Token | SDK/Provider 可观测的输入与输出 Token；确定性基线为 0 |
| 延迟 | 每个运行入口到结果返回的墙钟时间 |

原生和 LangGraph 版本使用确定性合成，目的是隔离并比较恢复机制，因此不能把它们的 Token 为 0 解读为同等生成质量下更省 Token。Agents SDK 与 Deep Agents 使用真实模型。正式结论至少运行三次，并报告波动。

## 9. 已知环境注意事项

安装 Deep Agents 0.7.5 后，当前共享虚拟环境中的 `e2b 2.35.0` 与 `wcmatch 11.0`、`vllm 0.13.0` 与 `anthropic 0.121.0` 存在依赖声明冲突。本实验不导入 `e2b` 或 `vllm`，未为此修改无关包。若后续需要同时运行它们，应新建独立虚拟环境或重新求解统一依赖锁文件。

## 参考资料

- [LangGraph Persistence](https://docs.langchain.com/oss/python/langgraph/persistence)
- [LangGraph Interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts)
- [OpenAI Agents SDK — Running agents](https://developers.openai.com/api/docs/guides/agents/running-agents)
- [Deep Agents overview](https://docs.langchain.com/oss/python/deepagents/overview)
