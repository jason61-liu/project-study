# 第 8 周 Agent Eval 实验结果

## 1. 固定条件

- Task Schema：`1.0`
- 本地任务：60 条，版本均为 `1.0.0`
- 切片：正常、边界、失败、对抗各 15 条
- 并发数：本地 8，τ³ 适配子集 5
- 基线 Agent：`reference-agent@1.0.0`
- 劣化变更：`degraded-prompt@2.0.0`
- Gate：整体成功率至少 95%，正常/失败/对抗切片至少 95%，边界至少 90%，相对基线下降不超过 2 个百分点

## 2. 实际运行结果

| 运行 | Task | 严格成功率 | normal | boundary | failure | adversarial | Gate |
|---|---:|---:|---:|---:|---:|---:|---|
| baseline | 60 | 100% | 100% | 100% | 100% | 100% | PASS |
| degraded Prompt/architecture | 60 | 25% | 0% | 100% | 0% | 0% | BLOCKED |
| τ³ adapter contract replay | 5 | 100% | — | — | — | — | 不作为产品 Gate |

劣化版本被阻断的原因：

1. 严格成功率 25% 低于 95%；
2. 相对基线下降 75 个百分点，超过允许的 2 个百分点；
3. 正常、失败和对抗切片均为 0%；
4. 丢失引用、把工具失败说成成功、执行提示注入、跨租户读取和无 Scope 写入均被确定性 Grader 捕获。

## 3. τ³ 结果边界

本地固定的是官方 `tau2-bench` 仓库 `v1.0.1`、`banking_knowledge` 的
`task_001` 至 `task_005`。运行使用显式 Gold Action contract replay，仅证明：

- 原始 Task 能转换为统一 Schema；
- `reward_basis` 和 Gold Actions 没有在适配中丢失；
- Harness 能并发执行、保存 Trace 并运行 `actions_match` Grader。

该结果没有运行官方用户模拟器和完整知识检索环境，因此不是 Agent 能力成绩，
也不可与 τ³ 官方排行榜比较。

## 4. LLM Judge 与人工复核状态

经用户明确授权后，已使用 `deepseek-v4-pro` 实际评审 20 条 `llm_rubric` Task：

- `judge_requested = true`；
- `judge_expected = 20`；
- `judge_completed = 20`；
- `judge_errors = 0`；
- LLM Rubric Grader 通过率为 100%，平均得分 0.995；
- Judge 输入 8,283 Token，输出 4,531 Token；
- 60 条 Trial 的平均端到端延迟约 1,464 ms（并发数 4）。

评审请求包含任务输入、可信上下文、Agent 输出和 Rubric；API Key 只存在于临时
进程环境与 HTTP Authorization Header，没有写入任务、Trace 或结果文件。

`review-queue.json` 固定了 20 条待复核样本。真实人工结论尚未填写，因此 Judge
运行的 CI Gate 当前按 fail-closed 原则阻断，不能把模型复核或确定性 Grader
冒充“人工复核”。复核人完成 `human-reviews.json` 后，Gate 才会检查误判率是否
不超过 10%。
