# τ³ banking_knowledge 适配子集来源

- 上游仓库：<https://github.com/sierra-research/tau2-bench>
- 固定版本：`v1.0.1`
- 领域：`banking_knowledge`
- 本地子集：`task_001` 至 `task_005`
- 下载日期：2026-08-14

`raw/` 保留上游 JSON，不修改字段。`benchmark_adapter.py` 将它们转换为本项目
统一的 Task/Trial/Grader Schema。当前运行使用 Gold Action contract replay 来验证
适配器和评分合同，不是 Agent 能力成绩，也不可与官方排行榜比较。
