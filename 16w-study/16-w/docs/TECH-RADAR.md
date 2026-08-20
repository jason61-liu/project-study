# 技术雷达：产品 → 机制 → 条件 → 证据

下次季度复核：2026-11-20。当前结论来自本仓库第 4、7、9、12、13、14 周材料与第 16 周本地实验；产品版本可能变化，复核时需重新查看官方文档和固定版本。

| 产品/类别 | 核心机制 | 适用条件 | 需要的证据 / 当前结论 |
|---|---|---|---|
| DeepSeek V3/R1 系列 | MLA、MoE、FP8、MTP；推理模型训练与蒸馏 | 成本敏感的推理/研究，需要可部署快照 | 固定模型、任务集、TTFT/TPOT、质量和成本；当前未接入 |
| Kimi K2/K2.5/k1.5/Linear | MoE、Agent 能力、RL、混合线性注意力路线 | 长上下文或工具研究，且语言/上下文质量达标 | 与 DeepSeek 在同 Harness 做质量/成本/长上下文比较；当前未接入 |
| vLLM | PagedAttention、连续批处理、OpenAI-compatible Serving | 通用高吞吐、自托管模型 | 并发扫描、Goodput、KV 命中与尾延迟；候选 |
| SGLang | Radix/Prefix Cache、结构化生成与运行时优化 | 前缀复用高、复杂生成程序 | 与 vLLM 同硬件同请求分布 A/B；候选 |
| LMCache | 跨请求/层级 KV 复用与传输 | 长前缀、高复用，传输成本低于重算 | 命中率、有效命中、传输延迟、一致性；暂不引入 |
| Agent Harness | Loop、Tool、预算、Checkpoint、Trace | 任意需要可控工具执行的 Agent | 轨迹/Outcome、恢复、幂等、安全 Gate；本项目采用自建最小 Harness |
| LangGraph / Deep Agents | 图状态、Checkpoint；文件上下文与子 Agent | 动态分支、长任务或多 Agent收益已证明 | 与单工作流消融；当前双路代理无收益，不采用 |
| Mem0 | 事实抽取、检索、更新与删除层 | 明确存在跨会话个性化价值 | 抽取准确率、召回、污染、租户删除；当前自建无跨任务记忆基线 |
| E2B/OpenSandbox 类 | 隔离执行、资源/网络/Secret 策略、销毁 | 执行不可信代码或 Shell | 逃逸、断网、文件、Secret、限额、冷启动与销毁测试；正式上线前必选 |

