# 补知识—重答—验证清单

| 优先级 | 不完整主题 | 补知识 | 重答形式 | 验证 |
|---|---|---|---|---|
| P0 | 官方 Benchmark 能力实跑 | 已有 tau3 v1.0.1 Gold contract replay；下一步接真实 Agent/用户模拟器 | 解释契约 Smoke 与能力分数、业务集的差异 | 保存命令、模型快照、原始结果和失败样本 |
| P0 | 真实 OAuth/OIDC | JWKS、轮换、撤销、强认证和 token exchange | 白板画四主体与 Token 流 | 双租户、过期、错 audience、撤销集成测试 |
| P0 | 强隔离 Sandbox | 选定产品的网络/挂载/Secret/销毁语义 | 威胁模型口述 | 逃逸、断网、资源与销毁红队 |
| P1 | OTel 正式 SDK/Collector | 固定 GenAI 语义约定版本 | 画 Trace/Span/Link | Collector 查询和内容泄漏测试 |
| P1 | vLLM/SGLang 实测 | 同硬件、同模型、同请求分布 | 解释 TTFT/TPOT/Goodput 权衡 | 并发扫描和置信区间报告 |
| P1 | PD 解耦与 KV 传输 | Mooncake/Dynamo/llm-d 实验配置 | 容量与路由白板 | 测 KV 传输、Decode 干扰和成本交叉点 |
| P2 | 模型训练细节 | GRPO、DSA、Kimi Linear 公式与实现 | 每题 3 分钟推导 | 从论文图表复现一个小实验或计算 |

复习协议：补完材料后，隔 48 小时无笔记重答；录音转写后按“结论、机制、权衡、证据”四项各 0–2 分，低于 6 分继续下一轮。
