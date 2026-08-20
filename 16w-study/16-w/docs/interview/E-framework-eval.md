# E. 框架、评测与可观测性（61–75）

61. State 是图中共享且类型化的数据；Node 读取状态并返回更新；Edge 决定控制流；Reducer 定义并发或多次更新如何合并同一字段。
62. Checkpoint 持久化节点位置、状态版本和已完成效果；HITL 在中断点保存后等待外部输入；恢复从快照继续；Time Travel 从旧快照派生新分支，但不能撤销已发生的外部副作用。
63. Interrupt 是设计内的暂停，状态可恢复且等待特定输入；异常表示非预期失败，需分类、重试或失败终止。不能用捕获异常冒充安全审批。
64. Runner 驱动循环，把 Agent 指令/模型、Tools、Handoffs 和 Guardrails 组合起来；Guardrail 在输入/输出或工具边界执行，Trace 记录每个决策和调用。
65. 按控制流可表达性、Checkpoint/HITL、工具生态、上下文隔离、多 Agent、Tracing、部署依赖、模型绑定和团队熟悉度选，不按 Demo 代码长度选；先用最小业务 Eval 做原型对比。
66. 一个用户请求形成 Trace，Workflow/Agent Span 包含 Model、Retrieval、Tool 子 Span，异步恢复用 context propagation 或 span link。默认不采集 Prompt、系统指令、工具正文、文档、PII 和凭证，只记录版本、Token、延迟、错误和策略证据。
67. Task 是评测合同；Trial 是一次运行；Grader 判断 Trace/Outcome/答案；Trace 是过程；Outcome 是真实终态；Harness 批量隔离运行、收集和聚合。
68. Capability Eval 探索系统能否完成目标和主要失败边界，会持续演化；Regression Eval 固定已知能力与失败样本，用硬门槛防版本倒退。
69. 模型采样、工具网络和环境都非确定，同一任务结果会波动。多 Trial 才能估计成功率分布、置信区间和尾部失败，而不是相信一次幸运结果。
70. 先用代码检查 Schema、安全、引用和 Outcome 硬条件；模型 Judge 评开放语义；人工校准高风险与分歧样本。三者按确定性、成本和风险分层。
71. Judge 有位置、长度、风格、自偏好、提示注入和版本漂移偏差。用盲评、顺序交换、锚点、多人标注集、模型快照和分歧抽检校准。
72. Agent 可能越权、重复副作用、超预算、读取跨租户数据或伪造完成，碰巧给出正确文本；轨迹和 Outcome 是安全与业务正确性的一部分。
73. Grader 检查业务不变量和可观察 Outcome，不绑定具体调用次数、内部节点名或措辞；只有安全/费用等必要轨迹约束才做硬断言。
74. BFCL 偏函数调用格式与选择；tau-bench 偏有状态工具交互和策略；GAIA 偏现实多步研究；业务集验证本地价值与风险。固定版本、隔离训练/测试、记录模型污染声明，并避免用榜单替代业务 KPI。
75. PydanticAI 强类型输出和依赖注入适合 Python 类型驱动团队；AutoGen 强多 Agent 对话；Google ADK 面向其 Agent/工具/部署生态；LangGraph 强显式状态图与恢复。按系统边界和 Eval 选型。

