# G. 推理 Serving、优化与训练边界（91–105）

91. TTFT 是首 token 前等待，受排队和 Prefill 主导；TPOT 是后续 token 间隔，反映 Decode；端到端延迟包含全链路；吞吐是单位时间处理请求或 token 数。
92. 吞吐只看完成量，Goodput 只计同时满足 TTFT/TPOT/正确性等 SLO 的有效完成量；拥塞下吞吐可能升而 Goodput 降。
93. 平均值会掩盖少量极慢请求。P50 表典型体验，P95/P99 揭示排队、长输入、GC 或资源争用等尾部风险，容量和 SLO 通常由尾延迟决定。
94. Continuous Batching 在每个迭代动态加入/移出请求，减少 GPU 空洞；高负载或长请求混入会增加排队和干扰，若调度不公平可能恶化 TTFT/TPOT 尾部。
95. PagedAttention 把每请求 KV 分页，减少连续预留和碎片；Radix/Prefix Cache 按前缀树复用不同请求的公共 KV。前者管内存分配，后者管跨请求复用。
96. 引擎 Prefix Cache 通常在单引擎进程内复用前缀；LMCache 面向跨引擎/层级存储与传输。要量有效命中率、节省重算时间、查找/序列化/网络传输、陈旧和租户一致性。
97. Chunked Prefill 把长 Prefill 切片与 Decode 交错，缓解阻塞但可能延长单请求 Prefill；PD 解耦把两类负载放不同 Worker，降低 Decode 干扰，代价是 KV 传输、路由和额外容量。
98. FlashAttention 通过分块和在线 softmax 减少标准 Attention 的 HBM IO；FlashMLA 针对 MLA 的压缩潜变量、变长和解码场景优化访问与计算。共同原则是 IO-aware，数据布局不同。
99. FP16 范围较窄，BF16 范围大且训练稳；FP8 更省带宽但需缩放；INT8/INT4 更省内存和算力但量化误差增。权重量化省模型带宽，激活量化更难，KV 量化影响长上下文缓存与注意力质量。
100. Draft 提议若干 token，目标模型并行验证并按接受规则修正，因此保持目标分布。收益取决于接受率、一次验证长度、Draft 延迟和目标模型并行效率；低接受率会倒亏。
101. MLA 压缩 KV 降缓存与带宽，DeepSeekMoE 用稀疏专家扩容量降激活计算，MTP 提供多 token 预测信号/推测机会，FP8 降训练推理成本，负载均衡避免专家热点。
102. R1 的 RL/GRPO 和蒸馏属于训练与能力迁移；V3.2 的 DSA 改变注意力稀疏选择，主要影响模型架构与推理计算路径。不要把训练算法和 Serving 调度混为一层。
103. Kimi K2/K2.5 是大规模 MoE Agent 模型路线，k1.5 强调推理 RL，Kimi Linear 探索混合线性注意力，Mooncake 是 KV-centric 解耦 Serving 系统；分别位于模型、训练、注意力架构和服务系统层。
104. vLLM/SGLang 是模型运行时与调度层；LMCache 是 KV 复用/传输层；TensorRT-LLM 是 NVIDIA 优化编译与运行时；Mooncake/Dynamo/llm-d 偏分布式 Serving、路由和 PD 解耦。可组合而非简单四选一。
105. 先按错误归因：知识缺失/新鲜度用 RAG，输入组织与跨会话事实用上下文/记忆，工具/控制失败改 Agent 架构，延迟吞吐改 Serving/路由，稳定能力缺口且有高质量数据才考虑微调/RL；每层用消融证明。

