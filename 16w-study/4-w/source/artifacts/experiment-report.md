# 第四周检索与记忆实验报告

> 生成时间：2026-08-11T01:37:41.779545+00:00
> Mem0 后端：`mem0_oss_local`；Cloud key invalid and Docker unavailable/unhealthy; fell back to process-local OSS

## 文档检索链

| 问题数 | 正确率 | Recall@K | 无答案正确率 | 平均延迟 ms |
|---:|---:|---:|---:|---:|
| 24 | 1.000 | 1.000 | 1.000 | 1239.632 |

模型输入/输出 Token：`2434/314`；估算费用：`$0.00133197`。

## 三种上下文策略

| 策略 | 正确率 | 平均输入 Token | 平均输出 Token | 平均总延迟 ms | 单位任务成本 USD |
|---|---:|---:|---:|---:|---:|
| full_history | 1.000 | 104.38 | 4.38 | 1201.452 | 0.00004921 |
| summary_history | 0.875 | 79.25 | 4.38 | 1126.296 | 0.00003828 |
| retrieval_memory | 1.000 | 69.00 | 4.00 | 1237.901 | 0.00003349 |

> 真实模型模式使用 DeepSeek API usage；无模型模式才使用 cl100k_base/字符近似。费用按官方 DeepSeek-V4-Pro 缓存命中、未命中和输出单价估算。

## 自建向量记忆与 Mem0

| 后端 | 写入正确率 | Recall@K | 错误记忆率 | 租户隔离 | 更新一致 | 删除一致 | 写入 p50 ms | 检索 p50 ms | Token |
|---|---:|---:|---:|---|---|---|---:|---:|---:|
| self_built_vector | 1.000 | 1.000 | 0.586 | True | True | True | 0.022 | 0.059 | 257 |
| mem0_oss_local | 1.000 | 1.000 | 0.625 | True | True | True | 0.835 | 0.322 | 257 |

## 环境与解释

- Docker 容器运行：`True`
- Docker Provider 可用：`False`
- Mem0 Cloud Key：`True`
- Mem0 Cloud Key 验证成功：`False`
- 真实模型：`True`；`deepseek-v4-pro`
- 两个记忆后端接收相同的原子事实；本轮比较存储、检索和生命周期，不比较云 LLM 事实抽取质量。
- 错误记忆率按 top-K 返回中不含 Gold 事实的条目比例计算；它与 Recall@K 是互补指标。
