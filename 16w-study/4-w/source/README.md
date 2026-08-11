# 第四周编码实验：安全 RAG、上下文策略与 Mem0 对比

本目录延续 `3-w/source` 的轻量结构，包含三个可以独立理解的实验：

```text
文档语料 + AuthContext
  -> tenant/ACL/删除/过期过滤
  -> 字符 TF-IDF + 词项覆盖混合召回
  -> authority/version 排序
  -> 最终授权门
  -> 抽取式回答 + source/version 引用

多轮历史
  -> Full History / Summary / Retrieval Memory
  -> 正确率、输入 Token、构造延迟、单位任务估算成本

同一组原子事实
  -> 自建 Hash Vector Memory
  -> Docker Mem0（健康时）或进程内 Mem0 OSS + Qdrant
  -> 写入、Recall@K、错误记忆、隔离、更新/删除、延迟、Token
```

## 一、文件说明

| 文件 | 作用 |
|---|---|
| `rag_chain.py` | 最小带引用检索链，包含时效、删除、双租户 ACL 和提示注入防护 |
| `context_eval.py` | 全历史、摘要历史、检索式记忆三种上下文策略 |
| `memory_backends.py` | 自建向量记忆、Docker Mem0、进程内 Mem0 OSS 三种实现 |
| `memory_eval.py` | 同一多轮事实集上的记忆后端统一评测 |
| `model_client.py` | DeepSeek-V4-Pro Chat Completions、真实 usage 和费用计算 |
| `run_experiment.py` | 运行完整实验并生成 JSON/Markdown 报告 |
| `data/documents.json` | 两租户文档、冲突证据、过期/删除/恶意样本 |
| `data/retrieval_questions.json` | 24 条问题及 Gold answer/evidence |
| `data/memory_tasks.json` | 6 组多轮事实写入和 12 条记忆查询 |
| `tests/` | 21 个默认无 Mock 测试及 2 个显式启用的真实 API 测试 |

## 二、检索链的安全顺序

检索结果不是仅按相似度决定：

```text
candidate = same_tenant
            AND ACL_allows_user
            AND deleted = false
            AND expires_at > now

final_context = top_ranked(candidate)
                AND final_authorize_with_current_ACL
```

双重权限门有意放在召回前和模型前：第一道门防止越权内容进入评分与日志，第二道门防止召回完成后 ACL 撤销或索引元数据陈旧。

引用结构包含：

```json
{
  "document_id": "doc-oauth",
  "source_uri": "kb://tenant-a/oauth",
  "version": 4,
  "updated_at": "2026-07-10T00:00:00+00:00",
  "score": 0.45
}
```

对超长/恶意文档，索引读取有字符上限，回答只选取相关句子，并移除常见 Prompt Injection 句。文档内容始终是 `DATA`，不会提升为系统指令。

## 三、三种上下文策略

### Full History

完整保留消息，事实 Recall 通常最好，但 Token 随轮数线性增加，也会保留重复和已经失效的信息。

### Summary History

把较早消息压缩为去重事实，并保留最近两轮。它节省 Token，但摘要器可能漏掉低频但关键的事实；本实验故意保留一个信息损失样本。

### Retrieval Memory

把消息写入 tenant/user 隔离的向量记忆，只召回与当前问题相关的 top-K。它通常减少输入 Token，但增加 Embedding/检索延迟，并依赖召回质量。

真实模型模式调用 `deepseek-v4-pro` 的 Chat Completions，并关闭 Thinking Mode 以控制延迟。Token 直接采用 API usage；费用依据 [DeepSeek 官方价格](https://api-docs.deepseek.com/quick_start/pricing)计算：缓存命中输入 `$0.003625/1M`、缓存未命中输入 `$0.435/1M`、输出 `$0.87/1M`。价格会变化，代码中的常量应随官方页面更新。

离线模式才使用 `cl100k_base` 估算；如果机器没有词表，则降级为字符近似。离线成本仅用于策略相对比较，不代表云厂商账单。

## 四、自建记忆与 Mem0 如何公平比较

两个后端接收完全相同、已经原子化的事实：

- 自建基线使用稳定 BLAKE2b 字符 n-gram Hash Embedding 和余弦相似度。
- Mem0 使用真实 Mem0 存储、搜索、更新和删除路径。
- 存在有效 `MEM0_API_KEY` 时自动优先使用 Mem0 Cloud；当前提供的 Key 在 `api.mem0.ai/v1/ping` 返回 `401 Invalid API key`，因此本轮只能回退 OSS。
- 进程内回退使用 Mem0 OSS、Qdrant local mode 和同一个本地 Hash Embedder，不下载模型、不调用云 API。
- 所有 Mem0 写入使用 `infer=False`，因此本轮结论比较的是记忆存储/检索/生命周期，不把某个云 LLM 的事实抽取波动混入结果。

租户作用域使用：

```text
storage_user_id = tenant_id + ":" + application_user_id
```

并额外保存/检查 `tenant_id` 与 `application_user_id`。搜索过滤只是第一道门，返回记录还要检查实际 metadata；知道 memory ID 也不能越权 update/delete。

## 五、Docker Mem0 状态与回退

代码优先探测 `mem0-dev-mem0-1`：

1. 容器必须处于 running；
2. 使用隔离身份执行一次无副作用搜索；
3. 认证、Embedding Provider 和向量存储都成功才选择 Docker。

当前检查中容器和认证 API 正常，但 Provider 调用返回 `502 provider_unavailable`，所以 `auto` 会回退到 `mem0_oss_local`。代码不会修改现有容器的全局 Provider 配置，也不会读取或打印 `ADMIN_API_KEY`；Docker 适配器让请求在容器内使用该环境变量完成认证。

## 六、运行

```bash
source /Users/shiyiliu/workspace/pyproject/.venv/bin/activate
cd /Users/shiyiliu/workspace/pyproject/16w-study/4-w/source
```

安装依赖：

```bash
python -m pip install -r requirements.txt
```

使用真实 DeepSeek，并按 Cloud -> 健康 Docker -> OSS 的顺序选择 Mem0：

```bash
export DEEPSEEK_API_KEY="你的 DeepSeek Key"
export MEM0_API_KEY="你的有效 Mem0 Key"
export OPENAI_BASE_URL="https://api.deepseek.com"
export AGENT_TEST_MODEL="deepseek-v4-pro"
MEM0_TELEMETRY=false python run_experiment.py --real-model --mem0 auto
```

强制使用进程内 OSS：

```bash
MEM0_TELEMETRY=false python run_experiment.py --mem0 local
```

只有 Docker Provider 健康时才强制 Docker：

```bash
python run_experiment.py --mem0 docker
```

完整测试：

```bash
MEM0_TELEMETRY=false python -m pytest -q -p no:cacheprovider
```

默认测试不会向外部服务发送语料。显式运行真实 API 测试：

```bash
RUN_LIVE_TESTS=1 python -m pytest -q -p no:cacheprovider tests/test_live_integrations.py
```

运行完整真实模型评测会把 `data/` 中的实验文档和多轮任务发送给 DeepSeek API，应先确认这些内容允许发送给外部服务商。

## 七、指标解释

- 写入正确率：写入后按授权 ID 读取，内容完全一致的比例。
- RAG 正确率：证据必须命中 Gold evidence，回答还必须包含评测集在调用前声明的 `answer_terms`；不使用运行后临时修改的 LLM Judge。
- Recall@K：Gold 事实是否出现在前 K 条记忆中。
- 错误记忆率：top-K 返回项中不包含本题 Gold 事实的比例。
- 租户隔离：tenant-a 查询是否完全看不到 tenant-b 的 `NEBULA` 事实。
- 更新一致性：新值可检索且旧值不再出现。
- 删除一致性：删除后 get/search 均不可见。
- 延迟：分别记录写入和搜索的 median/p95；小样本用于教学比较，不代表生产容量。
- Token：相同写入文本与查询的估算 Token；`infer=False` 下没有事实抽取 LLM Token。

## 八、已覆盖的异常与安全测试

- 无相关证据时显式拒答；
- 正式、非正式和过期冲突证据排序；
- 超长文档有上限，恶意提示不被执行；
- 双租户和文档 ACL 越权；
- 过期文档与 tombstone 在召回前过滤；
- 删除后同一检索链不再返回残留；
- 模型即使猜中 memory ID 也不能越权更新或删除；
- Mem0 更新时显式保留 tenant metadata，防止第三方 SDK 清空自定义字段。

当前验收结果：`21 passed, 2 skipped`；两个 skip 是必须显式设置 `RUN_LIVE_TESTS=1` 的外部 API 测试。DeepSeek-V4-Pro 全量真实评测已完成，Mem0 Cloud Key 验证失败后按设计回退到 OSS。完整数据见 `artifacts/experiment-report.json`，摘要见 `artifacts/experiment-report.md`。
