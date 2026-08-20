# 第 15 周：可上线的深度研究 Agent

这是一个小而完整的生产候选骨架：用户提交问题和允许访问的来源，系统分解问题、检索并去重证据、识别结构化声明冲突，最后生成每条结论可回查的引用报告。没有证据时明确拒答。它不会联网抓取、自动发布或执行高风险外部写操作。

## 已实现的验收证据

- 查询分解、词项检索、内容/URL 去重、确定性重排、证据集合；
- `[S1]` 形式的来源引用、引用集合校验、结构化 `claims` 冲突检测、无证据拒答；
- SQLite WAL Checkpoint、版本化 CAS、租户隔离、幂等提交、人工审批与进程重启后恢复；
- OTel 形状的安全 Trace、Prometheus 文本指标、费用估算和可查询运行记录；
- 无 Shell 的命令白名单 Sandbox 本地后备实现；
- 55 条功能评测定义、15 条安全场景、10 条故障场景和自动化单元/集成测试；
- 两实例 Docker Compose、健康检查、共享持久状态；
- 架构/状态/数据流/信任边界、SLO/容量/业务基线、安全模型和 4 份 ADR。

## 快速开始

只需要 Python 3.11+，运行时无第三方依赖：

```bash
cd 15-w
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m deep_research_agent.eval_harness --output evals/baseline.json
PYTHONPATH=src python3 -m deep_research_agent.eval_gate --baseline evals/baseline.json
PYTHONPATH=src python3 -m deep_research_agent.service --port 8080
```

提交研究任务：

```bash
curl -s http://localhost:8080/v1/runs \
  -H 'Content-Type: application/json' \
  -d '{
    "question":"Checkpoint 如何支持 Agent 恢复？",
    "identity":{"tenant_id":"tenant-a","user_id":"u-1","scopes":["research:run"]},
    "idempotency_key":"demo-1",
    "sources":[{
      "source_id":"doc-1",
      "title":"Checkpoint 指南",
      "url":"https://example.test/checkpoint",
      "content":"Checkpoint 保存已验证证据、当前步骤和剩余预算，使任务可从确定状态恢复。",
      "claims":{"恢复方式":"从最近检查点恢复"}
    }]
  }'
```

响应中的 `run_id` 可用于查询；调用方必须始终提供租户：

```bash
curl -H 'X-Tenant-Id: tenant-a' http://localhost:8080/v1/runs/RUN_ID
curl http://localhost:8080/metrics
```

若提交时设置 `"require_approval": true`，任务会停在 `waiting_approval`。审批者用 `research:approve` scope 恢复：

```bash
curl -X POST http://localhost:8080/v1/runs/RUN_ID/approve \
  -H 'Content-Type: application/json' \
  -d '{"identity":{"tenant_id":"tenant-a","user_id":"reviewer","scopes":["research:approve"]}}'
```

进程在证据 Checkpoint 后中断时，所有者或同租户 `operator` 可调用 `POST /v1/runs/RUN_ID/resume`；请求体只需包含 `identity`。若中断早于证据 Checkpoint，应以原幂等键重试原始请求。

## 容器与双实例演示

```bash
docker compose up --build
# 8080 创建 require_approval 任务，8081 查询并审批同一 run_id
docker compose down
```

两个实例共享 `research-data` 卷中的 SQLite WAL 数据库。它证明了示例规模下的共享状态与滚动替换路径；SQLite 单写者模型不是大规模生产数据库方案，容量阈值和迁移条件见 [SLO 与容量模型](docs/SLO-CAPACITY.md)。

## 输入契约

`question`、`identity.tenant_id/user_id/scopes` 和 `sources` 是核心字段。来源必须是 HTTP(S)，外部正文只作为无指令权限的证据。`claims` 是可选的结构化声明映射，用于确定性识别“同一声明、不同取值”。预算支持 `max_steps`、`max_sources`、`max_chars` 和 `max_cost_usd`。

错误按无证据拒答、策略拒绝、幂等冲突、并发版本冲突和内部失败区分。API 不接收原始 OAuth Token；生产网关应验证令牌后只传递最小身份上下文。

## 目录

- `src/deep_research_agent/`：运行时、存储、检索、安全、服务和评测 Harness；
- `tests/`：确定性单元与集成测试；
- `evals/baseline.json`：本机运行生成的评测基线；
- `docs/`：架构、安全、业务决策、SLO/容量与 ADR；
- `docs/schemas/`：版本化请求、运行状态与 Sandbox 工具 JSON Schema；
- `Dockerfile`、`compose.yaml`：干净环境启动和双实例演示。

## 已知限制

- 当前是“用户提供授权语料”的离线基线，不主动联网搜索；接入搜索工具时仍须在检索前强制租户和 ACL 过滤。
- 报告器是可复现的抽取式基线，不是开放域 LLM 综合器；这是业务基线和回归 Oracle，而非最终文案质量上限。
- SQLite 适用于单机/小规模共享卷演示；超过容量阈值迁移到带行级权限和备份恢复的 PostgreSQL。
- 本地 Sandbox 只提供 argv 白名单、超时、临时目录和 Linux 资源上限；生产不可信代码必须路由到独立容器/E2B/OpenSandbox，并默认断网和禁用 Secret。
- 本周不实现自动发布、OAuth Provider、联网抓取、长期个性化记忆或多 Agent；这些都不是当前业务闭环的必要条件。
