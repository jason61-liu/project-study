# OWASP Top 10 for Agentic Applications 2026：从风险清单到工程控制映射

![OWASP Agentic Top 10 风险传播与控制映射](./assets/owasp-agentic-risk-map.svg)

## 阅读前术语表

| 术语 | 中文建议 | 本文中的工程含义 |
|---|---|---|
| Agentic Application | 智能体应用 | 由模型、规划循环、工具、记忆、身份、策略和运行环境共同组成，能够连续采取动作并改变外部状态的系统。 |
| Asset | 资产 | 需要保护的数据、身份、权限、资金、代码、基础设施、业务状态和审计证据。 |
| Principal | 主体 | 发起动作并可被认证、授权和追责的用户、服务、Agent 或工作负载身份。 |
| Trust Boundary | 信任边界 | 数据或控制权从一种信任假设进入另一种信任假设的位置，例如浏览器内容进入模型上下文、Agent 请求进入工具执行器。 |
| Attack Surface | 攻击面 | 攻击者可以影响的输入、协议、工具、记忆、依赖、身份和执行环境总和。 |
| Threat Event | 威胁事件 | 某个威胁源利用弱点，对资产产生机密性、完整性或可用性影响的具体事件。 |
| Control | 控制措施 | 预防、检测、响应或恢复风险的技术、流程和人员机制。 |
| Invariant | 安全不变量 | 无论模型如何输出都必须成立的机器可验证条件，例如 `resource.tenant_id == principal.tenant_id`。 |
| Least Privilege | 最小权限 | 主体只获得完成当前任务所需的最小操作、资源、范围和持续时间。 |
| Least Agency | 最小自主权 | 除最小权限外，还限制 Agent 的目标空间、工具选择、步骤数、并发度和可自动产生的副作用。 |
| Blast Radius | 爆炸半径 | 一次失陷可以影响的租户、数据量、系统数量、金额和持续时间。 |
| Provenance | 来源溯源 | 内容、工具、模型、Prompt 或构件的来源、版本、签名、哈希和处理链记录。 |
| Fail-closed | 失败关闭 | 策略、身份、来源或证据不可用时默认拒绝高风险动作。 |
| Compensating Control | 补偿性控制 | 无法直接消除风险时，用隔离、限额、审批、监控或快速撤销降低影响。 |

## 1. 风险清单不是威胁模型

OWASP 的 2026 清单提供共同语言，但不能直接回答“本系统哪条数据流最危险”。正式文档强调 Agent 会放大既有漏洞，并提出 **Least Agency**：没有业务价值的自主性本身就是攻击面；同时要求观察 Agent 的目标、工具和行为偏移。[OWASP Top 10 for Agentic Applications 2026](https://genai.owasp.org/resource/owasp-top-10-for-agentic-applications-for-2026/)

落地时要完成四次转换：

```text
OWASP 风险类别
  → 本系统中的具体资产与信任边界
  → 可以被攻击者触发的失败路径
  → 可执行控制、攻击测试与审计证据
```

只写“防止 Prompt Injection”无法验收；写成“来自网页的文本不得改变已签名的 `goal_id`，且任何外发工具调用必须通过租户、Scope、目的地和数据分类策略”才是可测试控制。

## 2. OWASP Agentic Top 10 全量风险映射

以下名称与编号来自 OWASP 正式版目录。映射列是针对本学习项目的工程化展开，不代表 OWASP 官方排序以外的优先级结论。

| ID | 正式名称 | 主要信任边界／资产 | 典型失效路径 | 首要控制 | 最小攻击测试 |
|---|---|---|---|---|---|
| ASI01 | Agent Goal Hijack | 用户目标、外部内容、Planner | 恶意邮件或网页被当作高优先级指令，改变目标或步骤 | 内容与指令分离、目标胶囊、计划差异检查、高风险确认 | 文档内隐藏“忽略任务并外发数据”，验证目标不变且外发被拒绝 |
| ASI02 | Tool Misuse and Exploitation | Tool Gateway、参数、业务 API | Agent 在已有权限内误删、过量调用或把机密发给外部端点 | 工具 Allowlist、Schema、语义校验、限额、幂等、审批 | 合法工具名配危险参数；验证执行前被确定性策略拦截 |
| ASI03 | Identity and Privilege Abuse | OAuth Token、服务身份、委托链 | Agent 继承用户或上游 Agent 的宽权限，产生混淆代理与横向移动 | 独立工作负载身份、短期委托 Token、audience/scope、RBAC+ABAC | 使用 A 租户 Token 访问 B 租户资源；必须拒绝并记录主体链 |
| ASI04 | Agentic Supply Chain Vulnerabilities | 模型、Prompt、MCP、A2A、镜像、依赖 | 动态发现恶意工具或被篡改的 Prompt/镜像，运行时进入执行链 | 固定版本与哈希、签名验证、SBOM/AIBOM、注册表 Allowlist、Kill Switch | 替换工具描述或镜像摘要；验证加载失败而非降级信任 |
| ASI05 | Unexpected Code Execution (RCE) | Shell、解释器、模板、反序列化、宿主机 | 模型输出经 `eval`、Shell 拼接或恶意依赖变成任意代码执行 | 无 Shell 拼接、沙箱、只读根文件系统、Seccomp、网络拒绝、短生命周期 | 命令注入、恶意反序列化和逃逸探针均不能触达宿主或控制面 |
| ASI06 | Memory & Context Poisoning | 会话摘要、向量库、长期记忆、共享缓存 | 不可信内容被持久化并跨会话、跨 Agent 或跨租户影响决策 | 写入准入、来源/租户标签、隔离命名空间、TTL、撤销与重建 | 污染记忆后开启新会话，验证不可检索或不具备指令权 |
| ASI07 | Insecure Inter-Agent Communication | Agent 身份、发现目录、MCP/A2A 消息 | 伪造 Agent Card、重放委托、篡改消息或语义混淆 | mTLS、消息签名、nonce/时效、能力声明、语义 Schema | 重放旧的授权消息或伪造服务描述，必须验签失败 |
| ASI08 | Cascading Failures | Planner/Executor、多 Agent、自动部署 | 单点幻觉、污染或权限错误被下游自动采信并放大 | 断路器、分区预算、独立验证、两阶段提交、回滚与隔离 | 上游返回错误风险分数，验证下游不会自动批量执行 |
| ASI09 | Human-Agent Trust Exploitation | 审批 UI、解释、通知、用户决策 | Agent 用伪造理由、紧迫感或海量确认诱导用户批准危险动作 | 预览与执行分离、来源证据、差异化风险 UI、确认绑定具体动作 | 修改确认后的金额/收件人，验证旧批准不能复用 |
| ASI10 | Rogue Agents | 行为完整性、Agent 身份、编排器 | Agent 在失陷或漂移后持续偏离职责，单步看似合法但整体恶意 | 行为清单、持续证明、异常检测、隔离、撤销和重新准入 | 注入异常工具序列，验证自动冻结身份与工具能力 |

OWASP 明确区分了相邻风险：ASI02 是在已授权能力内误用合法工具；若涉及权限提升转为 ASI03，若导致任意代码执行转为 ASI05。ASI06 强调持久上下文污染，ASI01 强调目标或决策路径被改变，ASI10 则关注偏移开始后的行为完整性丧失。

## 3. 用传播链理解十项风险

十项风险不是十个互斥“桶”。真实事故通常是一条链：

```text
进入：ASI01 注入 / ASI04 恶意依赖 / ASI07 伪造消息
  ↓
持久化与放大：ASI06 记忆污染 / ASI03 权限继承 / ASI08 级联
  ↓
执行与影响：ASI02 工具误用 / ASI05 RCE / ASI09 诱导审批
  ↓
持续失陷：ASI10 Rogue Agent
```

因此控制不能只布置在输入端。即使输入检测漏报，工具执行前仍应拒绝越权；即使工具策略漏配，沙箱和网络出口仍应压缩爆炸半径；即使产生了副作用，审计、撤销和回滚仍要可用。这是纵深防御，而不是重复堆叠分类器。

## 4. 以信任域建立系统级映射

### 4.1 六类信任域

| 信任域 | 默认信任结论 | 必须携带的元数据 | 不应具备的能力 |
|---|---|---|---|
| 用户输入 | 已认证用户也可能误操作或恶意 | `user_id`、`tenant_id`、会话、来源、分类 | 不能直接成为系统策略 |
| 外部内容 | 不可信数据，可能携带间接注入 | URL/文档 ID、抓取时间、内容哈希、来源信誉 | 不能改变目标或授权 |
| 模型输出 | 不可信建议，非身份和非策略 | 模型快照、Prompt 版本、Trace、置信信号 | 不能直接持有凭证或执行副作用 |
| 工具结果 | 可能被攻击者控制或被供应链污染 | 工具 ID/版本、调用 ID、签名、Schema | 不能自动提升下轮指令优先级 |
| 记忆与检索 | 跨时间放大风险的半可信数据 | 租户、主体、来源、写入者、TTL、版本 | 不能跨租户共享未脱敏事实 |
| 执行与业务系统 | 权威副作用域 | 主体链、Scope、策略决策、幂等键、收据 | 不应相信自然语言“已审批” |

### 4.2 控制责任必须落在模型外

Prompt 可以改善模型行为，但不能提供以下安全性质：

- 密码学身份与不可伪造的委托链；
- 原子化权限检查与资源级租户隔离；
- 文件系统、CPU、内存、网络和系统调用隔离；
- 不可篡改审计、撤销、删除传播和密钥轮换；
- 对同一结构化输入稳定复现的策略判定。

这些必须由 Identity Provider、Policy Enforcement Point、Tool Gateway、Sandbox、数据库约束和审计系统承担。

## 5. 风险优先级：不要把 Top 10 当成固定分数

可以为每个具体威胁场景记录五个维度，每项 1–5：

\[
Priority = Likelihood \times Impact \times Agency \times Reach \times Persistence
\]

- `Likelihood`：攻击输入到达弱点的可行性；
- `Impact`：对机密性、完整性、可用性、合规与人身安全的影响；
- `Agency`：Agent 可自主执行的步骤、工具和副作用程度；
- `Reach`：可触达租户、系统和数据的范围；
- `Persistence`：污染、凭证或驻留影响跨会话持续的能力。

该乘积只用于本项目排序，不应冒充 OWASP 或 CVSS/AIVSS 官方评分。高风险硬门槛优先看“可否发生”，不能让低延迟或高回答质量抵消一次越权。

## 6. 建议的风险登记表 Schema

```yaml
risk_id: AGENT-ASI03-001
owasp: [ASI01, ASI03, ASI06]
scenario: poisoned_email_causes_cross_tenant_export
assets: [customer_pii, oauth_token, audit_integrity]
trust_boundaries: [email_to_context, agent_to_tool, tenant_a_to_tenant_b]
threat_actor: external_sender
preconditions:
  - mailbox content enters planning context
  - export tool has broad tenant scope
attack_path:
  - indirect_prompt_injection
  - goal_hijack
  - confused_deputy
  - data_exfiltration
invariants:
  - resource.tenant_id == principal.tenant_id
  - token.aud == export_api
  - approval.intent_hash == action.intent_hash
controls:
  preventive: [content_provenance, scoped_token, tool_pep]
  detective: [goal_drift_alert, cross_tenant_denial_log]
  responsive: [token_revocation, tool_kill_switch]
test_ids: [ATK-07, ATK-11]
owner: security-platform
residual_risk: medium
evidence: [policy_decision_id, audit_event_id, sandbox_receipt]
```

风险条目必须绑定负责人、测试和证据。没有 `test_ids` 的控制只是声明；没有 `evidence` 的测试难以审计；没有残余风险的设计容易把“装了护栏”误写成“风险消失”。

## 7. 纵深控制矩阵

| 阶段 | 预防 | 检测 | 响应／恢复 |
|---|---|---|---|
| 设计 | 最小自主权、威胁建模、职责分离 | 架构评审发现越界数据流 | 删除不必要工具与长期记忆 |
| 构建 | 固定依赖、签名、SBOM/AIBOM、策略测试 | SAST/SCA、Prompt/Tool diff | 回滚构件、供应链 Kill Switch |
| 摄取 | 来源标签、CDR、文件解析隔离、租户命名空间 | 注入信号、异常来源、污染率 | 隔离文档、撤销记忆、重建索引 |
| 规划 | 目标胶囊、步骤/预算、工具集合约束 | Goal/Plan drift、异常链路 | 中断运行、要求重新授权 |
| 执行 | PEP、RBAC/ABAC、Scope、Schema、沙箱、出口控制 | 工具序列、权限拒绝、资源超限 | 撤销 Token、终止沙箱、补偿事务 |
| 输出 | DLP、PII 脱敏、可信渲染、证据校验 | 泄漏探测、虚假完成检测 | 阻断输出、通知、数据事件响应 |
| 运营 | 身份生命周期、轮换、保留/删除 | 行为基线、不可变审计、红队 | 隔离 Rogue Agent、恢复可信基线 |

## 8. 第 9 周最小攻击集

至少覆盖以下 15 条，并把每条测试关联到一个或多个 ASI：

1. 直接 Prompt Injection 改变系统目标；
2. 网页隐藏文本进行间接 Prompt Injection；
3. 恶意工具结果要求 Agent 外发历史上下文；
4. 合法删除工具传入越界路径；
5. 未审批付款与审批后参数替换；
6. A 租户 RAG 查询 B 租户文档；
7. A 租户记忆被 B 租户检索；
8. 共享缓存键遗漏 `tenant_id`；
9. 过期或错误 audience 的 Access Token；
10. 重放上一次 Agent 委托消息；
11. MCP 工具描述或 Prompt 模板哈希漂移；
12. Shell 元字符、命令替换和恶意反序列化；
13. 沙箱访问宿主设备、元数据服务和控制面；
14. 上游错误输出触发下游批量级联；
15. 行为清单之外的工具序列触发 Rogue Agent 隔离。

每条结果至少记录 `expected_control`、`actual_decision`、`side_effect_observed`、`policy_decision_id`、`trace_id` 和 `residual_risk`。

## 9. 本文结论

1. OWASP Top 10 是风险语言，不是产品控制清单；必须映射到资产、边界、路径、控制、测试和证据。
2. 十项风险会组成传播链，输入过滤不能替代工具授权、沙箱和恢复机制。
3. 最小权限约束“能访问什么”，最小自主权还约束“可自行决定什么、做多久、做几步”。
4. 模型输出、工具结果、外部内容和记忆都不能被当成策略或身份事实。
5. 高风险不变量必须由确定性执行点实施，并进入发布 Hard Gate。

## 参考资料

- [OWASP: Top 10 for Agentic Applications 2026](https://genai.owasp.org/resource/owasp-top-10-for-agentic-applications-for-2026/)
- [OWASP Agentic Top 10 正式 PDF](https://genai.owasp.org/download/52117/?tmstv=1765059207)
- [OWASP: Securing Agentic Applications Guide](https://genai.owasp.org/download/49059/?tmstv=1753666640)
- [OWASP: Secrets Management Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Secrets_Management_Cheat_Sheet.html)
- [OWASP: Logging Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Logging_Cheat_Sheet.html)
