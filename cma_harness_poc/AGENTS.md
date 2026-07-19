# POC: CMA Harness — Phase 2 验证方案

## 背景

CMA Harness POC Phase 1 已完成 V1-V8 基础验证（Agent CRUD、Session CRUD、SSE、Event Stream、Tool Failure、Events→Messages、E2E、并发）。

Phase 2 验证 4 个未覆盖功能点：
1. Skills/MCP 支持
2. Thinking event 支持
3. 上下文压缩
4. 暂停/恢复 Session（对话不丢）

## 标准对齐

参考 wiki 文档确认的 CMA 标准：

| 特性 | CMA 标准 | 验证策略 |
|------|---------|---------|
| Skills | Agent 有 `skills` 字段（max 20） | Agent CRUD 透传 + Harness 实际加载 |
| MCP Servers | Agent 有 `mcp_servers` 字段（max 20） | Agent CRUD 透传 |
| Thinking Event | `agent.thinking` 独立事件，Extended thinking 默认开启 | SSE 推送 `agent.thinking.delta` + `agent.thinking` |
| 上下文压缩 | `BetaContextManagementConfig`：clear_thinking → clear_tool_uses → compact | 利用 Hermes 内置 ContextCompressor，提取 summary 到 CMA compact_ctx |
| 暂停/恢复 | Harness stateless，`wake(sessionId)` → `getEvents()` → resume | EventStore 保证事件不丢，Harness 每次重建 |

## 核心数据模型变更

### CompactContext（models.py 新增）

```python
@dataclass
class CompactContext:
    compacted_up_to: int = 0       # 已压缩到第几个 event（events 列表索引）
    summary: str = ""              # Hermes 生成的 [CONTEXT COMPACTION] 摘要文本
```

### SessionRecord 新增字段

```python
@dataclass
class SessionRecord:
    ...
    compact_context: Optional[CompactContext] = None
```

### AgentConfig 扩充

```python
@dataclass
class AgentConfig:
    ...
    mcp_servers: List[Dict[str, Any]] = field(default_factory=list)  # 新增
```

## 验证点

### V9: Skills/MCP

| ID | 验证项 | 类型 | 通过标准 |
|----|--------|------|---------|
| V9.1 | skills 字段 CRUD 透传 | stateless | POST Agent 带 skills → GET 返回相同 skills |
| V9.2 | mcp_servers 字段 CRUD 透传 | stateless | POST Agent 带 mcp_servers → GET 返回相同配置 |
| V9.3 | skills 激活工具集 | LLM | Agent 创建时带 skills → Harness 传 `enabled_toolsets=["terminal","file","web","skills"]` → Hermes 的 tool schema 中包含 skill_view/skill_manage |
| V9.4 | E2E: skills 在对话中生效 | LLM | Agent 带 skills → 对话中 agent 能调用 skill 工具并返回结果 |

**实现要点**：
- `harness_runner.py`: `enabled_toolsets` 根据 `agent_config.skills` 动态追加 "skills"
- `AgentConfig`: 新增 `mcp_servers` 字段

### V10: Thinking Event

| ID | 验证项 | 类型 | 通过标准 |
|----|--------|------|---------|
| V10.1 | `agent.thinking` 序列化 | stateless | `CmaEvent(type="agent.thinking", content=[{"type":"thinking","thinking":"..."}])` → `to_sse_dict()` 输出正确 |
| V10.2 | thinking.delta SSE 推送 | stateless | `agent.thinking.delta` 事件格式正确推送到 SSE 队列 |
| V10.3 | thinking + text 合并 | LLM | DeepSeek 返回 reasoning_content → SSE 流先出 `agent.thinking.delta` 再出 `agent.message`，thinking 内容不混入 text |
| V10.4 | eventsToMessages round-trip | stateless | 带 `{"type":"thinking","thinking":"..."}` content block 的 `agent.thinking` 事件 → 转 Hermes messages → `reasoning_content` 字段设置正确 |
| V10.5 | E2E: 真实 thinking 推 SSE | LLM | 用 DeepSeek 跑需要思考的 prompt（数学/推理题）→ SSE 收到 `agent.thinking.delta` + `agent.thinking` 事件 |

**实现要点**：
- `harness_runner.py`: 新增 `reasoning_cb(text)` callback，累加 buffer 后 flush 成 `agent.thinking` 事件
- `harness_runner.py`: `AIAgent(reasoning_callback=reasoning_cb)`
- `event_translator.py`: 在 `agent.thinking` → Hermes message 时设 `reasoning_content`；在 Hermes message → `agent.message` 时保留 thinking block
- `make_stream_delta_cb`: 不受 thinking 内容影响（stream_delta_callback 只收非 thinking text）

### V11: 上下文压缩

| ID | 验证项 | 类型 | 通过标准 |
|----|--------|------|---------|
| V11.1 | Hermes ContextCompressor 在 POC 中工作 | stateless | `agent.run_conversation()` 在 message 超阈值时触发 `should_compress()` → 返回结果含 `[CONTEXT COMPACTION` 消息 |
| V11.2 | compact_ctx 提取 | stateless | run_conversation 结果含压缩标记 → 提取 summary → 写入 SessionRecord.compact_context |
| V11.3 | compact_ctx 持久化 | stateless | 写入后 getSession 读回 → compact_context 内容一致 |
| V11.4 | 跨轮压缩状态生效 | stateless | turn N 写入 compact_ctx → turn N+1 eventsToMessages 使用 compact_ctx → 只转 compacted_up_to 之后的事件 |
| V11.5 | 压缩后 SSE 事件完整 | LLM | turn N 触发压缩 → turn N+1 用户发消息 → SSE 流包含完整的事件序列（含压缩前的） |
| V11.6 | E2E: 长对话压缩后继续 | LLM | 多轮对话触发压缩 → 后续 agent 仍能正确回答（使用 summary 上下文） |

**实现要点**：

`event_translator.py` 新增 `cma_events_to_hermes_messages` 重载：

```python
def cma_events_to_hermes_messages(
    events: List[CmaEvent],
    compact_ctx: Optional[CompactContext] = None,
) -> List[Dict]:
    messages = []
    start_idx = 0
    
    if compact_ctx and compact_ctx.compacted_up_to > 0:
        # 插入之前压缩的 summary
        messages.append({
            "role": "assistant",
            "content": compact_ctx.summary,
        })
        start_idx = compact_ctx.compacted_up_to
    
    # 处理剩余事件
    for event in events[start_idx:]:
        # ... 原有转换逻辑 ...
    
    return messages
```

`harness_runner.py` 压缩提取逻辑：

```python
# run_conversation 返回后
result = agent.run_conversation(...)
for msg in result.get("messages", []):
    content = msg.get("content", "") or ""
    if isinstance(content, str) and "[CONTEXT COMPACTION" in content:
        events = event_store.get_events(session_id)
        compact_ctx = CompactContext(
            compacted_up_to=len(events),
            summary=content,
        )
        session_service.update_compact_context(session_id, compact_ctx)
        break
```

### V12: 暂停/恢复 Session

| ID | 验证项 | 类型 | 通过标准 |
|----|--------|------|---------|
| V12.1 | user.interrupt 中断 Harness | stateless | 发送 `user.interrupt` → `_cancel_harness(session_id)` 取消 asyncio.Task |
| V12.2 | 中断后已有事件不丢失 | stateless | interrupt 前已 append 的 events 仍在 EventStore 中 |
| V12.3 | 中断后 Session 回到 idle | stateless | interrupt → session 状态变为 idle |
| V12.4 | 恢复后对话完整 | LLM | interrupt → 新 user.message → 新 Harness 从 EventStore 读全部 events → 重建 conversation_history → agent 知道之前讨论的内容 |
| V12.5 | 中断时正在执行 tool 的回退 | LLM | interrupt 时 terminal 命令正在运行 → Task cancelled → 命令输出不写入 EventStore → 下次 resume agent 重新发送 tool_use |

**实现要点**：

`api_server.py` 的 `_start_harness` 确认 cancel 行为：

```python
def _cancel_harness(self, session_id: str) -> None:
    task = self._harness_tasks.pop(session_id, None)
    if task is not None and not task.done():
        task.cancel()
    # 不修改 EventStore，不修改 Session 状态
    # 已有事件（已完成并 append 的）不动
    # 正在执行的 tool 的 tool_use 事件已写入 EventStore
    # 未完成的 tool_result 事件未写入 —— 下次 resume 时 agent 会重新发起 tool call
```

`_handle_post_events` 的 interrupt 处理：

```python
elif evt_type == "user.interrupt":
    self._cancel_harness(session_id)
    # Session 状态不在这里改 —— 让 Harness 自行退出后通过 session.status_idle 回 idle
    # 或者如果 Harness 不能干净退出，直接设 session → idle
```

## 实现计划

### 文件变更清单

| 文件 | 变更 |
|------|------|
| `models.py` | 新增 `CompactContext`、`AgentConfig.mcp_servers` |
| `event_translator.py` | 新增 `cma_events_to_hermes_messages(events, compact_ctx)` 重载、thinking block 处理 |
| `harness_runner.py` | 新增 `make_reasoning_cb`、`make_tool_start_cb` 已存在，改 `enabled_toolsets` 动态化、压缩提取逻辑 |
| `session_service.py` | 新增 `update_compact_context()` 方法 |
| `api_server.py` | interrupt 处理完善 |
| `verify/test_v9_skills_mcp.py` | 新增 |
| `verify/test_v10_thinking.py` | 新增 |
| `verify/test_v11_context_compact.py` | 新增 |
| `verify/test_v12_pause_resume.py` | 新增 |
| `verify/conftest.py` | 无变更（已有 `cma_server` fixture） |

### 执行顺序

1. V9 Skills/MCP — 最简单，先做，不依赖其他变更
2. V10 Thinking — 新增 callback 注册 + event 处理
3. V12 暂停/恢复 — interrupt 处理完善
4. V11 上下文压缩 — 依赖 event_translator 重载 + Session compact_context，最后做
