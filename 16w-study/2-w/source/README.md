# 最小模型适配层与 Agent 工具循环

这是第二周的编码实验，使用 Python 标准库实现：

- Responses 风格的非流式与流式模型响应适配；
- JSON Schema 结构化输出和工具参数校验；
- `model → tool → observation → model` 最小循环；
- 模型调用与工具调用的独立 Trace ID、起止时间和结果状态；
- 未知工具、参数错误、工具异常、工具超时、模型超时、拒答和最大步数处理；
- 显式提前终止条件。

## 运行测试

在本目录导出真实服务配置后执行：

```bash
source /Users/shiyiliu/workspace/pyproject/.venv/bin/activate
export OPENAI_API_KEY="你的密钥"
export OPENAI_BASE_URL="https://api.siliconflow.cn/v1"
export AGENT_TEST_MODEL="deepseek-ai/DeepSeek-V4-Pro"
python -m pytest -q -p no:cacheprovider
```

测试不使用 Mock：结构化输出、流式响应、工具循环、异常恢复、超时恢复、最大步数和提前终止都会调用真实模型。未知工具、非法参数和 Schema 拒绝属于确定性 Runtime 边界，不需要发起模型请求。未提供凭据时测试会跳过；运行真实测试会产生 API Token 消耗和费用。

## 核心结构

```text
AgentRuntime
  → ModelAdapter
      → ModelTransport.complete / stream
      → ModelReply(text / structured_output / tool_calls / refusal)
  → ToolSpec Schema 校验
  → Tool handler（独立超时）
  → function_call_output Observation
  → 下一次 ModelAdapter.call
```

`ModelTransport` 是连接真实供应商 SDK 的边界。`OpenAIResponsesTransport` 直接适配 Responses API；`OpenAIChatCompletionsTransport` 则把 Chat Completions 的消息、Tool Call 和流式 Chunk 归一化为同一套 Responses 风格内部事件。Schema 校验器只实现实验使用的安全子集；生产项目应使用完整 JSON Schema 库，并继续保留业务校验、鉴权和幂等控制。

SiliconFlow 的真实组装方式如下：

```python
from openai import OpenAI

from agent_runtime import AgentRuntime, ModelAdapter
from openai_transport import OpenAIChatCompletionsTransport


client = OpenAI(base_url="https://api.siliconflow.cn/v1")
transport = OpenAIChatCompletionsTransport(client)
adapter = ModelAdapter(transport, model="deepseek-ai/DeepSeek-V4-Pro")
agent = AgentRuntime(adapter)
result = agent.run("用三句话解释 Tool Call ID", stream=True)
print(result.status, result.text)
```

`ModelAdapter` 将模型超时传给 Transport，由底层 SDK 负责中止等待。工具超时使用线程 Future 限制 Runtime 的等待时间，但 Python 线程无法强制杀死已经运行的函数；涉及支付、发信等副作用的工具仍必须实现自身截止时间、幂等键和执行状态查询。

## 终止状态

`RunStatus` 明确区分：

- `completed`：模型产生通过校验的最终输出；
- `max_steps`：工具循环达到上限；
- `early_stopped`：应用的显式停止条件成立；
- `deadline`：Run 总 Deadline 耗尽；
- `model_refused`：模型明确拒答；
- `model_timeout` / `model_error`：模型调用失败；
- `invalid_model_output`：JSON 或 Schema 校验失败。

工具级错误不会直接伪装成 Run 成功或异常崩溃，而会转换成带稳定错误码的 Observation，让下一轮模型决定解释、修正参数或结束任务。

## 运行 10 分钟演示

配置真实模型环境变量后执行：

```bash
python demo_agent_trace.py
```

脚本会使用流式模式完成一次订单查询，打印最终结构化结果，以及两次模型调用和一次工具调用的 Trace ID、起止时间、耗时、状态与结果摘要。配套讲解稿见 [Agent边界、Loop设计题与10分钟验收.md](../Agent边界、Loop设计题与10分钟验收.md)。
