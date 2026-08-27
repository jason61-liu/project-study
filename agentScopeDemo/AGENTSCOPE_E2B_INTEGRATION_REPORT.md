# AgentScope 集成 E2B 沙箱测试报告

## 最终结论

| 方案 | 实现方式 | 测试结果 |
|---|---|---|
| 方案一 | AgentScope Agent + 自定义 FunctionTool + E2B SDK | 通过 |
| 方案二 | AgentScope Agent + E2BWorkspace 原生工具 | 通过 |

方案一将 `e2b.AsyncSandbox.commands.run()` 包装为一个 AgentScope
`FunctionTool`。已验证沙箱创建、Shell/Python 命令执行、Agent 自主工具调用、
人工确认、多轮对话复用同一个沙箱，以及退出后销毁沙箱。

方案二使用 AgentScope 2.0.7 的 `E2BWorkspace` 管理沙箱，并向 Agent 提供
`Bash`、`Read`、`Write`、`Edit`、`Glob`、`Grep` 等标准 Workspace 工具。
独立 E2BWorkspace 测试和多轮 Agent 测试均已通过。

方案二使用模板 `tpl-27b366cc77df425b82d51b46`。该模板已经内置 Python
3.12.12 和 uv 0.9.30，满足 AgentScope 2.0.7 的运行要求，不需要在沙箱中
升级、编译 Python，也不需要安装或升级 uv。

## 测试环境

| 项目 | 配置 |
|---|---|
| AgentScope | 2.0.7 |
| 宿主机 Python | 3.13.10 |
| E2B Python SDK | 2.34.0 |
| 模型 | `deepseek-v4-pro` |
| E2B API | `https://api.sandbox.ske-k8s251` |
| E2B Domain | `sandbox.ske-k8s251` |
| E2B LB | `10.103.246.175` |
| CA 证书 | `sandboxtest/test-sandbox-ca.crt` |

## 当前 E2BWorkspace 模板

| 项目 | 当前配置 |
|---|---|
| 模板 ID | `tpl-27b366cc77df425b82d51b46` |
| 源镜像 | `sandbox-code-interpreter:0.2.0` |
| 操作系统 | Debian GNU/Linux 12 |
| Python | 3.12.12 |
| uv | 0.9.30 |
| AgentScope 2.0.7 E2BWorkspace | 测试通过 |

## 项目目录与功能

```text
agentScopeDemo/
├── AGENTSCOPE_E2B_INTEGRATION_REPORT.md
├── sandboxtest/
├── agent-e2b-sdk/
└── agent-e2bworkspace/
```

| 文件或目录 | 功能 |
|---|---|
| `AGENTSCOPE_E2B_INTEGRATION_REPORT.md` | 汇总 AgentScope 集成 E2B 的两种实现方案、测试结果和差异 |
| `sandboxtest/` | E2B Python SDK 基础连通性测试，验证沙箱创建、命令执行和销毁 |
| `agent-e2b-sdk/` | 方案一实现；通过自定义 FunctionTool 直接调用 E2B SDK，提供多轮终端 Agent |
| `agent-e2bworkspace/` | 方案二实现；验证 AgentScope 原生 E2BWorkspace，并提供使用 Workspace 原生工具的多轮终端 Agent |

`agent-e2bworkspace/` 是当前 E2BWorkspace 方案的完整示例目录。它包含两个部分：

1. E2BWorkspace 独立测试，用于确认模板运行环境、Gateway 初始化和 Shell 后端。
2. 多轮 Agent 应用，用于确认 Agent 能在同一个 E2BWorkspace 沙箱中持续调用
   Bash 和文件工具。

## 方案一：FunctionTool + E2B SDK

### 实现原理

```text
用户多轮输入
  -> AgentScope Agent
  -> run_sandbox_command FunctionTool
  -> e2b.AsyncSandbox.commands.run()
  -> E2B 沙箱执行命令
  -> 返回 exit_code/stdout/stderr
```

应用在启动时通过 E2B SDK 创建一个沙箱，并将 `sandbox.commands.run()` 包装成
AgentScope 自定义工具。同一次终端会话中的所有工具调用复用同一个 sandbox ID。

### 实现目录

```text
agentScopeDemo/agent-e2b-sdk/
├── main.py
├── .env
├── .env.example
├── .gitignore
└── README.md
```

各文件功能：

| 文件 | 功能 |
|---|---|
| `main.py` | 创建 AgentScope 多轮终端 Agent；通过 E2B SDK 创建一个沙箱；将 `sandbox.commands.run()` 封装为 `run_sandbox_command` FunctionTool；退出时销毁沙箱 |
| `.env` | 当前本地运行配置，保存 E2B、模型服务和模板等实际配置；包含敏感信息，不应提交到代码仓库 |
| `.env.example` | 环境变量示例模板，只声明运行所需的配置项，不保存真实 API Key |
| `.gitignore` | 忽略 `.env`、Python 缓存等不应提交的本地文件 |
| `README.md` | 说明方案一的用途、启动命令、默认 E2B 配置和交互示例 |

### 已验证结果

| 测试项 | 结果 |
|---|---|
| 模型连接 | 成功 |
| E2B 沙箱创建 | 成功 |
| Agent 多轮对话 | 成功 |
| Agent 自主调用工具 | 成功 |
| 工具人工确认 | 成功 |
| Shell/Python 命令执行 | 成功，退出码为 0 |
| 多轮复用沙箱 | 成功，同一会话使用相同 sandbox ID |
| 退出清理 | 输入 `exit` 后保留 10 秒并销毁沙箱 |

### 运行

```bash
cd /root/workspace/pyproject/e2b-examples/agentScopeDemo/agent-e2b-sdk
source /root/workspace/pyproject/.venv/bin/activate
python -u main.py
```

## 方案二：AgentScope E2BWorkspace

### 实现原理

`E2BWorkspace` 内部仍然使用 E2B SDK 创建、连接和管理沙箱。它不是绕过 E2B
SDK，而是在 SDK 之上统一封装沙箱生命周期、文件系统、Shell 后端、Gateway、
MCP 和标准 Workspace 工具。

```text
用户多轮输入
  -> AgentScope Agent
  -> E2BWorkspace 原生工具
  -> E2BBackend
  -> E2B SDK / envd
  -> 沙箱内执行命令或文件操作
  -> 返回工具结果
```

首次初始化流程：

```text
创建或连接 E2B 沙箱
  -> 创建 /home/user/workspace 等目录
  -> 使用模板现有 uv 创建 /home/user/.agentscope/.venv
  -> 安装 Gateway Python 依赖
  -> 安装 agentscope==2.0.7
  -> 上传并启动 Gateway
  -> 返回 Bash/Edit/Glob/Grep/Read/Write 工具
```

Python、uv 和 AgentScope Gateway 是三个不同层次：

- Python 和 uv 已由当前模板提供。
- Gateway 虚拟环境由 E2BWorkspace 在首次初始化时创建。
- Gateway 所需 Python 包仍需要安装进这个虚拟环境。

### `agent-e2bworkspace` 实现目录

```text
agentScopeDemo/agent-e2bworkspace/
├── README.md
├── test_e2bworkspace_runtime.py
└── e2bworkspace_agent.py
```

各文件功能：

| 文件 | 功能 |
|---|---|
| `test_e2bworkspace_runtime.py` | 创建 E2BWorkspace，验证模板现有 Python/uv、Gateway 初始化和 `E2BBackend.exec_shell` 命令执行，最后销毁测试沙箱 |
| `e2bworkspace_agent.py` | 创建支持多轮终端对话的 Agent；初始化并复用一个 E2BWorkspace；注册 Bash/Edit/Glob/Grep/Read/Write 工具；退出时销毁沙箱 |
| `README.md` | 说明方案二的环境信息、配置加载、测试方法、Agent 启动方法、执行原理和网络兼容处理 |

### 模板兼容处理

当前实现继承 `E2BWorkspace` 并只调整首次 bootstrap 命令：

1. 使用模板内置的 Python 3.12.12 和 uv 0.9.30。
2. 跳过 Python、uv、apt 和 ripgrep 安装。
3. 使用 uv 创建 Gateway 虚拟环境。
4. 安装 `mcp<2.0.0`、`uvicorn`、`fastapi`、`httpx`。
5. 安装 `agentscope==2.0.7`。

当前沙箱不能通过 DNS 解析 `mirrors.deepseek.org`，因此安装 Gateway 依赖时
临时增加：

```text
200.200.1.241 mirrors.deepseek.org
```

并向沙箱传入：

```python
env={
    "UV_DEFAULT_INDEX": "http://mirrors.deepseek.org/pypi/simple",
    "UV_INSECURE_HOST": "mirrors.deepseek.org",
    "UV_HTTP_TIMEOUT": "600",
}
```

- `UV_DEFAULT_INDEX` 指定内网 PyPI。
- `UV_INSECURE_HOST` 允许使用当前 HTTP 镜像源。
- `UV_HTTP_TIMEOUT` 延长下载超时，可按网络情况调整。

这些不是 E2BWorkspace 的固定要求，而是当前沙箱 DNS 和软件源环境的兼容措施。
如果沙箱 DNS 和 PyPI 访问恢复正常，可以移除临时域名映射并调整 uv 配置。

### E2BWorkspace 独立测试结果

已验证：

1. 使用模板创建 E2B 沙箱。
2. 初始化 E2BWorkspace 和 Gateway。
3. Gateway 使用 Python 3.12.12。
4. 通过 `E2BBackend.exec_shell` 执行 Shell 命令。
5. 保留沙箱 10 秒后正常销毁。

实际输出：

```text
os=Debian GNU/Linux 12 (bookworm)
system=Python 3.12.12
uv=uv 0.9.30
gateway=Python 3.12.12
shell=e2bworkspace-ok
[PASS] E2BWorkspace initialization and Shell execution passed.
```

### E2BWorkspace 多轮 Agent 测试结果

多轮 Agent 使用一个 E2BWorkspace 贯穿整个终端会话，并注册以下原生工具：

```text
Bash / Edit / Glob / Grep / Read / Write
```

实际验证过程：

1. 第一轮要求 Agent 调用 Bash 执行 `pwd && python3 --version`。
2. Bash 返回 `/home/user/workspace` 和 `Python 3.12.12`。
3. 第二轮不允许调用工具，Agent 能从对话上下文回答 Python 版本为 3.12.12。
4. 输入 `exit` 后沙箱正常销毁，程序退出码为 0。

这证明 Agent 多轮上下文、工具调用和同一沙箱复用均正常。

### 运行

```bash
cd /root/workspace/pyproject/e2b-examples/agentScopeDemo/agent-e2bworkspace
source /root/workspace/pyproject/.venv/bin/activate
python -u e2bworkspace_agent.py
```

输入 `exit`、`quit` 或按 `Ctrl+D` 可以结束对话。`Ctrl+C` 默认只中断当前一轮
回复。

## 两种方案对比

| 对比项 | FunctionTool + E2B SDK | E2BWorkspace |
|---|---|---|
| 沙箱创建 | 应用直接调用 E2B SDK | E2BWorkspace 内部调用 E2B SDK |
| Agent 工具 | 自定义一个 `run_sandbox_command` | 原生 Bash/Read/Write/Edit/Glob/Grep |
| 文件操作 | 需要自行封装或通过 Shell 完成 | Workspace 提供统一文件工具 |
| Gateway | 不需要 | 首次初始化时在沙箱中创建并启动 |
| MCP 扩展 | 需要应用自行集成 | 可通过 Workspace Gateway 统一管理 |
| 多轮复用沙箱 | 应用自行持有 Sandbox 实例 | Workspace 实例负责持有和管理 |
| 实现复杂度 | 当前简单命令场景更直接 | 工具和 Workspace 能力更完整 |
| 当前测试结果 | 通过 | 通过 |

对于“Agent 多轮对话并执行一个 Shell 工具”的最小需求，方案一更简单。需要
标准文件工具、统一 Workspace 抽象或后续 MCP/Skills 扩展时，方案二更适合。

## 当前版本组合

| 组件 | 当前版本 | 结果 |
|---|---|---|
| AgentScope | 2.0.7 | 正常 |
| E2B Python SDK | 2.34.0 | 正常 |
| 宿主机 Python | 3.13.10 | 正常 |
| 沙箱及 Gateway Python | 3.12.12 | 正常 |
| 沙箱 uv | 0.9.30 | 正常 |
