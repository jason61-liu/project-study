# AgentScope E2BWorkspace 示例

## 结论

当前环境已验证可以使用 AgentScope 2.0.7 的 `E2BWorkspace`：

- 能通过指定模板创建 E2B 沙箱。
- 能在沙箱内初始化并启动 AgentScope Gateway。
- 能通过 E2BWorkspace 执行 Shell 命令和操作文件。
- 能创建一个支持多轮终端对话的 Agent，并在多轮对话中复用同一个沙箱。
- 退出 Agent 后能正常销毁沙箱。

当前模板已经内置 Python 3.12.12 和 uv 0.9.30，不需要在沙箱中安装、升级
或编译 Python，也不需要安装或升级 uv。

## 目录内容

```text
agent-e2bworkspace/
├── README.md
├── e2bworkspace_agent.py
└── test_e2bworkspace_runtime.py
```

| 文件 | 用途 |
|---|---|
| `test_e2bworkspace_runtime.py` | 独立验证 E2BWorkspace 初始化和 Shell 执行 |
| `e2bworkspace_agent.py` | 使用 E2BWorkspace 的多轮终端 Agent 应用 |

## 环境信息

| 项目 | 当前配置 |
|---|---|
| E2B API | `https://api.sandbox.ske-k8s251` |
| E2B Domain | `sandbox.ske-k8s251` |
| E2B LB | `10.103.246.175` |
| 模板 ID | `tpl-27b366cc77df425b82d51b46` |
| CA 证书 | `../sandboxtest/test-sandbox-ca.crt` |
| AgentScope | 2.0.7 |
| 沙箱系统 | Debian GNU/Linux 12 |
| 沙箱 Python | 3.12.12 |
| 沙箱 uv | 0.9.30 |

## 配置文件

多轮 Agent 按以下顺序查找 `.env`：

1. 当前目录的 `.env`。
2. `../agent-e2b-sdk/.env`。
3. 兼容旧目录结构的 `../agent/.env`。

当前环境可以直接复用 `../agent-e2b-sdk/.env`。运行 Agent 至少需要以下变量：

```dotenv
E2B_API_KEY=<E2B API Key>
MODEL_API_KEY=<模型 API Key>
MODEL_BASE_URL=<OpenAI 兼容接口地址>
MODEL_NAME=<模型名称>
```

可选变量：

```dotenv
E2B_LB_IP=10.103.246.175
E2B_TIMEOUT_SECONDS=900
E2B_KEEP_ALIVE_SECONDS=10
```

不要把真实 API Key 写入 README 或提交到代码仓库。

## 运行多轮 Agent

进入当前目录并激活虚拟环境：

```bash
cd /root/workspace/pyproject/e2b-examples/agentScopeDemo/agent-e2bworkspace
source /root/workspace/pyproject/.venv/bin/activate
python -u e2bworkspace_agent.py
```

启动过程包括：

1. 加载模型和 E2B 配置。
2. 创建一个 E2BWorkspace 沙箱。
3. 使用模板已有的 Python 和 uv 初始化 Gateway。
4. 为 Agent 注册 E2BWorkspace 原生工具。
5. 进入终端多轮交互。

当前 Agent 注册了以下工具，所有命令和文件操作都在沙箱中执行：

```text
Bash / Edit / Glob / Grep / Read / Write
```

示例对话：

```text
user> 请使用 Bash 执行 pwd 和 python3 --version。
user> 创建 hello.py，运行它并告诉我输出。
user> 刚才创建的文件叫什么？这次不要调用工具。
```

部分工具调用需要人工确认，终端出现确认提示时可以输入：

```text
y  本次允许
n  本次拒绝
a  当前会话后续同类调用始终允许
```

结束对话可以输入 `exit`、`quit`，或按 `Ctrl+D`。`Ctrl+C` 默认用于中断当前
一轮回复，不是退出整个程序。

同一次终端会话中的所有轮次共享一个沙箱和文件系统。退出后脚本默认保留沙箱
10 秒用于页面观察，然后自动销毁。

## 运行 E2BWorkspace 独立测试

测试脚本只需要 E2B 配置。当前目录没有单独的 `.env` 时，可先导出已有配置：

```bash
cd /root/workspace/pyproject/e2b-examples/agentScopeDemo/agent-e2bworkspace
source /root/workspace/pyproject/.venv/bin/activate
set -a
source ../agent-e2b-sdk/.env
set +a
python test_e2bworkspace_runtime.py
```

已验证输出：

```text
os=Debian GNU/Linux 12 (bookworm)
system=Python 3.12.12
uv=uv 0.9.30
gateway=Python 3.12.12
shell=e2bworkspace-ok
[PASS] E2BWorkspace initialization and Shell execution passed.
```

测试完成后，沙箱保留 10 秒并自动销毁。

## E2BWorkspace 执行原理

```text
用户多轮输入
  -> AgentScope Agent
  -> E2BWorkspace 原生工具
  -> E2BBackend
  -> E2B SDK / envd
  -> 沙箱内执行命令和文件操作
  -> 将结果返回 Agent
```

与 `agent-e2b-sdk/main.py` 的直接 SDK 方案相比：

| 方案 | 工具实现 |
|---|---|
| 直接 E2B SDK | 应用自行封装 `sandbox.commands.run()` 为 FunctionTool |
| E2BWorkspace | 使用 Workspace 统一提供的 Bash、Read、Write 等原生工具 |

E2BWorkspace 初始化时还会在沙箱中创建 Gateway 虚拟环境，用于承载 AgentScope
的 Workspace/Gateway 能力。Agent 程序和大模型仍然运行在宿主机，命令及文件
操作通过 E2BWorkspace 后端进入沙箱执行。

## 当前网络兼容处理

新创建的沙箱中，`mirrors.deepseek.org` 当前无法通过 DNS 正常解析。实测表现为：

```text
socket.gaierror: Temporary failure in name resolution
curl: Resolving timed out
```

因此两个脚本中的 `ExistingRuntimeE2BWorkspace` 对 Gateway 安装命令增加了临时
域名映射：

```text
200.200.1.241 mirrors.deepseek.org
```

同时向沙箱传入：

```python
env={
    "UV_DEFAULT_INDEX": "http://mirrors.deepseek.org/pypi/simple",
    "UV_INSECURE_HOST": "mirrors.deepseek.org",
    "UV_HTTP_TIMEOUT": "600",
}
```

- `UV_DEFAULT_INDEX`：让 uv 使用内网 PyPI。
- `UV_INSECURE_HOST`：允许访问 HTTP 镜像源。
- `UV_HTTP_TIMEOUT`：延长下载超时，可按实际网络情况调整。

这些配置不是 E2BWorkspace 的固定要求，而是当前沙箱网络环境的兼容措施。如果
沙箱 DNS 和 PyPI 访问恢复正常，可以移除临时域名映射，并按可访问的软件源调整
uv 配置。
