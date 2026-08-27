# AgentScope + E2B console agent

This is a minimal multi-turn AgentScope console agent. One E2B sandbox is
created when the process starts and reused by every `run_sandbox_command` tool
call in that console session. The sandbox is destroyed 10 seconds after the
console exits.

## Run

Copy and edit the environment file when setting up a new checkout:

```bash
cp .env.example .env
```

The current directory already has a configured local `.env`. Start the agent
in the existing virtual environment:

```bash
source /root/workspace/pyproject/.venv/bin/activate
python -u /root/workspace/pyproject/e2b-examples/agentScopeDemo/agent/main.py
```

Example request after the console starts:

```text
请在沙箱中执行 pwd、python3 --version 和 uname -a，并解释结果。
```

Custom function tools require confirmation by default. Approve the
`run_sandbox_command` call in the console to execute it.

## E2B defaults

- API URL: `https://api.sandbox.ske-k8s251`
- Domain: `sandbox.ske-k8s251`
- LB: `10.103.246.175`
- Template: `tpl-9f882069ddc2428bb5b62746`
- CA: `../sandboxtest/test-sandbox-ca.crt`

Override them with `E2B_API_URL`, `E2B_DOMAIN`, `E2B_LB_IP`,
`E2B_TEMPLATE_ID`, or `E2B_TIMEOUT_SECONDS` when necessary.

The E2B client trusts the private CubeSandbox CA, while the model client uses
the public certifi CA bundle. Their TLS trust settings are intentionally kept
separate.
