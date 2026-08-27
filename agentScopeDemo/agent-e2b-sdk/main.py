#!/usr/bin/env python3
"""A multi-turn AgentScope console agent with one persistent E2B sandbox."""

from __future__ import annotations

import asyncio
import json
import os
import socket
import ssl
from pathlib import Path

import certifi
import httpx
from dotenv import load_dotenv


APP_DIR = Path(__file__).resolve().parent
CA_FILE = APP_DIR.parent / "sandboxtest" / "test-sandbox-ca.crt"

# Local values are loaded before constructing either the model or E2B client.
# Existing process environment variables still take precedence.
load_dotenv(APP_DIR / ".env")

# Configure the private E2B-compatible deployment before importing ``e2b``.
os.environ.setdefault("E2B_API_URL", "https://api.sandbox.ske-k8s251")
os.environ.setdefault("E2B_DOMAIN", "sandbox.ske-k8s251")
os.environ.setdefault("E2B_VALIDATE_API_KEY", "false")
os.environ.setdefault("SSL_CERT_FILE", str(CA_FILE))

from agentscope.agent import Agent  # noqa: E402
from agentscope.console import launch_console  # noqa: E402
from agentscope.credential import OpenAICredential  # noqa: E402
from agentscope.model import OpenAIChatModel  # noqa: E402
from agentscope.tool import FunctionTool, Toolkit  # noqa: E402
from e2b import AsyncSandbox  # noqa: E402


E2B_TEMPLATE_ID = os.getenv(
    "E2B_TEMPLATE_ID",
    "tpl-9f882069ddc2428bb5b62746",
)
E2B_LB_IP = os.getenv("E2B_LB_IP", "10.103.246.175")
E2B_TIMEOUT_SECONDS = int(os.getenv("E2B_TIMEOUT_SECONDS", "900"))
MAX_COMMAND_TIMEOUT_SECONDS = 300


def require_env(name: str) -> str:
    """Return a required environment variable or raise a clear error."""
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Set the {name} environment variable.")
    return value


def check_e2b_configuration() -> None:
    """Validate local E2B configuration without exposing credentials."""
    require_env("E2B_API_KEY")
    if not CA_FILE.is_file():
        raise RuntimeError(f"CubeSandbox CA certificate not found: {CA_FILE}")

    domain = os.environ["E2B_DOMAIN"].split(":", maxsplit=1)[0]
    resolved_ips = {
        item[4][0]
        for item in socket.getaddrinfo(domain, None, family=socket.AF_INET)
    }
    if E2B_LB_IP not in resolved_ips:
        raise RuntimeError(
            f"{domain} resolves to {sorted(resolved_ips)}, "
            f"expected LB {E2B_LB_IP}",
        )


def build_model() -> OpenAIChatModel:
    """Build an OpenAI-compatible AgentScope chat model from environment."""
    api_key = require_env("MODEL_API_KEY")
    base_url = require_env("MODEL_BASE_URL")
    model_name = require_env("MODEL_NAME")
    return OpenAIChatModel(
        credential=OpenAICredential(
            api_key=api_key,
            base_url=base_url,
        ),
        model=model_name,
        stream=True,
        # ``SSL_CERT_FILE`` points at the private CubeSandbox CA for E2B.
        # The model endpoint uses a public certificate, so isolate it on the
        # normal certifi trust bundle instead of sharing E2B's TLS context.
        client_kwargs={
            "http_client": httpx.AsyncClient(
                verify=ssl.create_default_context(cafile=certifi.where()),
            ),
        },
    )


async def main() -> None:
    """Create one sandbox and expose it to a multi-turn console agent."""
    check_e2b_configuration()
    model = build_model()
    sandbox: AsyncSandbox | None = None

    try:
        print("[INFO] Creating E2B sandbox...")
        sandbox = await AsyncSandbox.create(
            template=E2B_TEMPLATE_ID,
            timeout=E2B_TIMEOUT_SECONDS,
        )
        sandbox_id = sandbox.sandbox_id
        print(f"[OK] Sandbox created: {sandbox_id}")
        print(f"[OK] envd URL: {sandbox.envd_api_url}")

        async def run_sandbox_command(
            command: str,
            timeout_seconds: int = 60,
        ) -> str:
            """Execute a shell command inside the current E2B sandbox.

            Use this tool whenever the user asks to execute a shell command,
            run Python code, inspect the runtime environment, or manipulate
            files. The command never runs on the host machine.

            Args:
                command: Shell command to execute inside the E2B sandbox.
                timeout_seconds: Command timeout from 1 to 300 seconds.

            Returns:
                A JSON string containing sandbox ID, exit code, stdout and
                stderr.
            """
            command = command.strip()
            if not command:
                raise ValueError("command must not be empty")
            if not 1 <= timeout_seconds <= MAX_COMMAND_TIMEOUT_SECONDS:
                raise ValueError(
                    "timeout_seconds must be between 1 and "
                    f"{MAX_COMMAND_TIMEOUT_SECONDS}",
                )

            result = await sandbox.commands.run(
                command,
                timeout=timeout_seconds,
            )
            return json.dumps(
                {
                    "sandbox_id": sandbox_id,
                    "command": command,
                    "exit_code": result.exit_code,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                },
                ensure_ascii=False,
            )

        agent = Agent(
            name="E2B Assistant",
            system_prompt=(
                "You are a helpful assistant with access to one persistent "
                "E2B Linux sandbox. For every request that requires running "
                "a command, executing code, inspecting the environment, or "
                "reading/writing files, you MUST call run_sandbox_command. "
                "Never claim a command was executed unless the tool actually "
                "ran. Use python3 for Python code. Explain the observed "
                "stdout, stderr, and exit code clearly. All command execution "
                f"must occur in sandbox {sandbox_id}; never on the host."
            ),
            model=model,
            toolkit=Toolkit(
                tools=[
                    FunctionTool(
                        run_sandbox_command,
                        is_concurrency_safe=False,
                    ),
                ],
            ),
        )

        print("[INFO] Agent is ready. Start chatting below.")
        await launch_console(agent)
    finally:
        try:
            if sandbox is not None:
                print(
                    "[INFO] Sandbox remains available for 10 seconds before "
                    f"cleanup: {sandbox.sandbox_id}",
                )
                await asyncio.sleep(10)
                await sandbox.kill()
                print(f"[OK] Sandbox destroyed: {sandbox.sandbox_id}")
        finally:
            await model.client.close()


if __name__ == "__main__":
    asyncio.run(main())
