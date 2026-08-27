#!/usr/bin/env python3
"""Multi-turn AgentScope console agent backed by one E2BWorkspace."""

from __future__ import annotations

import asyncio
import os
import socket
import ssl
import time
from pathlib import Path

import certifi
import httpx
from dotenv import load_dotenv


HERE = Path(__file__).resolve().parent
DEMO_DIR = HERE.parent
CA_FILE = DEMO_DIR / "sandboxtest" / "test-sandbox-ca.crt"
ENV_CANDIDATES = (
    HERE / ".env",
    DEMO_DIR / "agent-e2b-sdk" / ".env",
    DEMO_DIR / "agent" / ".env",
)
ENV_FILE = next(
    (path for path in ENV_CANDIDATES if path.is_file()),
    ENV_CANDIDATES[0],
)

# Read an explicit shell override before loading the older agent's .env,
# whose template value belongs to the direct-E2B-SDK example.
TEMPLATE_ID = os.getenv(
    "E2B_TEMPLATE_ID",
    "tpl-27b366cc77df425b82d51b46",
)

# Prefer this application's own .env. For the current workspace layout, fall
# back to the direct-E2B-SDK agent's existing .env without modifying it.
load_dotenv(ENV_FILE)

os.environ.setdefault("E2B_API_URL", "https://api.sandbox.ske-k8s251")
os.environ.setdefault("E2B_DOMAIN", "sandbox.ske-k8s251")
os.environ.setdefault("E2B_VALIDATE_API_KEY", "false")
os.environ.setdefault("SSL_CERT_FILE", str(CA_FILE))

from agentscope.agent import Agent  # noqa: E402
from agentscope.console import launch_console  # noqa: E402
from agentscope.credential import OpenAICredential  # noqa: E402
from agentscope.model import OpenAIChatModel  # noqa: E402
from agentscope.tool import Toolkit  # noqa: E402
from agentscope.workspace import E2BWorkspace  # noqa: E402
from e2b import AsyncSandbox  # noqa: E402


E2B_LB_IP = os.getenv("E2B_LB_IP", "10.103.246.175")
E2B_TIMEOUT_SECONDS = int(os.getenv("E2B_TIMEOUT_SECONDS", "900"))
E2B_KEEP_ALIVE_SECONDS = int(os.getenv("E2B_KEEP_ALIVE_SECONDS", "10"))


class ExistingRuntimeE2BWorkspace(E2BWorkspace):
    """Use the template's existing Python and uv for Gateway bootstrap."""

    def _bootstrap_commands(self) -> list[str]:
        commands = super()._bootstrap_commands()
        hosts_prefix = (
            "printf '200.200.1.241 mirrors.deepseek.org\\n' > "
            "/tmp/agentscope-hosts && "
            "mount --bind /tmp/agentscope-hosts /etc/hosts && "
        )
        pinned_agentscope = commands[4].replace(
            "'agentscope'",
            "'agentscope==2.0.7'",
        )
        return [
            "python3 --version && uv --version",
            commands[2],
            hosts_prefix + commands[3],
            hosts_prefix + pinned_agentscope,
        ]


def require_env(name: str) -> str:
    """Return a required environment variable or raise a clear error."""
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Set the {name} environment variable.")
    return value


def check_e2b_configuration() -> None:
    """Validate the private E2B endpoint without exposing credentials."""
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
    """Build the same OpenAI-compatible model used by the original agent."""
    return OpenAIChatModel(
        credential=OpenAICredential(
            api_key=require_env("MODEL_API_KEY"),
            base_url=require_env("MODEL_BASE_URL"),
        ),
        model=require_env("MODEL_NAME"),
        stream=True,
        # E2B uses a private CA, while the model endpoint uses public CAs.
        client_kwargs={
            "http_client": httpx.AsyncClient(
                verify=ssl.create_default_context(cafile=certifi.where()),
            ),
        },
    )


async def main() -> None:
    """Run a multi-turn console agent with one persistent E2BWorkspace."""
    check_e2b_configuration()
    model = build_model()
    workspace = ExistingRuntimeE2BWorkspace(
        workspace_id=f"console-agent-{int(time.time())}",
        template=TEMPLATE_ID,
        api_key=require_env("E2B_API_KEY"),
        domain=os.environ["E2B_DOMAIN"],
        timeout_seconds=E2B_TIMEOUT_SECONDS,
        env={
            "UV_DEFAULT_INDEX": "http://mirrors.deepseek.org/pypi/simple",
            "UV_INSECURE_HOST": "mirrors.deepseek.org",
            "UV_HTTP_TIMEOUT": "600",
        },
        sandbox_metadata={"purpose": "agentscope-e2bworkspace-console-agent"},
    )

    try:
        print("[INFO] Initializing E2BWorkspace...")
        await workspace.initialize()
        sandbox_id = workspace.sandbox_id
        print(f"[OK] E2BWorkspace initialized: {sandbox_id}")

        workspace_instructions = await workspace.get_instructions()
        agent = Agent(
            name="E2BWorkspace Assistant",
            system_prompt=(
                "You are a helpful assistant working in one persistent E2B "
                "Linux sandbox. Use the workspace tools whenever the user "
                "asks you to execute commands or code, inspect the runtime, "
                "or read and write files. Never claim that an operation ran "
                "unless a workspace tool actually returned its result. The "
                "same sandbox and filesystem are shared across all turns in "
                "this console session.\n\n"
                + workspace_instructions
            ),
            model=model,
            toolkit=Toolkit(tools=await workspace.list_tools()),
        )

        print("[INFO] Agent is ready. Type exit or quit to stop.")
        await launch_console(agent)
    finally:
        sandbox_id = workspace.sandbox_id
        try:
            if sandbox_id and workspace.is_alive:
                print(
                    f"[INFO] Keeping sandbox {sandbox_id} visible for "
                    f"{E2B_KEEP_ALIVE_SECONDS} seconds before cleanup.",
                )
                await asyncio.sleep(E2B_KEEP_ALIVE_SECONDS)
            if sandbox_id:
                await AsyncSandbox.kill(
                    sandbox_id,
                    api_key=require_env("E2B_API_KEY"),
                    domain=os.environ["E2B_DOMAIN"],
                )
                print(f"[OK] Sandbox destroyed: {sandbox_id}")
                # E2BWorkspace.close() pauses a live sandbox. This private E2B
                # deployment cannot delete a paused sandbox, so kill it while
                # running and detach the already-destroyed remote handle before
                # closing the local Gateway client.
                workspace._sandbox = None  # noqa: SLF001
            await workspace.close()
        finally:
            await model.client.close()


if __name__ == "__main__":
    asyncio.run(main())
