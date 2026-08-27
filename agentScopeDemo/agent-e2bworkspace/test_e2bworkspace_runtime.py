#!/usr/bin/env python3
"""Verify E2BWorkspace using the template's existing Python and uv."""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEMO_DIR = HERE.parent
ENV_FILE = DEMO_DIR / "agent" / ".env"
CA_FILE = DEMO_DIR / "sandboxtest" / "test-sandbox-ca.crt"

API_URL = "https://api.sandbox.ske-k8s251"
DOMAIN = "sandbox.ske-k8s251"
TEMPLATE_ID = os.getenv(
    "E2B_TEMPLATE_ID",
    "tpl-27b366cc77df425b82d51b46",
)
KEEP_ALIVE_SECONDS = int(os.getenv("E2B_KEEP_ALIVE_SECONDS", "10"))


def load_env_file(path: Path) -> None:
    """Load simple KEY=VALUE entries without overriding shell variables."""
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", maxsplit=1)
        key = key.strip()
        value = value.strip()
        if value[:1] == value[-1:] and value[:1] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


load_env_file(ENV_FILE)
os.environ["E2B_API_URL"] = API_URL
os.environ["E2B_DOMAIN"] = DOMAIN
os.environ["E2B_VALIDATE_API_KEY"] = "false"
os.environ["SSL_CERT_FILE"] = str(CA_FILE)

from agentscope.workspace import E2BWorkspace  # noqa: E402
from e2b import AsyncSandbox  # noqa: E402


class ExistingRuntimeE2BWorkspace(E2BWorkspace):
    """Bootstrap Gateway without installing Python, uv, apt packages."""

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
            # Verification only: these commands do not install or upgrade
            # the Python and uv already provided by the template.
            "python3 --version && uv --version",
            commands[2],
            hosts_prefix + commands[3],
            hosts_prefix + pinned_agentscope,
        ]


async def main() -> None:
    """Create E2BWorkspace, execute Shell, then destroy the sandbox."""
    api_key = os.getenv("E2B_API_KEY", "")
    if not api_key:
        raise RuntimeError("E2B_API_KEY is required")

    workspace = ExistingRuntimeE2BWorkspace(
        workspace_id=f"existing-runtime-test-{int(time.time())}",
        template=TEMPLATE_ID,
        api_key=api_key,
        domain=DOMAIN,
        timeout_seconds=1800,
        env={
            "UV_DEFAULT_INDEX": "http://mirrors.deepseek.org/pypi/simple",
            "UV_INSECURE_HOST": "mirrors.deepseek.org",
            "UV_HTTP_TIMEOUT": "600",
        },
    )
    sandbox_id: str | None = None

    try:
        print("[STEP] Initialize E2BWorkspace with existing Python/uv")
        await workspace.initialize()
        sandbox_id = workspace.sandbox_id
        print(f"[OK] E2BWorkspace initialized: {sandbox_id}")

        result = await workspace.get_backend().exec_shell(
            [
                "sh",
                "-c",
                "set -eu; "
                ". /etc/os-release; echo \"os=$PRETTY_NAME\"; "
                "echo system=$(python3 --version 2>&1); "
                "echo uv=$(uv --version 2>&1); "
                "echo gateway=$(/home/user/.agentscope/.venv/bin/python --version 2>&1); "
                "echo shell=e2bworkspace-ok",
            ],
            timeout=60,
        )
        stdout = result.stdout.decode(errors="replace")
        stderr = result.stderr.decode(errors="replace")
        print(stdout.rstrip())
        if stderr:
            print(stderr.rstrip())
        if not result.ok():
            raise RuntimeError(f"E2BWorkspace shell failed: {result.exit_code}")
        if "shell=e2bworkspace-ok" not in stdout:
            raise RuntimeError("Shell execution evidence is missing")

        print("[PASS] E2BWorkspace initialization and Shell execution passed.")
    finally:
        if sandbox_id is None:
            sandbox_id = workspace.sandbox_id
        if sandbox_id:
            print(
                f"[INFO] Keeping sandbox {sandbox_id} visible for "
                f"{KEEP_ALIVE_SECONDS} seconds before cleanup.",
            )
            await asyncio.sleep(KEEP_ALIVE_SECONDS)
            await AsyncSandbox.kill(
                sandbox_id,
                api_key=api_key,
                domain=DOMAIN,
            )
            print(f"[OK] sandbox destroyed: {sandbox_id}")
            workspace._sandbox = None  # noqa: SLF001
        if workspace.is_alive:
            await workspace.close()


if __name__ == "__main__":
    asyncio.run(main())
