#!/usr/bin/env python3
"""Create a CubeSandbox with the official E2B SDK and run commands in it."""

from __future__ import annotations

import os
import socket
import time
from pathlib import Path


# These variables must be configured before importing ``e2b`` because the SDK
# reads them while constructing its control-plane and data-plane clients.
os.environ.setdefault("E2B_API_URL", "https://api.sandbox.ske-k8s251")
os.environ.setdefault("E2B_DOMAIN", "sandbox.ske-k8s251")
os.environ.setdefault("E2B_VALIDATE_API_KEY", "false")
os.environ.setdefault(
    "SSL_CERT_FILE",
    str(Path(__file__).with_name("test-sandbox-ca.crt")),
)

from e2b import Sandbox  # noqa: E402  (environment must be set first)


TEMPLATE_ID = os.getenv(
    "E2B_TEMPLATE_ID",
    "tpl-9f882069ddc2428bb5b62746",
)
LB_IP = os.getenv("E2B_LB_IP", "10.103.246.175")
SANDBOX_TIMEOUT_SECONDS = int(os.getenv("E2B_TIMEOUT_SECONDS", "300"))


def check_configuration() -> None:
    """Validate credentials and confirm that the data-plane uses the LB."""
    if not os.getenv("E2B_API_KEY"):
        raise RuntimeError(
            "E2B_API_KEY is required. Export it before running this script.",
        )

    domain = os.environ["E2B_DOMAIN"].split(":", maxsplit=1)[0]
    resolved_ips = {
        item[4][0]
        for item in socket.getaddrinfo(domain, None, family=socket.AF_INET)
    }
    if LB_IP not in resolved_ips:
        raise RuntimeError(
            f"{domain} resolves to {sorted(resolved_ips)}, expected LB {LB_IP}",
        )

    print(f"[OK] control plane: {os.environ['E2B_API_URL']}")
    print(f"[OK] data plane: {domain} -> {LB_IP}")
    print(f"[OK] template: {TEMPLATE_ID}")


def main() -> None:
    """Create a sandbox, execute commands through envd, and clean it up."""
    check_configuration()
    sandbox: Sandbox | None = None

    try:
        sandbox = Sandbox.create(
            template=TEMPLATE_ID,
            timeout=SANDBOX_TIMEOUT_SECONDS,
        )
        print(f"[OK] sandbox created: {sandbox.sandbox_id}")
        print(f"[OK] envd URL: {sandbox.envd_api_url}")

        result = sandbox.commands.run(
            "pwd && echo 'hello from E2B sandbox' && python3 --version",
            timeout=60,
        )
        print(f"[OK] command exit code: {result.exit_code}")
        if result.stdout:
            print("--- stdout ---")
            print(result.stdout.rstrip())
        if result.stderr:
            print("--- stderr ---")
            print(result.stderr.rstrip())

        if result.exit_code != 0:
            raise RuntimeError(
                f"Sandbox command failed with exit code {result.exit_code}",
            )
    finally:
        if sandbox is not None:
            print(
                "[INFO] sandbox remains available for 10 seconds before "
                f"cleanup: {sandbox.sandbox_id}",
            )
            time.sleep(10)
            sandbox.kill()
            print(f"[OK] sandbox destroyed: {sandbox.sandbox_id}")


if __name__ == "__main__":
    main()
