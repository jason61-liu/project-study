from __future__ import annotations

import resource
import subprocess
import sys
import tempfile
from dataclasses import dataclass


@dataclass(frozen=True)
class SandboxResult:
    returncode: int
    stdout: str
    stderr: str


class Sandbox:
    """Minimal local fallback. Production deployments should swap this for E2B/OpenSandbox."""

    ALLOWED = {"python3", "jq"}

    @staticmethod
    def _limits() -> None:
        resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
        resource.setrlimit(resource.RLIMIT_AS, (128 * 1024 * 1024, 128 * 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))

    def execute(self, argv: list[str], timeout_seconds: float = 3.0) -> SandboxResult:
        if not argv or argv[0] not in self.ALLOWED:
            raise PermissionError("command is not allowlisted")
        if timeout_seconds > 5:
            raise ValueError("sandbox timeout exceeds policy")
        with tempfile.TemporaryDirectory(prefix="research-sandbox-") as directory:
            completed = subprocess.run(
                argv,
                cwd=directory,
                env={"PATH": "/usr/bin:/bin"},
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                # macOS rejects RLIMIT_AS in some managed runtimes; a production
                # deployment uses the container profile documented in SECURITY.md.
                preexec_fn=self._limits if sys.platform.startswith("linux") else None,
                check=False,
            )
        return SandboxResult(completed.returncode, completed.stdout[:16_000], completed.stderr[:16_000])
