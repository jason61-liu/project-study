"""Managed E2B adapter with fail-closed command and lifecycle controls."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import PurePosixPath
import shlex
from typing import Any, Callable, Protocol
from uuid import uuid4


class SandboxPolicyError(Exception):
    """The request or provisioned sandbox violates the declared policy."""


@dataclass(frozen=True)
class SandboxPolicy:
    template: str = "week9-secure-1c-512m"
    cpu_count: int = 1
    memory_mb: int = 512
    command_timeout_s: float = 5.0
    lifetime_s: int = 30
    max_output_bytes: int = 32_768
    max_file_bytes: int = 1_048_576
    workdir: str = "/home/user/work"
    allowed_executables: frozenset[str] = frozenset({"python3", "printf", "ls", "cat", "wc"})


@dataclass(frozen=True)
class SandboxResult:
    exit_code: int
    stdout: str
    stderr: str
    sandbox_id: str
    truncated: bool = False


class SandboxExecutor(Protocol):
    def run(self, argv: list[str], *, files: dict[str, str] | None = None) -> SandboxResult: ...


class ShellPolicy:
    """Turn a shell-like input into one allowlisted argv without shell syntax."""

    META = frozenset({"|", "||", "&", "&&", ";", ">", ">>", "<", "<<", "`", "$(`"})

    def __init__(self, allowed_executables: frozenset[str]) -> None:
        self.allowed_executables = allowed_executables

    def parse(self, command: str) -> list[str]:
        if not command or len(command) > 8_000:
            raise SandboxPolicyError("command is empty or too large")
        try:
            argv = shlex.split(command, posix=True)
        except ValueError as exc:
            raise SandboxPolicyError("invalid shell quoting") from exc
        return self.validate(argv)

    def validate(self, argv: list[str]) -> list[str]:
        if not argv or argv[0] not in self.allowed_executables:
            raise SandboxPolicyError("executable is not allowlisted")
        for token in argv:
            if token in self.META or any(marker in token for marker in ("$(`", "${", "\n", "\r")):
                raise SandboxPolicyError("shell metacharacters are forbidden")
            if token.startswith("/") or ".." in PurePosixPath(token).parts:
                raise SandboxPolicyError("absolute paths and traversal are forbidden")
        return list(argv)


def _safe_relative_path(path: str) -> str:
    value = PurePosixPath(path)
    if value.is_absolute() or not value.parts or ".." in value.parts:
        raise SandboxPolicyError("file path must stay under the sandbox workdir")
    return str(value)


class E2BSandboxExecutor:
    """Create one disposable, network-denied E2B microVM per execution.

    CPU/RAM are properties of the named E2B template.  The adapter verifies
    the provisioned values before writing files or running code.  No volume is
    mounted and no credential is injected into the guest environment.
    """

    def __init__(
        self,
        policy: SandboxPolicy | None = None,
        *,
        sandbox_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.policy = policy or SandboxPolicy(template=os.getenv("E2B_TEMPLATE", SandboxPolicy.template))
        self._factory = sandbox_factory
        self.shell_policy = ShellPolicy(self.policy.allowed_executables)

    def run_shell(self, command: str, *, files: dict[str, str] | None = None) -> SandboxResult:
        return self.run(self.shell_policy.parse(command), files=files)

    def run_code(self, code: str, *, files: dict[str, str] | None = None) -> SandboxResult:
        if not code or len(code.encode()) > 32_768:
            raise SandboxPolicyError("code is empty or too large")
        return self.run(["python3", "-I", "-c", code], files=files)

    def run(self, argv: list[str], *, files: dict[str, str] | None = None) -> SandboxResult:
        argv = self.shell_policy.validate(argv)
        checked_files = {_safe_relative_path(name): content for name, content in (files or {}).items()}
        if any(len(content.encode()) > self.policy.max_file_bytes for content in checked_files.values()):
            raise SandboxPolicyError("input file exceeds size limit")

        if self._factory is None:
            from e2b import Sandbox

            factory = Sandbox.create
        else:
            factory = self._factory

        sandbox = factory(
            template=self.policy.template,
            timeout=self.policy.lifetime_s,
            secure=True,
            allow_internet_access=False,
            network={
                "allow_public_traffic": False,
                "deny_out": ["0.0.0.0/0", "::/0"],
            },
            lifecycle={"on_timeout": "kill", "auto_resume": False},
            envs={"WEEK9_TASK_ID": uuid4().hex},
        )
        try:
            info = sandbox.get_info()
            if info.cpu_count > self.policy.cpu_count or info.memory_mb > self.policy.memory_mb:
                raise SandboxPolicyError(
                    f"template resources exceed policy: {info.cpu_count} CPU/{info.memory_mb} MiB"
                )
            if info.allow_internet_access is not False:
                raise SandboxPolicyError("sandbox did not confirm network denial")
            if getattr(info, "volume_mounts", []):
                raise SandboxPolicyError("persistent volume mounts are forbidden")

            sandbox.files.make_dir(self.policy.workdir, user="user")
            for name, content in checked_files.items():
                sandbox.files.write(f"{self.policy.workdir}/{name}", content, user="user")

            # argv is quoted after allowlist validation. ulimit constrains guest
            # CPU seconds and file creation; RAM is constrained by the template.
            command = (
                f"ulimit -t {max(1, int(self.policy.command_timeout_s))}; "
                f"ulimit -f {max(1, self.policy.max_file_bytes // 512)}; "
                f"exec {shlex.join(argv)}"
            )
            result = sandbox.commands.run(
                command,
                cwd=self.policy.workdir,
                user="user",
                timeout=self.policy.command_timeout_s,
            )
            stdout, out_cut = self._bounded(result.stdout or "")
            stderr, err_cut = self._bounded(result.stderr or "")
            return SandboxResult(
                exit_code=int(result.exit_code),
                stdout=stdout,
                stderr=stderr,
                sandbox_id=str(sandbox.sandbox_id),
                truncated=out_cut or err_cut,
            )
        finally:
            # kill is permanent; timeout=kill is a second lifecycle backstop.
            sandbox.kill()

    def _bounded(self, value: str) -> tuple[str, bool]:
        raw = value.encode()
        if len(raw) <= self.policy.max_output_bytes:
            return value, False
        return raw[: self.policy.max_output_bytes].decode(errors="replace"), True


class UnavailableSandboxExecutor:
    """Explicit offline sentinel; it never executes code on the host."""

    def run(self, argv: list[str], *, files: dict[str, str] | None = None) -> SandboxResult:
        raise SandboxPolicyError("E2B live execution requires E2B_API_KEY")

    def run_shell(self, command: str, *, files: dict[str, str] | None = None) -> SandboxResult:
        raise SandboxPolicyError("E2B live execution requires E2B_API_KEY")

    def run_code(self, code: str, *, files: dict[str, str] | None = None) -> SandboxResult:
        raise SandboxPolicyError("E2B live execution requires E2B_API_KEY")
