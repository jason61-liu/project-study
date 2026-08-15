from __future__ import annotations

from dataclasses import dataclass

import pytest

from sandbox import E2BSandboxExecutor, SandboxPolicy, SandboxPolicyError, ShellPolicy


@dataclass
class FakeInfo:
    cpu_count: int = 1
    memory_mb: int = 512
    allow_internet_access: bool = False
    volume_mounts: list = None

    def __post_init__(self):
        self.volume_mounts = self.volume_mounts or []


class FakeFiles:
    def __init__(self):
        self.writes = []

    def make_dir(self, path, **kwargs):
        self.directory = (path, kwargs)

    def write(self, path, content, **kwargs):
        self.writes.append((path, content, kwargs))


class FakeCommands:
    def run(self, command, **kwargs):
        self.call = (command, kwargs)
        return type("CommandResult", (), {"stdout": "ok", "stderr": "", "exit_code": 0})()


class FakeSandbox:
    sandbox_id = "fake-sandbox"

    def __init__(self):
        self.files = FakeFiles()
        self.commands = FakeCommands()
        self.killed = False

    def get_info(self):
        return FakeInfo()

    def kill(self):
        self.killed = True


def test_e2b_create_call_enforces_network_resources_secret_and_lifecycle():
    captured = {}
    sandbox = FakeSandbox()

    def factory(**kwargs):
        captured.update(kwargs)
        return sandbox

    executor = E2BSandboxExecutor(SandboxPolicy(), sandbox_factory=factory)
    result = executor.run_shell("printf hello", files={"input.txt": "safe"})

    assert result.stdout == "ok" and sandbox.killed
    assert captured["allow_internet_access"] is False
    assert captured["network"] == {"allow_public_traffic": False, "deny_out": ["0.0.0.0/0", "::/0"]}
    assert captured["lifecycle"] == {"on_timeout": "kill", "auto_resume": False}
    assert captured["timeout"] == 30 and captured["secure"] is True
    assert set(captured["envs"]) == {"WEEK9_TASK_ID"}
    assert "E2B_API_KEY" not in captured["envs"]
    command, options = sandbox.commands.call
    assert "ulimit -t 5" in command and "ulimit -f" in command
    assert options["timeout"] == 5.0 and options["cwd"] == "/home/user/work"


@pytest.mark.parametrize("command", ["bash -c id", "cat ../../etc/passwd", "printf ok | cat", "curl https://example.com"])
def test_shell_policy_blocks_escape_and_unallowlisted_executables(command):
    with pytest.raises(SandboxPolicyError):
        ShellPolicy(SandboxPolicy().allowed_executables).parse(command)


def test_file_api_rejects_absolute_and_traversal_paths():
    executor = E2BSandboxExecutor(SandboxPolicy(), sandbox_factory=lambda **_kwargs: FakeSandbox())
    with pytest.raises(SandboxPolicyError):
        executor.run_shell("printf hello", files={"../../escape": "x"})
    with pytest.raises(SandboxPolicyError):
        executor.run_shell("printf hello", files={"/etc/escape": "x"})


def test_resource_mismatch_fails_before_execution_and_still_kills():
    sandbox = FakeSandbox()
    sandbox.get_info = lambda: FakeInfo(cpu_count=2, memory_mb=1024)
    executor = E2BSandboxExecutor(SandboxPolicy(), sandbox_factory=lambda **_kwargs: sandbox)
    with pytest.raises(SandboxPolicyError, match="resources exceed"):
        executor.run_shell("printf hello")
    assert sandbox.killed
