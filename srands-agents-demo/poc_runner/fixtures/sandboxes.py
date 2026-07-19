"""测试夹具：Mock Sandbox 实现"""

from typing import Any, AsyncGenerator

from strands.sandbox.base import Sandbox
from strands.sandbox.types import ExecutionResult, FileInfo


class MockDockerSandbox(Sandbox):
    """模拟 Docker Sandbox — 所有结果带 [Docker] 标识"""

    def __init__(self):
        self._execute_log: list[dict[str, Any]] = []

    @property
    def execute_log(self) -> list[dict[str, Any]]:
        return list(self._execute_log)

    async def execute_streaming(
        self, command: str, *, timeout=None, cwd=None, env=None, **kwargs
    ) -> AsyncGenerator[Any, None]:
        self._execute_log.append({"command": command, "type": "docker"})
        yield ExecutionResult(
            exit_code=0,
            stdout=f"[Docker] executed: {command}",
            stderr="",
        )

    async def execute_code_streaming(
        self, code: str, language: str, *, timeout=None, cwd=None, env=None, **kwargs
    ) -> AsyncGenerator[Any, None]:
        yield ExecutionResult(
            exit_code=0,
            stdout=f"[Docker] code: {code[:50]}...",
            stderr="",
        )

    async def read_file(self, path: str, **kwargs) -> bytes:
        return f"[Docker] content of {path}".encode()

    async def write_file(self, path: str, content: bytes, **kwargs) -> None:
        pass

    async def remove_file(self, path: str, **kwargs) -> None:
        pass

    async def list_files(self, path: str, **kwargs) -> list[FileInfo]:
        return [FileInfo(name="docker_file.txt", is_dir=False, size=100)]


class MockSshSandbox(Sandbox):
    """模拟 SSH Sandbox — 所有结果带 [SSH] 标识"""

    def __init__(self):
        self._execute_log: list[dict[str, Any]] = []

    @property
    def execute_log(self) -> list[dict[str, Any]]:
        return list(self._execute_log)

    async def execute_streaming(
        self, command: str, *, timeout=None, cwd=None, env=None, **kwargs
    ) -> AsyncGenerator[Any, None]:
        self._execute_log.append({"command": command, "type": "ssh"})
        yield ExecutionResult(
            exit_code=0,
            stdout=f"[SSH] executed: {command}",
            stderr="",
        )

    async def execute_code_streaming(
        self, code: str, language: str, *, timeout=None, cwd=None, env=None, **kwargs
    ) -> AsyncGenerator[Any, None]:
        yield ExecutionResult(
            exit_code=0,
            stdout=f"[SSH] code: {code[:50]}...",
            stderr="",
        )

    async def read_file(self, path: str, **kwargs) -> bytes:
        return f"[SSH] content of {path}".encode()

    async def write_file(self, path: str, content: bytes, **kwargs) -> None:
        pass

    async def remove_file(self, path: str, **kwargs) -> None:
        pass

    async def list_files(self, path: str, **kwargs) -> list[FileInfo]:
        return [FileInfo(name="ssh_file.txt", is_dir=False, size=200)]
