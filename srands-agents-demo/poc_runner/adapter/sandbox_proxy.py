"""CMA Sandbox Proxy — 拦截 bash 调用重定向到 CMA Sandbox（gRPC stub）"""

from typing import Any, AsyncGenerator

from strands.hooks.events import BeforeToolCallEvent
from strands.sandbox.base import Sandbox
from strands.sandbox.types import ExecutionResult, FileInfo


class CmaSandboxProxy(Sandbox):
    """Sandbox 代理：将命令执行转发到 CMA Sandbox（gRPC stub）

    当前实现为本地 stub，用文件系统模拟远程执行。
    CMA 集成时替换为真正的 gRPC 调用。
    """

    def __init__(self, *, prefix: str = "[CMA Sandbox]", record_log: bool = True):
        self._prefix = prefix
        self._record_log = record_log
        self._execute_log: list[dict[str, Any]] = []

    @property
    def execute_log(self) -> list[dict[str, Any]]:
        """获取所有 execute 调用日志"""
        return list(self._execute_log)

    def clear_log(self) -> None:
        self._execute_log.clear()

    async def execute_streaming(
        self,
        command: str,
        *,
        timeout: float | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[Any, None]:
        """执行命令（stub 实现：返回标记性结果）"""
        import subprocess
        import sys

        if self._record_log:
            self._execute_log.append({
                "command": command,
                "cwd": cwd,
                "timeout": timeout,
            })

        # Stub: 本地执行命令（CMA 集成时替换为 gRPC 调用）
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd,
                env=env,
            )
            yield ExecutionResult(
                exit_code=result.returncode,
                stdout=f"{self._prefix} stdout: {result.stdout}",
                stderr=result.stderr,
            )
        except subprocess.TimeoutExpired as e:
            yield ExecutionResult(
                exit_code=1,
                stdout="",
                stderr=f"{self._prefix} timeout: {e}",
            )

    async def execute_code_streaming(
        self,
        code: str,
        language: str,
        *,
        timeout: float | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[Any, None]:
        """执行代码（stub）"""
        yield ExecutionResult(
            exit_code=0,
            stdout=f"{self._prefix} code executed ({language})",
            stderr="",
        )

    async def read_file(self, path: str, **kwargs: Any) -> bytes:
        import pathlib
        return pathlib.Path(path).read_bytes()

    async def write_file(self, path: str, content: bytes, **kwargs: Any) -> None:
        import pathlib
        p = pathlib.Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content)

    async def remove_file(self, path: str, **kwargs: Any) -> None:
        import pathlib
        p = pathlib.Path(path)
        if p.exists():
            p.unlink()

    async def list_files(self, path: str, **kwargs: Any) -> list[FileInfo]:
        import pathlib
        p = pathlib.Path(path)
        if not p.exists():
            return []
        result = []
        for item in p.iterdir():
            result.append(FileInfo(
                name=item.name,
                is_dir=item.is_dir(),
                size=item.stat().st_size if item.is_file() else None,
            ))
        return result


def make_cma_redirected_bash(sandbox_proxy: CmaSandboxProxy):
    """创建重定向到 CMA Sandbox Proxy 的 bash 工具"""
    from strands.vended_tools.bash.bash import make_bash

    return make_bash(sandbox=sandbox_proxy, name="bash")


def create_sandbox_redirect_hook(
    sandbox_proxy: CmaSandboxProxy,
    cma_bash_tool=None,
):
    """创建 BeforeToolCallEvent hook，拦截 bash 调用并重定向"""
    if cma_bash_tool is None:
        cma_bash_tool = make_cma_redirected_bash(sandbox_proxy)

    def redirect_bash(event: BeforeToolCallEvent) -> None:
        if event.tool_use["name"] in ("bash", "execute"):
            event.selected_tool = cma_bash_tool

    return redirect_bash
