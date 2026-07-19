# cma_harness_poc/mcp_manager.py — Per-agent MCP connection manager
"""
Minimal MCP (Model Context Protocol) client - JSON-RPC 2.0 over stdio.

No external dependencies. Implements just enough of the protocol for
tool discovery and tool calling.

Protocol flow:
  Client -> Server:  initialize (handshake)
  Server -> Client:  initialize result
  Client -> Server:  notifications/initialized (no response)
  Client -> Server:  tools/list
  Server -> Client:  tools list with schemas
  Client -> Server:  tools/call {name, arguments}
  Server -> Client:  tool result
"""
from __future__ import annotations
import json
import logging
import os
import subprocess
import threading
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Tool name prefix to avoid collision with Hermes built-in tools
_MCP_PREFIX = "mcp_"


class McpServerProcess:
    """A running MCP server subprocess (stdio transport)."""

    def __init__(
        self, name: str, command: str, args: List[str],
        env: Optional[Dict[str, str]] = None,
    ):
        self.name = name
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._req_id = 0
        self._tool_schemas: List[Dict[str, Any]] = []
        self._disconnected = False

        merged_env = os.environ.copy()
        if env:
            merged_env.update(env)
        self._env = merged_env

        try:
            self._proc = subprocess.Popen(
                [command] + args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=self._env,
                text=True,
                bufsize=1,
            )
        except FileNotFoundError as e:
            raise RuntimeError(
                f"MCP server '{name}': command '{command}' not found"
            ) from e

    def _send(self, msg: dict) -> None:
        """Send one JSON-RPC message to the server process."""
        if self._proc is None or self._proc.stdin is None:
            raise RuntimeError(f"MCP server '{self.name}' not running")
        line = json.dumps(msg, ensure_ascii=False)
        self._proc.stdin.write(line + "\n")
        self._proc.stdin.flush()

    def _recv(self, timeout: float = 10.0) -> dict:
        """Read one JSON-RPC response from the server process."""
        if self._proc is None or self._proc.stdout is None:
            raise RuntimeError(f"MCP server '{self.name}' not running")

        import time as _time
        deadline = _time.monotonic() + timeout
        while _time.monotonic() < deadline:
            line = self._proc.stdout.readline()
            if not line:
                raise RuntimeError(
                    f"MCP server '{self.name}' stdout closed unexpectedly"
                )
            line = line.strip()
            if not line:
                continue
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                logger.warning(
                    "MCP server '%s': ignoring non-JSON stdout line: %s",
                    self.name, line[:200],
                )
        raise TimeoutError(
            f"MCP server '{self.name}': no response within {timeout}s"
        )

    def _call(
        self, method: str, params: Optional[dict] = None,
        timeout: float = 10.0,
    ) -> dict:
        """Send a JSON-RPC request and return the result dict."""
        with self._lock:
            self._req_id += 1
            req_id = self._req_id
            msg: dict = {
                "jsonrpc": "2.0",
                "id": req_id,
                "method": method,
            }
            if params is not None:
                msg["params"] = params
            self._send(msg)
            resp = self._recv(timeout=timeout)
            if "error" in resp:
                err = resp["error"]
                raise RuntimeError(
                    f"MCP server '{self.name}' error: "
                    f"{err.get('message', '')} (code={err.get('code')})"
                )
            return resp.get("result", {})

    def initialize(self, timeout: float = 15.0) -> None:
        """Perform MCP handshake: initialize + tools/list."""
        self._call("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "cma-harness", "version": "0.1.0"},
        }, timeout=timeout)
        # initialized notification (fire-and-forget, no response expected)
        try:
            self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        except Exception:
            pass
        result = self._call("tools/list", timeout=timeout)
        self._tool_schemas = result.get("tools", [])

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Return the tool schemas from initialization."""
        return list(self._tool_schemas)

    def call_tool(
        self, tool_name: str, arguments: dict, timeout: float = 30.0,
    ) -> dict:
        """Call a tool on this server and return the result dict."""
        if self._disconnected:
            raise RuntimeError(f"MCP server '{self.name}' already disconnected")
        result = self._call("tools/call", {
            "name": tool_name,
            "arguments": arguments,
        }, timeout=timeout)
        return result

    def shutdown(self) -> None:
        """Terminate the server process."""
        self._disconnected = True
        if self._proc:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=3)
            except Exception:
                try:
                    self._proc.kill()
                    self._proc.wait(timeout=2)
                except Exception:
                    pass
            self._proc = None

    @property
    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None


def _find_command(cmd: str) -> str:
    """Find command in PATH or return as-is if absolute."""
    if os.path.isabs(cmd) and os.path.isfile(cmd):
        return cmd
    for p in os.environ.get("PATH", "").split(os.pathsep):
        full = os.path.join(p, cmd)
        if os.path.isfile(full):
            return full
    return cmd  # let subprocess raise the error


class CmaMcpManager:
    """
    Per-agent MCP connection pool.

    Architecture:
        _connections[agent_id][server_name] = McpServerProcess

    Each agent gets its own set of MCP server connections, fully isolated
    from other agents. Connections are established on connect_agent() and
    torn down on disconnect_agent().

    MCP tools are NOT registered in the global Hermes ToolRegistry.
    Instead, they are managed entirely within this class - the harness loop
    calls get_tool_schemas() for schema injection and call_tool() for dispatch.
    """

    def __init__(self):
        self._connections: Dict[str, Dict[str, McpServerProcess]] = {}
        self._lock = threading.Lock()

    def connect_agent(
        self, agent_id: str, mcp_servers: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Connect all MCP servers for an agent.

        Args:
            agent_id: Unique agent identifier.
            mcp_servers: List of MCP server config dicts. Each dict may have:
                - name: server identifier (used in tool name prefix)
                - command: executable path (required for stdio transport)
                - args: list of command arguments (optional)
                - env: dict of environment variables (optional)
                - timeout: connection timeout in seconds (optional, default 15)

        Returns:
            {server_name: [tool_schema_dict, ...]} for successfully connected servers.
        """
        if not mcp_servers:
            return {}

        result: Dict[str, List[Dict[str, Any]]] = {}

        for cfg in mcp_servers:
            name = cfg.get("name", "unknown")
            command = cfg.get("command", "")
            if not command:
                logger.warning(
                    "MCP agent %s server '%s': missing 'command'",
                    agent_id, name,
                )
                continue

            # Idempotent: skip if already connected
            with self._lock:
                existing_conns = self._connections.get(agent_id, {})
                if name in existing_conns:
                    logger.debug(
                        "MCP server '%s' already connected for agent %s",
                        name, agent_id,
                    )
                    result[name] = existing_conns[name].get_tool_schemas()
                    continue

            cmd_path = _find_command(command)
            args = cfg.get("args", [])
            env = cfg.get("env", {})
            connect_timeout = cfg.get("timeout", 15.0)

            try:
                proc = McpServerProcess(name, cmd_path, args, env=env)
                proc.initialize(timeout=connect_timeout)
            except Exception as e:
                logger.warning(
                    "Failed to connect MCP server '%s' for agent %s: %s",
                    name, agent_id, e,
                )
                continue

            with self._lock:
                self._connections.setdefault(agent_id, {})[name] = proc
            schemas = proc.get_tool_schemas()
            result[name] = schemas
            logger.info(
                "MCP server '%s' for agent %s: %d tool(s) registered",
                name, agent_id, len(schemas),
            )

        return result

    def get_tool_schemas(self, agent_id: str) -> List[Dict[str, Any]]:
        """
        Return OpenAI-format tool schemas for this agent's MCP servers.

        Tool names are prefixed with ``mcp_{server_name}_`` to avoid
        collision with Hermes built-in tools and between agents.
        """
        schemas: List[Dict[str, Any]] = []
        with self._lock:
            conns = self._connections.get(agent_id, {})

        for srv_name, proc in conns.items():
            for tool in proc.get_tool_schemas():
                tool_name = tool.get("name", "")
                if not tool_name:
                    continue
                prefixed = f"{_MCP_PREFIX}{srv_name}_{tool_name}"
                input_schema = tool.get("inputSchema", {"type": "object"})
                schemas.append({
                    "type": "function",
                    "function": {
                        "name": prefixed,
                        "description": tool.get("description", ""),
                        "parameters": (
                            input_schema
                            if isinstance(input_schema, dict)
                            else {"type": "object"}
                        ),
                    },
                })
        return schemas

    def call_tool(
        self, agent_id: str, tool_name: str,
        arguments: dict, timeout: float = 30.0,
    ) -> str:
        """
        Route a tool call to the correct MCP server for this agent.

        Returns JSON string with the result or error.
        """
        srv_name, mcp_tool_name = self._parse_tool_name(tool_name)
        if srv_name is None:
            return json.dumps({"error": f"Cannot parse MCP tool name: {tool_name}"})

        with self._lock:
            conns = self._connections.get(agent_id, {})

        proc = conns.get(srv_name)
        if proc is None:
            return json.dumps({
                "error": f"MCP server '{srv_name}' not connected for agent '{agent_id}'"
            })

        try:
            result = proc.call_tool(mcp_tool_name, arguments, timeout=timeout)
            content = result.get("content", [])
            texts = []
            for block in content if isinstance(content, list) else [content]:
                if isinstance(block, dict):
                    text = block.get("text", "")
                    if text:
                        texts.append(text)
                elif isinstance(block, str):
                    texts.append(block)
            output = "\n".join(texts) if texts else json.dumps(result)
            return output
        except Exception as e:
            logger.warning(
                "MCP tool call failed: agent=%s tool=%s error=%s",
                agent_id, tool_name, e,
            )
            return json.dumps({"error": f"MCP tool call failed: {e}"})

    def disconnect_agent(self, agent_id: str) -> None:
        """Disconnect all MCP servers for an agent and clean up."""
        with self._lock:
            conns = self._connections.pop(agent_id, {})
        for name, proc in conns.items():
            try:
                proc.shutdown()
                logger.debug(
                    "Disconnected MCP server '%s' for agent %s", name, agent_id,
                )
            except Exception as e:
                logger.warning("Error disconnecting MCP server '%s': %s", name, e)

    def get_connected_servers(self, agent_id: str) -> List[str]:
        """Return list of connected server names for an agent."""
        with self._lock:
            conns = self._connections.get(agent_id, {})
        return list(conns.keys())

    @staticmethod
    def _parse_tool_name(
        tool_name: str,
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Parse ``mcp_{server}_{tool}`` -> (server_name, tool_name).
        Returns (None, None) if parsing fails.
        """
        if not tool_name.startswith(_MCP_PREFIX):
            return None, None
        rest = tool_name[len(_MCP_PREFIX):]
        parts = rest.split("_")
        if len(parts) < 2:
            return None, None
        srv_name = parts[0]
        tool = "_".join(parts[1:])
        return srv_name, tool
