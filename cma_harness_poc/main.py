# cma_harness_poc/main.py — Entry point
from __future__ import annotations
import asyncio
import logging
import os
import sys

# Ensure Hermes source and own package are importable
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
# hermes-core 已安装在 .venv-poc 中，无需 path hack
if os.path.isdir(SCRIPT_DIR):
    sys.path.insert(0, PROJECT_ROOT)

# Python version check (Hermes requires 3.10+)
if sys.version_info < (3, 10):
    print(f"Error: Hermes requires Python 3.10+ (got {sys.version})")
    print(f"Tip: Use uv: source ~/.hermes/.env && ./cma_harness_poc/run.sh")
    sys.exit(1)

from cma_harness_poc.agent_store import AgentStore
from cma_harness_poc.event_store import CmaEventStore
from cma_harness_poc.session_service import CmaSessionService
from cma_harness_poc.mcp_manager import CmaMcpManager
from cma_harness_poc.api_server import CmaApiServer, DEFAULT_HOST, DEFAULT_PORT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("cma_harness_poc")


async def main():
    agent_store = AgentStore()
    event_store = CmaEventStore()
    session_service = CmaSessionService(event_store)
    mcp_manager = CmaMcpManager()
    server = CmaApiServer(agent_store, event_store, session_service,
                          mcp_manager=mcp_manager)

    host = os.environ.get("CMA_HOST", DEFAULT_HOST)
    port = int(os.environ.get("CMA_PORT", str(DEFAULT_PORT)))
    await server.start(host, port)

    logger.info("CMA Harness POC ready on http://%s:%d", host, port)
    logger.info("Web test console: http://%s:%d/", host, port)
    logger.info("")
    logger.info("Endpoints:")
    logger.info("  POST /v1/agents                           — create agent")
    logger.info("  POST /v1/sessions                         — create session")
    logger.info("  POST /v1/sessions/{id}/events             — send events")
    logger.info("  GET  /v1/sessions/{id}/events/stream      — SSE event stream")
    logger.info("  GET  /v1/sessions/{id}/events             — query event history")
    logger.info("  GET  /health                               — health check")
    logger.info("")
    logger.info("Step 1 — create agent:")
    logger.info("  curl -X POST http://%s:%d/v1/agents \\", host, port)
    logger.info("    -H 'Content-Type: application/json' \\")
    logger.info("    -d '{\"name\":\"demo\",\"model\":\"deepseek-chat\"}'")
    logger.info("")
    logger.info("Step 2 — create session:")
    logger.info("  curl -X POST http://%s:%d/v1/sessions \\", host, port)
    logger.info("    -H 'Content-Type: application/json' \\")
    logger.info("    -d '{\"agent\": {\"type\": \"agent\", \"id\": \"<agent_id>\"}}'")
    logger.info("")
    logger.info("Step 3 — connect SSE (in another terminal):")
    logger.info("  curl -N http://%s:%d/v1/sessions/<session_id>/events/stream", host, port)
    logger.info("")
    logger.info("Step 4 — send user message:")
    logger.info("  curl -X POST http://%s:%d/v1/sessions/<session_id>/events \\", host, port)
    logger.info("    -H 'Content-Type: application/json' \\")
    logger.info("    -d '{\"events\":[{\"type\":\"user.message\",\"content\":[{\"type\":\"text\",\"text\":\"echo hello\"}]}]}'")
    logger.info("")

    # Keep running
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
