# cma_harness_poc/api_server.py — CMA-compatible REST API server (aiohttp)
from __future__ import annotations
import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, Optional

try:
    from aiohttp import web
except ImportError:
    web = None

from cma_harness_poc.models import CmaEvent, SessionState
from cma_harness_poc.agent_store import AgentStore
from cma_harness_poc.event_store import CmaEventStore
from cma_harness_poc.session_service import CmaSessionService
from cma_harness_poc.harness_runner import run_harness, push_cma_event
from cma_harness_poc.mcp_manager import CmaMcpManager

logger = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8643


def _extract_text(content: Any) -> str:
    """Extract plain text from CMA content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                texts.append(block.get("text", ""))
        return " ".join(texts)
    return str(content) if content else ""


class CmaApiServer:
    """aiohttp-based CMA REST API server with SSE streaming."""

    def __init__(
        self,
        agent_store: AgentStore,
        event_store: CmaEventStore,
        session_service: CmaSessionService,
        mcp_manager: Optional[CmaMcpManager] = None,
    ):
        self._agent_store = agent_store
        self._event_store = event_store
        self._session_service = session_service
        self._mcp_manager = mcp_manager
        # One asyncio.Queue per session for SSE broadcast
        self._sse_queues: Dict[str, asyncio.Queue] = {}
        # Track running harness tasks per session
        self._harness_tasks: Dict[str, asyncio.Task] = {}
        # Expose AIAgent reference per session for native Hermes interrupt
        self._harness_agents: Dict[str, list] = {}

    def _build_app(self) -> web.Application:
        app = web.Application()

        # Agent endpoints
        app.router.add_post("/v1/agents", self._handle_create_agent)
        app.router.add_get("/v1/agents", self._handle_list_agents)
        app.router.add_get("/v1/agents/{agent_id}", self._handle_get_agent)

        # Session endpoints
        app.router.add_post("/v1/sessions", self._handle_create_session)
        app.router.add_get("/v1/sessions/{session_id}", self._handle_get_session)

        # Event endpoints
        app.router.add_post(
            "/v1/sessions/{session_id}/events", self._handle_post_events
        )
        app.router.add_get(
            "/v1/sessions/{session_id}/events/stream", self._handle_events_stream
        )
        # Event history query (new)
        app.router.add_get(
            "/v1/sessions/{session_id}/events", self._handle_get_events
        )

        # Health
        app.router.add_get("/health", self._handle_health)

        # Web test console (root)
        app.router.add_get("/", self._handle_root)

        return app

    async def start(
        self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT
    ) -> None:
        """Start the aiohttp TCPSite."""
        app = self._build_app()
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, host, port)
        await site.start()
        logger.info("CMA API server listening on http://%s:%d", host, port)

    # ------------------------------------------------------------------
    # Agent handlers
    # ------------------------------------------------------------------

    async def _handle_create_agent(self, request: web.Request) -> web.Response:
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON body"}, status=400)

        name = body.get("name", "")
        model_raw = body.get("model", {})
        if isinstance(model_raw, dict):
            model = model_raw.get("id", "")
        else:
            model = str(model_raw)
        system = body.get("system", "")
        tools = body.get("tools")
        skills_raw = body.get("skills", [])
        skills = [
            s.get("name", s) if isinstance(s, dict) else s for s in skills_raw
        ]
        mcp_servers = body.get("mcp_servers", [])
        metadata = body.get("metadata", {})

        if not name:
            return web.json_response({"error": "name is required"}, status=400)
        if not model:
            return web.json_response(
                {"error": "model.id is required"}, status=400
            )

        config = self._agent_store.create(
            name=name,
            model=model,
            system=system,
            tools=tools,
            skills=skills,
            mcp_servers=mcp_servers,
            metadata=metadata,
        )
        return web.json_response(
            self._agent_store.to_dict(config), status=201
        )

    async def _handle_list_agents(self, request: web.Request) -> web.Response:
        agents = [
            self._agent_store.to_dict(a) for a in self._agent_store.list()
        ]
        return web.json_response(agents)

    async def _handle_get_agent(self, request: web.Request) -> web.Response:
        agent_id = request.match_info["agent_id"]
        config = self._agent_store.get(agent_id)
        if config is None:
            return web.json_response(
                {"error": "Agent not found"}, status=404
            )
        return web.json_response(self._agent_store.to_dict(config))

    # ------------------------------------------------------------------
    # Session handlers
    # ------------------------------------------------------------------

    async def _handle_create_session(self, request: web.Request) -> web.Response:
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON body"}, status=400)

        agent_ref = body.get("agent", {})
        agent_id = (
            agent_ref.get("id", "") if isinstance(agent_ref, dict) else ""
        )
        agent_version = (
            agent_ref.get("version", 1) if isinstance(agent_ref, dict) else 1
        )
        environment_id = body.get("environment_id", "env_default")

        if not agent_id:
            return web.json_response(
                {"error": "agent.id is required"}, status=400
            )
        if self._agent_store.get(agent_id) is None:
            return web.json_response(
                {"error": f"Agent '{agent_id}' not found"}, status=404
            )

        record = self._session_service.create_session(
            agent_id=agent_id,
            agent_version=agent_version,
            environment_id=environment_id,
        )
        return web.json_response(
            {
                "id": record.id,
                "status": record.status.value,
                "agent_id": record.agent_id,
                "agent_version": record.agent_version,
                "environment_id": record.environment_id,
                "created_at": record.created_at,
                "updated_at": record.updated_at,
                "usage": record.usage,
            },
            status=201,
        )

    async def _handle_get_session(self, request: web.Request) -> web.Response:
        session_id = request.match_info["session_id"]
        record = self._session_service.get_session(session_id)
        if record is None:
            return web.json_response(
                {"error": "Session not found"}, status=404
            )
        return web.json_response(
            {
                "id": record.id,
                "status": record.status.value,
                "agent_id": record.agent_id,
                "agent_version": record.agent_version,
                "environment_id": record.environment_id,
                "created_at": record.created_at,
                "updated_at": record.updated_at,
                "usage": record.usage,
                "compact_context": record.compact_context,
            }
        )

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    async def _handle_post_events(
        self, request: web.Request
    ) -> web.Response:
        session_id = request.match_info["session_id"]
        record = self._session_service.get_session(session_id)
        if record is None:
            return web.json_response(
                {"error": "Session not found"}, status=404
            )

        try:
            body = await request.json()
        except Exception:
            return web.json_response(
                {"error": "Invalid JSON body"}, status=400
            )

        events_data = body.get("events", [])
        if not events_data:
            return web.json_response(
                {"error": "events list is required"}, status=400
            )

        for evt_data in events_data:
            evt_type = evt_data.get("type", "")
            if evt_type == "user.message":
                text = _extract_text(evt_data.get("content", []))
                self._start_harness(session_id, text, record)
            elif evt_type == "user.interrupt":
                self._cancel_harness(session_id)
                self._event_store.append(session_id, CmaEvent(
                    session_id=session_id, type="session.interrupt",
                    content=[{"type": "text", "text": "User interrupted"}],
                ))
                self._session_service.update_session_state(
                    session_id, SessionState.IDLE)
            else:
                logger.warning(
                    "Ignoring unsupported event type: %s", evt_type
                )

        return web.json_response({"status": "accepted"})

    async def _handle_events_stream(
        self, request: web.Request
    ) -> web.StreamResponse:
        """SSE stream: past events first, then live events, then None-terminated."""
        session_id = request.match_info["session_id"]
        record = self._session_service.get_session(session_id)
        if record is None:
            return web.json_response(
                {"error": "Session not found"}, status=404
            )

        # Get or create the SSE queue for this session
        queue = self._sse_queues.get(session_id)
        if queue is None:
            queue = asyncio.Queue()
            self._sse_queues[session_id] = queue

        # Push all past events onto the queue so the client catches up
        existing = self._event_store.get_events(session_id)
        for evt in existing:
            await queue.put(evt)

        response = web.StreamResponse(
            status=200,
            reason="OK",
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
        await response.prepare(request)

        try:
            keepalive_interval = 30
            last_keepalive = time.monotonic()
            while True:
                # Compute timeout until next keepalive
                elapsed = time.monotonic() - last_keepalive
                timeout = max(0.0, keepalive_interval - elapsed)

                try:
                    evt = await asyncio.wait_for(queue.get(), timeout=timeout)
                except asyncio.TimeoutError:
                    # 30-second keepalive heartbeat
                    await response.write(b": keepalive\n\n")
                    last_keepalive = time.monotonic()
                    continue

                if evt is None:
                    # Sentinel: harness finished, no more events
                    break

                data = json.dumps(evt.to_sse_dict(), ensure_ascii=False)
                await response.write(f"data: {data}\n\n".encode("utf-8"))
                last_keepalive = time.monotonic()

        except asyncio.CancelledError:
            # Client disconnected
            pass
        except Exception:
            logger.exception(
                "SSE stream error for session %s", session_id
            )
        finally:
            # If no harness is running, we can clean up the queue;
            # otherwise leave it for future SSE connections.
            if session_id not in self._harness_tasks:
                self._sse_queues.pop(session_id, None)

        return response

    async def _handle_get_events(
        self, request: web.Request
    ) -> web.Response:
        """GET /v1/sessions/{session_id}/events — 查询完整事件历史。"""
        session_id = request.match_info["session_id"]
        record = self._session_service.get_session(session_id)
        if record is None:
            return web.json_response(
                {"error": "Session not found"}, status=404
            )

        events = self._event_store.get_events(session_id)
        return web.json_response({
            "session_id": session_id,
            "status": record.status.value,
            "events": [e.to_sse_dict() for e in events],
        })

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    async def _handle_health(
        self, request: web.Request
    ) -> web.Response:
        return web.json_response(
            {"status": "ok", "timestamp": time.time()}
        )

    async def _handle_root(
        self, request: web.Request
    ) -> web.Response:
        """Serve the web test console at /"""
        html_path = os.path.join(
            os.path.dirname(__file__), "test_console.html"
        )
        if os.path.isfile(html_path):
            with open(html_path, "r") as f:
                content = f.read()
            return web.Response(
                text=content, content_type="text/html"
            )
        return web.json_response(
            {"error": "test_console.html not found"},
            status=404,
        )

    # ------------------------------------------------------------------
    # Harness lifecycle
    # ------------------------------------------------------------------

    def _start_harness(
        self, session_id: str, user_message: str, record,
    ) -> None:
        """Launch (or restart) the Hermes agent harness in an executor thread."""
        # Cancel any existing harness for this session
        self._cancel_harness(session_id)

        agent_config = self._agent_store.get(record.agent_id)
        if agent_config is None:
            logger.error(
                "Agent %s not found for session %s",
                record.agent_id, session_id,
            )
            return

        # Create a mutable container that run_harness will populate
        # with the AIAgent reference once it's created.
        agent_container = [None]
        self._harness_agents[session_id] = agent_container

        # Ensure an SSE queue exists
        queue = self._sse_queues.get(session_id)
        if queue is None:
            queue = asyncio.Queue()
            self._sse_queues[session_id] = queue

        # Persist the user.message event and push to SSE
        push_cma_event(
            self._event_store,
            self._session_service,
            queue,
            session_id,
            "user.message",
            content=[{"type": "text", "text": user_message}],
        )

        loop = asyncio.get_event_loop()

        def _run() -> dict:
            return run_harness(
                session_id=session_id,
                user_message=user_message,
                agent_config=agent_config,
                event_store=self._event_store,
                session_service=self._session_service,
                sse_queue=queue,
                compact_callback=self._on_compacted,
                mcp_manager=self._mcp_manager,
                agent_container=agent_container,
            )

        # Run the synchronous harness in a thread pool executor
        task = loop.run_in_executor(None, _run)

        def _done_callback(fut: asyncio.Future) -> None:
            """Called when the executor future completes (in executor thread)."""
            try:
                # Re-raise any exception so it gets logged
                fut.result()
            except asyncio.CancelledError:
                logger.info("Harness cancelled for session %s", session_id)
            except Exception:
                logger.exception(
                    "Harness task failed for session %s", session_id
                )
            finally:
                # Signal SSE stream to end — must be threadsafe
                loop.call_soon_threadsafe(
                    lambda: queue.put_nowait(None)
                )
                self._harness_tasks.pop(session_id, None)

        task.add_done_callback(_done_callback)
        self._harness_tasks[session_id] = task

    def _on_compacted(self, session_id: str, compact_ctx) -> None:
        """Callback when Hermes internal compression occurred."""
        self._session_service.update_compact_context(session_id, compact_ctx)
        logger.info(
            "Session %s compacted: %d events truncated",
            session_id, compact_ctx.compacted_up_to,
        )

    def _cancel_harness(self, session_id: str) -> None:
        """Cancel the running harness for *session_id* gracefully.

        1. Call Hermes native AIAgent.interrupt() — thread-safe, signals
           the agent loop and in-flight tools to stop at the next safe point.
        2. Cancel the asyncio Future as a fallback (the thread continues
           but we no longer wait for it).
        """
        # Step 1: graceful Hermes-native interrupt (跨线程安全)
        agent_container = self._harness_agents.pop(session_id, None)
        if agent_container and agent_container[0] is not None:
            agent_container[0].interrupt("User interrupted")
            logger.info("Hermes native interrupt sent for session %s", session_id)

        # Step 2: cancel the asyncio-level Future (fallback)
        task = self._harness_tasks.pop(session_id, None)
        if task is not None and not task.done():
            task.cancel()
