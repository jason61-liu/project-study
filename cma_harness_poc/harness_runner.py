# cma_harness_poc/harness_runner.py — Hermes AIAgent integration
# Design: AIAgent.run_conversation() as core loop, callback hooks for CMA events
from __future__ import annotations
import asyncio
import logging
import os
import sys
from typing import Any, Callable, Dict, List, Optional

# Add hermes-core vendor path so transitive imports (tools.*, agent.*)
# resolve correctly when importing AIAgent
_HERMES_VENDOR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "hermes-core", "hermes_core", "_vendor",
)
if _HERMES_VENDOR not in sys.path:
    sys.path.insert(0, _HERMES_VENDOR)

# AIAgent from hermes-core vendored copy (no direct hermes-agent dep)
from hermes_core._vendor.run_agent import AIAgent

# Auto-load ~/.hermes/.env so the API key is available regardless of
# how the POC is started (run.sh, docker, or direct python).
_env_path = os.path.expanduser("~/.hermes/.env")
if os.path.isfile(_env_path):
    try:
        with open(_env_path) as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _v = _line.split("=", 1)
                    _k = _k.strip()
                    _v = _v.strip().strip("\"'")
                    if _k not in os.environ and _v:
                        os.environ[_k] = _v
    except Exception:
        pass

from cma_harness_poc.models import CmaEvent, AgentConfig
from cma_harness_poc.event_store import CmaEventStore
from cma_harness_poc.session_service import CmaSessionService
from cma_harness_poc.event_translator import cma_events_to_hermes_messages

logger = logging.getLogger(__name__)


def _resolve_model_base_url(model: str) -> str:
    return os.environ.get("CMA_LLM_BASE_URL", "https://api.deepseek.com")


def _resolve_api_key(model: str) -> str:
    return os.environ.get("CMA_LLM_API_KEY",
           os.environ.get("DEEPSEEK_API_KEY", ""))


def push_cma_event(
    event_store: CmaEventStore,
    session_service: CmaSessionService,
    sse_queue: "asyncio.Queue",
    session_id: str,
    event_type: str,
    **kwargs,
) -> CmaEvent:
    """Create CMA event → write to EventStore + push to SSE queue."""
    event = CmaEvent(session_id=session_id, type=event_type, **kwargs)
    stored = event_store.append(session_id, event)
    session_service.on_event_appended(session_id, stored)
    try:
        sse_queue.put_nowait(stored)
    except Exception:
        pass
    return stored


def _make_stream_cb(
    event_store: CmaEventStore,
    session_service: CmaSessionService,
    sse_queue: "asyncio.Queue",
    session_id: str,
):
    """
    stream_delta_callback: receives text deltas from Hermes streaming.
    Accumulates deltas — flush() emits a consolidated agent.message event
    into EventStore + SSE.
    """
    buffer: list[str] = []

    def cb(text):
        if text:
            buffer.append(text)
        try:
            sse_queue.put_nowait(CmaEvent(
                session_id=session_id, type="agent.message.delta",
                content=[{"type": "text", "text": text}],
            ))
        except Exception:
            pass

    def flush():
        if not buffer:
            return ""
        full_text = "".join(buffer)
        buffer.clear()
        push_cma_event(event_store, session_service, sse_queue, session_id,
                       "agent.message",
                       content=[{"type": "text", "text": full_text}])
        return full_text

    cb.flush = flush
    return cb


def _make_tool_start_cb(
    event_store: CmaEventStore,
    session_service: CmaSessionService,
    sse_queue: "asyncio.Queue",
    session_id: str,
    flush_fn=None,
):
    """
    tool_start_callback: called by Hermes before executing a tool.
    Flushes any accumulated stream text, then emits agent.tool_use event.
    """
    _es, _ss, _sq, _sid = event_store, session_service, sse_queue, session_id
    _flush = flush_fn

    def cb(tool_call_id, name, args):
        if _flush:
            _flush()
        push_cma_event(_es, _ss, _sq, _sid, "agent.tool_use",
                       id=tool_call_id, name=name, input=args)

    return cb


def _make_tool_complete_cb(
    event_store: CmaEventStore,
    session_service: CmaSessionService,
    sse_queue: "asyncio.Queue",
    session_id: str,
):
    """
    tool_complete_callback: called by Hermes after a tool executes.
    Emits agent.tool_result event with is_error flag.
    """
    _es, _ss, _sq, _sid = event_store, session_service, sse_queue, session_id

    def cb(tool_call_id, name, args, result):
        is_error = _detect_failure(name, result)
        push_cma_event(_es, _ss, _sq, _sid, "agent.tool_result",
                       tool_use_id=tool_call_id,
                       content=[{"type": "text", "text": str(result)}],
                       is_error=is_error)

    return cb


def _make_reasoning_cb(
    event_store: CmaEventStore,
    session_service: CmaSessionService,
    sse_queue: "asyncio.Queue",
    session_id: str,
):
    """
    reasoning_callback: receives reasoning/thinking deltas from Hermes.
    Accumulates deltas — flush() returns accumulated text for later
    consolidated agent.thinking event emission.
    """
    buffer: list[str] = []

    def cb(text: str):
        if text:
            buffer.append(text)
        try:
            sse_queue.put_nowait(CmaEvent(
                session_id=session_id, type="agent.thinking.delta",
                content=[{"type": "thinking", "thinking": text}],
            ))
        except Exception:
            pass

    def flush():
        if not buffer:
            return ""
        full = "".join(buffer)
        buffer.clear()
        # Consolidated agent.thinking is emitted from _extract_thinking_from_result
        return full

    cb.flush = flush
    return cb


def _detect_failure(tool_name: str, result: Any) -> bool:
    """Heuristic: treat None or known error prefixes as failure."""
    if result is None:
        return True
    result_str = str(result)
    if result_str.startswith("Error executing tool"):
        return True
    if result_str.startswith("Tool '"):
        return True
    return False


def _extract_thinking_from_result(
    result: Dict[str, Any],
    event_store: CmaEventStore,
    session_service: CmaSessionService,
    sse_queue: "asyncio.Queue",
    session_id: str,
) -> None:
    """Extract reasoning_content/thinking from run_conversation result and emit agent.thinking."""
    last_reasoning = result.get("last_reasoning")
    if last_reasoning and isinstance(last_reasoning, str) and last_reasoning.strip():
        push_cma_event(
            event_store, session_service, sse_queue,
            session_id, "agent.thinking",
            content=[{"type": "thinking", "thinking": last_reasoning}],
        )
        return

    # Fallback: scan messages for reasoning
    for msg in result.get("messages", []):
        reasoning = msg.get("reasoning_content") or msg.get("reasoning")
        if reasoning and isinstance(reasoning, str) and reasoning.strip():
            push_cma_event(
                event_store, session_service, sse_queue,
                session_id, "agent.thinking",
                content=[{"type": "thinking", "thinking": reasoning}],
            )
            return


def _extract_compacted_summary(
    result: Dict[str, Any],
) -> Optional[str]:
    """Scan run_conversation result for Hermes compaction summary."""
    for msg in result.get("messages", []):
        content = msg.get("content", "") or ""
        if isinstance(content, str) and "[CONTEXT COMPACTION" in content:
            return content
    return None


def run_harness(
    session_id: str,
    user_message: str,
    agent_config: AgentConfig,
    event_store: CmaEventStore,
    session_service: CmaSessionService,
    sse_queue: "asyncio.Queue",
    compact_callback: Optional[Callable] = None,
    mcp_manager=None,  # kept for signature compat; MCP is out of scope per design doc
    agent_container: Optional[list] = None,  # [None] → [agent] after creation
) -> Dict[str, Any]:
    """
    Run one turn of the Hermes agent loop.

    Uses AIAgent.run_conversation() as the core engine with callback hooks
    to inject CMA events (tool_use, tool_result, agent.message, etc.)
    into the EventStore + SSE stream.

    Design doc: hermes-as-cma-poc-design.md §4
    """
    # 1. Rebuild conversation history from past CMA events
    compact_ctx = session_service.get_compact_context(session_id)
    events = event_store.get_events(session_id)
    conversation_history = cma_events_to_hermes_messages(events, compact_ctx)
    # Remove trailing user message (it will be re-appended by run_conversation)
    if conversation_history and conversation_history[-1].get("role") == "user":
        conversation_history = conversation_history[:-1]

    # 2. Resolve LLM endpoint
    base_url = _resolve_model_base_url(agent_config.model)
    api_key = _resolve_api_key(agent_config.model)

    # 3. Build callback hooks per design doc §4.1-4.2
    stream_cb = _make_stream_cb(event_store, session_service, sse_queue, session_id)
    tool_start_cb = _make_tool_start_cb(
        event_store, session_service, sse_queue, session_id,
        flush_fn=stream_cb.flush,
    )
    tool_complete_cb = _make_tool_complete_cb(
        event_store, session_service, sse_queue, session_id,
    )
    reasoning_cb = _make_reasoning_cb(
        event_store, session_service, sse_queue, session_id,
    )

    # 4. Create AIAgent — design doc §4.2
    #    Determine toolsets: base tools + skills if agent has any.
    toolsets = ["terminal", "file", "web", "browser"]
    if agent_config.skills:
        toolsets.append("skills")

    #    Build system prompt: CMA system + skill preload content
    base_system = agent_config.system or ""
    preloaded_prompt = ""
    if agent_config.skills:
        try:
            from hermes_core._vendor.agent.skill_commands import (
                build_preloaded_skills_prompt,
            )
            prompt, loaded, missing = build_preloaded_skills_prompt(
                agent_config.skills,
                task_id=session_id,
            )
            preloaded_prompt = prompt
            if missing:
                logger.warning(
                    "Skills not found (skipped): %s", ", ".join(missing)
                )
        except Exception as exc:
            logger.warning("Skill preload failed: %s", exc)

    if preloaded_prompt:
        ephemeral_system = (
            f"{base_system}\n\n{preloaded_prompt}"
            if base_system
            else preloaded_prompt
        )
    else:
        ephemeral_system = base_system or None

    #    Pass system message via ephemeral_system_prompt so it's included
    #    in Hermes' system prompt assembly.
    agent = AIAgent(
        base_url=base_url,
        model=agent_config.model,
        api_key=api_key,
        max_iterations=90,
        # Callback hooks (核心对接点)
        tool_start_callback=tool_start_cb,
        tool_complete_callback=tool_complete_cb,
        stream_delta_callback=stream_cb,
        reasoning_callback=reasoning_cb,
        # CMA 不需要的能力 → 禁用
        skip_memory=True,
        skip_context_files=True,
        # 动态 toolsets：有 skills 时追加 skills 工具集
        enabled_toolsets=toolsets,
        quiet_mode=True,
        verbose_logging=False,
        # Hermes session_db 不写（用 CMA EventStore 替代）
        session_db=None,
        # Inject CMA agent's system prompt + skill preload into Hermes system prompt
        ephemeral_system_prompt=ephemeral_system,
    )

    # Expose agent reference so the API server can call agent.interrupt()
    if agent_container is not None:
        agent_container[:] = [agent]

    # 5. Emit session.status_running
    push_cma_event(event_store, session_service, sse_queue, session_id,
                   "session.status_running")

    # 6. Run the Hermes agent loop via AIAgent.run_conversation()
    try:
        result = agent.run_conversation(
            user_message=user_message,
            conversation_history=conversation_history,
        )

        # 7. Flush any remaining stream text into consolidated agent.message
        #    (needed for pure-text responses where no tool_start_cb triggers flush)
        stream_cb.flush()

        # 8. Extract thinking/reasoning from result if present
        _extract_thinking_from_result(
            result, event_store, session_service, sse_queue, session_id,
        )

        # 9. Check for context compaction and notify callback
        if compact_callback:
            compact_summary = _extract_compacted_summary(result)
            if compact_summary:
                from cma_harness_poc.models import CompactContext
                events_after = event_store.get_events(session_id)
                ctx = CompactContext(
                    compacted_up_to=len(events_after),
                    summary=compact_summary,
                )
                compact_callback(session_id, ctx)

    except Exception as exc:
        logger.exception("Harness loop failed for session %s", session_id)
        push_cma_event(event_store, session_service, sse_queue, session_id,
                       "session.error",
                       error={"message": str(exc), "fatal": True})
        push_cma_event(event_store, session_service, sse_queue, session_id,
                       "session.status_terminated")
        return {"error": str(exc)}

    # 9. Emit session.status_idle
    push_cma_event(event_store, session_service, sse_queue, session_id,
                   "session.status_idle",
                   stop_reason={"type": "end_turn"})

    return {"messages": result.get("messages", [])}
