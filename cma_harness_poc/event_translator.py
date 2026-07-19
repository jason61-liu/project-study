# cma_harness_poc/event_translator.py — CMA event ↔ Hermes OpenAI-format message
from __future__ import annotations
from typing import Any, Dict, List, Optional

from cma_harness_poc.models import CmaEvent, CompactContext


def extract_text_from_blocks(blocks: Any) -> str:
    """Extract text from CMA content blocks array."""
    if isinstance(blocks, str):
        return blocks
    if isinstance(blocks, list):
        texts = []
        for block in blocks:
            if isinstance(block, dict) and block.get("type") == "text":
                texts.append(block.get("text", ""))
        return " ".join(texts)
    return str(blocks) if blocks else ""


def extract_thinking_from_blocks(blocks: Any) -> str:
    """Extract thinking/reasoning content from CMA content blocks array."""
    if isinstance(blocks, list):
        parts = []
        for block in blocks:
            if isinstance(block, dict) and block.get("type") == "thinking":
                parts.append(block.get("thinking", ""))
        return "\n".join(parts)
    return ""


def _convert_events_to_messages(
    events: List[CmaEvent],
) -> List[Dict[str, Any]]:
    """Convert CMA events to Hermes OpenAI-format messages (no compact_ctx)."""
    messages: List[Dict[str, Any]] = []
    pending_tool_calls: List[CmaEvent] = []

    def _flush_tool_calls():
        if not pending_tool_calls:
            return
        tool_calls = []
        for tc in pending_tool_calls:
            tool_calls.append({
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name or "",
                    "arguments": tc.input or {},
                },
            })
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": tool_calls,
        })
        pending_tool_calls.clear()

    for event in events:
        if event.type == "user.message":
            _flush_tool_calls()
            messages.append({
                "role": "user",
                "content": extract_text_from_blocks(event.content),
            })

        elif event.type == "agent.tool_use":
            pending_tool_calls.append(event)

        elif event.type == "agent.tool_result":
            _flush_tool_calls()
            text = extract_text_from_blocks(event.content)
            messages.append({
                "role": "tool",
                "tool_call_id": event.tool_use_id or "",
                "content": text,
            })

        elif event.type == "agent.message":
            _flush_tool_calls()
            text = extract_text_from_blocks(event.content)
            msg: Dict[str, Any] = {
                "role": "assistant",
                "content": text,
            }
            reasoning = extract_thinking_from_blocks(event.content)
            if reasoning:
                msg["reasoning_content"] = reasoning
            messages.append(msg)

    _flush_tool_calls()
    return messages


def cma_events_to_hermes_messages(
    events: List[CmaEvent],
    compact_ctx: Optional[CompactContext] = None,
) -> List[Dict[str, Any]]:
    """
    Convert CMA events to Hermes OpenAI-format messages.

    If compact_ctx is provided and has compacted_up_to > 0 and non-empty
    summary, the summary is inserted as an assistant message and only
    events after compacted_up_to are converted individually.
    """
    messages: List[Dict[str, Any]] = []

    if compact_ctx and compact_ctx.compacted_up_to > 0 and compact_ctx.summary:
        messages.append({
            "role": "assistant",
            "content": compact_ctx.summary,
        })
        remaining = events[compact_ctx.compacted_up_to:]
    else:
        remaining = events

    messages.extend(_convert_events_to_messages(remaining))
    return messages
