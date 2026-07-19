"""Approval Bridge — 工具审批三种模式的实现桥接"""

from typing import Any, Callable, Optional

from strands import Agent
from strands.hooks.events import BeforeToolCallEvent


class ApprovalConfig:
    """工具审批配置"""

    def __init__(
        self,
        tool_approvals: dict[str, str] | None = None,
        default_mode: str = "allowed",
    ):
        """
        Args:
            tool_approvals: {"bash": "manual", "readFile": "allowed", ...}
            default_mode: 未在 tool_approvals 中指定的工具的默认模式
        """
        self.tool_approvals = tool_approvals or {}
        self.default_mode = default_mode

    def get_mode(self, tool_name: str) -> str:
        return self.tool_approvals.get(tool_name, self.default_mode)


def create_approval_hook(
    approval_config: ApprovalConfig,
    on_interrupt: Optional[Callable] = None,
):
    """创建审批 hook 回调

    Args:
        approval_config: 审批配置
        on_interrupt: manual 模式中断时的回调，接收 (tool_use, interrupt_id)
    """

    def approval_hook(event: BeforeToolCallEvent) -> None:
        tool_name = event.tool_use["name"]
        mode = approval_config.get_mode(tool_name)

        if mode == "forbidden":
            # TC-APR-05: 直接拒绝
            event.cancel_tool = f"Tool '{tool_name}' is forbidden"

        elif mode == "manual":
            # TC-APR-02/03/04: 触发 interrupt 暂停
            interrupt = event.interrupt(
                name=f"approval_{tool_name}",
                reason={
                    "tool_name": tool_name,
                    "tool_input": event.tool_use["input"],
                    "message": f"Approve execution of '{tool_name}'?",
                },
            )
            # 标记中断（供 event_translator 使用）
            setattr(event, "_interrupt_triggered", True)

            if on_interrupt:
                on_interrupt(event.tool_use, interrupt.id if hasattr(interrupt, 'id') else None)

        # allowed 模式：不做任何操作，自动放行

    return approval_hook


def resume_agent_after_interrupt(
    agent: Agent,
    interrupts: list[Any],
    approved: bool = True,
) -> Any:
    """中断恢复：构建 InterruptResponse 并重新调用 agent()

    Args:
        agent: 被中断的 Agent 实例
        interrupts: AgentResult.interrupts 列表
        approved: True=确认执行, False=拒绝执行

    Returns:
        AgentResult
    """
    responses = []
    for interrupt in interrupts:
        interrupt_id = interrupt.id if hasattr(interrupt, 'id') else interrupt.get("id", "")
        response = "APPROVED" if approved else "DENIED"
        responses.append({
            "interruptResponse": {
                "interruptId": interrupt_id,
                "response": response,
            }
        })

    return agent(responses)
