"""生成 MCP 连接与能力协商时序图。"""

from html import escape
from pathlib import Path


OUTPUT = Path(__file__).with_name("mcp-connection-capability-negotiation.svg")
lines: list[str] = []


def add(value: str) -> None:
    lines.append(value)


def text(x: int, y: int, value: str, css: str = "label", anchor: str = "middle") -> None:
    add(
        f'<text x="{x}" y="{y}" class="{css}" text-anchor="{anchor}">'
        f"{escape(value)}</text>"
    )


def participant(x: int, title: str, subtitle: str, fill: str, stroke: str) -> None:
    add(
        f'<g data-graph-role="node"><rect x="{x - 104}" y="88" width="208" '
        f'height="58" rx="10" fill="{fill}" stroke="{stroke}" stroke-width="1.8"/>'
    )
    text(x, 114, title, "participant")
    text(x, 134, subtitle, "small")
    add(f'<line x1="{x}" y1="146" x2="{x}" y2="850" class="lifeline"/></g>')


def frame(y: int, height: int, title: str, note: str) -> None:
    add(
        f'<rect data-graph-role="container" x="48" y="{y}" width="1104" height="{height}" rx="10" '
        'fill="#f9fafb" fill-opacity="0.58" stroke="#cbd5e1" '
        'stroke-width="1.2" stroke-dasharray="7 5"/>'
    )
    add(f'<rect x="64" y="{y + 12}" width="154" height="24" rx="12" fill="#111827"/>')
    text(141, y + 29, title, "frame-title")
    text(232, y + 29, note, "frame-note", anchor="start")


def activation(x: int, y: int, height: int, fill: str) -> None:
    add(
        f'<rect x="{x - 5}" y="{y}" width="10" height="{height}" rx="3" '
        f'fill="{fill}" stroke="#ffffff" stroke-width="1"/>'
    )


def message(
    source: int,
    target: int,
    y: int,
    label: str,
    color: str,
    marker: str,
    detail: str | None = None,
    dashed: bool = False,
) -> None:
    direction = 1 if target > source else -1
    start = source + 8 * direction
    end = target - 12 * direction
    dash = ' stroke-dasharray="7 5"' if dashed else ""
    add(
        f'<line x1="{start}" y1="{y}" x2="{end}" y2="{y}" '
        f'stroke="{color}" stroke-width="2" marker-end="url(#{marker})"{dash} '
        f'data-graph-role="edge"/>'
    )
    label_y = y - 9
    text((source + target) // 2, label_y, label, "message")
    if detail:
        text((source + target) // 2, y + 16, detail, "detail")


add('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 940" width="1200" height="940">')
add("<style>")
add("text { font-family: 'PingFang SC', 'Hiragino Sans GB', 'Heiti SC', 'Helvetica Neue', Arial, sans-serif; }")
add(".title { fill:#111827; font-size:24px; font-weight:700; }")
add(".subtitle { fill:#6b7280; font-size:13px; }")
add(".participant { fill:#111827; font-size:15px; font-weight:700; }")
add(".label { fill:#111827; font-size:13px; }")
add(".small { fill:#6b7280; font-size:11px; }")
add(".message { fill:#111827; font-size:13px; font-weight:600; }")
add(".detail { fill:#6b7280; font-size:11px; }")
add(".frame-title { fill:#ffffff; font-size:12px; font-weight:700; }")
add(".frame-note { fill:#64748b; font-size:12px; }")
add(".lifeline { stroke:#94a3b8; stroke-width:1.2; stroke-dasharray:6 6; }")
add(".legend { fill:#475569; font-size:12px; }")
add("</style>")
add("<defs>")
for marker_id, color in (
    ("arrow-blue", "#2563eb"),
    ("arrow-orange", "#ea580c"),
    ("arrow-green", "#16a34a"),
    ("arrow-red", "#dc2626"),
):
    add(
        f'<marker id="{marker_id}" markerWidth="10" markerHeight="7" refX="9" '
        f'refY="3.5" orient="auto"><polygon points="0 0, 10 3.5, 0 7" '
        f'fill="{color}"/></marker>'
    )
add("</defs>")
add('<rect width="1200" height="940" fill="#ffffff"/>')
text(600, 40, "MCP 连接建立与能力协商（2026-07-28）", "title")
text(600, 64, "无状态 JSON-RPC：发现可缓存，但每个请求仍携带版本、身份与客户端能力", "subtitle")

# 先放阶段容器和生命线，再放消息与文字，确保结构清晰。
frame(166, 174, "阶段 1 · 连接", "专用 Client + Transport")
frame(352, 330, "阶段 2 · 发现", "可选调用，可按返回提示缓存")
frame(696, 166, "阶段 3 · 就绪", "按双方能力注册")

participants = (
    (140, "MCP Host", "AI 应用 / Client Manager", "#eff6ff", "#93c5fd"),
    (420, "MCP Client", "每个 Server 一个专用连接", "#faf5ff", "#c4b5fd"),
    (730, "Transport", "stdio 或 Streamable HTTP", "#fff7ed", "#fdba74"),
    (1050, "MCP Server", "本地或远程能力提供者", "#f0fdf4", "#86efac"),
)
for args in participants:
    participant(*args)

activation(420, 198, 638, "#ddd6fe")
activation(730, 246, 440, "#fed7aa")
activation(1050, 280, 556, "#bbf7d0")

message(140, 420, 224, "创建专用 MCP Client", "#2563eb", "arrow-blue", "读取 Server 配置")
message(420, 730, 270, "打开 Transport", "#ea580c", "arrow-orange", "stdio: 启动子进程；HTTP: 建连并认证")
message(730, 1050, 316, "建立可交换 JSON-RPC 的通道", "#ea580c", "arrow-orange")

message(420, 730, 404, "server/discover", "#2563eb", "arrow-blue", "_meta: protocolVersion + clientInfo + clientCapabilities")
message(730, 1050, 442, "转发带帧请求", "#ea580c", "arrow-orange")
message(1050, 730, 488, "Discover Result", "#16a34a", "arrow-green", "supportedVersions + capabilities + serverInfo + ttlMs")
message(730, 420, 526, "返回发现结果", "#16a34a", "arrow-green")

# 版本不匹配分支使用红色虚线，表示只有协商失败时才发生。
add('<rect data-graph-role="decoration" x="378" y="550" width="716" height="112" rx="8" fill="#fff7f7" stroke="#fecaca" stroke-width="1"/>')
text(394, 570, "alt · 请求版本不受支持", "detail", anchor="start")
message(1050, 730, 594, "UnsupportedProtocolVersionError", "#dc2626", "arrow-red", "附带 Server 支持的版本列表", dashed=True)
message(730, 420, 628, "转发版本错误", "#dc2626", "arrow-red", dashed=True)
text(432, 656, "Client 选择交集版本并重试；若无交集则连接不可用", "detail", anchor="start")

message(420, 140, 752, "缓存能力矩阵并标记 Ready", "#16a34a", "arrow-green", "缓存受 ttlMs / cacheScope 约束")
message(140, 420, 790, "按已协商能力注册功能", "#2563eb", "arrow-blue", "例如 tools / resources / prompts / elicitation")
message(420, 1050, 828, "tools/list 或 subscriptions/listen", "#2563eb", "arrow-blue", "每个请求继续携带同一组 _meta")

# 图例与版本提示位于业务流下方，避开所有生命线消息。
add('<line x1="70" y1="902" x2="104" y2="902" stroke="#2563eb" stroke-width="2" marker-end="url(#arrow-blue)"/>')
text(116, 906, "协议请求", "legend", anchor="start")
add('<line x1="226" y1="902" x2="260" y2="902" stroke="#16a34a" stroke-width="2" marker-end="url(#arrow-green)"/>')
text(272, 906, "能力/结果", "legend", anchor="start")
add('<line x1="392" y1="902" x2="426" y2="902" stroke="#ea580c" stroke-width="2" marker-end="url(#arrow-orange)"/>')
text(438, 906, "Transport 控制", "legend", anchor="start")
add('<line x1="594" y1="902" x2="628" y2="902" stroke="#dc2626" stroke-width="2" stroke-dasharray="7 5" marker-end="url(#arrow-red)"/>')
text(640, 906, "协商失败分支", "legend", anchor="start")
text(1130, 906, "注意：当前版本不是 initialize/initialized 握手", "legend", anchor="end")

add("</svg>")
OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"generated: {OUTPUT}")
