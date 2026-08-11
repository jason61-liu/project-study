"""生成前四周 Agent、工具、RAG 与记忆端到端数据流图。"""

from __future__ import annotations

from html import escape
from pathlib import Path


ROOT = Path(__file__).parent
SVG_PATH = ROOT / "four-week-agent-rag-memory-data-flow.svg"


def text(x: int, y: int, value: str, css: str = "label", anchor: str = "middle") -> str:
    """生成 XML 安全的单行 SVG 文本。"""

    return f'<text x="{x}" y="{y}" class="{css}" text-anchor="{anchor}">{escape(value)}</text>'


def card(x: int, y: int, w: int, h: int, title: str, detail: str, fill: str, stroke: str) -> str:
    """生成带标题与说明的 Flat Icon 卡片。"""

    return (
        f'<g data-graph-role="node"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>'
        f'<text x="{x + w / 2}" y="{y + 29}" class="node-title" text-anchor="middle">{escape(title)}</text>'
        f'<text x="{x + w / 2}" y="{y + 52}" class="detail" text-anchor="middle">{escape(detail)}</text></g>'
    )


def database(x: int, y: int, w: int, h: int, title: str, detail: str, fill: str, stroke: str) -> str:
    """生成数据库圆柱形节点。"""

    cx = x + w / 2
    return (
        f'<g data-graph-role="node"><rect x="{x}" y="{y + 12}" width="{w}" height="{h - 24}" fill="{fill}"/>'
        f'<ellipse cx="{cx}" cy="{y + 12}" rx="{w / 2}" ry="12" fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>'
        f'<line x1="{x}" y1="{y + 12}" x2="{x}" y2="{y + h - 12}" stroke="{stroke}" stroke-width="1.6"/>'
        f'<line x1="{x + w}" y1="{y + 12}" x2="{x + w}" y2="{y + h - 12}" stroke="{stroke}" stroke-width="1.6"/>'
        f'<ellipse cx="{cx}" cy="{y + h - 12}" rx="{w / 2}" ry="12" fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>'
        f'<text x="{cx}" y="{y + 42}" class="node-title" text-anchor="middle">{escape(title)}</text>'
        f'<text x="{cx}" y="{y + 64}" class="detail" text-anchor="middle">{escape(detail)}</text></g>'
    )


def edge(points: list[tuple[int, int]], label: str, color: str, marker: str, dashed: bool = False) -> tuple[str, str]:
    """生成折线路径与偏移标签；标签独立放在覆盖层。"""

    path = "M " + " L ".join(f"{x},{y}" for x, y in points)
    dash = ' stroke-dasharray="7 5"' if dashed else ""
    line = (
        f'<path data-graph-role="edge" d="{path}" fill="none" stroke="{color}" stroke-width="2.2" '
        f'marker-end="url(#{marker})"{dash}/>'
    )
    longest = max(zip(points, points[1:]), key=lambda pair: abs(pair[1][0] - pair[0][0]) + abs(pair[1][1] - pair[0][1]))
    (x1, y1), (x2, y2) = longest
    lx, ly = (x1 + x2) // 2, (y1 + y2) // 2
    if y1 == y2:
        ly -= 9
    else:
        lx += 9
    width = max(52, len(label) * 7 + 14)
    overlay = (
        f'<rect x="{lx - width / 2}" y="{ly - 14}" width="{width}" height="19" rx="4" fill="#ffffff" opacity="0.96"/>'
        + text(lx, ly, label, "edge-label")
    )
    return line, overlay


def build() -> None:
    """按背景、连线、节点、覆盖文字四层构建 SVG。"""

    background: list[str] = []
    edges: list[str] = []
    nodes: list[str] = []
    overlays: list[str] = []

    lanes = [
        (30, 90, 1340, 225, "主请求路径：消息信任边界与模型推理", "#2563eb", "W1 + W2"),
        (30, 340, 1340, 230, "Agent 行动路径：预算、Tool Call、MCP 与 Observation", "#ea580c", "W2 + W3"),
        (30, 595, 1340, 260, "上下文与数据路径：安全 RAG、线程状态、长期记忆、权威数据", "#16a34a", "W3 + W4"),
    ]
    for x, y, w, h, title_value, color, week in lanes:
        background.append(
            f'<rect data-graph-role="container" x="{x}" y="{y}" width="{w}" height="{h}" rx="12" '
            f'fill="{color}" fill-opacity="0.035" stroke="{color}" stroke-width="1.2" stroke-dasharray="8 5"/>'
        )
        background.append(f'<rect x="{x + 16}" y="{y + 14}" width="390" height="27" rx="13" fill="{color}"/>')
        overlays.append(text(x + 30, y + 33, title_value, "lane-title", "start"))
        overlays.append(text(x + w - 22, y + 33, week, "week", "end"))

    # 主路径节点。
    nodes.extend(
        [
            card(55, 160, 135, 82, "用户输入", "query · intent", "#eff6ff", "#93c5fd"),
            card(235, 160, 160, 82, "Agent Host", "identity · trace", "#eff6ff", "#60a5fa"),
            card(440, 160, 180, 82, "消息组装", "system > developer > user", "#eff6ff", "#60a5fa"),
            card(665, 150, 190, 102, "Context Compiler", "选择 · 去重 · 压缩 · ACL", "#f0fdf4", "#4ade80"),
            card(905, 150, 160, 102, "LLM", "Token · Attention · Decode", "#faf5ff", "#c4b5fd"),
            card(1120, 160, 210, 82, "输出守卫", "Schema · citation · refusal", "#f0fdfa", "#5eead4"),
        ]
    )
    # Tool/Agent Loop 节点。
    nodes.extend(
        [
            card(245, 420, 170, 82, "预算控制器", "steps · token · timeout", "#fff7ed", "#fdba74"),
            card(500, 410, 175, 102, "Tool Call", "name · args · call_id", "#fff7ed", "#fb923c"),
            card(760, 410, 190, 102, "Tool Runtime", "Schema · Scope · retry", "#fff7ed", "#fb923c"),
            card(1040, 410, 210, 102, "MCP / Tools", "discover · call · result", "#fef2f2", "#fca5a5"),
        ]
    )
    # 数据与记忆节点。
    nodes.extend(
        [
            card(55, 680, 150, 82, "文档源", "version · delete event", "#eff6ff", "#93c5fd"),
            card(245, 680, 160, 82, "增量摄取", "chunk · embed · dedup", "#eff6ff", "#93c5fd"),
            database(450, 670, 165, 100, "安全索引", "dense + BM25 + ACL", "#eef2ff", "#818cf8"),
            card(655, 680, 165, 82, "Hybrid Retrieval", "filter · fuse · rerank", "#f0fdf4", "#4ade80"),
            card(865, 680, 150, 82, "线程状态", "history · checkpoint", "#f0fdf4", "#86efac"),
            database(1050, 670, 150, 100, "长期记忆", "semantic · episodic", "#f0fdf4", "#4ade80"),
            database(1230, 670, 110, 100, "业务库", "权威事实", "#faf5ff", "#a78bfa"),
        ]
    )
    def add(points: list[tuple[int, int]], label: str, color: str, marker: str, dashed: bool = False) -> None:
        line, overlay = edge(points, label, color, marker, dashed)
        edges.append(line)
        overlays.append(overlay)

    # 顶部主数据流。
    add([(190, 201), (235, 201)], "user message", "#2563eb", "arrow-blue")
    add([(395, 201), (440, 201)], "trusted envelope", "#2563eb", "arrow-blue")
    add([(620, 201), (665, 201)], "messages + identity", "#2563eb", "arrow-blue")
    add([(855, 201), (905, 201)], "budgeted tokens", "#7c3aed", "arrow-purple")
    add([(1065, 201), (1120, 201)], "answer / refusal", "#2563eb", "arrow-blue")

    # 模型到工具，再由 observation 回到消息/上下文。
    add([(985, 252), (985, 330), (700, 330), (700, 365), (588, 365), (588, 410)], "action", "#ea580c", "arrow-orange")
    add([(675, 461), (760, 461)], "tool request", "#ea580c", "arrow-orange")
    add([(950, 461), (1040, 461)], "JSON-RPC / MCP", "#ea580c", "arrow-orange")
    add([(1145, 410), (1145, 365), (1090, 365), (1090, 220), (1065, 220)], "observation", "#7c3aed", "arrow-purple")
    add([(315, 242), (315, 420)], "budgets", "#ea580c", "arrow-orange", True)

    # RAG 摄取与读取路径。
    add([(205, 721), (245, 721)], "source event", "#2563eb", "arrow-blue")
    add([(405, 721), (450, 721)], "chunks + lineage", "#7c3aed", "arrow-purple")
    add([(615, 721), (655, 721)], "authorized candidates", "#16a34a", "arrow-green")
    add([(738, 680), (738, 620), (450, 620), (450, 280), (665, 280), (665, 220)], "evidence + memory", "#16a34a", "arrow-green")

    # 长期记忆先按当前身份检索，再与线程状态一起进入 Context Retrieval。
    add([(1050, 721), (1015, 721)], "retrieve", "#16a34a", "arrow-green")
    add([(865, 721), (820, 721)], "recent state", "#16a34a", "arrow-green")
    add([(835, 512), (835, 585), (940, 585), (940, 680)], "checkpoint write", "#16a34a", "arrow-green", True)

    # Tool 读取/写入权威业务数据；记忆不能绕过这条边界。
    add([(1145, 512), (1145, 640), (1285, 640), (1285, 670)], "authorized API", "#9333ea", "arrow-violet")
    # 关键安全不变量与图例。
    overlays.append('<rect x="195" y="805" width="1010" height="34" rx="8" fill="#fff7ed" stroke="#fdba74"/>')
    overlays.append(text(700, 827, "最终上下文 = 相关证据 AND 当前授权；长期记忆是辅助上下文，业务库才是权威事实源", "invariant"))
    legend = [
        ("#2563eb", "请求/文档"),
        ("#ea580c", "行动/控制"),
        ("#16a34a", "上下文/记忆"),
        ("#7c3aed", "转换/Observation"),
        ("#9333ea", "权威业务访问"),
    ]
    lx = 70
    for color, label in legend:
        overlays.append(f'<line x1="{lx}" y1="910" x2="{lx + 32}" y2="910" stroke="{color}" stroke-width="3"/>')
        overlays.append(text(lx + 42, 914, label, "legend", "start"))
        lx += 210

    svg: list[str] = []
    svg.append('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1400 950" width="1400" height="950">')
    svg.append("""<style>
text{font-family:'PingFang SC','Hiragino Sans GB','Microsoft YaHei','Helvetica Neue',Arial,sans-serif}
.title{fill:#111827;font-size:27px;font-weight:700}.subtitle{fill:#6b7280;font-size:14px}
.node-title{fill:#111827;font-size:15px;font-weight:650}.detail{fill:#6b7280;font-size:11.5px}
.label{fill:#111827;font-size:13px}.lane-title{fill:#ffffff;font-size:12px;font-weight:700}
.week{fill:#475569;font-size:12px;font-weight:700}.edge-label{fill:#374151;font-size:10.5px;font-weight:600}
.invariant{fill:#9a3412;font-size:12.5px;font-weight:700}.legend{fill:#475569;font-size:11.5px}
</style>""")
    svg.append("""<defs>
<marker id="arrow-blue" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0,10 3.5,0 7" fill="#2563eb"/></marker>
<marker id="arrow-orange" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0,10 3.5,0 7" fill="#ea580c"/></marker>
<marker id="arrow-green" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0,10 3.5,0 7" fill="#16a34a"/></marker>
<marker id="arrow-purple" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0,10 3.5,0 7" fill="#7c3aed"/></marker>
<marker id="arrow-violet" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0,10 3.5,0 7" fill="#9333ea"/></marker>
<marker id="arrow-gray" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0,10 3.5,0 7" fill="#6b7280"/></marker>
</defs>""")
    svg.append('<rect width="1400" height="950" fill="#ffffff"/>')
    svg.append(text(700, 40, "前四周完整数据流：用户输入、工具、上下文、记忆与模型输出", "title"))
    svg.append(text(700, 66, "概率性模型负责选择与生成；确定性 Host / Runtime 负责身份、权限、预算、Schema 与副作用", "subtitle"))
    svg.extend(background)
    svg.extend(edges)
    svg.extend(nodes)
    svg.extend(overlays)
    svg.append("</svg>")
    ROOT.mkdir(parents=True, exist_ok=True)
    SVG_PATH.write_text("\n".join(svg), encoding="utf-8")
    print(SVG_PATH)


if __name__ == "__main__":
    build()
