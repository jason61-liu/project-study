"""生成第四周五张 Flat Icon 风格技术图。所有 SVG 均使用 Python 列表逐行构建。"""

from __future__ import annotations

from html import escape
from pathlib import Path


ROOT = Path(__file__).parent


class Diagram:
    """小型 SVG 画布：分离背景、连线、节点和覆盖文字，保持稳定层级。"""

    def __init__(self, name: str, title: str, subtitle: str, width: int = 1200, height: int = 820) -> None:
        self.path = ROOT / f"{name}.svg"
        self.width = width
        self.height = height
        self.background: list[str] = []
        self.edges: list[str] = []
        self.nodes: list[str] = []
        self.overlays: list[str] = []
        self.title = title
        self.subtitle = subtitle

    def text(self, x: float, y: float, value: str, css: str = "label", anchor: str = "middle", layer: str = "nodes") -> None:
        getattr(self, layer).append(
            f'<text x="{x}" y="{y}" class="{css}" text-anchor="{anchor}">{escape(value)}</text>'
        )

    def lane(self, x: int, y: int, w: int, h: int, title: str, color: str) -> None:
        self.background.append(
            f'<rect data-graph-role="container" x="{x}" y="{y}" width="{w}" height="{h}" rx="10" '
            f'fill="{color}" fill-opacity="0.035" stroke="{color}" stroke-width="1.2" stroke-dasharray="7 5"/>'
        )
        self.background.append(f'<rect x="{x+14}" y="{y+12}" width="150" height="24" rx="12" fill="{color}"/>')
        self.text(x + 89, y + 29, title, "lane-title", layer="overlays")

    def card(self, x: int, y: int, w: int, h: int, title: str, detail: str, fill: str = "#ffffff", stroke: str = "#d1d5db") -> None:
        self.nodes.append(
            f'<g data-graph-role="node"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'
            f'<text x="{x+w/2}" y="{y+28}" class="node-title" text-anchor="middle">{escape(title)}</text>'
            f'<text x="{x+w/2}" y="{y+49}" class="detail" text-anchor="middle">{escape(detail)}</text></g>'
        )

    def db(self, x: int, y: int, w: int, h: int, title: str, detail: str, fill: str, stroke: str) -> None:
        cx = x + w / 2
        ry = 10
        self.nodes.append(
            f'<g data-graph-role="node"><rect x="{x}" y="{y+ry}" width="{w}" height="{h-2*ry}" fill="{fill}"/>'
            f'<ellipse cx="{cx}" cy="{y+ry}" rx="{w/2}" ry="{ry}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'
            f'<line x1="{x}" y1="{y+ry}" x2="{x}" y2="{y+h-ry}" stroke="{stroke}" stroke-width="1.5"/>'
            f'<line x1="{x+w}" y1="{y+ry}" x2="{x+w}" y2="{y+h-ry}" stroke="{stroke}" stroke-width="1.5"/>'
            f'<ellipse cx="{cx}" cy="{y+h-ry}" rx="{w/2}" ry="{ry}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'
            f'<text x="{cx}" y="{y+36}" class="node-title" text-anchor="middle">{escape(title)}</text>'
            f'<text x="{cx}" y="{y+56}" class="detail" text-anchor="middle">{escape(detail)}</text></g>'
        )

    def arrow(self, points: list[tuple[int, int]], label: str = "", color: str = "#2563eb", dashed: bool = False, label_at: tuple[int, int] | None = None) -> None:
        marker = {"#2563eb": "blue", "#16a34a": "green", "#9333ea": "purple", "#dc2626": "red", "#ea580c": "orange"}.get(color, "gray")
        path = "M " + " L ".join(f"{x},{y}" for x, y in points)
        dash = ' stroke-dasharray="7 5"' if dashed else ""
        self.edges.append(
            f'<path data-graph-role="edge" d="{path}" fill="none" stroke="{color}" stroke-width="2" '
            f'marker-end="url(#arrow-{marker})"{dash}/>'
        )
        if label:
            lx, ly = label_at or ((points[0][0] + points[-1][0]) // 2, (points[0][1] + points[-1][1]) // 2 - 8)
            self.overlays.append(f'<rect x="{lx-48}" y="{ly-13}" width="96" height="18" rx="4" fill="#ffffff" opacity="0.95"/>')
            self.text(lx, ly, label, "edge-label", layer="overlays")

    def note(self, x: int, y: int, w: int, title: str, text: str, color: str = "#fff7ed", stroke: str = "#fdba74") -> None:
        self.overlays.append(f'<rect x="{x}" y="{y}" width="{w}" height="52" rx="8" fill="{color}" stroke="{stroke}"/>')
        self.text(x + 14, y + 21, title, "note-title", "start", "overlays")
        self.text(x + 14, y + 40, text, "note-text", "start", "overlays")

    def legend(self, items: list[tuple[str, str]]) -> None:
        x = 45
        y = self.height - 25
        for color, name in items:
            self.overlays.append(f'<line x1="{x}" y1="{y}" x2="{x+28}" y2="{y}" stroke="{color}" stroke-width="3"/>')
            self.text(x + 36, y + 4, name, "legend", "start", "overlays")
            x += 145

    def write(self) -> None:
        lines: list[str] = []
        lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.width} {self.height}" width="{self.width}" height="{self.height}">')
        lines.append("""<style>
text{font-family:'PingFang SC','Hiragino Sans GB','Microsoft YaHei','Helvetica Neue',Helvetica,Arial,sans-serif}
.title{fill:#111827;font-size:25px;font-weight:700}.subtitle{fill:#6b7280;font-size:13px}
.node-title{fill:#111827;font-size:14px;font-weight:650}.detail{fill:#6b7280;font-size:11px}
.label{fill:#111827;font-size:13px}.lane-title{fill:#fff;font-size:12px;font-weight:700}
.edge-label{fill:#374151;font-size:11px;font-weight:600}.note-title{fill:#9a3412;font-size:12px;font-weight:700}
.note-text{fill:#7c2d12;font-size:11px}.legend{fill:#475569;font-size:11px}
</style>""")
        lines.append("<defs>")
        for name, color in (("blue", "#2563eb"), ("green", "#16a34a"), ("purple", "#9333ea"), ("red", "#dc2626"), ("orange", "#ea580c"), ("gray", "#6b7280")):
            lines.append(f'<marker id="arrow-{name}" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0,10 3.5,0 7" fill="{color}"/></marker>')
        lines.append("</defs>")
        lines.append(f'<rect width="{self.width}" height="{self.height}" fill="#ffffff"/>')
        lines.append(f'<text x="{self.width/2}" y="38" class="title" text-anchor="middle">{escape(self.title)}</text>')
        lines.append(f'<text x="{self.width/2}" y="63" class="subtitle" text-anchor="middle">{escape(self.subtitle)}</text>')
        lines.extend(self.background)
        lines.extend(self.edges)
        lines.extend(self.nodes)
        lines.extend(self.overlays)
        lines.append("</svg>")
        self.path.write_text("\n".join(lines), encoding="utf-8")


def ingestion() -> None:
    d = Diagram("secure-incremental-rag", "增量摄取、权限安全检索与删除传播", "相关性只决定候选顺序；ACL 决定候选是否允许进入模型上下文", height=900)
    d.lane(35, 90, 1130, 235, "写入：增量摄取", "#2563eb")
    cards = [(55,"源文档/事件","etag · mtime · CDC"),(255,"版本与去重","content hash · source id"),(455,"解析与 Chunk","结构边界 · overlap"),(655,"双路表征","Embedding + BM25"),(880,"索引提交","ACL · lineage · version")]
    for x,t,s in cards: d.card(x,165,165 if x<880 else 245,72,t,s,"#eff6ff","#93c5fd")
    for a,b,l in ((220,255,"change"),(420,455,"new version"),(620,655,"chunks"),(820,880,"atomic batch")): d.arrow([(a,201),(b,201)],l)
    d.card(455,245,165,55,"ACL 快照","principal/group/policy v", "#fff7ed","#fdba74")
    # ACL 元数据从两个卡片之间的 8px 走廊进入索引，避免穿过 Chunk 节点。
    d.arrow([(537,245),(537,241),(1002,241),(1002,237)],"inherit", "#ea580c", label_at=(765,239))

    d.lane(35,350,1130, 225, "读取：安全检索", "#16a34a")
    qcards=[(55,"Query + Identity","user · tenant · groups"),(245,"查询理解","rewrite · filters"),(435,"Hybrid Recall","dense + sparse"),(625,"Fusion/Rerank","RRF · cross-encoder"),(815,"ACL 最终门","deny by default"),(1005,"Context","citation + provenance")]
    for x,t,s in qcards: d.card(x,425,145,72,t,s,"#f0fdf4","#86efac")
    for a,b,l in ((200,245,"query"),(390,435,"filtered"),(580,625,"top-N"),(770,815,"ranked"),(960,1005,"allowed")): d.arrow([(a,461),(b,461)],l,"#16a34a")
    d.note(330,515,540,"安全不变量","final_context = top-ranked candidates AND currently-authorized resources")

    d.lane(35,600,1130, 235, "删除与重建", "#dc2626")
    dels=[(55,"Tombstone","source/version deleted"),(260,"删除事件","outbox + idempotency"),(465,"全派生面清理","chunk/vector/BM25/cache"),(700,"Shadow Rebuild","snapshot + delta replay"),(940,"Alias Swap","校验后原子切换")]
    for x,t,s in dels: d.card(x,675,175 if x<940 else 185,72,t,s,"#fef2f2","#fca5a5")
    for a,b,l in ((230,260,"delete"),(435,465,"fan-out")): d.arrow([(a,711),(b,711)],l,"#dc2626",True)
    d.arrow([(640,711),(700,711)],"rebuild", "#9333ea")
    d.arrow([(875,711),(940,711)],"verified", "#9333ea")
    d.note(245,770,710,"重建期间","双写 delta 或记录变更日志；切换前比较文档数、版本水位、ACL 覆盖率与抽样检索")
    d.legend([("#2563eb","摄取"),("#16a34a","安全读取"),("#dc2626","删除"),("#9333ea","重建")])
    d.write()


def evaluation() -> None:
    d=Diagram("rag-evaluation-stack","RAG 评估：从召回到有依据的回答","离线检索指标、引用/忠实度和无答案检测必须分别测量",height=820)
    d.lane(35,90,1130,165,"评测数据", "#2563eb")
    for x,t,s in [(70,"Query Set","answerable + unanswerable"),(310,"Graded Qrels","相关性等级"),(550,"ACL Oracle","allowed / denied"),(790,"Gold Evidence","source spans"),(1010,"Gold Answer","可选参考答案")]:
        d.card(x,155,150 if x<1010 else 135,65,t,s,"#eff6ff","#93c5fd")
    d.lane(35,280,1130,245,"分层指标", "#16a34a")
    metrics=[(55,"Recall@K","命中任一相关证据"),(275,"MRR","首个相关结果位置"),(495,"引用正确率","claim-to-cited-span"),(715,"答案忠实度","claims entailed by context"),(935,"无答案检测","precision / recall / F1")]
    for x,t,s in metrics: d.card(x,355,190,76,t,s,"#f0fdf4","#86efac")
    for x in (150,370,590,810,1030): d.arrow([(x,220),(x,345)],"evaluate","#2563eb")
    d.lane(35,550,1130,180,"发布门与切片", "#9333ea")
    d.card(70,615,210,70,"相关性门","Recall/MRR 达标", "#faf5ff","#c4b5fd")
    d.card(360,615,210,70,"安全门","ACL leakage = 0", "#fef2f2","#fca5a5")
    d.card(650,615,210,70,"生成门","citation + faithfulness", "#faf5ff","#c4b5fd")
    d.card(940,615,170,70,"拒答门","answerability F1", "#fff7ed","#fdba74")
    for a,b in ((280,360),(570,650),(860,940)): d.arrow([(a,650),(b,650)],"AND","#9333ea")
    d.note(235,715,730,"必须分层报告","按 tenant、语言、文档类型、时间窗口、query 难度和 answerable 分片，平均值不能掩盖安全泄漏")
    d.legend([("#2563eb","Gold/Oracle"),("#16a34a","指标"),("#9333ea","发布门"),("#dc2626","权限失败")])
    d.write()


def context_engineering() -> None:
    d=Diagram("context-engineering-compiler","Context Engineering：把有限注意力预算当作编译目标","从候选信息宇宙中选择最小、高信号、可溯源且未过期的 Token 集",height=850)
    d.lane(35,90,1130,185,"候选上下文宇宙", "#6b7280")
    for x,t,s in [(55,"System/Policy","稳定高优先级"),(270,"Tool Schemas","最小可区分集合"),(485,"History","近期消息与决策"),(700,"Retrieved Data","JIT / progressive disclosure"),(915,"Memory/Notes","跨窗口持久状态")]:
        d.card(x,160,180,70,t,s,"#f9fafb","#d1d5db")
    d.lane(35,300,1130,260,"Context Compiler", "#2563eb")
    stages=[(55,"1 选择","task relevance + ACL"),(245,"2 去重","content/entity/citation"),(435,"3 优先级/过期","authority * freshness"),(625,"4 摘要/压缩","保留决策与未决项"),(815,"5 隔离","sub-agent / scratchpad"),(1005,"6 溯源","source + version + span")]
    for x,t,s in stages: d.card(x,385,150,76,t,s,"#eff6ff","#93c5fd")
    for a,b in ((205,245),(395,435),(585,625),(775,815),(965,1005)): d.arrow([(a,423),(b,423)],"", "#2563eb")
    for x in (145,360,575,790,1005): d.arrow([(x,230),(x,375)],"candidates","#6b7280",True)
    d.note(255,485,690,"排序原则","policy > current task evidence > recent decisions > optional examples；过期或低置信内容降级/剔除")
    d.lane(35,590,1130,175,"Inference + Feedback", "#16a34a")
    d.card(140,650,220,70,"Budgeted Context","token cap + reserved output", "#f0fdf4","#86efac")
    d.card(490,650,220,70,"LLM / Agent","reason + tool use", "#faf5ff","#c4b5fd")
    d.card(840,650,220,70,"Trace & Notes","measure · persist · expire", "#fff7ed","#fdba74")
    d.arrow([(360,685),(490,685)],"smallest useful set","#16a34a")
    d.arrow([(710,685),(840,685)],"outcome","#16a34a")
    d.arrow([(950,650),(950,615),(250,615),(250,640)],"next-turn feedback","#9333ea",True,label_at=(600,607))
    d.legend([("#6b7280","候选"),("#2563eb","编译/裁剪"),("#16a34a","推理"),("#9333ea","反馈")])
    d.write()


def memory_taxonomy() -> None:
    d=Diagram("agent-state-memory-cache-taxonomy","Agent 状态、记忆、业务事实与缓存的边界","不要按“都能保存东西”归为 Memory；应按语义、作用域、权威性和生命周期区分",height=900)
    d.lane(35,90,1130,190,"当前推理与恢复", "#2563eb")
    for x,t,s in [(60,"Conversation History","消息序列；可裁剪"),(335,"Checkpoint","thread state；可恢复"),(610,"Prompt Cache","输入前缀复用"),(885,"KV Cache","单次解码注意力状态")]:
        d.card(x,165,220,75,t,s,"#eff6ff","#93c5fd")
    d.lane(35,310,1130,225,"长期 Agent Memory", "#16a34a")
    for x,t,s in [(100,"Semantic","用户/世界事实"),(390,"Episodic","过去事件与轨迹"),(680,"Procedural","规则/策略/示例")]:
        d.db(x,385,210,85,t,s,"#f0fdf4","#86efac")
    d.card(950,390,155,75,"Memory Policy","extract · merge · expire", "#fff7ed","#fdba74")
    d.arrow([(310,427),(390,427)],"retrieve","#16a34a")
    d.arrow([(600,427),(680,427)],"retrieve","#16a34a")
    d.arrow([(890,427),(950,427)],"govern","#ea580c")
    d.lane(35,565,1130,220,"权威业务数据", "#9333ea")
    d.db(115,640,245,90,"Business Database","订单/余额/权限/审计", "#faf5ff","#c4b5fd")
    d.card(485,645,230,75,"Tool/API Boundary","transaction + authorization", "#ffffff","#d1d5db")
    d.card(840,645,230,75,"Grounded Observation","version + provenance", "#f0fdfa","#5eead4")
    d.arrow([(360,682),(485,682)],"read/write","#9333ea")
    d.arrow([(715,682),(840,682)],"facts","#9333ea")
    d.note(185,775,830,"核心约束","Memory 是辅助上下文，不是订单/余额/ACL 的 source of truth；Cache 可随时丢弃，Checkpoint 负责恢复而非知识抽取")
    d.legend([("#2563eb","推理状态/缓存"),("#16a34a","长期记忆"),("#9333ea","权威业务事实"),("#ea580c","治理")])
    d.write()


def langgraph_mem0() -> None:
    d=Diagram("langgraph-vs-mem0-memory","LangGraph Memory 与 Mem0：编排持久化原语 vs 事实记忆层","二者可组合：LangGraph 管线程/图状态，Mem0 管跨会话事实抽取与检索",height=900)
    d.lane(35,90,545,660,"LangGraph", "#1c3c3c")
    d.card(80,165,210,75,"thread_id","一次会话/执行线程", "#f0fdfa","#5eead4")
    d.db(335,160,190,88,"Checkpointer","thread-scoped state", "#eff6ff","#93c5fd")
    d.arrow([(290,202),(335,202)],"checkpoint","#2563eb")
    d.card(80,300,210,75,"namespace + key","user/org/app scope", "#f0fdfa","#5eead4")
    d.db(335,295,190,88,"Store","JSON + semantic search", "#f0fdf4","#86efac")
    d.arrow([(290,337),(335,337)],"put/search","#16a34a")
    d.card(80,450,210,75,"Hot Path","立即可见；增加延迟", "#fff7ed","#fdba74")
    d.card(335,450,190,75,"Background","解耦；存在新鲜度延迟", "#faf5ff","#c4b5fd")
    d.arrow([(290,487),(335,487)],"trade-off","#9333ea")
    d.note(75,575,465,"开发者负责","定义 state/schema、何时写、提取什么、namespace ACL、冲突合并与遗忘策略")

    d.lane(620,90,545,660,"Mem0", "#6366f1")
    d.card(665,165,185,75,"add(messages)","user/agent/run scope", "#faf5ff","#c4b5fd")
    d.card(920,165,190,75,"Fact Extraction","LLM + context lookup", "#faf5ff","#c4b5fd")
    d.arrow([(850,202),(920,202)],"extract","#9333ea")
    d.card(665,300,185,75,"Dedup / Entity","clean durable facts", "#faf5ff","#c4b5fd")
    d.db(920,292,190,92,"Stores","SQL + vector + entity", "#eff6ff","#93c5fd")
    d.arrow([(850,337),(920,337)],"persist","#2563eb")
    d.card(665,450,185,75,"search(query)","filters required", "#f0fdf4","#86efac")
    d.card(920,450,190,75,"Hybrid Rank","semantic + BM25 + entity", "#f0fdf4","#86efac")
    d.arrow([(850,487),(920,487)],"retrieve","#16a34a")
    d.note(655,575,465,"平台/SDK负责较多","事实抽取、去重、Embedding 和检索；应用仍负责授权、作用域过滤、更新/删除与质量评估")
    d.arrow([(580,675),(620,675)],"可组合","#ea580c")
    d.note(260,780,680,"组合方式","Checkpoint 保存当前图执行状态；LangGraph Store/Mem0 提供跨线程记忆。任何 namespace/filter 都必须从已验证 tenant/user 构造")
    d.legend([("#2563eb","持久化"),("#16a34a","检索"),("#9333ea","提取/异步"),("#ea580c","组合边界")])
    d.write()


if __name__ == "__main__":
    ROOT.mkdir(parents=True, exist_ok=True)
    ingestion()
    evaluation()
    context_engineering()
    memory_taxonomy()
    langgraph_mem0()
    print("generated:", *sorted(path.name for path in ROOT.glob("*.svg")))
