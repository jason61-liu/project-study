"""生成第六周五张 Dark Luxury 技术图，并导出高分辨率 PNG。"""

from pathlib import Path
from xml.sax.saxutils import escape

import cairosvg


OUT = Path(__file__).parent
W, H = 1400, 860
COLORS = {
    "gold": "#d4a574", "mint": "#6ee7b7", "orange": "#fdba74",
    "violet": "#a78bfa", "blue": "#38bdf8", "rose": "#f87171",
    "gray": "#94a3b8", "amber": "#fbbf24",
}


def base(title: str, subtitle: str) -> list[str]:
    lines: list[str] = []
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">')
    lines.append('<style>')
    lines.append('text{font-family:"Arial Unicode MS","Hiragino Sans GB","Heiti SC","Helvetica Neue",Arial,sans-serif}.ttl{font-family:"Songti SC","Arial Unicode MS",Georgia,serif;font-size:30px;font-weight:700;fill:#f5f0eb}.sub{font-size:13px;fill:#a39787}.sec{font-family:"Songti SC","Arial Unicode MS",Georgia,serif;font-size:15px;font-weight:700;fill:#c9a96e}.nm{font-size:15px;font-weight:650;fill:#f5f0eb}.sm{font-size:11px;fill:#a39787}.xs{font-size:10px;fill:#6b5f53}.al{font-size:11px;fill:#c9b8a0}.mono{font-family:"Arial Unicode MS","SF Mono","Courier New",monospace;font-size:11px;fill:#d7c6b5}</style>')
    lines.append('<defs>')
    for name, color in COLORS.items():
        lines.append(f'<marker id="arr-{name}" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0,10 3.5,0 7" fill="{color}"/></marker>')
    lines.append('<radialGradient id="glow"><stop offset="0" stop-color="#d4a574" stop-opacity=".07"/><stop offset="1" stop-color="#d4a574" stop-opacity="0"/></radialGradient>')
    lines.append('</defs>')
    lines.append(f'<rect width="{W}" height="{H}" fill="#0a0a0a"/>')
    lines.append('<ellipse cx="700" cy="400" rx="620" ry="340" fill="url(#glow)" data-graph-role="decoration" data-owner="canvas"/>')
    lines.append(f'<text x="700" y="48" text-anchor="middle" class="ttl">{escape(title)}</text>')
    lines.append(f'<text x="700" y="74" text-anchor="middle" class="sub">{escape(subtitle)}</text>')
    return lines


def panel(lines: list[str], x: int, y: int, w: int, h: int, title: str, color: str = "#d4a574") -> None:
    lines.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="#0e0e0e" stroke="{color}" stroke-width="1" stroke-dasharray="7 5"/>')
    lines.append(f'<text x="{x+18}" y="{y+28}" class="sec" fill="{color}">{escape(title)}</text>')


def node(lines: list[str], x: int, y: int, w: int, h: int, title: str, detail: str, color: str, double: bool = False) -> None:
    lines.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="#111111" stroke="{color}" stroke-width="1.7"/>')
    if double:
        lines.append(f'<rect x="{x+4}" y="{y+4}" width="{w-8}" height="{h-8}" rx="6" fill="none" stroke="{color}" stroke-width=".6" opacity=".55"/>')
    lines.append(f'<text x="{x+w/2}" y="{y+29}" text-anchor="middle" class="nm" fill="{color}">{escape(title)}</text>')
    lines.append(f'<text x="{x+w/2}" y="{y+51}" text-anchor="middle" class="sm">{escape(detail)}</text>')


def label(lines: list[str], x: float, y: float, value: str, cls: str = "al", anchor: str = "middle") -> None:
    lines.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" class="{cls}">{escape(value)}</text>')


def arrow(lines: list[str], d: str, kind: str = "gold", dashed: bool = False, width: float = 2.0) -> None:
    dash = ' stroke-dasharray="6 4"' if dashed else ""
    lines.append(f'<path d="{d}" fill="none" stroke="{COLORS[kind]}" stroke-width="{width}"{dash} marker-end="url(#arr-{kind})"/>')


def footer(lines: list[str], items: list[tuple[str, str]]) -> None:
    x = 65
    for kind, value in items:
        lines.append(f'<line x1="{x}" y1="824" x2="{x+34}" y2="824" stroke="{COLORS[kind]}" stroke-width="2" marker-end="url(#arr-{kind})"/>')
        label(lines, x+46, 828, value, "sm", "start")
        x += 235
    lines.append('</svg>')


def task_graph() -> list[str]:
    lines = base("任务分解：从目标到可执行依赖图", "子目标不是自然语言清单；它必须有依赖、完成谓词、证据和重规划触发器")
    panel(lines, 35, 105, 940, 610, "TASK DAG · VERSION v7", COLORS["blue"])
    panel(lines, 1025, 105, 340, 610, "CONTROL CONTRACT", COLORS["rose"])
    node(lines, 80, 165, 210, 78, "G · 业务目标", "机器可验证 outcome", COLORS["gold"], True)
    node(lines, 365, 165, 210, 78, "D · 分解器", "生成子目标与依赖", COLORS["violet"])
    node(lines, 650, 165, 265, 78, "V · 图验证器", "无环 · 覆盖 · 资源 · 权限", COLORS["mint"])
    arrow(lines, "M290 204 H365", "gold"); label(lines, 327, 192, "goal")
    arrow(lines, "M575 204 H650", "violet"); label(lines, 612, 192, "TaskGraph")
    tasks = [(100,340,"T1 · 取证","evidence_set",COLORS["blue"]),(390,340,"T2 · 分析","analysis_artifact",COLORS["orange"]),(680,340,"T3 · 校验","verified_report",COLORS["mint"]),(390,505,"T4 · 发布","published_id",COLORS["rose"])]
    for x,y,t,d,c in tasks: node(lines,x,y,210,78,t,d,c)
    arrow(lines, "M470 243 V300 H205 V340", "blue"); label(lines, 325, 290, "depends_on = none")
    arrow(lines, "M310 379 H390", "orange"); label(lines, 350, 367, "evidence")
    arrow(lines, "M600 379 H680", "mint"); label(lines, 640, 367, "artifact")
    arrow(lines, "M785 418 V455 H495 V505", "rose"); label(lines, 642, 445, "verified")
    lines.append('<rect x="100" y="625" width="815" height="58" rx="8" fill="#1a1a1a" stroke="#d4a574"/>')
    label(lines,507,649,"关键路径：T1 → T2 → T3 → T4；并行只允许在无数据依赖的节点之间","sm")
    label(lines,507,670,"完成 = 节点谓词成立 AND Artifact 可追溯；不是模型说 done","xs")
    contracts=[("Subgoal","objective · non-goals",COLORS["violet"]),("Done Predicate","boolean over evidence",COLORS["mint"]),("Budgets","step · token · deadline",COLORS["amber"]),("Replan Trigger","version · blocker · drift",COLORS["rose"]),("Graph Version","CAS: expected_version",COLORS["blue"])]
    for i,(t,d,c) in enumerate(contracts): node(lines,1060,155+i*103,270,68,t,d,c)
    arrow(lines,"M975 410 H1000 V395 H1025","rose",True); label(lines,1000,383,"gate")
    footer(lines, [("blue","依赖/数据"),("violet","规划"),("mint","验证"),("rose","重规划/控制")])
    return lines


def verifier_layers() -> list[str]:
    lines = base("反思与验证：三层证据强度", "自我反思改善候选，外部验证提供独立信号，确定性检查执行不可协商规则")
    node(lines, 70, 175, 250, 88, "Candidate", "模型生成的方案或答案", COLORS["gold"], True)
    layers=[(390,130,"L1 · Self-Reflection","同模型批评与修订","低独立性 · 高覆盖",COLORS["violet"]),(390,300,"L2 · External Verifier","独立模型 / 环境 / 人工","中高独立性 · 语义判断",COLORS["blue"]),(390,470,"L3 · Deterministic Checker","Schema / tests / policy engine","高确定性 · 覆盖有限",COLORS["mint"])]
    for x,y,t,d,e,c in layers:
        node(lines,x,y,315,88,t,d,c)
        label(lines,x+157,y+72,e,"xs")
    node(lines, 790, 275, 240, 100, "Evidence Reducer", "合并 verdict + provenance", COLORS["orange"])
    node(lines, 1110, 175, 220, 88, "Accept", "满足硬门禁与质量阈值", COLORS["mint"])
    node(lines, 1110, 455, 220, 88, "Reject / Revise", "结构化反馈或终止", COLORS["rose"])
    arrow(lines,"M320 219 H355 V174 H390","violet"); arrow(lines,"M320 219 H350 V344 H390","blue"); arrow(lines,"M320 219 H345 V514 H390","mint")
    arrow(lines,"M705 174 H750 V315 H790","violet"); arrow(lines,"M705 344 H790","blue"); arrow(lines,"M705 514 H750 V335 H790","mint")
    arrow(lines,"M1030 315 H1070 V219 H1110","mint"); label(lines,1070,205,"hard gates pass")
    arrow(lines,"M1030 335 H1070 V499 H1110","rose"); label(lines,1070,486,"fail / uncertainty")
    arrow(lines,"M1110 499 H1060 V650 H195 V263","gold"); label(lines,620,638,"bounded optimize loop · max rounds · min improvement")
    panel(lines, 70, 700, 1260, 70, "TRUST ORDER", COLORS["gold"])
    label(lines,700,738,"事实与权限：权威数据源 / Policy Engine > 外部验证器 > 自我反思；三者不能互相替代","sm")
    label(lines,700,758,"反思通过不代表测试通过；Judge 高分不能覆盖 Schema、ACL 或业务状态失败","xs")
    footer(lines, [("violet","自我反馈"),("blue","独立验证"),("mint","确定性门禁"),("rose","拒绝/修订")])
    return lines


def control_boundaries() -> list[str]:
    lines = base("五种编排形态的控制权边界", "关键问题：谁选择下一步、谁持有会话、谁能调用工具、谁对最终结果负责")
    methods=[
        ("State Machine","代码按 event + guard 转移","Runtime","固定、强审计",COLORS["blue"]),
        ("Task Queue","生产者投递；Worker claim","Scheduler","异步、可削峰",COLORS["mint"]),
        ("Manager","Manager 分解、委派、合并","Manager Agent","动态集中控制",COLORS["gold"]),
        ("Handoff","当前 Agent 转移会话所有权","Receiving Agent","对话/责任转移",COLORS["rose"]),
        ("Agent-as-Tool","父 Agent 调用受限子 Agent","Parent Agent","子 Agent 无最终控制",COLORS["violet"]),
    ]
    for i,(name,flow,owner,fit,color) in enumerate(methods):
        y=120+i*120
        lines.append(f'<rect x="45" y="{y}" width="1310" height="94" rx="10" fill="#101010" stroke="{color}"/>')
        lines.append(f'<rect x="45" y="{y}" width="250" height="94" rx="10" fill="#171717"/>')
        lines.append(f'<text x="70" y="{y+37}" class="nm" fill="{color}">{escape(name)}</text>')
        label(lines,70,y+63,fit,"xs","start")
        label(lines,340,y+35,flow,"sm","start")
        label(lines,340,y+63,f"control_owner = {owner}","mono","start")
        lines.append(f'<rect x="1040" y="{y+19}" width="270" height="55" rx="7" fill="#1a1a1a" stroke="{color}"/>')
        label(lines,1175,y+42,"FINAL ACCOUNTABILITY","xs")
        final={"State Machine":"Runtime / service owner","Task Queue":"Workflow coordinator","Manager":"Manager + Runtime","Handoff":"Receiving Agent + Runtime","Agent-as-Tool":"Parent Agent + Runtime"}[name]
        label(lines,1175,y+64,final,"sm")
    lines.append('<rect x="45" y="735" width="1310" height="48" rx="8" fill="#1a1a1a" stroke="#d4a574"/>')
    label(lines,700,758,"Handoff 转移会话责任；Agent-as-Tool 只返回结果。Queue 解耦时间，不自动解决业务依赖与权限","sm")
    footer(lines, [("blue","确定性控制"),("mint","异步调度"),("gold","Manager"),("rose","所有权转移"),("violet","受限调用")])
    return lines


def orchestration_cost() -> list[str]:
    lines = base("串行、并行与多 Agent：性能收益和协调税", "并行只压缩无依赖关键路径；多 Agent 额外支付 Context、调度、聚合和恢复成本")
    panel(lines,35,105,420,580,"SERIAL · STRONG DEPENDENCY",COLORS["blue"])
    for i,name in enumerate(["A · Retrieve","B · Analyze","C · Verify","D · Publish"]):
        node(lines,105,155+i*115,280,66,name,"L = sum(L_i) · P ≈ prod(p_i)",COLORS["blue"] if i<3 else COLORS["rose"])
        if i<3: arrow(lines,f"M245 {221+i*115} V{270+i*115}","blue")
    panel(lines,490,105,420,580,"PARALLEL · INDEPENDENT WORK",COLORS["mint"])
    node(lines,580,155,240,66,"Fan-out","dependency check first",COLORS["gold"])
    for i,(x,t) in enumerate([(525,"Worker A"),(680,"Worker B")]): node(lines,x,310,135,66,t,"isolated",COLORS["mint"])
    arrow(lines,"M700 221 V270 H592 V310","mint"); arrow(lines,"M700 221 V270 H747 V310","mint")
    node(lines,580,465,240,76,"Aggregator","conflict · partial · lineage",COLORS["orange"])
    arrow(lines,"M592 376 V425 H660 V465","orange"); arrow(lines,"M747 376 V425 H740 V465","orange")
    label(lines,700,580,"L ≈ max(L_i) + L_agg","mono"); label(lines,700,606,"C ≈ sum(C_i) + C_coord","mono")
    panel(lines,945,105,420,580,"MULTI-AGENT · COORDINATION TAX",COLORS["rose"])
    costs=[("Context Duplication","N × system/tools/history",COLORS["violet"]),("Scheduling","queue · priority · backpressure",COLORS["amber"]),("Error Propagation","decompose → worker → aggregate",COLORS["rose"]),("Cancellation","parent → workers → tools",COLORS["blue"]),("Recovery","partial commit · replay · merge",COLORS["mint"])]
    for i,(t,d,c) in enumerate(costs): node(lines,985,150+i*98,340,66,t,d,c)
    lines.append('<rect x="85" y="715" width="1230" height="66" rx="9" fill="#1a1a1a" stroke="#d4a574"/>')
    label(lines,700,742,"并行收益条件：saved critical-path latency > fan-out + queue + aggregate + retry overhead","sm")
    label(lines,700,764,"默认单控制器；只有独立性、隔离或专业化收益被实验验证后才升级多 Agent","xs")
    footer(lines, [("blue","串行依赖"),("mint","并行执行"),("orange","聚合"),("rose","错误/协调税")])
    return lines


def recoverable_execution() -> list[str]:
    lines = base("可恢复执行协议：版本、幂等、断点与补偿", "Exactly-once 通常不可得；目标是 at-least-once 调度下获得业务级 effect-once")
    panel(lines,35,105,860,620,"VERSIONED STATE MACHINE",COLORS["blue"])
    states=[(90,170,"READY","state_version=7",COLORS["gray"]),(350,170,"CLAIMED","lease + worker_id",COLORS["blue"]),(610,170,"EXECUTING","attempt=3",COLORS["orange"]),(610,390,"COMMITTED","effect_receipt",COLORS["mint"]),(350,390,"COMPENSATING","saga action",COLORS["rose"]),(90,390,"FAILED","terminal reason",COLORS["rose"])]
    for x,y,t,d,c in states: node(lines,x,y,210,82,t,d,c,double=t in {"EXECUTING","COMMITTED"})
    arrow(lines,"M300 211 H350","blue"); label(lines,325,198,"CAS claim")
    arrow(lines,"M560 211 H610","orange"); label(lines,585,198,"execute")
    arrow(lines,"M715 252 V390","mint"); label(lines,730,326,"commit receipt","al","start")
    arrow(lines,"M610 431 H560","rose"); label(lines,585,418,"compensate")
    arrow(lines,"M350 431 H300","rose"); label(lines,325,418,"failed")
    arrow(lines,"M455 390 V330 H195 V252","gold"); label(lines,330,319,"retry from checkpoint")
    lines.append('<rect x="90" y="555" width="730" height="112" rx="9" fill="#1a1a1a" stroke="#d4a574"/>')
    label(lines,455,580,"原子写入：new_state + outbox_event + checkpoint","sm")
    label(lines,455,604,"CAS: UPDATE ... WHERE state_version = expected_version","mono")
    label(lines,455,628,"idempotency_key = hash(task_id, operation, business_scope)","mono")
    label(lines,455,650,"恢复先 reconcile 外部状态，再决定 retry / compensate / manual","xs")
    panel(lines,945,105,420,620,"RECOVERY LEDGER",COLORS["gold"])
    ledgers=[("State Version","拒绝 stale writer",COLORS["blue"]),("Idempotency Record","相同 key 返回旧 receipt",COLORS["mint"]),("Checkpoint","恢复状态而非聊天文本",COLORS["violet"]),("Effect Receipt","区分 unknown / committed",COLORS["orange"]),("Compensation","语义逆操作，不是 DB rollback",COLORS["rose"])]
    for i,(t,d,c) in enumerate(ledgers): node(lines,985,150+i*103,340,70,t,d,c)
    arrow(lines,"M895 420 H920 V392 H945","gold",True); label(lines,920,379,"persist")
    lines.append('<rect x="65" y="747" width="1270" height="42" rx="8" fill="#1a1a1a" stroke="#d4a574"/>')
    label(lines,700,772,"断点必须同时描述：逻辑状态、证据版本、预算余额、在途调用和已提交副作用","sm")
    footer(lines, [("blue","状态转移"),("mint","提交"),("gold","恢复"),("rose","补偿/失败")])
    return lines


def planning_recovery_interview() -> list[str]:
    lines = base(
        "规划、恢复与任务编排：架构选择和消融证据",
        "先确定控制权与失败语义，再用配对实验判断多 Agent 的收益是否覆盖协调税",
    )
    panel(lines, 35, 110, 390, 570, "ARCHITECTURE ADOPTION", COLORS["blue"])
    choices = [
        (75, 160, "Single Agent", "路径动态 · 工具有限 · 单一 owner", COLORS["blue"]),
        (75, 285, "Workflow", "路径可枚举 · 硬门禁 · 强审计", COLORS["mint"]),
        (75, 410, "Multi-Agent", "可独立并行 · 隔离/专业化收益", COLORS["violet"]),
    ]
    for x, y, title, detail, color in choices:
        node(lines, x, y, 310, 78, title, detail, color, double=title == "Single Agent")
    lines.append('<rect x="75" y="555" width="310" height="82" rx="8" fill="#1a1a1a" stroke="#f87171"/>')
    label(lines, 230, 580, "反例门禁", "nm")
    label(lines, 230, 605, "稳定路径不用 Agent · 强依赖不并行", "sm")
    label(lines, 230, 625, "无可测增益不升级多 Agent", "xs")

    panel(lines, 455, 110, 500, 570, "CONTROL OWNERSHIP", COLORS["gold"])
    label(lines, 485, 160, "MANAGER · retained ownership", "sec", "start")
    node(lines, 485, 190, 180, 76, "Manager", "plan · delegate · merge", COLORS["gold"], True)
    node(lines, 745, 190, 180, 76, "Specialist", "bounded artifact", COLORS["violet"])
    arrow(lines, "M665 218 H745", "gold"); label(lines, 705, 207, "delegate")
    arrow(lines, "M745 244 H665", "mint"); label(lines, 705, 262, "return artifact")
    lines.append('<rect x="485" y="285" width="440" height="48" rx="7" fill="#1a1a1a" stroke="#d4a574"/>')
    label(lines, 705, 315, "conversation_owner = Manager", "mono")

    label(lines, 485, 390, "HANDOFF · transferred ownership", "sec", "start")
    node(lines, 485, 420, 160, 76, "Agent A", "current owner", COLORS["blue"])
    lines.append('<polygon points="705,420 755,458 705,496 655,458" fill="#111111" stroke="#fdba74" stroke-width="1.7"/>')
    label(lines, 705, 454, "accept", "sm")
    label(lines, 705, 471, "gate", "xs")
    node(lines, 795, 420, 130, 76, "Agent B", "new owner", COLORS["rose"], True)
    arrow(lines, "M645 458 H655", "orange"); arrow(lines, "M755 458 H795", "rose")
    lines.append('<rect x="485" y="520" width="440" height="74" rx="7" fill="#1a1a1a" stroke="#f87171"/>')
    label(lines, 705, 547, "conversation_owner: A → B", "mono")
    label(lines, 705, 571, "pending commitments · scope · deadline must transfer", "xs")

    panel(lines, 985, 110, 380, 570, "ABLATION EVIDENCE", COLORS["rose"])
    evidence = [
        (1020, 160, "Benefit", "Δ success · recovery · quality", COLORS["mint"]),
        (1020, 280, "Cost", "Δ token · latency · coordination", COLORS["orange"]),
        (1020, 400, "Variation", "SD · p95 · paired 95% CI", COLORS["violet"]),
        (1020, 520, "Decision", "CI clears effect + cost SLO?", COLORS["gold"]),
    ]
    for x, y, title, detail, color in evidence:
        node(lines, x, y, 310, 76, title, detail, color)
    arrow(lines, "M1175 236 V280", "orange")
    arrow(lines, "M1175 356 V400", "violet")
    arrow(lines, "M1175 476 V520", "gold")

    lines.append('<rect x="65" y="715" width="1270" height="66" rx="9" fill="#1a1a1a" stroke="#d4a574"/>')
    label(lines, 700, 741, "Checkpoint → classify(error) → retry | replan | wait-human | stop", "mono")
    label(lines, 700, 765, "统计单位是 task；三次重复衡量模型波动，不应伪装成 60 个独立样本", "sm")
    footer(lines, [("blue", "控制/基线"), ("mint", "恢复/收益"), ("orange", "代价"), ("rose", "所有权/风险")])
    return lines


DIAGRAMS = {
    "task-decomposition-dependency-replanning": task_graph,
    "reflection-external-deterministic-verification": verifier_layers,
    "orchestration-control-boundaries": control_boundaries,
    "serial-parallel-multi-agent-cost": orchestration_cost,
    "versioned-idempotent-recoverable-execution": recoverable_execution,
    "planning-recovery-orchestration-interview": planning_recovery_interview,
}


def main() -> None:
    for stem, builder in DIAGRAMS.items():
        svg = OUT / f"{stem}.svg"
        png = OUT / f"{stem}.png"
        svg.write_text("\n".join(builder()), encoding="utf-8")
        cairosvg.svg2png(url=str(svg), write_to=str(png), output_width=2100)
        print(svg)


if __name__ == "__main__":
    main()
