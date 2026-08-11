"""生成第 5 周五张 Dark Luxury 技术图，并导出 PNG 供视觉验收。"""

from pathlib import Path
from xml.sax.saxutils import escape

import cairosvg


OUT = Path(__file__).parent
W, H = 1400, 840


def base(title: str, subtitle: str) -> list[str]:
    """返回公共 SVG 头；使用逐行列表，便于检查 XML 与几何。"""
    lines: list[str] = []
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">')
    lines.append('<style>')
    lines.append('text{font-family:"Arial Unicode MS","Hiragino Sans GB","Heiti SC","Helvetica Neue",Arial,sans-serif}.ttl{font-family:"Songti SC","Arial Unicode MS",Georgia,"Times New Roman",serif;font-size:30px;font-weight:700;fill:#f5f0eb}.sub{font-size:13px;fill:#a39787}.sec{font-family:"Songti SC","Arial Unicode MS",Georgia,"Times New Roman",serif;font-size:15px;font-weight:700;fill:#c9a96e}.nm{font-size:15px;font-weight:650;fill:#f5f0eb}.sm{font-size:11px;fill:#a39787}.xs{font-size:10px;fill:#6b5f53}.al{font-size:11px;fill:#c9b8a0}.metric{font-family:"Arial Unicode MS","SF Mono","Courier New",monospace;font-size:11px;fill:#d7c6b5}</style>')
    lines.append('<defs>')
    for name, color in [('gold','#d4a574'),('mint','#6ee7b7'),('orange','#fdba74'),('violet','#a78bfa'),('blue','#38bdf8'),('rose','#f87171'),('gray','#94a3b8')]:
        lines.append(f'<marker id="arr-{name}" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0,10 3.5,0 7" fill="{color}"/></marker>')
    lines.append('<radialGradient id="glow"><stop offset="0" stop-color="#d4a574" stop-opacity=".07"/><stop offset="1" stop-color="#d4a574" stop-opacity="0"/></radialGradient>')
    lines.append('</defs>')
    lines.append(f'<rect width="{W}" height="{H}" fill="#0a0a0a"/>')
    lines.append(f'<ellipse cx="700" cy="390" rx="620" ry="330" fill="url(#glow)" data-graph-role="decoration" data-owner="canvas"/>')
    lines.append(f'<text x="700" y="48" text-anchor="middle" class="ttl">{escape(title)}</text>')
    lines.append(f'<text x="700" y="73" text-anchor="middle" class="sub">{escape(subtitle)}</text>')
    return lines


def panel(lines: list[str], x: int, y: int, w: int, h: int, title: str, accent: str = '#d4a574') -> None:
    lines.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="#0e0e0e" stroke="{accent}" stroke-width="1" stroke-dasharray="7 5" opacity=".9"/>')
    lines.append(f'<text x="{x+18}" y="{y+28}" class="sec" fill="{accent}">{escape(title)}</text>')


def node(lines: list[str], x: int, y: int, w: int, h: int, title: str, detail: str, color: str, *, double: bool = False) -> None:
    lines.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="#111111" stroke="{color}" stroke-width="1.7"/>')
    if double:
        lines.append(f'<rect x="{x+4}" y="{y+4}" width="{w-8}" height="{h-8}" rx="6" fill="none" stroke="{color}" stroke-width=".6" opacity=".55"/>')
    lines.append(f'<text x="{x+w/2}" y="{y+30}" text-anchor="middle" class="nm" fill="{color}">{escape(title)}</text>')
    lines.append(f'<text x="{x+w/2}" y="{y+53}" text-anchor="middle" class="sm">{escape(detail)}</text>')


def label(lines: list[str], x: int, y: int, text: str, cls: str = 'al', anchor: str = 'middle') -> None:
    lines.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" class="{cls}">{escape(text)}</text>')


COLORS = {'gold':'#d4a574','mint':'#6ee7b7','orange':'#fdba74','violet':'#a78bfa','blue':'#38bdf8','rose':'#f87171','gray':'#94a3b8'}


def arrow(lines: list[str], d: str, kind: str = 'gold', *, dashed: bool = False, width: float = 2.0) -> None:
    dash = ' stroke-dasharray="6 4"' if dashed else ''
    lines.append(f'<path d="{d}" fill="none" stroke="{COLORS[kind]}" stroke-width="{width}"{dash} marker-end="url(#arr-{kind})"/>')


def footer(lines: list[str], items: list[tuple[str, str]]) -> None:
    x = 70
    for kind, text in items:
        lines.append(f'<line x1="{x}" y1="808" x2="{x+34}" y2="808" stroke="{COLORS[kind]}" stroke-width="2" marker-end="url(#arr-{kind})"/>')
        label(lines, x+45, 812, text, 'sm', 'start')
        x += 215
    lines.append('</svg>')


def workflow_agent_boundary() -> list[str]:
    lines = base('Workflow 与 Agent：控制权边界', '边界不由是否调用 LLM 决定，而由运行时路径和下一步选择权决定')
    panel(lines, 34, 100, 400, 590, 'DETERMINISTIC CONTROL PLANE', '#38bdf8')
    panel(lines, 500, 100, 400, 590, 'MODEL-DIRECTED CONTROL PLANE', '#d4a574')
    panel(lines, 966, 100, 400, 590, 'RUNTIME SAFETY PLANE', '#f87171')
    node(lines, 84, 160, 300, 82, 'Single LLM Call', '固定输入 → 固定输出契约', '#94a3b8')
    node(lines, 84, 302, 300, 82, 'Workflow', '代码预定义路径、分支与终止', '#38bdf8', double=True)
    node(lines, 84, 444, 300, 82, 'Workflow + LLM Router', '模型分类；代码执行有限分支', '#a78bfa')
    node(lines, 550, 235, 300, 96, 'Agent Loop', '模型选择下一步、工具与停止时机', '#d4a574', double=True)
    node(lines, 550, 430, 300, 96, 'Environment Feedback', 'Observation 校正计划与动作', '#6ee7b7')
    node(lines, 1016, 154, 300, 74, 'Policy & Identity', 'Scope · ACL · approval', '#f87171')
    node(lines, 1016, 270, 300, 74, 'Budget & Stop', 'steps · token · cost · timeout', '#fbbf24')
    node(lines, 1016, 386, 300, 74, 'Schema & Tool Runtime', 'validation · idempotency · sandbox', '#5a9e6f')
    node(lines, 1016, 502, 300, 74, 'Trace & Recovery', 'checkpoint · retry · compensate', '#a78bfa')
    arrow(lines, 'M234 242 V302', 'blue'); label(lines, 250, 278, '复杂度递增', 'al', 'start')
    arrow(lines, 'M234 384 V444', 'violet'); label(lines, 250, 420, '有限动态性', 'al', 'start')
    arrow(lines, 'M384 343 H468 V283 H550', 'gold'); label(lines, 468, 267, '路径不可预定义')
    arrow(lines, 'M700 331 V430', 'mint'); label(lines, 718, 385, 'action')
    arrow(lines, 'M550 478 H505 V283 H550', 'gold'); label(lines, 516, 462, 'observe → replan', 'al', 'start')
    arrow(lines, 'M850 283 H938 V191 H1016', 'rose', dashed=True); label(lines, 934, 176, '不可委托的边界')
    arrow(lines, 'M850 478 H938 V539 H1016', 'violet', dashed=True); label(lines, 932, 560, 'trace / checkpoint')
    lines.append('<rect x="84" y="608" width="766" height="54" rx="8" fill="#1a1a1a" stroke="#d4a574" stroke-width="1"/>')
    label(lines, 467, 632, '选择原则：先单调用；固定路径用 Workflow；只有步骤未知且环境反馈可验证时才升级 Agent', 'sm')
    label(lines, 467, 651, '模型可以拥有策略选择权，但身份、权限、预算与副作用控制必须留在 Runtime', 'xs')
    footer(lines, [('blue','代码路径'),('gold','模型控制'),('mint','环境证据'),('rose','安全约束')])
    return lines


def five_patterns() -> list[str]:
    lines = base('五种可组合的 Agentic Workflow 模式', '区别在依赖拓扑、并发边界、动态分解位置以及质量反馈是否成环')
    rows = [
        (112, 'Prompt Chaining', '#38bdf8', 'A → Gate → B → Gate → C', '可分解、强依赖；准确率换延迟'),
        (232, 'Routing', '#a78bfa', 'Classifier → {Specialist A | B | C}', '类别可分；错路由是上界'),
        (352, 'Parallelization', '#6ee7b7', 'Fan-out(A,B,C) → Aggregate / Vote', '独立子任务或多样性投票'),
        (472, 'Orchestrator–Workers', '#d4a574', 'Orchestrator ⇢ dynamic workers → Synthesis', '子任务事前未知；需动态委派'),
        (592, 'Evaluator–Optimizer', '#f87171', 'Generate → Evaluate → Refine ↺', '评价标准清楚且迭代确有增益'),
    ]
    for y, name, color, topology, fit in rows:
        lines.append(f'<rect x="45" y="{y}" width="1310" height="92" rx="10" fill="#101010" stroke="{color}" stroke-width="1"/>')
        lines.append(f'<rect x="45" y="{y}" width="250" height="92" rx="10" fill="#171717" stroke="none"/>')
        label(lines, 70, y+37, name, 'nm', 'start'); lines[-1] = lines[-1].replace('class="nm"', f'class="nm" fill="{color}"')
        label(lines, 70, y+62, fit, 'xs', 'start')
        label(lines, 330, y+39, topology, 'metric', 'start')
        metric = {'Prompt Chaining':'L≈ΣLi · C≈ΣCi','Routing':'L≈Lr+Lb · P≤Proute','Parallelization':'L≈max(Li)+La','Orchestrator–Workers':'N 动态 · 合并成本显著','Evaluator–Optimizer':'C≈Σ(Cgen+Ceval)'}[name]
        label(lines, 330, y+67, metric, 'sm', 'start')
        risk = {'Prompt Chaining':'上游格式漂移','Routing':'分类错误','Parallelization':'相关失败/合并冲突','Orchestrator–Workers':'任务重叠/遗漏','Evaluator–Optimizer':'评价器偏差/死循环'}[name]
        lines.append(f'<rect x="1040" y="{y+20}" width="270" height="50" rx="7" fill="#1a1a1a" stroke="{color}" stroke-width=".8"/>')
        label(lines, 1175, y+41, 'PRIMARY FAILURE', 'xs')
        label(lines, 1175, y+61, risk, 'sm')
    lines.append('<rect x="45" y="716" width="1310" height="54" rx="8" fill="#1a1a1a" stroke="#d4a574"/>')
    label(lines, 700, 740, '组合不是堆叠：先写依赖 DAG，再决定串行、并行、动态分解和反馈闭环', 'sm')
    label(lines, 700, 758, '每增加一次模型调用，都要证明成功率增益覆盖延迟、成本与新故障面', 'xs')
    footer(lines, [('blue','固定依赖'),('mint','并行执行'),('gold','动态编排'),('rose','质量反馈')])
    return lines


def react_loop() -> list[str]:
    lines = base('ReAct：证据闭环与轨迹风险', 'Thought/State → Action → Observation 的价值来自外部校正，风险来自上下文和错误的累积')
    panel(lines, 35, 105, 850, 575, 'ONLINE DECISION LOOP', '#d4a574')
    node(lines, 90, 215, 210, 90, 'State / Thought', '目标 · 假设 · 下一步', '#d4a574', double=True)
    node(lines, 355, 215, 210, 90, 'Action', 'tool + validated args', '#fdba74')
    node(lines, 620, 215, 210, 90, 'Observation', '外部事实或结构化错误', '#6ee7b7')
    arrow(lines, 'M300 260 H355', 'orange'); label(lines, 328, 248, 'decide')
    arrow(lines, 'M565 260 H620', 'mint'); label(lines, 592, 248, 'execute')
    arrow(lines, 'M725 305 V375 H195 V305', 'gold'); label(lines, 460, 365, 'update belief / replan')
    node(lines, 90, 455, 210, 86, 'Progress State', 'done · pending · blockers', '#38bdf8')
    node(lines, 355, 455, 210, 86, 'Runtime Guard', 'budget · policy · schema', '#f87171')
    node(lines, 620, 455, 210, 86, 'Completion Check', 'explicit evidence predicate', '#a78bfa')
    arrow(lines, 'M195 375 V455', 'blue'); arrow(lines, 'M460 375 V455', 'rose'); arrow(lines, 'M725 375 V455', 'violet')
    panel(lines, 930, 105, 435, 575, 'ERROR COMPOUNDING', '#f87171')
    levels = [(160,'e₁','错误观察'),(255,'e₂','错误状态摘要'),(350,'e₃','错误工具选择'),(445,'e₄','不可逆副作用')]
    risk_label_positions = [(1145, 'start'), (1165, 'start'), (1135, 'end'), (1200, 'end')]
    for i,(y,e,t) in enumerate(levels):
        x=980+i*78
        lines.append(f'<circle cx="{x+55}" cy="{y}" r="{28+i*7}" fill="#1a1111" stroke="#f87171" stroke-width="1.4"/>')
        label(lines,x+55,y+4,e,'nm')
        risk_x, risk_anchor = risk_label_positions[i]
        label(lines,risk_x,y+5,t,'sm',risk_anchor)
    lines.append('<line x1="952" y1="160" x2="952" y2="445" stroke="#f87171" stroke-width="1.5" stroke-dasharray="5 4"/>')
    label(lines, 944, 305, 'propagate', 'al', 'end')
    lines.append('<rect x="975" y="548" width="345" height="90" rx="8" fill="#111111" stroke="#d4a574"/>')
    label(lines, 1148, 575, '控制轨迹，而不是隐藏轨迹', 'nm')
    label(lines, 1148, 598, 'typed state · evidence refs · compaction', 'sm')
    label(lines, 1148, 618, 'max steps · approval · idempotency', 'sm')
    footer(lines, [('orange','Action'),('mint','Observation'),('gold','状态更新'),('rose','错误传播')])
    return lines


def planning_comparison() -> list[str]:
    lines = base('Plan-and-Execute vs 逐步决策', '核心变量是承诺视野：一次规划多远、环境变化后多快重新获得反馈')
    panel(lines, 35, 105, 640, 590, 'PLAN-AND-EXECUTE · LONG HORIZON', '#38bdf8')
    node(lines, 80, 170, 160, 74, 'Global Plan', 'P=[s₁…sₙ]', '#38bdf8')
    for i, t in enumerate(['Execute s₁','Execute s₂','Execute s₃','Verify']):
        x=80+i*140
        node(lines,x,330,115,66,t,'固定计划', '#94a3b8')
        if i<3: arrow(lines,f'M{x+115} 363 H{x+140}','blue',width=1.5)
    arrow(lines,'M160 244 V300 H137 V330','blue'); label(lines,285,287,'一次生成全局依赖与顺序')
    lines.append('<path d="M220 413 H565" stroke="#f87171" stroke-width="2" stroke-dasharray="6 4"/>')
    label(lines,392,438,'计划漂移区：旧假设继续影响后续步骤','sm')
    node(lines, 155, 515, 400, 82, '优势 / 风险', '全局一致、少调用 / 脆弱、重规划昂贵', '#f87171')
    panel(lines, 725, 105, 640, 590, 'STEPWISE · SHORT HORIZON', '#d4a574')
    for i,(x,t,c) in enumerate([(770,'Decide','#d4a574'),(930,'Act','#fdba74'),(1090,'Observe','#6ee7b7')]):
        node(lines,x,240,125,76,t,'one step',c,double=(i==0))
    arrow(lines,'M895 278 H930','orange'); arrow(lines,'M1055 278 H1090','mint')
    arrow(lines,'M1152 316 V380 H832 V316','gold'); label(lines,995,370,'每步用新证据更新')
    node(lines, 805, 470, 450, 82, '优势 / 风险', '适应变化、易恢复 / 局部贪心、调用多', '#a78bfa')
    lines.append('<rect x="255" y="625" width="890" height="100" rx="12" fill="#15120f" stroke="#d4a574" stroke-width="1.5"/>')
    label(lines,700,653,'推荐：Rolling-Horizon Hybrid','nm')
    label(lines,700,678,'先产出目标、约束、里程碑和依赖；只承诺未来 k 步；每个里程碑用证据重规划','sm')
    label(lines,700,701,'不可逆动作前缩短视野并请求确认，可逆探索阶段允许更长批次','xs')
    footer(lines, [('blue','长期计划'),('gold','短期反馈'),('mint','环境事实'),('rose','计划漂移')])
    return lines


def architecture_dimensions() -> list[str]:
    lines = base('Agent 架构选择：六维约束与 Pareto 决策', '不要用“最 Agentic”做目标；先设硬约束，再用可复现实验比较候选方案')
    panel(lines, 35, 105, 350, 580, 'CANDIDATE ARCHITECTURES', '#94a3b8')
    for i,(name,detail,color) in enumerate([
        ('A · Single Call','retrieval + schema','#94a3b8'),('B · Fixed Workflow','chain / route / parallel','#38bdf8'),('C · Hybrid','plan + bounded loop','#d4a574'),('D · Autonomous Agent','dynamic tools + replanning','#f87171')]):
        node(lines,75,155+i*118,270,76,name,detail,color,double=(i==2))
    panel(lines, 435, 105, 520, 580, 'MEASUREMENT CONTROL PANEL', '#d4a574')
    dims=[('成功率','任务级完成 + 质量阈值','↑'),('可控性','越权/偏航/人工干预率','↑'),('延迟','p50 / p95 / deadline miss','↓'),('成本','token + tool + retry / success','↓'),('可观测性','可归因 trace 覆盖率','↑'),('恢复能力','MTTR + resume/compensate 成功率','↑')]
    for i,(n,d,trend) in enumerate(dims):
        y=150+i*82
        lines.append(f'<rect x="475" y="{y}" width="440" height="58" rx="7" fill="#111111" stroke="{["#6ee7b7","#a78bfa","#fdba74","#f87171","#38bdf8","#d4a574"][i]}" stroke-width="1"/>')
        label(lines,495,y+24,n,'nm','start'); label(lines,590,y+24,d,'sm','start'); label(lines,890,y+35,trend,'nm')
    panel(lines, 1005, 105, 360, 580, 'DECISION GATES', '#6ee7b7')
    gates=[('1. HARD SLO','权限、预算、p95、合规'),('2. EVAL SET','正常 + 边界 + 对抗'),('3. PARETO','剔除被全面支配方案'),('4. FAILURE TEST','超时、半成功、恢复'),('5. ADR','证据、权衡、回退条件')]
    for i,(n,d) in enumerate(gates):
        y=155+i*100
        node(lines,1045,y,280,70,n,d,'#6ee7b7' if i<4 else '#d4a574')
        if i<4: arrow(lines,f'M1185 {y+70} V{y+100}','mint',width=1.4)
    arrow(lines,'M385 395 H435','gold'); label(lines,410,382,'eval')
    arrow(lines,'M955 395 H1005','mint'); label(lines,980,382,'gate')
    lines.append('<rect x="270" y="718" width="860" height="52" rx="8" fill="#1a1a1a" stroke="#d4a574"/>')
    label(lines,700,742,'最终选择 = 满足硬约束的 Pareto 前沿方案，而不是加权平均分最高的幻觉精度','sm')
    label(lines,700,760,'指标必须绑定 trace、数据集版本、模型版本与成本价格快照','xs')
    footer(lines, [('gray','候选方案'),('gold','测量'),('mint','约束门禁'),('rose','风险/成本')])
    return lines


DIAGRAMS = {
    'workflow-agent-control-boundary': workflow_agent_boundary,
    'five-agentic-workflow-patterns': five_patterns,
    'react-evidence-loop-and-risk': react_loop,
    'plan-execute-vs-stepwise': planning_comparison,
    'agent-architecture-six-dimensions': architecture_dimensions,
}


def main() -> None:
    for stem, builder in DIAGRAMS.items():
        svg = OUT / f'{stem}.svg'
        png = OUT / f'{stem}.png'
        svg.write_text('\n'.join(builder()), encoding='utf-8')
        cairosvg.svg2png(url=str(svg), write_to=str(png), output_width=2100)
        print(svg)


if __name__ == '__main__':
    main()
