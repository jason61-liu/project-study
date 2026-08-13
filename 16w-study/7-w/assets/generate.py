"""生成第七周五张 Dark Luxury 技术图并导出 PNG。"""

from pathlib import Path
from xml.sax.saxutils import escape
import cairosvg

OUT = Path(__file__).parent
W, H = 1400, 860
C = {"gold":"#d4a574","mint":"#6ee7b7","orange":"#fdba74","violet":"#a78bfa","blue":"#38bdf8","rose":"#f87171","gray":"#94a3b8","amber":"#fbbf24"}

def base(title, subtitle):
    x=[]
    x.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">')
    x.append('<style>text{font-family:"Arial Unicode MS","Hiragino Sans GB","Heiti SC","Helvetica Neue",Arial,sans-serif}.ttl{font-family:"Songti SC","Arial Unicode MS",Georgia,serif;font-size:30px;font-weight:700;fill:#f5f0eb}.sub{font-size:13px;fill:#a39787}.sec{font-family:"Songti SC","Arial Unicode MS",Georgia,serif;font-size:15px;font-weight:700;fill:#c9a96e}.nm{font-size:15px;font-weight:650;fill:#f5f0eb}.sm{font-size:11px;fill:#a39787}.xs{font-size:10px;fill:#76695d}.mono{font-family:"Arial Unicode MS","SF Mono","Courier New",monospace;font-size:11px;fill:#d7c6b5}.al{font-size:11px;fill:#c9b8a0}</style>')
    x.append('<defs>')
    for n,c in C.items(): x.append(f'<marker id="arr-{n}" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0,10 3.5,0 7" fill="{c}"/></marker>')
    x.append('<radialGradient id="glow"><stop offset="0" stop-color="#d4a574" stop-opacity=".07"/><stop offset="1" stop-color="#d4a574" stop-opacity="0"/></radialGradient></defs>')
    x.append(f'<rect width="{W}" height="{H}" fill="#0a0a0a"/><ellipse cx="700" cy="400" rx="620" ry="340" fill="url(#glow)" data-graph-role="decoration" data-owner="canvas"/>')
    x.append(f'<text x="700" y="48" text-anchor="middle" class="ttl">{escape(title)}</text><text x="700" y="74" text-anchor="middle" class="sub">{escape(subtitle)}</text>')
    return x

def panel(x,a,b,w,h,title,color):
    x.append(f'<rect x="{a}" y="{b}" width="{w}" height="{h}" rx="10" fill="#0e0e0e" stroke="{color}" stroke-dasharray="7 5"/><text x="{a+18}" y="{b+28}" class="sec">{escape(title)}</text>')
def node(x,a,b,w,h,title,detail,color,double=False):
    x.append(f'<rect x="{a}" y="{b}" width="{w}" height="{h}" rx="8" fill="#111111" stroke="{color}" stroke-width="1.7"/>')
    if double:x.append(f'<rect x="{a+4}" y="{b+4}" width="{w-8}" height="{h-8}" rx="6" fill="none" stroke="{color}" opacity=".55"/>')
    x.append(f'<text x="{a+w/2}" y="{b+29}" text-anchor="middle" class="nm">{escape(title)}</text><text x="{a+w/2}" y="{b+52}" text-anchor="middle" class="sm">{escape(detail)}</text>')
def arrow(x,d,k="gold",dash=False):x.append(f'<path d="{d}" fill="none" stroke="{C[k]}" stroke-width="2" {"stroke-dasharray=\"6 4\"" if dash else ""} marker-end="url(#arr-{k})"/>')
def label(x,a,b,s,cls="al",anchor="middle"):x.append(f'<text x="{a}" y="{b}" text-anchor="{anchor}" class="{cls}">{escape(s)}</text>')
def finish(x,items):
    a=65
    for k,s in items:
        x.append(f'<line x1="{a}" y1="824" x2="{a+34}" y2="824" stroke="{C[k]}" stroke-width="2" marker-end="url(#arr-{k})"/>');label(x,a+46,828,s,"sm","start");a+=245
    x.append('</svg>');return x

def graph_model():
    x=base("一次 LangGraph 研究任务怎样运行","Node 做一步工作；State 留下结果；Edge 选下一步；Reducer 合并两个并行搜索结果")
    panel(x,35,105,940,610,"运行路径：研究工具授权问题",C["blue"]);panel(x,1025,105,340,610,"每一步对 State 做了什么",C["violet"])
    node(x,75,165,200,76,"START","question + attempt=0",C["gold"])
    node(x,355,145,220,76,"准备查询 Node","返回两个 query + attempt=1",C["blue"],True)
    node(x,355,315,220,76,"固定 Edge","同时启动两个搜索",C["orange"])
    node(x,660,145,250,76,"规范搜索 Node","返回 evidence=[P1]",C["mint"]);node(x,660,315,250,76,"案例搜索 Node","返回 evidence=[C1]",C["mint"])
    node(x,500,505,260,84,"Reducer 合并","[] + [P1] + [C1]",C["violet"],True)
    arrow(x,"M275 203 H355","gold");arrow(x,"M465 221 V315","orange");arrow(x,"M575 353 H620 V183 H660","blue");arrow(x,"M575 353 H660","blue")
    arrow(x,"M785 221 H940 V470 H650 V505","mint");arrow(x,"M785 391 V450 H620 V505","mint")
    label(x,630,448,"同一轮结束后统一合并","mono")
    rules=[("State 输入","question, attempt=0",C["gray"]),("Node 更新","只返回改动字段",C["mint"]),("Edge 调度","决定下一批节点",C["blue"]),("Reducer 结果","evidence=[P1,C1]",C["violet"]),("下一步","evaluate 读取合并结果",C["rose"])]
    for i,(a,b,c) in enumerate(rules):node(x,1060,150+i*105,270,68,a,b,c)
    x.append('<rect x="75" y="625" width="835" height="58" rx="8" fill="#1a1a1a" stroke="#d4a574"/>');label(x,492,650,"new_evidence = merge(old_evidence, node_update)","mono");label(x,492,672,"Conditional Edge 会在 evaluate 后读取 evidence_ok，选择重试或回答","xs")
    return finish(x,[("gold","初始输入"),("blue","调度下一步"),("mint","节点更新"),("violet","字段合并")])

def persistence():
    x=base("报告审批怎样暂停一天后继续","Thread 关联两次 HTTP 请求；Checkpoint 保存位置；恢复可能重跑节点，发布必须幂等")
    panel(x,35,105,860,610,"thread_id = report-42 的状态时间线",C["blue"]);panel(x,945,105,420,610,"请求、存储与安全责任",C["rose"])
    pts=[(90,"C0","收到问题"),(270,"C1","草稿完成"),(450,"C2","等待审批"),(630,"C3","批准通过"),(810,"C4","发布完成")]
    for a,t,d in pts:node(x,a,190,135,72,t,d,C["rose"] if t=="C2" else C["blue"])
    for a in [225,405,585,765]:arrow(x,f"M{a} 226 H{a+45}","blue")
    node(x,350,350,250,84,"请求 1：interrupt","保存草稿 v3 后 API 返回",C["rose"],True);arrow(x,"M517 262 V350","rose")
    node(x,650,350,205,84,"请求 2：resume","同一 thread_id + approve",C["mint"]);arrow(x,"M600 392 H650","mint")
    node(x,120,515,250,84,"从 C1 创建分支","修改语气，不改原时间线",C["violet"]);arrow(x,"M337 262 V480 H245 V515","violet",True)
    node(x,535,515,300,84,"发布 Node 可能重跑","用 publish:draft-7:v3 幂等",C["orange"]);arrow(x,"M717 434 V515","orange")
    rules=[("Thread ID","关联两次请求，不是凭证",C["blue"]),("Checkpoint","State + 下一节点 + 元数据",C["violet"]),("Interrupt","持久暂停，进程可退出",C["rose"]),("Resume","验证用户、Scope、草稿版本",C["mint"]),("外部副作用","幂等键或补偿",C["orange"])]
    for i,(a,b,c) in enumerate(rules):node(x,985,150+i*104,340,70,a,b,c)
    return finish(x,[("blue","时间线"),("rose","暂停"),("mint","恢复"),("violet","分叉/Time Travel")])

def openai_sdk():
    x=base("模型提出 Tool Call 后，究竟是谁执行","模型只返回 name + arguments + call_id；Runner 调用宿主 Tool，并把 observation 回传模型")
    panel(x,35,105,800,610,"一次 research report 的 Runner 循环",C["blue"]);panel(x,885,105,480,610,"同一次运行的 Trace 证据",C["violet"])
    node(x,75,170,180,76,"宿主调用 Runner","max_turns=8",C["gold"],True);node(x,335,170,190,76,"模型 API","建议下一步行动",C["blue"])
    node(x,620,140,170,70,"最终报告","Schema 校验后完成",C["mint"]);node(x,620,270,170,70,"宿主执行 Tool","search(call_7F3)",C["orange"]);node(x,620,400,170,70,"Handoff","合规 Agent 接管",C["rose"])
    arrow(x,"M255 208 H335","gold");arrow(x,"M525 208 H575 V175 H620","mint");arrow(x,"M525 208 H575 V305 H620","orange");arrow(x,"M525 208 H575 V435 H620","rose")
    arrow(x,"M705 340 V370 H430 V246","orange");arrow(x,"M705 470 V520 H430 V246","rose");label(x,480,360,"next turn","mono")
    node(x,75,535,330,78,"确定性边界","Schema · Scope · 幂等 · Guardrail",C["violet"]);node(x,475,535,315,78,"本地 Context","Token/tenant/client 不给模型",C["gray"])
    node(x,925,155,400,70,"Trace: research-report-42","串起模型、工具与接管",C["gold"],True)
    spans=[("Agent Span","当前负责人 researcher",C["blue"]),("Generation Span","模型提出 search",C["violet"]),("Function Span","call_7F3 + result",C["orange"]),("Handoff Span","researcher → compliance",C["rose"]),("Guardrail Span","最终引用检查",C["mint"])]
    for i,(a,b,c) in enumerate(spans):node(x,965,260+i*82,320,58,a,b,c)
    arrow(x,"M1125 225 V250","gold")
    return finish(x,[("blue","模型循环"),("orange","Tool"),("rose","Handoff"),("violet","Trace/Guardrail")])

def harness_compare():
    x=base("同一份研究任务，三种委派方式发生了什么","比较的不是品牌功能数量，而是任务输入、子上下文、返回方式、文件权限与协调成本")
    headers=[("Deep Agents",35,C["mint"]),("Claude Subagent",490,C["blue"]),("Agent Team",945,C["violet"])]
    for t,a,c in headers:panel(x,a,105,420,610,t,c)
    rows1=[("父 Agent","task(规范研究契约)"),("子上下文","新上下文，独立检索"),("返回","一份结构化结果"),("文件","虚拟 Backend + 权限"),("适合","有界长任务与材料隔离")]
    rows2=[("父 Agent","按 description 委派"),("子上下文","独立窗口 + 任务 Prompt"),("返回","报告给父 Agent"),("文件","通常共享工作区"),("适合","编码项目内专项调查")]
    rows3=[("Lead","共享任务表分配工作"),("成员","独立 Session + mailbox"),("通信","成员可横向发送消息"),("文件","必须划分 ownership"),("代价","多上下文 + 协调故障")]
    for col,rows,color in [(35,rows1,C["mint"]),(490,rows2,C["blue"]),(945,rows3,C["violet"])]:
        for i,(a,b) in enumerate(rows):node(x,col+40,155+i*103,340,68,a,b,color)
    x.append('<rect x="75" y="700" width="1250" height="64" rx="9" fill="#1a1a1a" stroke="#d4a574"/>');label(x,700,727,"子 Agent 的核心是隔离上下文并返回结果；隔离上下文不等于隔离文件和权限","sm");label(x,700,750,"若两个固定搜索函数可直接并发，就不必引入 Agent；父 Agent始终负责验收返回证据","xs")
    return finish(x,[("mint","Deep Agents"),("blue","Claude Subagent"),("violet","Agent Team"),("rose","协调风险")])

def mapping():
    x=base("一条报告任务从循环到产品，需要增加哪些责任","上层包含下层：循环决定行动，Runtime 保持执行，SDK 暴露接口，Harness 提供长期工作环境")
    layers=[(90,"L0 · 原生循环","模型 → Tool → Observation","让一次任务向前走",C["orange"]),(350,"L1 · 持久 Runtime","State + Checkpoint + Interrupt","崩溃或审批后继续",C["blue"]),(610,"L2 · Agent SDK","Agent + Runner + Event","让应用可编程接入",C["violet"]),(870,"L3 · 产品 Harness","Workspace + Session + 权限","让用户长期安全使用",C["gold"])]
    for a,t,d,e,c in layers:
        node(x,a,185,220,100,t,d,c,True);label(x,a+110,270,e,"xs")
        if a<870:arrow(x,f"M{a+220} 235 H{a+260}","gold")
    panel(x,60,350,1280,320,"沿调用链定位 Hermes 与 Kimi 的执行入口",C["rose"])
    entries=[("Hermes CLI","hermes_cli.main:main","参数 / 配置 / Session"),("Hermes 装配","run_agent:main","创建 Agent 与 Runtime"),("Hermes 循环","agent/conversation_loop.py","模型 / Tool / 重试"),("Kimi SDK","Session.create / resume","应用接入面"),("Kimi CLI Runtime","KimiCLI + soul/agent","工具 / Skills / 审批")]
    for i,(a,b,c) in enumerate(entries):
        y=400+i*48;x.append(f'<rect x="100" y="{y}" width="1200" height="38" rx="6" fill="#151515" stroke="#3a342f"/>');label(x,130,y+24,a,"sm","start");label(x,420,y+24,b,"mono","start");label(x,1130,y+24,c,"xs")
    x.append('<rect x="90" y="705" width="1220" height="64" rx="8" fill="#1a1a1a" stroke="#d4a574"/>');label(x,700,732,"故障定位：Tool 错看循环；恢复错看 Runtime；接入错看 SDK；工作区/权限错看 Harness","sm");label(x,700,754,"Hermes 展示循环怎样产品化；Kimi Agent SDK 通过 Session 复用 Kimi CLI Runtime","xs")
    return finish(x,[("orange","原生循环"),("blue","持久编排"),("violet","SDK"),("gold","产品 Harness")])

DIAGRAMS={"langgraph-state-node-edge-reducer":graph_model,"langgraph-checkpoint-interrupt-time-travel":persistence,"openai-agents-runner-tracing":openai_sdk,"deepagents-claude-subagents-team":harness_compare,"agent-framework-entrypoint-mapping":mapping}
def main():
    for stem,fn in DIAGRAMS.items():
        svg=OUT/f"{stem}.svg";png=OUT/f"{stem}.png";svg.write_text("\n".join(fn()),encoding="utf-8");cairosvg.svg2png(url=str(svg),write_to=str(png),output_width=2100);print(svg)
if __name__=="__main__":main()
