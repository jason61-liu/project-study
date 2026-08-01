from pathlib import Path


OUTPUT = Path(__file__).with_name("minimal-agent-runtime.svg")
lines = []
lines.append('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 920" width="1200" height="920">')
lines.append('  <style>')
lines.append("    text { font-family: 'Helvetica Neue', Helvetica, Arial, 'PingFang SC', 'Microsoft YaHei', sans-serif; }")
lines.append('    .title { font-size: 26px; font-weight: 700; fill: #111827; }')
lines.append('    .subtitle { font-size: 14px; fill: #6b7280; }')
lines.append('    .section { font-size: 17px; font-weight: 700; fill: #111827; }')
lines.append('    .node { font-size: 14px; font-weight: 700; fill: #111827; }')
lines.append('    .small { font-size: 12px; fill: #6b7280; }')
lines.append('    .label { font-size: 12px; font-weight: 600; fill: #374151; }')
lines.append('  </style>')
lines.append('  <defs>')
for name, color in [("blue", "#2563eb"), ("green", "#16a34a"), ("purple", "#9333ea"), ("red", "#dc2626"), ("gray", "#6b7280")]:
    lines.append(f'    <marker id="arrow-{name}" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">')
    lines.append(f'      <polygon points="0 0, 10 3.5, 0 7" fill="{color}"/>')
    lines.append('    </marker>')
lines.append('  </defs>')
lines.append('  <rect width="1200" height="920" fill="#ffffff"/>')
lines.append('  <text x="600" y="44" text-anchor="middle" class="title">最小 Agent：协议、运行循环与有界执行</text>')
lines.append('  <text x="600" y="70" text-anchor="middle" class="subtitle">Schema 保证形状，Call ID 保证因果关联，ReAct 负责推进，预算与终止条件负责收敛</text>')

containers = [
    (40, 105, 540, 335, "protocol", "A. Schema 与工具结果回传", "#eff6ff", "#93c5fd"),
    (620, 105, 540, 335, "stream", "B. 流式生命周期与韧性", "#f0fdfa", "#5eead4"),
    (40, 475, 540, 335, "react", "C. ReAct：推理—行动—观察", "#faf5ff", "#d8b4fe"),
    (620, 475, 540, 335, "budget", "D. 多维预算与显式终止", "#fff7ed", "#fdba74"),
]
for x, y, w, h, ident, title, fill, stroke in containers:
    lines.append(f'  <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" stroke="{stroke}" stroke-width="1.5" data-graph-role="container" data-container-id="{ident}" data-semantic-role="boundary" data-graph-bounds="{x},{y},{x+w},{y+h}"/>')
    lines.append(f'  <text x="{x+24}" y="{y+32}" class="section">{title}</text>')

def node(x, y, w, h, title, detail, fill="#ffffff", stroke="#d1d5db"):
    lines.append(f'  <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
    lines.append(f'  <text x="{x+w/2}" y="{y+29}" text-anchor="middle" class="node">{title}</text>')
    lines.append(f'  <text x="{x+w/2}" y="{y+50}" text-anchor="middle" class="small">{detail}</text>')

def arrow(path, color, marker, dash=None):
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ''
    lines.append(f'  <path d="{path}" fill="none" stroke="{color}" stroke-width="2"{dash_attr} marker-end="url(#arrow-{marker})"/>')

node(68, 170, 130, 66, "JSON Schema", "类型 / required / enum", "#ffffff", "#93c5fd")
node(245, 170, 130, 66, "模型输出", "strict 约束形状", "#ffffff", "#93c5fd")
node(422, 170, 130, 66, "function_call", "name + call_id + args", "#ffffff", "#93c5fd")
node(422, 300, 130, 66, "工具执行器", "鉴权 + 业务校验", "#ffffff", "#86efac")
node(245, 300, 130, 66, "call_output", "相同 call_id", "#ffffff", "#86efac")
arrow("M 198 203 L 241 203", "#2563eb", "blue")
arrow("M 375 203 L 418 203", "#2563eb", "blue")
arrow("M 487 236 L 487 296", "#2563eb", "blue")
arrow("M 422 333 L 379 333", "#16a34a", "green")
arrow("M 310 300 L 310 240", "#16a34a", "green")
lines.append('  <text x="400" y="322" text-anchor="middle" class="small">执行结果</text>')
lines.append('  <text x="324" y="275" class="small">回传下一轮</text>')
lines.append('  <rect x="68" y="275" width="142" height="91" rx="8" fill="#fef2f2" stroke="#fca5a5" stroke-width="1.2"/>')
lines.append('  <text x="139" y="301" text-anchor="middle" class="label">三层校验</text>')
lines.append('  <text x="84" y="324" class="small">1. JSON 可解析</text>')
lines.append('  <text x="84" y="344" class="small">2. Schema 合法</text>')
lines.append('  <text x="84" y="364" class="small">3. 业务语义安全</text>')

stream_nodes = [(650, "created"), (774, "delta × N"), (898, "item.done"), (1022, "completed")]
for x, title in stream_nodes:
    node(x, 175, 105, 60, title, "有序事件", "#ffffff", "#5eead4")
for x1, x2 in [(755, 774), (879, 898), (1003, 1022)]:
    arrow(f"M {x1} 205 L {x2-4} 205", "#2563eb", "blue")
node(664, 305, 120, 60, "Timeout", "截止时间耗尽", "#ffffff", "#fca5a5")
node(830, 305, 120, 60, "Cancel", "客户端停止消费", "#ffffff", "#fca5a5")
node(996, 305, 120, 60, "429", "RPM / TPM 耗尽", "#ffffff", "#fca5a5")
arrow("M 826 235 L 826 267 L 724 267 L 724 301", "#dc2626", "red", "5,4")
arrow("M 950 235 L 950 267 L 890 267 L 890 301", "#dc2626", "red", "5,4")
arrow("M 1074 235 L 1074 301", "#dc2626", "red", "5,4")
lines.append('  <text x="890" y="395" text-anchor="middle" class="small">只重试可恢复错误；遵守 Retry-After，并限制次数与总耗时</text>')

node(205, 530, 150, 66, "Thought", "分解目标 / 更新计划", "#ffffff", "#d8b4fe")
node(395, 530, 150, 66, "Action", "选择工具与参数", "#ffffff", "#d8b4fe")
node(395, 675, 150, 66, "Environment", "工具或外部环境", "#ffffff", "#86efac")
node(205, 675, 150, 66, "Observation", "事实、错误与状态变化", "#ffffff", "#86efac")
arrow("M 355 563 L 391 563", "#9333ea", "purple")
arrow("M 470 596 L 470 671", "#9333ea", "purple")
arrow("M 395 708 L 359 708", "#16a34a", "green")
arrow("M 280 675 L 280 600", "#16a34a", "green")
lines.append('  <polygon points="100,635 145,595 190,635 145,675" fill="#ffffff" stroke="#dc2626" stroke-width="1.5"/>')
lines.append('  <text x="145" y="630" text-anchor="middle" class="label">完成？</text>')
lines.append('  <text x="145" y="647" text-anchor="middle" class="small">证据充分</text>')
arrow("M 205 563 L 145 563 L 145 591", "#dc2626", "red", "5,4")
arrow("M 145 675 L 145 752 L 280 752 L 280 745", "#9333ea", "purple")
lines.append('  <text x="170" y="769" class="small">否：带观察进入下一步</text>')
lines.append('  <text x="126" y="582" text-anchor="end" class="small">是 → Final</text>')

budget_cards = [(648, "Step", "n ≤ N"), (772, "Token", "Σtokens ≤ T"), (896, "Cost", "Σcost ≤ C"), (1020, "Time", "now ≤ deadline")]
for x, title, detail in budget_cards:
    node(x, 530, 105, 58, title, detail, "#ffffff", "#fdba74")
lines.append('  <polygon points="845,665 890,625 935,665 890,705" fill="#ffffff" stroke="#ea580c" stroke-width="1.8"/>')
lines.append('  <text x="890" y="661" text-anchor="middle" class="label">继续？</text>')
lines.append('  <text x="890" y="679" text-anchor="middle" class="small">先检查完成</text>')
for x in [700, 824, 948, 1072]:
    arrow(f"M {x} 588 L {x} 610 L 890 610 L 890 621", "#ea580c", "gray", "4,3")
node(650, 728, 170, 58, "CONTINUE", "仍有预算且可推进", "#ffffff", "#86efac")
node(960, 728, 170, 58, "STOP", "完成或预算耗尽", "#ffffff", "#fca5a5")
arrow("M 845 665 L 735 665 L 735 724", "#16a34a", "green")
arrow("M 935 665 L 1045 665 L 1045 724", "#dc2626", "red")
lines.append('  <text x="760" y="650" class="small">有剩余</text>')
lines.append('  <text x="982" y="650" class="small">完成 / 耗尽</text>')

lines.append('  <line x1="54" y1="858" x2="86" y2="858" stroke="#2563eb" stroke-width="2" marker-end="url(#arrow-blue)"/>')
lines.append('  <text x="96" y="863" class="small">协议与事件流</text>')
lines.append('  <line x1="238" y1="858" x2="270" y2="858" stroke="#16a34a" stroke-width="2" marker-end="url(#arrow-green)"/>')
lines.append('  <text x="280" y="863" class="small">结果 / 继续</text>')
lines.append('  <line x1="416" y1="858" x2="448" y2="858" stroke="#9333ea" stroke-width="2" marker-end="url(#arrow-purple)"/>')
lines.append('  <text x="458" y="863" class="small">Agent 循环</text>')
lines.append('  <line x1="590" y1="858" x2="622" y2="858" stroke="#dc2626" stroke-width="2" stroke-dasharray="5,4" marker-end="url(#arrow-red)"/>')
lines.append('  <text x="632" y="863" class="small">失败 / 终止</text>')
lines.append('  <text x="1144" y="863" text-anchor="end" class="small">正确性来自约束与验证；可靠性来自有界状态机，而不是更长的 Prompt</text>')
lines.append('</svg>')

OUTPUT.write_text("\n".join(lines), encoding="utf-8")
print(f"SVG generated: {OUTPUT}")
