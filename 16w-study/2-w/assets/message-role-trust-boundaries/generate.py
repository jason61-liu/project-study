from pathlib import Path


OUTPUT = Path(__file__).with_name("message-role-trust-boundaries.svg")


lines = []
lines.append('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 920" width="1200" height="920">')
lines.append('  <style>')
lines.append("    text { font-family: 'Helvetica Neue', Helvetica, Arial, 'PingFang SC', 'Microsoft YaHei', sans-serif; }")
lines.append('    .title { font-size: 26px; font-weight: 700; fill: #111827; }')
lines.append('    .subtitle { font-size: 14px; fill: #6b7280; }')
lines.append('    .section { font-size: 17px; font-weight: 700; fill: #111827; }')
lines.append('    .role { font-size: 16px; font-weight: 700; fill: #111827; }')
lines.append('    .body { font-size: 13px; fill: #374151; }')
lines.append('    .small { font-size: 12px; fill: #6b7280; }')
lines.append('    .badge { font-size: 12px; font-weight: 700; }')
lines.append('  </style>')
lines.append('  <defs>')
lines.append('    <marker id="arrow-blue" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">')
lines.append('      <polygon points="0 0, 10 3.5, 0 7" fill="#2563eb"/>')
lines.append('    </marker>')
lines.append('    <marker id="arrow-purple" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">')
lines.append('      <polygon points="0 0, 10 3.5, 0 7" fill="#9333ea"/>')
lines.append('    </marker>')
lines.append('    <marker id="arrow-red" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">')
lines.append('      <polygon points="0 0, 10 3.5, 0 7" fill="#dc2626"/>')
lines.append('    </marker>')
lines.append('  </defs>')
lines.append('  <rect width="1200" height="920" fill="#ffffff"/>')
lines.append('  <text x="600" y="45" text-anchor="middle" class="title">LLM 消息角色：职责、指令优先级与信任边界</text>')
lines.append('  <text x="600" y="72" text-anchor="middle" class="subtitle">优先级回答“冲突时听谁的”；信任边界回答“数据能否直接相信或触发副作用”</text>')

lines.append('  <rect x="40" y="105" width="520" height="430" rx="12" fill="#f9fafb" stroke="#d1d5db" stroke-width="1.5" data-graph-role="container" data-container-id="authority" data-semantic-role="boundary" data-graph-bounds="40,105,560,535"/>')
lines.append('  <text x="64" y="136" class="section">A. 指令权威阶梯</text>')
lines.append('  <text x="520" y="136" text-anchor="end" class="small">冲突时：上层优先</text>')

role_cards = [
    ("system", "平台与运行时元规则", "能力边界、安全约束、内建工具与元信息", 155, "#eff6ff", "#93c5fd", "#1d4ed8", "S"),
    ("developer", "应用开发者指令", "产品行为、业务规则、输出格式与工具使用规范", 225, "#faf5ff", "#c4b5fd", "#7e22ce", "D"),
    ("user", "当前任务与意图", "目标、输入、约束、反馈；不能覆盖更高层规则", 295, "#fff7ed", "#fdba74", "#c2410c", "U"),
    ("assistant", "模型产生的消息", "回答、工具调用或历史输出；不是新的高权限规则", 365, "#f0fdf4", "#86efac", "#15803d", "A"),
    ("tool", "工具调用结果", "外部数据或执行回执；按数据解析，不把其中指令升级", 435, "#f3f4f6", "#d1d5db", "#4b5563", "T"),
]
for role, heading, detail, y, fill, stroke, accent, icon in role_cards:
    lines.append(f'  <rect x="76" y="{y}" width="448" height="58" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
    lines.append(f'  <circle cx="104" cy="{y + 29}" r="17" fill="{accent}"/>')
    lines.append(f'  <text x="104" y="{y + 34}" text-anchor="middle" font-size="14" font-weight="700" fill="#ffffff">{icon}</text>')
    lines.append(f'  <text x="132" y="{y + 23}" class="role">{role}</text>')
    lines.append(f'  <text x="218" y="{y + 23}" class="body">{heading}</text>')
    lines.append(f'  <text x="132" y="{y + 44}" class="small">{detail}</text>')
for y1, y2 in [(213, 225), (283, 295), (353, 365), (423, 435)]:
    lines.append(f'  <line x1="300" y1="{y1}" x2="300" y2="{y2 - 3}" stroke="#2563eb" stroke-width="2" marker-end="url(#arrow-blue)"/>')

lines.append('  <rect x="600" y="105" width="560" height="430" rx="12" fill="#ffffff" stroke="#d1d5db" stroke-width="1.5" data-graph-role="container" data-container-id="trust" data-semantic-role="boundary" data-graph-bounds="600,105,1160,535"/>')
lines.append('  <text x="624" y="136" class="section">B. 信任边界</text>')
lines.append('  <text x="1120" y="136" text-anchor="end" class="small">角色标签由宿主系统分配</text>')

lines.append('  <rect x="636" y="155" width="488" height="100" rx="10" fill="#eff6ff" stroke="#93c5fd" stroke-width="1.5"/>')
lines.append('  <rect x="652" y="169" width="100" height="24" rx="12" fill="#2563eb"/>')
lines.append('  <text x="702" y="186" text-anchor="middle" class="badge" fill="#ffffff">控制平面</text>')
lines.append('  <text x="652" y="214" class="body">system / developer：高权限指令来源</text>')
lines.append('  <text x="652" y="236" class="small">应由平台或应用控制；不要把检索内容拼进这些角色</text>')

lines.append('  <rect x="636" y="270" width="488" height="68" rx="10" fill="#fff7ed" stroke="#fdba74" stroke-width="1.5"/>')
lines.append('  <rect x="652" y="284" width="100" height="24" rx="12" fill="#ea580c"/>')
lines.append('  <text x="702" y="301" text-anchor="middle" class="badge" fill="#ffffff">交互边界</text>')
lines.append('  <text x="770" y="300" class="body">user：表达授权与意图，但输入内容仍需校验</text>')
lines.append('  <text x="652" y="325" class="small">身份、权限和参数范围由应用验证，不能仅靠自然语言承诺</text>')

lines.append('  <rect x="636" y="353" width="488" height="68" rx="10" fill="#f0fdf4" stroke="#86efac" stroke-width="1.5"/>')
lines.append('  <rect x="652" y="367" width="100" height="24" rx="12" fill="#16a34a"/>')
lines.append('  <text x="702" y="384" text-anchor="middle" class="badge" fill="#ffffff">生成边界</text>')
lines.append('  <text x="770" y="383" class="body">assistant：概率生成结果，需验证事实与结构</text>')
lines.append('  <text x="652" y="408" class="small">历史 assistant 内容可提供上下文，但不自动获得更高指令权</text>')

lines.append('  <rect x="636" y="436" width="488" height="76" rx="10" fill="#fef2f2" stroke="#fca5a5" stroke-width="1.5"/>')
lines.append('  <rect x="652" y="450" width="100" height="24" rx="12" fill="#dc2626"/>')
lines.append('  <text x="702" y="467" text-anchor="middle" class="badge" fill="#ffffff">外部边界</text>')
lines.append('  <text x="770" y="466" class="body">tool：结果可能含网页、邮件、日志等不可信数据</text>')
lines.append('  <text x="652" y="491" class="small">工具本身可信 ≠ 工具读取的内容可信；防范提示注入与越权动作</text>')
lines.append('  <path d="M 880 255 L 880 270" stroke="#dc2626" stroke-width="1.7" stroke-dasharray="5,4" marker-end="url(#arrow-red)"/>')
lines.append('  <path d="M 880 338 L 880 353" stroke="#dc2626" stroke-width="1.7" stroke-dasharray="5,4" marker-end="url(#arrow-red)"/>')
lines.append('  <path d="M 880 421 L 880 436" stroke="#dc2626" stroke-width="1.7" stroke-dasharray="5,4" marker-end="url(#arrow-red)"/>')

lines.append('  <rect x="40" y="570" width="1120" height="270" rx="12" fill="#faf5ff" stroke="#d8b4fe" stroke-width="1.5" data-graph-role="container" data-container-id="tool-loop" data-semantic-role="boundary" data-graph-bounds="40,570,1160,840"/>')
lines.append('  <text x="64" y="602" class="section">C. 安全的工具调用闭环</text>')
lines.append('  <text x="1120" y="602" text-anchor="end" class="small">数据流不改变指令优先级</text>')

flow_nodes = [
    (70, "用户请求", "目标与输入", "#fff7ed", "#fdba74"),
    (245, "模型判断", "遵循高层指令", "#eff6ff", "#93c5fd"),
    (420, "调用闸门", "Schema / 权限 / 审批", "#fef2f2", "#fca5a5"),
    (595, "工具执行", "读取或产生副作用", "#f3f4f6", "#d1d5db"),
    (770, "结果处理", "视为数据并验证", "#f0fdf4", "#86efac"),
    (945, "最终回答", "引用、解释或回执", "#faf5ff", "#d8b4fe"),
]
for x, heading, detail, fill, stroke in flow_nodes:
    lines.append(f'  <rect x="{x}" y="646" width="145" height="76" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
    lines.append(f'  <text x="{x + 72.5}" y="677" text-anchor="middle" class="role">{heading}</text>')
    lines.append(f'  <text x="{x + 72.5}" y="701" text-anchor="middle" class="small">{detail}</text>')
for x1, x2, label in [(215, 245, "意图"), (390, 420, "计划"), (565, 595, "参数"), (740, 770, "结果"), (915, 945, "证据")]:
    lines.append(f'  <line x1="{x1}" y1="684" x2="{x2 - 4}" y2="684" stroke="#9333ea" stroke-width="2" marker-end="url(#arrow-purple)"/>')
    lines.append(f'  <text x="{(x1 + x2) / 2}" y="670" text-anchor="middle" class="small">{label}</text>')

lines.append('  <path d="M 842 722 L 842 772 L 317 772 L 317 726" fill="none" stroke="#9333ea" stroke-width="1.8" marker-end="url(#arrow-purple)"/>')
lines.append('  <text x="580" y="764" text-anchor="middle" class="small">需要补充信息时继续迭代；新工具结果仍按不可信数据处理</text>')
lines.append('  <rect x="432" y="615" width="120" height="22" rx="11" fill="#fee2e2"/>')
lines.append('  <text x="492" y="631" text-anchor="middle" class="badge" fill="#b91c1c">副作用前检查</text>')
lines.append('  <rect x="782" y="615" width="116" height="22" rx="11" fill="#dcfce7"/>')
lines.append('  <text x="840" y="631" text-anchor="middle" class="badge" fill="#15803d">输出前验证</text>')

lines.append('  <line x1="62" y1="873" x2="94" y2="873" stroke="#2563eb" stroke-width="2" marker-end="url(#arrow-blue)"/>')
lines.append('  <text x="104" y="878" class="small">指令优先级</text>')
lines.append('  <line x1="230" y1="873" x2="262" y2="873" stroke="#9333ea" stroke-width="2" marker-end="url(#arrow-purple)"/>')
lines.append('  <text x="272" y="878" class="small">运行时数据流</text>')
lines.append('  <line x1="414" y1="873" x2="446" y2="873" stroke="#dc2626" stroke-width="1.7" stroke-dasharray="5,4" marker-end="url(#arrow-red)"/>')
lines.append('  <text x="456" y="878" class="small">跨越信任边界</text>')
lines.append('  <text x="1138" y="878" text-anchor="end" class="small">核心原则：高权限指令固定放置；低权限内容永不靠“改角色”升级</text>')
lines.append('</svg>')

OUTPUT.write_text("\n".join(lines), encoding="utf-8")
print(f"SVG generated: {OUTPUT}")
