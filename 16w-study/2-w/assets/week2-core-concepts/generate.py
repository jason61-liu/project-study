from html import escape
from pathlib import Path


ROOT = Path(__file__).parent


def start(title: str, subtitle: str, height: int = 680) -> list[str]:
    lines: list[str] = []
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 {height}" width="1200" height="{height}">')
    lines.append('  <style>')
    lines.append("    text { font-family: 'Helvetica Neue', Helvetica, Arial, 'PingFang SC', 'Microsoft YaHei', sans-serif; }")
    lines.append('    .title { font-size: 25px; font-weight: 700; fill: #111827; }')
    lines.append('    .subtitle { font-size: 14px; fill: #6b7280; }')
    lines.append('    .heading { font-size: 16px; font-weight: 700; fill: #111827; }')
    lines.append('    .body { font-size: 13px; fill: #374151; }')
    lines.append('    .small { font-size: 12px; fill: #6b7280; }')
    lines.append('    .badge { font-size: 12px; font-weight: 700; }')
    lines.append('  </style>')
    lines.append('  <defs>')
    for name, color in [('blue', '#2563eb'), ('green', '#16a34a'), ('purple', '#9333ea'), ('red', '#dc2626'), ('gray', '#6b7280')]:
        lines.append(f'    <marker id="arrow-{name}" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">')
        lines.append(f'      <polygon points="0 0, 10 3.5, 0 7" fill="{color}"/>')
        lines.append('    </marker>')
    lines.append('  </defs>')
    lines.append(f'  <rect width="1200" height="{height}" fill="#ffffff"/>')
    lines.append(f'  <text x="600" y="44" text-anchor="middle" class="title">{escape(title)}</text>')
    lines.append(f'  <text x="600" y="70" text-anchor="middle" class="subtitle">{escape(subtitle)}</text>')
    return lines


def container(lines, x, y, w, h, ident, fill='#f9fafb', stroke='#d1d5db'):
    lines.append(f'  <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" stroke="{stroke}" stroke-width="1.5" data-graph-role="container" data-container-id="{ident}" data-semantic-role="boundary" data-graph-bounds="{x},{y},{x+w},{y+h}"/>')


def box(lines, x, y, w, h, title, detail, fill, stroke, ident):
    lines.append(f'  <rect id="{ident}" x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
    lines.append(f'  <text x="{x+w/2}" y="{y+32}" text-anchor="middle" class="heading">{escape(title)}</text>')
    lines.append(f'  <text x="{x+w/2}" y="{y+56}" text-anchor="middle" class="small">{escape(detail)}</text>')


def arrow(lines, d, color='blue', dash=None):
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ''
    colors = {'blue': '#2563eb', 'green': '#16a34a', 'purple': '#9333ea', 'red': '#dc2626', 'gray': '#6b7280'}
    lines.append(f'  <path d="{d}" fill="none" stroke="{colors[color]}" stroke-width="2"{dash_attr} marker-end="url(#arrow-{color})"/>')


def label(lines, x, y, value, color='#6b7280'):
    lines.append(f'  <text x="{x}" y="{y}" text-anchor="middle" class="small" fill="{color}">{escape(value)}</text>')


def legend(lines, y, entries):
    x = 70
    for color, value in entries:
        colors = {'blue': '#2563eb', 'green': '#16a34a', 'purple': '#9333ea', 'red': '#dc2626', 'gray': '#6b7280'}
        lines.append(f'  <line x1="{x}" y1="{y}" x2="{x+32}" y2="{y}" stroke="{colors[color]}" stroke-width="2" marker-end="url(#arrow-{color})"/>')
        lines.append(f'  <text x="{x+44}" y="{y+5}" class="small">{escape(value)}</text>')
        x += 190


def finish(lines, filename):
    lines.append('</svg>')
    path = ROOT / filename
    path.write_text('\n'.join(lines), encoding='utf-8')
    print(f'generated: {path}')


def structured_output_diagram():
    lines = start('结构化输出与工具结果回传链路', 'Schema 约束结构；应用校验语义与权限；call_id 只负责关联一次 Action 和 Observation')
    container(lines, 35, 105, 1130, 475, 'protocol')
    lines.append('  <text x="60" y="138" class="heading">协议主链</text>')
    xs = [45, 205, 365, 525, 685, 845, 1005]
    arrows = [(180, 201), (340, 361), (500, 521), (660, 681), (820, 841), (980, 1001)]
    for x1, x2 in arrows:
        arrow(lines, f'M {x1} 290 L {x2} 290', 'blue')
    arrow(lines, 'M 432 335 L 432 395 L 912 395 L 912 335', 'green', '6,4')
    label(lines, 672, 385, '同一个 call_id 关联请求与结果', '#15803d')
    cards = [
        ('Tool Schema', '限制动作空间', '#eff6ff', '#93c5fd', 'schema'),
        ('Model', '选择工具或回答', '#faf5ff', '#d8b4fe', 'model'),
        ('function_call', 'call_id + arguments', '#fff7ed', '#fdba74', 'call'),
        ('校验闸门', 'Schema / 业务 / 权限', '#fef2f2', '#fca5a5', 'gate'),
        ('Tool', '执行与幂等', '#f3f4f6', '#d1d5db', 'tool'),
        ('call_output', 'call_id + output', '#f0fdf4', '#86efac', 'output'),
        ('Final', '验证后提交', '#faf5ff', '#c4b5fd', 'final'),
    ]
    for x, card in zip(xs, cards):
        title, detail, fill, stroke, ident = card
        box(lines, x, 245, 135, 90, title, detail, fill, stroke, ident)
    checks = [
        (85, '1. JSON 语法', '完整终态后解析'),
        (355, '2. Schema', '类型、必填、枚举'),
        (625, '3. 业务语义', '跨字段与外部状态'),
        (895, '4. 授权副作用', '身份、审批、幂等'),
    ]
    for x, title, detail in checks:
        box(lines, x, 455, 220, 76, title, detail, '#ffffff', '#d1d5db', title)
    legend(lines, 625, [('blue', '协议数据流'), ('green', '关联键')])
    lines.append('  <text x="1135" y="630" text-anchor="end" class="small">结构正确 ≠ 事实正确 ≠ 获得授权</text>')
    finish(lines, 'structured-output-tool-loop.svg')


def streaming_diagram():
    lines = start('流式响应与韧性状态机', '只有 completed 才能正式提交；断流、取消和超时都必须保留明确终态')
    container(lines, 40, 105, 1120, 475, 'stream')
    lines.append('  <text x="65" y="138" class="heading">SSE 生命周期</text>')
    xs = [60, 245, 430, 615, 800, 985]
    for x1, x2 in zip([200, 385, 570, 755, 940], [241, 426, 611, 796, 981]):
        arrow(lines, f'M {x1} 260 L {x2} 260', 'blue')
    cards = [
        ('Request', '整体 Deadline', '#eff6ff', '#93c5fd', 'request'),
        ('created', '记录 response_id', '#eff6ff', '#93c5fd', 'created'),
        ('delta × N', '按 Item 键重组', '#faf5ff', '#d8b4fe', 'delta'),
        ('item.done', '参数才可解析', '#fff7ed', '#fdba74', 'done'),
        ('completed', '取得 usage', '#f0fdf4', '#86efac', 'completed'),
        ('Commit', '正式结果', '#f0fdf4', '#86efac', 'commit'),
    ]
    for x, card in zip(xs, cards):
        box(lines, x, 215, 140, 90, *card)
    arrow(lines, 'M 500 305 L 500 405', 'red')
    arrow(lines, 'M 685 305 L 685 405', 'red')
    arrow(lines, 'M 870 305 L 870 405', 'red')
    box(lines, 420, 405, 160, 82, '断流 / error', '部分输出不提交', '#fef2f2', '#fca5a5', 'error')
    box(lines, 605, 405, 160, 82, '取消竞争', '查询最终状态', '#fef2f2', '#fca5a5', 'cancel')
    box(lines, 790, 405, 160, 82, 'incomplete', '记录停止原因', '#fef2f2', '#fca5a5', 'incomplete')
    lines.append('  <rect x="80" y="510" width="1040" height="45" rx="8" fill="#f3f4f6" stroke="#d1d5db"/>')
    lines.append('  <text x="600" y="538" text-anchor="middle" class="body">Connect → 首字节 → Idle → Overall Deadline → Tool Timeout；每层超时独立计量</text>')
    legend(lines, 625, [('blue', '正常事件流'), ('red', '异常终态')])
    lines.append('  <text x="1135" y="630" text-anchor="end" class="small">429：Retry-After 或指数退避 + 抖动 + 总重试上限</text>')
    finish(lines, 'streaming-resilience-state-machine.svg')


def react_diagram():
    lines = start('ReAct：推理—行动—观察闭环', '语言推理维护计划；Action 接触环境；Observation 用外部事实更新下一步决策')
    container(lines, 40, 105, 820, 480, 'loop', '#f9fafb', '#d1d5db')
    container(lines, 890, 105, 270, 480, 'evidence', '#ffffff', '#d1d5db')
    lines.append('  <text x="65" y="140" class="heading">方法循环</text>')
    lines.append('  <text x="915" y="140" class="heading">论文证据</text>')
    arrow(lines, 'M 290 270 L 380 270', 'blue')
    label(lines, 335, 254, '选择动作')
    arrow(lines, 'M 570 270 L 650 270', 'blue')
    label(lines, 610, 254, '执行')
    arrow(lines, 'M 740 315 L 740 455 L 570 455', 'green')
    label(lines, 655, 444, '环境结果', '#15803d')
    arrow(lines, 'M 380 455 L 195 455 L 195 315', 'purple')
    label(lines, 300, 444, '更新状态', '#7e22ce')
    box(lines, 100, 225, 190, 90, 'Thought', '分解、跟踪、纠错', '#faf5ff', '#d8b4fe', 'thought')
    box(lines, 380, 225, 190, 90, 'Action', '有限动作空间', '#eff6ff', '#93c5fd', 'action')
    box(lines, 650, 225, 180, 90, 'Environment', 'Wikipedia / 工具 / UI', '#f3f4f6', '#d1d5db', 'environment')
    box(lines, 380, 410, 190, 90, 'Observation', '事实、状态或错误', '#f0fdf4', '#86efac', 'observation')
    lines.append('  <rect x="115" y="520" width="630" height="40" rx="8" fill="#fff7ed" stroke="#fdba74"/>')
    lines.append('  <text x="430" y="546" text-anchor="middle" class="body">完成 / 请求输入 / 最大步数 / 无进展 / 超时 / 预算耗尽</text>')
    evidence = [
        ('HotpotQA', 'ReAct 27.4 < CoT 29.4'),
        ('FEVER', 'ReAct 60.9 > CoT 56.3'),
        ('ALFWorld', 'ReAct best 71%'),
        ('WebShop', 'ReAct SR 40.0%'),
    ]
    y = 170
    for title, detail in evidence:
        box(lines, 915, y, 220, 76, title, detail, '#ffffff', '#d1d5db', title)
        y += 92
    legend(lines, 625, [('blue', '决策与行动'), ('green', '环境观察'), ('purple', '反馈更新')])
    lines.append('  <text x="1135" y="630" text-anchor="end" class="small">外部落地减少幻觉，但增加搜索错误、循环和副作用风险</text>')
    finish(lines, 'react-loop-and-evidence.svg')


def budget_diagram():
    lines = start('Agent 多预算终止控制器', '每次行动前做最坏情况准入；为最终回答和清理保留 Token、费用与时间')
    container(lines, 40, 105, 1120, 500, 'budget')
    inputs = [
        (70, 'Step / Calls', '循环与调用次数'),
        (350, 'Token', '输入、输出、推理'),
        (630, 'Cost', '模型与工具费用'),
        (910, 'Deadline', '总时限与子超时'),
    ]
    for x, title, detail in inputs:
        box(lines, x, 145, 220, 82, title, detail, '#eff6ff', '#93c5fd', title)
        arrow(lines, f'M {x+110} 227 L {x+110} 295', 'blue')
    lines.append('  <rect x="70" y="295" width="1060" height="92" rx="10" fill="#faf5ff" stroke="#d8b4fe" stroke-width="1.5"/>')
    lines.append('  <text x="600" y="330" text-anchor="middle" class="heading">Admission Controller</text>')
    lines.append('  <text x="600" y="358" text-anchor="middle" class="body">完成谓词 → 安全策略 → 剩余预算 → 收尾预留 → 重复与无进展检测</text>')
    arrow(lines, 'M 600 387 L 600 425', 'purple')
    lines.append('  <polygon points="600,425 690,470 600,515 510,470" fill="#fff7ed" stroke="#fdba74" stroke-width="1.5"/>')
    lines.append('  <text x="600" y="466" text-anchor="middle" class="heading">允许下一步？</text>')
    lines.append('  <text x="600" y="487" text-anchor="middle" class="small">先判完成，再判预算</text>')
    arrow(lines, 'M 510 470 L 255 470 L 255 525', 'green')
    arrow(lines, 'M 600 515 L 600 525', 'red')
    arrow(lines, 'M 690 470 L 945 470 L 945 525', 'blue')
    box(lines, 145, 525, 220, 72, 'COMPLETED', '满足显式完成谓词', '#f0fdf4', '#86efac', 'completed')
    box(lines, 490, 525, 220, 72, 'STOP', '预算 / 超时 / 取消', '#fef2f2', '#fca5a5', 'stop')
    box(lines, 835, 525, 220, 72, 'NEXT STEP', '执行后按实际 usage 对账', '#eff6ff', '#93c5fd', 'next')
    legend(lines, 645, [('blue', '预算输入与继续'), ('green', '成功完成'), ('red', '受控终止')])
    finish(lines, 'agent-budget-termination-controller.svg')


structured_output_diagram()
streaming_diagram()
react_diagram()
budget_diagram()
