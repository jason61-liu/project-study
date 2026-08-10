"""生成“用户授权后 Agent 调用工具”的 OAuth/OIDC 时序图。"""

from html import escape
from pathlib import Path


OUTPUT = Path(__file__).with_name("user-authorized-agent-tool-call.svg")
parts: list[str] = []


def add(value: str) -> None:
    parts.append(value)


def label(x: int, y: int, value: str, css: str = "label", anchor: str = "middle") -> None:
    add(f'<text x="{x}" y="{y}" class="{css}" text-anchor="{anchor}">{escape(value)}</text>')


def participant(x: int, title: str, subtitle: str, fill: str, stroke: str) -> None:
    add(f'<g data-graph-role="node"><rect x="{x-112}" y="100" width="224" height="62" rx="10" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1.8"/>')
    label(x, 127, title, "participant")
    label(x, 148, subtitle, "small")
    add(f'<line x1="{x}" y1="162" x2="{x}" y2="1000" class="lifeline"/></g>')


def frame(y: int, height: int, title: str, note: str) -> None:
    add(f'<rect data-graph-role="container" x="45" y="{y}" width="1110" height="{height}" rx="10" '
        'fill="#f8fafc" fill-opacity="0.68" stroke="#cbd5e1" stroke-width="1.2" stroke-dasharray="7 5"/>')
    add(f'<rect x="61" y="{y+12}" width="172" height="25" rx="12" fill="#111827"/>')
    label(147, y + 30, title, "frame-title")
    label(247, y + 30, note, "frame-note", "start")


def message(source: int, target: int, y: int, title: str, detail: str, color: str, marker: str, dashed: bool = False) -> None:
    direction = 1 if target > source else -1
    dash = ' stroke-dasharray="7 5"' if dashed else ""
    add(f'<line data-graph-role="edge" x1="{source + 8*direction}" y1="{y}" '
        f'x2="{target - 13*direction}" y2="{y}" stroke="{color}" stroke-width="2" '
        f'marker-end="url(#{marker})"{dash}/>')
    label((source + target)//2, y - 10, title, "message")
    label((source + target)//2, y + 17, detail, "detail")


add('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 1110" width="1200" height="1110">')
add("""<style>
text { font-family:'PingFang SC','Hiragino Sans GB','Heiti SC','Helvetica Neue',Arial,sans-serif; }
.title{fill:#111827;font-size:25px;font-weight:700}.subtitle{fill:#64748b;font-size:13px}
.participant{fill:#111827;font-size:15px;font-weight:700}.small{fill:#64748b;font-size:11px}
.label{fill:#111827;font-size:13px}.message{fill:#111827;font-size:12px;font-weight:650}
.detail{fill:#64748b;font-size:10.5px}.frame-title{fill:#fff;font-size:12px;font-weight:700}
.frame-note{fill:#64748b;font-size:12px}.lifeline{stroke:#94a3b8;stroke-width:1.2;stroke-dasharray:6 6}
.note-title{fill:#7c2d12;font-size:12px;font-weight:700}.note{fill:#9a3412;font-size:11px}.legend{fill:#475569;font-size:11.5px}
</style>""")
add('<defs>')
for marker, color in (("blue", "#2563eb"), ("orange", "#ea580c"), ("green", "#16a34a"), ("red", "#dc2626")):
    add(f'<marker id="arrow-{marker}" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">'
        f'<polygon points="0 0,10 3.5,0 7" fill="{color}"/></marker>')
add('</defs>')

label(600, 40, "用户授权后 Agent 调用工具：身份与令牌边界", "title")
label(600, 66, "Authorization Code + PKCE / 面向资源的委托令牌 / 工具鉴权 / 过期与撤销", "subtitle")

xs = [155, 450, 750, 1045]
participant(xs[0], "用户", "资源所有者 / subject", "#fff7ed", "#ea580c")
participant(xs[1], "Agent 应用", "OAuth Client / actor", "#eff6ff", "#2563eb")
participant(xs[2], "授权服务器 / STS", "认证、同意、签发、撤销", "#f5f3ff", "#7c3aed")
participant(xs[3], "工具 API / MCP Server", "Resource Server", "#f0fdf4", "#16a34a")

frame(180, 300, "1. 用户授权", "Authorization Code + PKCE；OIDC 只负责登录身份")
message(xs[0], xs[1], 235, "请求执行受保护操作", "用户意图；尚未授予工具权限", "#2563eb", "arrow-blue")
message(xs[1], xs[2], 290, "跳转 /authorize", "state + nonce + code_challenge + scope + resource", "#2563eb", "arrow-blue")
message(xs[2], xs[0], 345, "认证并征得同意", "展示具体资源、Scope 与 Agent 应用", "#ea580c", "arrow-orange")
message(xs[2], xs[1], 400, "回调：code + state", "Agent 必须校验 state；code 是一次性凭证", "#16a34a", "arrow-green", True)
message(xs[1], xs[2], 455, "POST /token", "code + code_verifier + client authentication", "#2563eb", "arrow-blue")

frame(495, 185, "2. 建立会话与委托", "不要把 ID Token 当作访问工具的 Access Token")
message(xs[2], xs[1], 550, "签发 OIDC / OAuth 令牌", "ID Token（登录）+ Access Token + 可选 Refresh Token", "#16a34a", "arrow-green", True)
message(xs[1], xs[2], 615, "可选：Token Exchange", "subject_token=user；actor=agent；resource=tool；缩小 scope", "#2563eb", "arrow-blue")
message(xs[2], xs[1], 660, "短期委托 Access Token", "sub=user，act=agent，aud=tool，scope=最小权限", "#16a34a", "arrow-green", True)

frame(700, 180, "3. 调用工具", "Resource Server 同时校验令牌、业务策略和工具参数")
message(xs[1], xs[3], 755, "tools/call / HTTPS API", "Bearer delegated token + idempotency-key + trace-id", "#2563eb", "arrow-blue")
message(xs[3], xs[2], 810, "可选：Introspection / JWKS", "验证 active 或签名、iss、aud、exp；密钥与状态可缓存", "#ea580c", "arrow-orange", True)
message(xs[3], xs[1], 860, "结构化结果 + 审计标识", "授权还要检查 scope、tenant、sub/act、资源归属与参数", "#16a34a", "arrow-green", True)

frame(900, 105, "4. 过期 / 撤销", "取消请求不等于撤销令牌，撤销令牌也不等于回滚已发生的副作用")
message(xs[3], xs[1], 958, "401 invalid_token / 403 insufficient_scope", "刷新或重新交换后，仅在幂等或确认未执行时重试", "#dc2626", "arrow-red", True)

add('<rect x="58" y="1025" width="1084" height="46" rx="9" fill="#fff7ed" stroke="#fdba74"/>')
label(76, 1044, "关键约束", "note-title", "start")
label(76, 1062, "ID Token 的 audience 是 Agent Client；工具只接受面向自身 audience 的 Access Token。Token Exchange 是可选模式，需授权服务器支持。", "note", "start")

for x, color, name in ((365,"#2563eb","请求 / 调用"),(500,"#ea580c","用户控制 / 校验"),(665,"#16a34a","令牌 / 结果"),(800,"#dc2626","异常 / 撤销")):
    add(f'<line x1="{x}" y1="1082" x2="{x+28}" y2="1082" stroke="{color}" stroke-width="3"/>')
    label(x+36,1086,name,"legend","start")
add('</svg>')

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text("\n".join(parts), encoding="utf-8")
print(OUTPUT)
