"""Generate the three Week 9 security diagrams with explicit geometry.

The template router could not place several cross-boundary labels without a
collision, so this focused renderer uses the Fireworks Style 1 tokens and the
mandatory line-list construction method.
"""

from __future__ import annotations

from html import escape
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


COLORS = {
    "data": "#2563eb",
    "control": "#7c3aed",
    "write": "#10b981",
    "risk": "#ef4444",
}


def node(node_id, x, y, width, height, title, subtitle, fill="#ffffff", stroke="#d1d5db"):
    return {"id": node_id, "x": x, "y": y, "w": width, "h": height, "title": title, "subtitle": subtitle, "fill": fill, "stroke": stroke}


def edge(edge_id, source, target, path, label, lx, ly, flow="data"):
    return {"id": edge_id, "source": source, "target": target, "path": path, "label": label, "lx": lx, "ly": ly, "flow": flow}


def render(name, title, subtitle, containers, nodes, edges, footer, *, diagram_type="architecture"):
    lines = []
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 760" width="960" height="760" data-generator="fireworks-tech-graph" data-schema-version="1" data-style-id="1" data-visual-theme="Flat Icon" data-diagram-type="{diagram_type}" data-semantic-profile="generic" data-semantic-valid="true" data-quality-profile="standard" data-max-bends-per-edge="4" data-max-total-bends="40" data-max-route-stretch="5.0" data-max-bridged-crossings="0" data-min-node-gap="0" data-min-container-gutter="0" data-min-label-clearance="2" data-min-segment-length="0">')
    lines.append('  <defs>')
    for flow, color in COLORS.items():
        lines.append(f'    <marker id="arrow-{flow}" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0, 10 3.5, 0 7" fill="{color}"/></marker>')
    lines.append("    <style>")
    lines.append("      text { font-family: 'Helvetica Neue', Helvetica, Arial, 'PingFang SC', 'Microsoft YaHei', sans-serif; }")
    lines.append("      .title { font-size: 30px; font-weight: 700; fill: #111827; }")
    lines.append("      .subtitle { font-size: 14px; font-weight: 500; fill: #6b7280; }")
    lines.append("      .section { font-size: 13px; font-weight: 700; fill: #2563eb; letter-spacing: 1.2px; }")
    lines.append("      .node-title { font-size: 15px; font-weight: 700; fill: #111827; }")
    lines.append("      .node-sub { font-size: 12px; font-weight: 500; fill: #6b7280; }")
    lines.append("      .edge-label { font-size: 11px; font-weight: 600; fill: #6b7280; }")
    lines.append("      .legend, .footer { font-size: 11px; font-weight: 500; fill: #64748b; }")
    lines.append("    </style>")
    lines.append("  </defs>")
    lines.append('  <rect data-graph-role="background" width="960" height="760" fill="#ffffff"/>')
    lines.append(f'  <text x="480" y="52" text-anchor="middle" class="title">{escape(title)}</text>')
    lines.append(f'  <text x="480" y="78" text-anchor="middle" class="subtitle">{escape(subtitle)}</text>')

    for item in containers:
        cid, y, height, label = item
        lines.append(f'  <g id="{cid}" data-graph-role="container" data-container-id="{cid}" data-semantic-role="boundary" data-graph-bounds="40,{y},920,{y + height}">')
        lines.append(f'    <rect data-graph-role="container" x="40" y="{y}" width="880" height="{height}" rx="16" fill="none" stroke="#dbe5f1" stroke-width="1.4" stroke-dasharray="6 5"/>')
        lines.append(f'    <text x="58" y="{y + 24}" class="section">{escape(label)}</text>')
        lines.append("  </g>")

    for item in edges:
        color = COLORS[item["flow"]]
        bends = max(0, item["path"].count(" L ") - 1)
        lines.append(f'  <path id="{item["id"]}" data-graph-role="edge" data-edge-id="{item["id"]}" data-source="{item["source"]}" data-target="{item["target"]}" data-edge-kind="flow" data-flow="{item["flow"]}" data-bends="{bends}" data-route-stretch="1.0" data-bridges="" d="{item["path"]}" fill="none" stroke="{color}" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" marker-end="url(#arrow-{item["flow"]})"/>')

    for item in nodes:
        x, y, width, height = item["x"], item["y"], item["w"], item["h"]
        lines.append(f'  <g id="node-{item["id"]}" data-graph-role="node" data-node-id="{item["id"]}" data-semantic-role="component" data-graph-bounds="{x},{y},{x + width},{y + height}">')
        lines.append(f'    <rect x="{x}" y="{y}" width="{width}" height="{height}" rx="10" fill="{item["fill"]}" stroke="{item["stroke"]}" stroke-width="2"/>')
        lines.append(f'    <text x="{x + width / 2}" y="{y + 27}" text-anchor="middle" class="node-title">{escape(item["title"])}</text>')
        lines.append(f'    <text x="{x + width / 2}" y="{y + 48}" text-anchor="middle" class="node-sub">{escape(item["subtitle"])}</text>')
        lines.append("  </g>")

    for item in edges:
        lines.append(f'  <text x="{item["lx"]}" y="{item["ly"]}" text-anchor="middle" class="edge-label">{escape(item["label"])}</text>')

    legend = [("data", "data"), ("control", "control"), ("write", "audit/write"), ("risk", "deny/impact")]
    for index, (flow, label) in enumerate(legend):
        x = 90 + index * 190
        lines.append(f'  <line x1="{x}" y1="708" x2="{x + 28}" y2="708" stroke="{COLORS[flow]}" stroke-width="2.4" marker-end="url(#arrow-{flow})"/>')
        lines.append(f'  <text x="{x + 38}" y="712" class="legend">{label}</text>')
    lines.append(f'  <text x="48" y="742" class="footer">{escape(footer)}</text>')
    lines.append("</svg>")
    (ROOT / f"{name}.svg").write_text("\n".join(lines) + "\n", encoding="utf-8")
    report = {"diagram": name, "nodes": len(nodes), "edges": len(edges), "containers": len(containers), "generator": "focused-line-list", "visual_review": False}
    (ROOT / f"{name}.layout.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def data_flow():
    containers = [("entries", 108, 112, "TRUST BOUNDARY 1 · UNTRUSTED ENTRIES"), ("control", 258, 112, "TRUST BOUNDARY 2 · AGENT AND DETERMINISTIC CONTROL"), ("assets", 408, 112, "TRUST BOUNDARY 3 · TENANT ASSETS AND ISOLATED EXECUTION"), ("recovery", 558, 120, "AUDIT, IMPACT CONTAINMENT, AND DELETION")]
    nodes = [
        node("entry", 70, 142, 190, 60, "[ENTRY] User/Web/File", "request + untrusted artifact", "#fff7ed", "#f97316"),
        node("ingress", 370, 142, 190, 60, "Input Envelope", "provenance + PII mask", "#eff6ff", "#3b82f6"),
        node("model", 670, 142, 190, 60, "Model / Planner", "no raw credentials", "#faf5ff", "#8b5cf6"),
        node("approval", 70, 292, 180, 60, "Human Approval", "exact action hash", "#fff1f2", "#ef4444"),
        node("pep", 370, 292, 190, 60, "RBAC / ABAC PEP", "scope + tenant + role", "#eff6ff", "#2563eb"),
        node("gateway", 670, 292, 190, 60, "Tool Gateway", "allowlist + schema", "#eff6ff", "#2563eb"),
        node("tenant", 70, 442, 190, 60, "[ASSET] Tenant Data", "RAG · memory · cache", "#ecfdf5", "#10b981"),
        node("sandbox", 370, 442, 190, 60, "[ASSET] E2B", "1 CPU · 512 MiB · TTL", "#f3f4f6", "#374151"),
        node("egress", 670, 442, 190, 60, "Network Egress", "deny_out = all", "#fff1f2", "#ef4444"),
        node("audit", 210, 602, 220, 60, "Sanitized Audit", "subject · actor · jti", "#f0fdfa", "#14b8a6"),
        node("lifecycle", 550, 602, 250, 60, "Export / Delete", "approval + tombstone", "#fff1f2", "#ef4444"),
    ]
    edges = [
        edge("input", "entry", "ingress", "M 260,172 L 370,172", "input data", 315, 163),
        edge("context", "ingress", "model", "M 560,172 L 670,172", "safe context", 615, 163),
        edge("intent", "model", "gateway", "M 765,202 L 765,292", "tool intent", 808, 250, "control"),
        edge("typed", "gateway", "pep", "M 670,322 L 560,322", "typed call", 615, 313, "control"),
        edge("approve", "pep", "approval", "M 370,322 L 250,322", "high risk", 310, 313, "risk"),
        edge("authorize", "pep", "sandbox", "M 465,352 L 465,442", "authorized", 510, 400, "control"),
        edge("tenant-data", "tenant", "sandbox", "M 260,472 L 370,472", "tenant rows", 315, 463, "data"),
        edge("deny", "sandbox", "egress", "M 560,472 L 670,472", "no route", 615, 463, "risk"),
        edge("audit", "sandbox", "audit", "M 465,502 L 465,544 L 320,544 L 320,602", "redacted event", 390, 536, "write"),
        edge("delete", "audit", "lifecycle", "M 430,632 L 550,632", "delete proof", 490, 623, "write"),
    ]
    render("week9-secure-data-flow", "Week 9 Secure Agent Data Flow", "untrusted content becomes data; identity and policy gate every asset and side effect", containers, nodes, edges, "Assets: tenant data + sandbox · Subjects: user + agent · Impacts: leak, cross-tenant access, destructive action, RCE", diagram_type="data-flow")


def identity_flow():
    containers = [("identity", 108, 112, "IDENTITY TRUST DOMAIN"), ("host", 258, 112, "AGENT HOST · RAW CREDENTIAL STOPS HERE"), ("enforce", 408, 112, "TOOL ENFORCEMENT BOUNDARY"), ("resource", 558, 120, "RESOURCE AND AUDIT DOMAIN")]
    nodes = [
        node("token", 70, 142, 190, 60, "Delegated Token", "tenant · scope · jti · exp", "#fff7ed", "#f97316"),
        node("idp", 370, 142, 190, 60, "OIDC / OAuth IdP", "iss · aud · sub · act", "#eff6ff", "#3b82f6"),
        node("subject", 670, 142, 190, 60, "[SUBJECT] Alice", "tenant-a member", "#f0fdfa", "#14b8a6"),
        node("verifier", 70, 292, 190, 60, "Host Token Verifier", "signature + revoke", "#eff6ff", "#2563eb"),
        node("principal", 370, 292, 190, 60, "Verified Principal", "sub + act + tenant + scope", "#f0fdfa", "#14b8a6"),
        node("model", 670, 292, 190, 60, "Model / Planner", "business args only", "#faf5ff", "#8b5cf6"),
        node("approval", 70, 442, 190, 60, "Security Approver", "separation of duties", "#fff1f2", "#ef4444"),
        node("pdp", 370, 442, 190, 60, "PDP / PEP", "RBAC · ABAC · scope", "#eff6ff", "#2563eb"),
        node("gateway", 670, 442, 190, 60, "Tool Gateway", "allowlist + schema", "#eff6ff", "#2563eb"),
        node("tenant", 280, 602, 190, 60, "[TENANT] tenant-a", "RAG · memory · cache", "#ecfdf5", "#10b981"),
        node("runtime", 540, 602, 150, 60, "E2B Runtime", "ephemeral", "#f3f4f6", "#374151"),
        node("receipt", 760, 602, 130, 60, "Audit Receipt", "jti only", "#f0fdfa", "#14b8a6"),
    ]
    edges = [
        edge("login", "subject", "idp", "M 670,172 L 560,172", "authenticate", 615, 163, "control"),
        edge("issue", "idp", "token", "M 370,172 L 260,172", "delegation", 315, 163),
        edge("verify", "token", "verifier", "M 165,202 L 165,292", "raw token", 206, 250, "control"),
        edge("claims", "verifier", "principal", "M 260,322 L 370,322", "verified claims", 315, 313),
        edge("model-view", "principal", "model", "M 560,322 L 670,322", "no credential", 615, 313, "risk"),
        edge("intent", "model", "gateway", "M 765,352 L 765,442", "tool + args", 807, 400, "control"),
        edge("decision", "gateway", "pdp", "M 670,472 L 560,472", "principal + resource", 615, 463, "control"),
        edge("hitl", "pdp", "approval", "M 370,472 L 260,472", "high risk", 315, 463, "risk"),
        edge("tenant-bind", "pdp", "tenant", "M 465,502 L 465,552 L 375,552 L 375,602", "tenant scoped", 418, 544),
        edge("execute", "tenant", "runtime", "M 470,632 L 540,632", "bounded", 505, 623, "control"),
        edge("receipt", "runtime", "receipt", "M 690,632 L 760,632", "redacted", 725, 623, "write"),
    ]
    render("week9-identity-propagation", "Delegated Identity Propagation", "raw credential stops at the Host; immutable subject, actor, tenant and scope continue", containers, nodes, edges, "Effective authority = delegation AND role AND scope AND resource attributes AND approved action")


def threat_model():
    containers = [("entries", 108, 112, "ATTACK ENTRIES · UNTRUSTED"), ("decision", 258, 112, "AGENT DECISION AND AUTHORIZATION BOUNDARY"), ("assets", 408, 112, "ASSETS · TENANT AND RUNTIME BOUNDARY"), ("impacts", 558, 120, "IMPACTS AND CONTAINMENT")]
    nodes = [
        node("direct", 70, 142, 180, 60, "[ENTRY] Direct PI", "user-controlled text", "#fff7ed", "#f97316"),
        node("indirect", 390, 142, 180, 60, "[ENTRY] Indirect PI", "web · file · RAG", "#fff7ed", "#f97316"),
        node("tool-result", 710, 142, 180, 60, "[ENTRY] Tool Result", "malicious observation", "#fff7ed", "#f97316"),
        node("goal", 70, 292, 180, 60, "Goal Hijack", "planner drift", "#faf5ff", "#8b5cf6"),
        node("misuse", 390, 292, 180, 60, "Tool Misuse", "valid API, unsafe effect", "#fff1f2", "#ef4444"),
        node("identity", 710, 292, 180, 60, "Identity Abuse", "scope · role · tenant", "#fff1f2", "#ef4444"),
        node("tenant", 70, 442, 180, 60, "[ASSET] Tenant Data", "RAG · memory · cache", "#ecfdf5", "#10b981"),
        node("secret", 390, 442, 180, 60, "[ASSET] Credentials", "Host-only Secret", "#fff7ed", "#f97316"),
        node("runtime", 710, 442, 180, 60, "[ASSET] Sandbox", "CPU · FS · network", "#f3f4f6", "#374151"),
        node("leak", 70, 602, 180, 60, "[IMPACT] Data Leak", "PII / Secret", "#fff1f2", "#ef4444"),
        node("cross", 390, 602, 180, 60, "[IMPACT] Cross Tenant", "confidentiality breach", "#fff1f2", "#ef4444"),
        node("rce", 710, 602, 180, 60, "[IMPACT] RCE", "guest compromise", "#fff1f2", "#ef4444"),
    ]
    edges = [
        edge("direct-goal", "direct", "goal", "M 160,202 L 160,292", "instruction", 204, 250, "control"),
        edge("indirect-misuse", "indirect", "misuse", "M 480,202 L 480,292", "tainted data", 529, 250, "control"),
        edge("tool-identity", "tool-result", "identity", "M 800,202 L 800,292", "false authority", 853, 250, "control"),
        edge("goal-data", "goal", "tenant", "M 160,352 L 160,442", "unsafe query", 207, 400, "risk"),
        edge("misuse-secret", "misuse", "secret", "M 480,352 L 480,442", "secret request", 529, 400, "risk"),
        edge("identity-runtime", "identity", "runtime", "M 800,352 L 800,442", "over-privilege", 853, 400, "risk"),
        edge("data-leak", "tenant", "leak", "M 160,502 L 160,602", "exfiltration", 205, 555, "risk"),
        edge("secret-cross", "secret", "cross", "M 480,502 L 480,602", "credential abuse", 531, 555, "risk"),
        edge("runtime-rce", "runtime", "rce", "M 800,502 L 800,602", "escape attempt", 852, 555, "risk"),
    ]
    render("week9-threat-model", "Agent Threat Model and Impact Paths", "entries cross trust boundaries toward assets; deterministic controls bound impact", containers, nodes, edges, "Containment: revoke token · kill sandbox · deny egress · delete tenant derivatives · preserve sanitized audit")


if __name__ == "__main__":
    data_flow()
    identity_flow()
    threat_model()
    print("generated 3 diagrams")
