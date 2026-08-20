from __future__ import annotations

import hashlib
import re
from collections import defaultdict

from .guardrails import sanitize_excerpt, source_taints
from .models import Evidence, Source


TOKEN_RE = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)


def tokens(text: str) -> set[str]:
    result = {item.lower() for item in TOKEN_RE.findall(text) if len(item) > 1}
    cjk = "".join(char for char in text if "\u4e00" <= char <= "\u9fff")
    result.update(cjk[index : index + 2] for index in range(max(0, len(cjk) - 1)))
    return result


def decompose(question: str) -> list[str]:
    pieces = [part.strip(" ，,。？?；;\n") for part in re.split(r"[？?；;\n]|以及|并且|同时", question)]
    unique: list[str] = []
    for piece in pieces:
        if piece and piece not in unique:
            unique.append(piece)
    return unique[:6] or [question.strip()]


def retrieve(question: str, subqueries: list[str], sources: tuple[Source, ...], limit: int) -> list[Evidence]:
    query_tokens = tokens(" ".join([question, *subqueries]))
    seen_urls: set[str] = set()
    seen_content: set[str] = set()
    ranked: list[tuple[float, Source]] = []
    for source in sources:
        digest = hashlib.sha256(" ".join(source.content.split()).encode()).hexdigest()
        canonical = source.url.rstrip("/").lower()
        if canonical in seen_urls or digest in seen_content or source_taints(source.content):
            continue
        seen_urls.add(canonical)
        seen_content.add(digest)
        source_tokens = tokens(f"{source.title} {source.content} {' '.join(source.claims)}")
        overlap = len(query_tokens & source_tokens)
        coverage = overlap / max(len(query_tokens), 1)
        title_bonus = len(query_tokens & tokens(source.title)) * 0.15
        score = coverage + title_bonus
        if score > 0:
            ranked.append((score, source))
    ranked.sort(key=lambda item: (-item[0], item[1].source_id))
    return [
        Evidence(
            evidence_id=f"S{index}",
            source_id=source.source_id,
            title=source.title,
            url=source.url,
            excerpt=sanitize_excerpt(source.content),
            score=round(score, 4),
            claims=source.claims,
        )
        for index, (score, source) in enumerate(ranked[:limit], start=1)
    ]


def detect_conflicts(evidence: list[Evidence]) -> list[dict[str, object]]:
    values: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for item in evidence:
        for claim, value in item.claims.items():
            values[claim][value].append(item.evidence_id)
    return [
        {"claim": claim, "values": [{"value": value, "sources": ids} for value, ids in variants.items()]}
        for claim, variants in values.items()
        if len(variants) > 1
    ]
