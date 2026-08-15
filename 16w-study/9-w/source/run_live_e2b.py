"""Explicit managed-E2B smoke run; no host-local execution fallback exists."""

from __future__ import annotations

import json

from attack_cases import live_e2b_status


if __name__ == "__main__":
    result = live_e2b_status()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["status"] in {"PASS", "SKIPPED"} else 1)
