"""真实 DeepSeek OpenAI-compatible JSON 客户端。"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from openai import OpenAI

from models import ModelReply


class DeepSeekResearchModel:
    """三种架构必须复用同一实例配置，避免模型差异污染实验。"""

    def __init__(self, *, timeout_s: float = 90.0) -> None:
        key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("缺少 DEEPSEEK_API_KEY/OPENAI_API_KEY；正式实验禁止使用测试替身")
        self.model = os.getenv("AGENT_TEST_MODEL", "deepseek-v4-pro")
        self.client = OpenAI(
            api_key=key,
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com"),
            timeout=timeout_s,
            max_retries=0,  # 实验层显式记录重试，避免 SDK 隐式重试污染步骤和延迟。
        )

    def complete_json(self, *, system: str, user: str, purpose: str) -> ModelReply:
        """要求 JSON Object；若服务返回 Markdown fence，则做有限兼容解析。"""

        started = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0,
            max_tokens=700,
            response_format={"type": "json_object"},
            extra_body={"thinking": {"type": "disabled"}},
        )
        elapsed = (time.perf_counter() - started) * 1000
        content = (response.choices[0].message.content or "{}").strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("```", 1)[0]
        data: dict[str, Any] = json.loads(content)
        usage = response.usage
        return ModelReply(
            data=data,
            input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            latency_ms=elapsed,
            model=response.model or self.model,
        )

