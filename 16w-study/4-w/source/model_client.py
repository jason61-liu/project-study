"""DeepSeek-V4-Pro 的最小 OpenAI-compatible 客户端与真实用量记录。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
import time

from openai import OpenAI


@dataclass(frozen=True)
class ModelResult:
    """一次真实模型调用的文本、Token、延迟和按官方价格估算的费用。"""

    text: str
    input_tokens: int
    output_tokens: int
    cache_hit_input_tokens: int
    cache_miss_input_tokens: int
    latency_ms: float
    estimated_cost_usd: float
    model: str

    def to_dict(self) -> dict:
        return asdict(self)


class DeepSeekModel:
    """调用 DeepSeek Chat Completions；API Key 只从进程环境读取。"""

    # 2026-08-11 DeepSeek 官方价格，单位 USD/1M tokens。
    CACHE_HIT_INPUT_PRICE = 0.003625
    CACHE_MISS_INPUT_PRICE = 0.435
    OUTPUT_PRICE = 0.87

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout_s: float = 90.0,
    ) -> None:
        key = api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("缺少 DEEPSEEK_API_KEY/OPENAI_API_KEY")
        self.model = model or os.getenv("AGENT_TEST_MODEL", "deepseek-v4-pro")
        self.client = OpenAI(
            api_key=key,
            base_url=base_url or os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com"),
            timeout=timeout_s,
            max_retries=2,
        )

    def complete(self, system: str, user: str, *, max_tokens: int = 220) -> ModelResult:
        """以非思考模式生成短答案，并优先使用服务端返回的实际 Token。"""

        started = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0,
            max_tokens=max_tokens,
            extra_body={"thinking": {"type": "disabled"}},
        )
        latency_ms = (time.perf_counter() - started) * 1000
        text = response.choices[0].message.content or ""
        usage = response.usage
        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        # DeepSeek 返回 cache_hit/cache_miss 字段时采用真实拆分；否则保守地全部按
        # cache miss 计费，避免低估成本。
        cache_hit = int(getattr(usage, "prompt_cache_hit_tokens", 0) or 0)
        cache_miss_value = getattr(usage, "prompt_cache_miss_tokens", None)
        cache_miss = int(cache_miss_value) if cache_miss_value is not None else max(0, input_tokens - cache_hit)
        cost = (
            cache_hit * self.CACHE_HIT_INPUT_PRICE
            + cache_miss * self.CACHE_MISS_INPUT_PRICE
            + output_tokens * self.OUTPUT_PRICE
        ) / 1_000_000
        return ModelResult(
            text=text.strip(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_hit_input_tokens=cache_hit,
            cache_miss_input_tokens=cache_miss,
            latency_ms=latency_ms,
            estimated_cost_usd=cost,
            model=response.model or self.model,
        )

