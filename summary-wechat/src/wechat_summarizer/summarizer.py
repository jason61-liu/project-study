"""LLM summarizer module using DeepSeek API."""
import os
import time
from typing import Optional

from openai import OpenAI


class WeChatSummarizerError(Exception):
    """Base exception for summarizer errors."""


class WeChatSummarizer:
    """Summarizer using DeepSeek API."""

    DEFAULT_MODEL = "deepseek-chat"
    DEFAULT_PROMPT = """请为以下微信公众号文章写一个简洁准确的总结。

要求：
1. 用2-3句话概括文章核心内容
2. 提炼文章的主要观点
3. 总结要简洁、准确、易读

文章标题：{title}

文章内容：
{content}

总结："""

    DETAILED_PROMPT = """请为以下微信公众号文章写一个详细全面的总结。

要求：
1. 全面覆盖文章的所有核心观点和要点
2. 按照文章的逻辑结构组织总结，使用清晰的分段和层级
3. 对于文章中的重要概念、方法、案例等要详细阐述
4. 保留文章的关键信息和数据
5. 使用要点列表和适当的格式使总结更易读
6. 总结要详实、准确、完整，不少于300字

文章标题：{title}

文章内容：
{content}

详细总结："""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        max_retries: int = 3,
        base_url: str = "https://api.deepseek.com",
    ):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise WeChatSummarizerError(
                "API key not provided. Set DEEPSEEK_API_KEY environment variable or pass api_key parameter."
            )
        self.model = model
        self.max_retries = max_retries
        self.client = OpenAI(api_key=self.api_key, base_url=base_url)

    def summarize(self, title: str, content: str, prompt: Optional[str] = None, detailed: bool = False) -> str:
        """Summarize article content.

        Args:
            title: Article title
            content: Article content
            prompt: Custom prompt (optional)
            detailed: Whether to generate detailed summary

        Returns:
            Generated summary
        """
        prompt = prompt or (self.DETAILED_PROMPT if detailed else self.DEFAULT_PROMPT)
        user_prompt = prompt.format(title=title, content=content[:16000] if detailed else content[:8000])

        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=4000 if detailed else 2000,
                    temperature=0.7,
                )

                summary = response.choices[0].message.content.strip()
                return summary

            except Exception as e:
                last_error = str(e)
                error_str = str(e).lower()
                if "rate" in error_str or "quota" in error_str or "429" in error_str:
                    # Rate limit error, retry with exponential backoff
                    if attempt < self.max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                else:
                    raise WeChatSummarizerError(f"Summarization failed: {e}")

        raise WeChatSummarizerError(
            f"Failed after {self.max_retries} retries. Last error: {last_error}"
        )
