"""WeChat article scraper module."""
import asyncio
import time
from typing import Optional

import httpx


DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://mp.weixin.qq.com/",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}


class RateLimiter:
    """Simple rate limiter to prevent too many requests."""

    def __init__(self, requests_per_second: float = 1.0):
        self._min_interval = 1.0 / requests_per_second
        self._last_request_time = 0.0

    def acquire(self) -> None:
        """Wait if necessary to maintain rate limit."""
        now = time.time()
        time_since_last = now - self._last_request_time
        if time_since_last < self._min_interval:
            sleep_time = self._min_interval - time_since_last
            time.sleep(sleep_time)
        self._last_request_time = time.time()


class WeChatScraperError(Exception):
    """Base exception for scraper errors."""


class WeChatScraper:
    """Scraper for WeChat Official Account articles."""

    def __init__(
        self,
        timeout: float = 30.0,
        max_retries: int = 3,
        requests_per_second: float = 1.0,
    ):
        self.timeout = timeout
        self.max_retries = max_retries
        self.rate_limiter = RateLimiter(requests_per_second)
        self._client: Optional[httpx.Client] = None

    def __enter__(self):
        self._client = httpx.Client(timeout=self.timeout, follow_redirects=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            self._client.close()

    def fetch(self, url: str) -> str:
        """Fetch article HTML."""
        if not self._client:
            raise RuntimeError("Scraper not initialized. Use context manager.")

        self.rate_limiter.acquire()

        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self._client.get(url, headers=DEFAULT_HEADERS)
                response.raise_for_status()
                return response.text
            except httpx.HTTPStatusError as e:
                last_error = e
                if e.response.status_code in (429, 503, 504):
                    # Rate limited or temporary server error, retry with backoff
                    time.sleep(2 ** attempt)
                else:
                    # Other HTTP errors, don't retry
                    raise WeChatScraperError(f"HTTP error: {e.response.status_code}") from e
            except httpx.RequestError as e:
                last_error = e
                # Network error, retry with backoff
                time.sleep(2 ** attempt)

        raise WeChatScraperError(f"Failed after {self.max_retries} retries: {last_error}")
