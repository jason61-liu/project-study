"""Configuration constants for WeChat article summarizer."""

# HTTP Client Settings
DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_REQUESTS_PER_SECOND = 1.0

# LLM Settings
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_MAX_TOKENS = 2000
DEFAULT_TEMPERATURE = 0.7
MAX_CONTENT_LENGTH = 8000

# WeChat URL Pattern
WECHAT_URL_PATTERN = r"https?://mp\.weixin\.qq\.com/"

# Default Headers for HTTP requests
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://mp.weixin.qq.com/",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}
