"""WeChat Article Summarizer - A CLI tool to summarize WeChat articles."""

__version__ = "0.1.0"

from wechat_summarizer.scraper import WeChatScraper, WeChatScraperError
from wechat_summarizer.parser import WeChatParser, WeChatParserError
from wechat_summarizer.summarizer import WeChatSummarizer, WeChatSummarizerError

__all__ = [
    "__version__",
    "WeChatScraper",
    "WeChatScraperError",
    "WeChatParser",
    "WeChatParserError",
    "WeChatSummarizer",
    "WeChatSummarizerError",
]
