"""WeChat article parser module."""
import re
from typing import Optional

from bs4 import BeautifulSoup


class WeChatParserError(Exception):
    """Base exception for parser errors."""


class WeChatParser:
    """Parser for WeChat Official Account articles."""

    def __init__(self, html: str):
        self.soup = BeautifulSoup(html, "lxml")
        self._title: Optional[str] = None
        self._content: Optional[str] = None

    def parse(self) -> tuple[str, str]:
        """Parse article and return (title, content)."""
        title = self.get_title()
        content = self.get_content()

        if not title:
            raise WeChatParserError("Could not extract article title")
        if not content:
            raise WeChatParserError("Could not extract article content")

        return title, content

    def get_title(self) -> Optional[str]:
        """Extract article title."""
        if self._title is not None:
            return self._title

        # Method 1: Look for rich_media_title class
        title_elem = self.soup.find(class_="rich_media_title")
        if title_elem:
            self._title = self._clean_text(title_elem.get_text())
            return self._title

        # Method 2: Look for meta og:title
        meta_og_title = self.soup.find("meta", property="og:title")
        if meta_og_title and meta_og_title.get("content"):
            self._title = meta_og_title["content"]
            return self._title

        # Method 3: Look for meta twitter:title
        meta_twitter_title = self.soup.find("meta", attrs={"name": "twitter:title"})
        if meta_twitter_title and meta_twitter_title.get("content"):
            self._title = meta_twitter_title["content"]
            return self._title

        # Method 4: Look for h1 tag
        h1_elem = self.soup.find("h1")
        if h1_elem:
            self._title = self._clean_text(h1_elem.get_text())
            return self._title

        return None

    def get_content(self) -> Optional[str]:
        """Extract article content."""
        if self._content is not None:
            return self._content

        # Method 1: Look for js_content div (most common)
        content_div = self.soup.find("div", id="js_content")
        if content_div:
            self._content = self._extract_text_from_div(content_div)
            return self._content

        # Method 2: Look for rich_media_content class
        content_div = self.soup.find(class_="rich_media_content")
        if content_div:
            self._content = self._extract_text_from_div(content_div)
            return self._content

        return None

    def _extract_text_from_div(self, div) -> str:
        """Extract and clean text from a div element."""
        paragraphs = []

        for elem in div.find_all(["p", "div", "section"], recursive=False):
            text = self._clean_text(elem.get_text())
            if text:
                paragraphs.append(text)

        return "\n\n".join(paragraphs)

    def _clean_text(self, text: str) -> str:
        """Clean text by removing extra whitespace and special characters."""
        # Remove leading/trailing whitespace
        text = text.strip()
        # Replace multiple spaces with single space
        text = re.sub(r"\s+", " ", text)
        # Remove common WeChat artifacts
        text = re.sub(r"\s*【.*?】\s*", "", text)
        text = re.sub(r"\s*\[.*?\]\s*", "", text)
        return text
