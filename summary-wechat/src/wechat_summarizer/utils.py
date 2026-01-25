"""Utility functions for WeChat article summarizer."""
import re
from typing import Optional


def validate_wechat_url(url: str) -> bool:
    """Validate if URL is a WeChat Official Account article URL.

    Args:
        url: URL to validate

    Returns:
        True if valid WeChat URL, False otherwise
    """
    pattern = r"https?://mp\.weixin\.qq\.com/s/[a-zA-Z0-9_-]+"
    return bool(re.match(pattern, url))


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """Truncate text to maximum length.

    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def sanitize_filename(filename: str) -> str:
    """Sanitize filename by removing invalid characters.

    Args:
        filename: Filename to sanitize

    Returns:
        Sanitized filename
    """
    # Remove invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', "", filename)
    # Replace multiple spaces with single space
    filename = re.sub(r"\s+", " ", filename)
    # Limit length
    return truncate_text(filename.strip(), 200)


def format_summary(title: str, summary: str) -> str:
    """Format summary output.

    Args:
        title: Article title
        summary: Generated summary

    Returns:
        Formatted summary string
    """
    return f"# {title}\n\n{summary}"


def is_valid_html(html: str) -> bool:
    """Check if string appears to be valid HTML.

    Args:
        html: String to check

    Returns:
        True if appears to be HTML, False otherwise
    """
    if not html or len(html) < 10:
        return False
    return bool(re.search(r"<(html|body|div|p|h1|h2|h3)", html, re.IGNORECASE))
