"""CLI interface for WeChat article summarizer."""
import sys
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv

from wechat_summarizer.scraper import WeChatScraper, WeChatScraperError
from wechat_summarizer.parser import WeChatParser, WeChatParserError
from wechat_summarizer.summarizer import WeChatSummarizer, WeChatSummarizerError
from wechat_summarizer.utils import sanitize_filename

load_dotenv()

app = typer.Typer(
    name="wechat-summarizer",
    help="CLI tool to summarize WeChat Official Account articles",
)


@app.command()
def summarize(
    url: str = typer.Argument(..., help="WeChat article URL"),
    model: str = typer.Option(
        "deepseek-chat",
        "--model", "-m",
        help="LLM model to use for summarization"
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output", "-o",
        help="Output file path (if not specified, prints to stdout)"
    ),
    save: bool = typer.Option(
        False,
        "--save", "-s",
        help="Save summary to markdown file with article title as filename"
    ),
    detailed: bool = typer.Option(
        False,
        "--detailed", "-d",
        help="Generate detailed summary covering all core points"
    ),
    api_key: Optional[str] = typer.Option(
        None,
        "--api-key", "-k",
        help="DeepSeek API key (overrides DEEPSEEK_API_KEY env var)"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Enable verbose output"
    ),
) -> None:
    """Summarize a WeChat Official Account article.

    Example:
        wechat-summarizer summarize "https://mp.weixin.qq.com/s/xxxxx"
    """
    def log(msg: str) -> None:
        if verbose:
            typer.echo(f"[INFO] {msg}", err=True)

    try:
        # Step 1: Fetch article
        log(f"Fetching article from: {url}")
        with WeChatScraper() as scraper:
            html = scraper.fetch(url)

        # Step 2: Parse article
        log("Parsing article content...")
        parser = WeChatParser(html)
        title, content = parser.parse()
        log(f"Found article: {title}")

        # Step 3: Summarize
        log("Generating summary...")
        summarizer = WeChatSummarizer(api_key=api_key, model=model)
        summary = summarizer.summarize(title, content, detailed=detailed)

        # Step 4: Output result
        result = f"# {title}\n\n{summary}"

        # Determine output path
        output_path = None
        if save:
            # Use article title as filename (sanitized)
            filename = sanitize_filename(title) + ".md"
            output_path = Path(filename)
        elif output:
            output_path = Path(output)

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(result, encoding="utf-8")
            typer.echo(f"Summary saved to: {output_path}")
        else:
            typer.echo("\n" + "=" * 60)
            typer.echo(result)
            typer.echo("=" * 60)

    except WeChatScraperError as e:
        typer.echo(f"Error fetching article: {e}", err=True)
        sys.exit(1)
    except WeChatParserError as e:
        typer.echo(f"Error parsing article: {e}", err=True)
        sys.exit(1)
    except WeChatSummarizerError as e:
        typer.echo(f"Error generating summary: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        typer.echo(f"Unexpected error: {e}", err=True)
        sys.exit(1)


@app.command()
def version() -> None:
    """Show version information."""
    typer.echo("wechat-summarizer v0.1.0")


if __name__ == "__main__":
    app()
