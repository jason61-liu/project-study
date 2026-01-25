# WeChat Article Summarizer

A CLI tool to automatically fetch and summarize WeChat Official Account articles using LLM.

## Features

- Fetch WeChat articles with anti-scraping measures
- Parse article content using BeautifulSoup
- Summarize using DeepSeek API
- CLI interface with multiple options
- Save summaries to file

## Installation

```bash
# Install in editable mode
pip install -e .

# Or install dependencies manually
pip install httpx beautifulsoup4 lxml typer openai python-dotenv
```

## Configuration

Create a `.env` file based on `.env.example`:

```bash
cp .env.example .env
```

Edit `.env` and add your DeepSeek API key:

```env
DEEPSEEK_API_KEY=your_actual_api_key_here
```

Get your API key from: https://platform.deepseek.com/api_keys

## Usage

### Basic Usage

```bash
# Summarize an article (output to terminal)
wechat-summarizer summarize "https://mp.weixin.qq.com/s/xxxxx"

# Or use python -m
python -m wechat_summarizer summarize "https://mp.weixin.qq.com/s/xxxxx"
```

### Save to File

```bash
wechat-summarizer summarize "https://mp.weixin.qq.com/s/xxxxx" --output summary.md
```

### Advanced Options

```bash
# Use different model (deepseek-chat or deepseek-coder)
wechat-summarizer summarize "https://mp.weixin.qq.com/s/xxxxx" --model deepseek-coder

# Provide API key directly
wechat-summarizer summarize "https://mp.weixin.qq.com/s/xxxxx" --api-key YOUR_KEY

# Verbose output
wechat-summarizer summarize "https://mp.weixin.qq.com/s/xxxxx" --verbose
```

### Short Options

```bash
wechat-summarizer summarize "https://mp.weixin.qq.com/s/xxxxx" -o summary.md -m deepseek-chat -v
```

## Project Structure

```
wechat/
├── src/
│   └── wechat_summarizer/
│       ├── __init__.py
│       ├── __main__.py           # python -m entry point
│       ├── cli.py                # CLI commands
│       ├── scraper.py            # HTTP fetcher
│       ├── parser.py             # HTML parser
│       ├── summarizer.py         # LLM API client
│       ├── config.py             # Configuration
│       └── utils.py              # Utilities
├── pyproject.toml                # Project config
├── .env.example                  # Env template
└── README.md
```

## Requirements

- Python 3.9+
- DeepSeek API key

## Error Handling

The tool handles common errors:

- Network timeouts and retries
- Invalid article URLs
- Parsing failures
- API rate limits
- Missing API keys

## License

MIT
