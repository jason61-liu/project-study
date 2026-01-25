# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**WeChat Article Summarizer** - A CLI tool to fetch and summarize WeChat Official Account articles using DeepSeek API.

## Project Structure

```
wechat/
├── src/wechat_summarizer/
│   ├── __init__.py
│   ├── __main__.py           # python -m 入口
│   ├── cli.py                # Typer CLI 命令定义
│   ├── scraper.py            # 抓取微信文章 (使用 httpx)
│   ├── parser.py             # 解析 HTML 提取内容 (BeautifulSoup)
│   ├── summarizer.py         # LLM API 调用 (DeepSeek)
│   ├── config.py             # 配置常量
│   └── utils.py              # 工具函数
├── tests/
├── pyproject.toml            # 项目配置
├── .env                      # 环境变量 (DEEPSEEK_API_KEY)
└── README.md
```

## Important Instructions

### Virtual Environment

**每次执行 Python 命令时，必须先激活虚拟环境：**

```bash
source ~/workspace/pyproject/.venv/bin/activate
```

所有 Python 相关命令都应该使用此模式：
```bash
source ~/workspace/pyproject/.venv/bin/activate && <python命令>
```

### Install Package

```bash
source ~/workspace/pyproject/.venv/bin/activate && uv pip install -e .
```

### Run CLI

```bash
# Basic usage
source ~/workspace/pyproject/.venv/bin/activate && wechat-summarizer summarize "https://mp.weixin.qq.com/s/xxxxx"

# With verbose output
source ~/workspace/pyproject/.venv/bin/activate && wechat-summarizer summarize "URL" -v

# Save to file
source ~/workspace/pyproject/.venv/bin/activate && wechat-summarizer summarize "URL" -o output.md
```

## Tech Stack

| 组件 | 选择 |
|------|------|
| HTTP 客户端 | `httpx` |
| HTML 解析 | `BeautifulSoup4 + lxml` |
| CLI 框架 | `Typer` |
| LLM API | `DeepSeek (OpenAI-compatible)` |
| 环境变量 | `python-dotenv` |

## API Key

DeepSeek API key 通过环境变量 `DEEPSEEK_API_KEY` 配置，存储在 `.env` 文件中。
