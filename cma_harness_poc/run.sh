#!/bin/bash
# cma_harness_poc/run.sh — 用 uv 启动 CMA Harness POC 测试服务
# 用法: source ~/.hermes/.env && ./cma_harness_poc/run.sh
#
# 前置条件:
#   uv 已安装 (/opt/homebrew/bin/uv)

set -euo pipefail
cd "$(dirname "$0")/.."

VENV=".venv-poc"
# 首次运行自动创建 venv + 安装依赖
if [ ! -f "$VENV/bin/python3" ]; then
  echo "⏳ 创建 uv venv (Python 3.12)..."
  /opt/homebrew/bin/uv venv --python 3.12 "$VENV"
  echo "⏳ 安装依赖..."
  https_proxy=http://127.0.0.1:7897 /opt/homebrew/bin/uv pip install --python "$VENV" \
    aiohttp pyyaml python-dotenv httpx jinja2 markdown pydantic tenacity requests 2>&1 | tail -3
  echo "⏳ 安装 hermes-core (editable)..."
  /opt/homebrew/bin/uv pip install --python "$VENV" -e ./hermes-core 2>&1 | tail -3
fi

export PYTHONPATH="."
exec "$VENV/bin/python3" cma_harness_poc/main.py "$@"
