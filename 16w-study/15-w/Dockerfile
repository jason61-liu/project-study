FROM python:3.13-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

RUN useradd --create-home --uid 10001 appuser && mkdir -p /data && chown appuser:appuser /data
USER appuser
ENV DATABASE_PATH=/data/research.db PYTHONUNBUFFERED=1
EXPOSE 8080
CMD ["research-agent"]

