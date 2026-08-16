"""OpenTelemetry Trace/Metric/Log JSONL export with content-safe defaults."""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import re
import threading
from typing import Any, Iterator, Sequence

from opentelemetry import trace
from opentelemetry._logs import SeverityNumber
from opentelemetry.context import get_current
from opentelemetry.sdk._logs import LoggerProvider, ReadableLogRecord
from opentelemetry.sdk._logs.export import LogRecordExportResult, LogRecordExporter, SimpleLogRecordProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    MetricExportResult,
    MetricExporter,
    MetricsData,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExportResult, SpanExporter

from versioning import VersionManifest


class UnsafeTelemetry(ValueError):
    pass


class _JsonlWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append_json(self, payload: dict[str, Any]) -> None:
        line = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


class JsonlSpanExporter(SpanExporter):
    def __init__(self, path: Path) -> None:
        self.writer = _JsonlWriter(path)

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        for span in spans:
            self.writer.append_json(json.loads(span.to_json()))
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        return None


class JsonlMetricExporter(MetricExporter):
    def __init__(self, path: Path) -> None:
        super().__init__()
        self.writer = _JsonlWriter(path)

    def export(self, metrics_data: MetricsData, timeout_millis: float = 10_000, **kwargs) -> MetricExportResult:
        self.writer.append_json(json.loads(metrics_data.to_json()))
        return MetricExportResult.SUCCESS

    def force_flush(self, timeout_millis: float = 10_000) -> bool:
        return True

    def shutdown(self, timeout_millis: float = 30_000, **kwargs) -> None:
        return None


class JsonlLogExporter(LogRecordExporter):
    def __init__(self, path: Path) -> None:
        self.writer = _JsonlWriter(path)

    def export(self, batch: Sequence[ReadableLogRecord]) -> LogRecordExportResult:
        for record in batch:
            self.writer.append_json(json.loads(record.to_json()))
        return LogRecordExportResult.SUCCESS

    def shutdown(self) -> None:
        return None


class Telemetry:
    """Owns isolated providers so tests can create multiple runtimes safely."""

    _SENSITIVE_KEY = re.compile(
        r"(?i)(gen_ai\.(input\.messages|output\.messages|system_instructions|prompt\.variable)"
        r"|gen_ai\.tool\.call\.(arguments|result)"
        r"|app\.(prompt|tool_result|sandbox\.(stdout|stderr|code))"
        r"|(^|\.)(access_token|api_key|authorization|secret|password)$)"
    )
    _SENSITIVE_VALUE = re.compile(
        r"(?i)(bearer\s+\S+|\bsk_[A-Za-z0-9_-]{8,}\b"
        r"|\beyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{8,}\b"
        r"|\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b|(?<!\d)1[3-9]\d{9}(?!\d))"
    )

    def __init__(self, output_dir: Path, *, service_version: str = "10w-v1") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.trace_path = self.output_dir / "traces.jsonl"
        self.metric_path = self.output_dir / "metrics.jsonl"
        self.log_path = self.output_dir / "logs.jsonl"
        resource = Resource.create(
            {
                "service.name": "week10-agent-runtime",
                "service.version": service_version,
                "deployment.environment.name": "local-lab",
            }
        )

        self.tracer_provider = TracerProvider(resource=resource)
        self.tracer_provider.add_span_processor(SimpleSpanProcessor(JsonlSpanExporter(self.trace_path)))
        self.tracer = self.tracer_provider.get_tracer("week10.runtime", service_version)

        reader = PeriodicExportingMetricReader(
            JsonlMetricExporter(self.metric_path), export_interval_millis=60_000
        )
        self.meter_provider = MeterProvider(resource=resource, metric_readers=[reader])
        meter = self.meter_provider.get_meter("week10.runtime", service_version)
        self.request_counter = meter.create_counter("agent.request.total", unit="{request}")
        self.error_counter = meter.create_counter("agent.error.total", unit="{error}")
        self.token_counter = meter.create_counter("gen_ai.client.token.usage", unit="{token}")
        self.cost_counter = meter.create_counter("agent.cost", unit="{microUSD}")
        self.duration_histogram = meter.create_histogram("agent.operation.duration", unit="ms")

        self.logger_provider = LoggerProvider(resource=resource)
        self.logger_provider.add_log_record_processor(SimpleLogRecordProcessor(JsonlLogExporter(self.log_path)))
        self._logger = self.logger_provider.get_logger("week10.runtime", service_version)

    @contextmanager
    def span(
        self,
        name: str,
        *,
        attributes: dict[str, Any],
        kind: trace.SpanKind = trace.SpanKind.INTERNAL,
    ) -> Iterator[trace.Span]:
        self._validate(attributes)
        with self.tracer.start_as_current_span(
            name,
            kind=kind,
            attributes=attributes,
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            yield span

    def safe_log(self, event: str, attributes: dict[str, Any]) -> None:
        self._validate(attributes)
        self._logger.emit(
            body=event,
            attributes=attributes,
            severity_number=SeverityNumber.INFO,
            severity_text="INFO",
            context=get_current(),
        )

    def record_summary(
        self,
        *,
        tenant_id: str,
        versions: VersionManifest,
        outcome: str,
        duration_ms: float,
        input_tokens: int,
        output_tokens: int,
        cost_microusd: int,
    ) -> None:
        labels = {
            "tenant.id": tenant_id,
            "app.version.fingerprint": versions.fingerprint,
            "app.version.model": versions.model,
            "app.outcome": outcome,
        }
        self.request_counter.add(1, labels)
        if outcome != "success":
            self.error_counter.add(1, labels)
        self.token_counter.add(input_tokens, {**labels, "gen_ai.token.type": "input"})
        self.token_counter.add(output_tokens, {**labels, "gen_ai.token.type": "output"})
        self.cost_counter.add(cost_microusd, labels)
        self.duration_histogram.record(duration_ms, labels)

    def force_flush(self) -> None:
        self.tracer_provider.force_flush()
        self.meter_provider.force_flush()
        self.logger_provider.force_flush()

    def shutdown(self) -> None:
        self.force_flush()
        self.tracer_provider.shutdown()
        self.meter_provider.shutdown()
        self.logger_provider.shutdown()

    @classmethod
    def _validate(cls, attributes: dict[str, Any]) -> None:
        for key, value in attributes.items():
            if cls._SENSITIVE_KEY.search(key):
                raise UnsafeTelemetry(f"content-bearing telemetry attribute is forbidden: {key}")
            if isinstance(value, str) and cls._SENSITIVE_VALUE.search(value):
                raise UnsafeTelemetry(f"sensitive telemetry value is forbidden: {key}")
