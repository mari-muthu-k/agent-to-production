"""OpenTelemetry for the service (Day 4, 6.2-6.3).

The code marks its steps once with TRACER.start_as_current_span(...): parse_pdf, embed_chunks, agent.run,
llm.call, tool.<name>, retrieve, citation_check. Until setup_tracing() runs, those spans go nowhere (OTel's
default no-op provider). setup_tracing() decides where they go: in memory (the notebook), the console (the
api service's default) or OTLP to Jaeger/Datadog. FastAPI 0.142 adds its own request spans as soon as a
tracer provider is set: no instrumentation package needed.
"""
import json
import os
from typing import Optional

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor, SpanExporter, SpanExportResult

TRACER = trace.get_tracer("paper_agent")


class _Switch(SpanExporter):
    """One exporter slot: a global tracer provider can be set only once per process, so re-running
    setup_tracing() (a catch-up cell) swaps the exporter instead of stacking a second provider."""

    def __init__(self):
        self.target: Optional[SpanExporter] = None

    def export(self, spans):
        return self.target.export(spans) if self.target else SpanExportResult.SUCCESS

    def shutdown(self):
        if self.target:
            self.target.shutdown()


_SWITCH = _Switch()


HTTP_DETAILS = ("http.", "url.", "server.", "network.")


def one_line(span) -> str:
    """ConsoleSpanExporter's format for the api service: one JSON line per span."""
    attributes = {k: v for k, v in (span.attributes or {}).items() if not k.startswith(HTTP_DETAILS)}
    return json.dumps({"span": span.name, "trace_id": f"{span.context.trace_id:032x}"[-8:],
                       "ms": round((span.end_time - span.start_time) / 1e6, 1), **attributes}) + "\n"


def default_exporter() -> SpanExporter:
    """OTLP when OTEL_EXPORTER_OTLP_ENDPOINT is set (Jaeger, Datadog ...), else one line per span on stdout."""
    if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        return OTLPSpanExporter()
    return ConsoleSpanExporter(formatter=one_line)


def setup_tracing(exporter: Optional[SpanExporter] = None) -> SpanExporter:
    """Send every span from now on to `exporter` (default: default_exporter()). Safe to call again."""
    if not isinstance(trace.get_tracer_provider(), TracerProvider):
        provider = TracerProvider(resource=Resource.create({"service.name": "paper-api"}))
        provider.add_span_processor(SimpleSpanProcessor(_SWITCH))
        trace.set_tracer_provider(provider)
    _SWITCH.target = exporter or default_exporter()
    return _SWITCH.target


def describe(span) -> str:
    """The attributes worth reading next to a span's name."""
    a = span.attributes or {}
    if span.name == "llm.call":
        return (f"{a.get('llm.deployment', '?')} · {a.get('llm.input_tokens', 0):,} in + "
                f"{a.get('llm.output_tokens', 0):,} out tokens · ${a.get('llm.cost_usd', 0):.6f}")
    if span.name == "retrieve":
        return ", ".join(a.get("retrieve.chunk_ids", ()))
    if span.name == "citation_check":
        return "valid" if a.get("citation.valid") else "INVALID"
    return ""


def show_trace(exporter, trace_id: Optional[int] = None, width: int = 32) -> list:
    """Print one request's spans as a tree with a timeline bar each (default: the latest trace).
    Returns the spans, parents first."""
    spans = list(exporter.get_finished_spans())
    if not spans:
        print("no spans yet: run setup_tracing() first, then send a request")
        return []
    if trace_id is None:
        trace_id = max(spans, key=lambda s: s.end_time).context.trace_id
    spans = [s for s in spans if s.context.trace_id == trace_id]
    ids = {s.context.span_id for s in spans}
    children: dict = {}
    for s in sorted(spans, key=lambda s: s.start_time):
        parent = s.parent.span_id if s.parent and s.parent.span_id in ids else None
        children.setdefault(parent, []).append(s)
    roots = children.get(None, [])
    t0 = min(s.start_time for s in spans)
    total = max(max(s.end_time for s in spans) - t0, 1)
    ordered = []

    def walk(span, depth):
        ordered.append(span)
        start = round((span.start_time - t0) / total * width)
        length = max(1, round((span.end_time - span.start_time) / total * width))
        bar = " " * start + "█" * min(length, width - start)
        label = ("  " * depth + span.name)[:44]
        ms = (span.end_time - span.start_time) / 1e6
        print(f"{label:44} {ms:8.0f} ms  |{bar:<{width}}|  {describe(span)}")
        for child in children.get(span.context.span_id, []):
            walk(child, depth + 1)

    for root in roots:
        walk(root, 0)
    return ordered
