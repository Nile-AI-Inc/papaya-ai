from __future__ import annotations

import contextvars
import inspect
import json
import os
import platform
import re
import sys
import urllib.error
import urllib.request
from contextlib import AbstractContextManager
from datetime import datetime, timezone
from typing import Any, Callable, Literal
from uuid import uuid4

from .version import __version__

CaptureMode = Literal["metadata", "redacted", "full"]
SpanStatus = Literal["success", "failed", "partial", "unknown"]
SpanKind = Literal[
    "workflow",
    "agent",
    "llm",
    "tool",
    "retrieval",
    "embedding",
    "reranker",
    "memory",
    "state_transition",
    "guardrail",
    "router",
    "human",
    "handoff",
    "evaluator",
    "other",
]
Transport = Callable[[str, dict[str, str], bytes], tuple[int, str]]

_active_run: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar("papaya_active_run", default=None)


def _iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid4()}"


def _json_default(value: Any) -> str:
    return str(value)


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, default=_json_default, separators=(",", ":")).encode("utf-8")


def _json_text(value: Any) -> str:
    return json.dumps(value, default=_json_default, separators=(",", ":"))


def _byte_length(value: Any) -> int:
    if isinstance(value, str):
        return len(value.encode("utf-8"))
    return len(_json_bytes(value))


def _redact_string(value: str) -> str:
    value = re.sub(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", "[redacted-email]", value, flags=re.I)
    value = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[redacted-ssn]", value)
    value = re.sub(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", "[redacted-phone]", value)
    value = re.sub(r"\b(?:sk|pk|papaya|openai|anthropic|gemini|aws)[-_][A-Za-z0-9_-]{12,}\b", "[redacted-secret]", value, flags=re.I)
    value = re.sub(r"Bearer\s+[A-Za-z0-9._~+/-]+=*", "Bearer [redacted-token]", value, flags=re.I)
    return value


def _redact_value(value: Any) -> Any:
    if isinstance(value, str):
        return _redact_string(value)
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    if isinstance(value, tuple):
        return [_redact_value(item) for item in value]
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if re.search(r"^(authorization|api[-_]?key|access[-_]?token|refresh[-_]?token|id[-_]?token|secret|password)$", key_text, flags=re.I):
                result[key_text] = "[redacted-secret]"
            else:
                result[key_text] = _redact_value(item)
        return result
    return value


def _content_type(value: Any) -> str:
    if isinstance(value, str):
        return "text"
    if isinstance(value, list) and all(isinstance(item, dict) and "role" in item for item in value):
        return "messages"
    return "json"


def _payload(value: Any, capture: CaptureMode) -> dict[str, Any]:
    if capture == "metadata":
        return {
            "contentType": _content_type(value),
            "redactionState": "metadata",
            "byteLength": _byte_length(value),
        }
    captured = _redact_value(value) if capture == "redacted" else value
    return {
        "contentType": _content_type(captured),
        "value": captured,
        "redactionState": capture,
        "byteLength": _byte_length(captured),
    }


def _error_payload(error: BaseException | Any) -> dict[str, str]:
    if isinstance(error, BaseException):
        return {"type": error.__class__.__name__, "message": str(error)}
    return {"message": str(error)}


def _merge_options(*option_sets: dict[str, Any] | None) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for options in option_sets:
        if not options:
            continue
        for key, value in options.items():
            if value is None:
                continue
            if key == "metadata" and isinstance(value, dict):
                result["metadata"] = {**result.get("metadata", {}), **value}
            else:
                result[key] = value
    return result


_RUN_OPTION_KEYS = (
    "traceId",
    "runId",
    "sessionId",
    "conversationId",
    "userId",
    "organizationId",
    "workflowKey",
    "workflowLabel",
    "conversational",
    "metadata",
)


def _run_options_from_trace(trace: dict[str, Any] | None) -> dict[str, Any] | None:
    if not trace:
        return None
    return {key: trace[key] for key in _RUN_OPTION_KEYS if key in trace}


def _provider_args_and_options(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[tuple[Any, ...], dict[str, Any], dict[str, Any] | None]:
    provider_args = args
    provider_kwargs = dict(kwargs)
    call_options = provider_kwargs.pop("papaya", None)

    if provider_args and isinstance(provider_args[0], dict) and "papaya" in provider_args[0]:
        request = dict(provider_args[0])
        call_options = request.pop("papaya", call_options)
        provider_args = (request, *provider_args[1:])

    return provider_args, provider_kwargs, call_options if isinstance(call_options, dict) else None


def _call_input(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    if kwargs and args:
        return {"args": list(args), "kwargs": kwargs}
    if kwargs:
        return kwargs
    if len(args) == 1:
        return args[0]
    return list(args)


def _value_get(value: Any, *keys: str) -> Any:
    for key in keys:
        if isinstance(value, dict) and key in value:
            return value[key]
        if hasattr(value, key):
            return getattr(value, key)
    return None


def _number_value(value: Any) -> int | float | None:
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _usage_from_record(value: Any) -> dict[str, Any] | None:
    usage = (
        _value_get(value, "usage")
        or _value_get(value, "token_usage")
        or _value_get(value, "usage_metadata")
        or _value_get(value, "usageMetadata")
    )
    if usage is None:
        usage = value
    input_tokens = _number_value(_value_get(usage, "input_tokens", "prompt_tokens", "inputTokens", "promptTokenCount"))
    output_tokens = _number_value(_value_get(usage, "output_tokens", "completion_tokens", "outputTokens", "candidatesTokenCount"))
    total_tokens = _number_value(_value_get(usage, "total_tokens", "totalTokens", "totalTokenCount"))
    if input_tokens is None and output_tokens is None and total_tokens is None:
        return None
    cost_usd = _number_value(_value_get(usage, "cost_usd", "costUsd"))
    return {
        "inputTokens": input_tokens,
        "outputTokens": output_tokens,
        "totalTokens": total_tokens if total_tokens is not None else (input_tokens or 0) + (output_tokens or 0),
        "cacheReadInputTokens": _number_value(_value_get(usage, "cache_read_input_tokens", "cached_input_tokens", "cacheReadInputTokens")),
        "cacheCreationInputTokens": _number_value(_value_get(usage, "cache_creation_input_tokens", "cacheCreationInputTokens")),
        "reasoningTokens": _number_value(_value_get(usage, "reasoning_tokens", "reasoningTokens")),
        "costUsd": cost_usd,
        "pricingSource": "provider" if cost_usd is not None else None,
    }


def _usage_from_result(value: Any, seen: set[int] | None = None) -> dict[str, Any] | None:
    if value is None:
        return None
    seen = seen or set()
    if id(value) in seen:
        return None
    seen.add(id(value))
    usage = _usage_from_record(value)
    if usage:
        return usage
    if isinstance(value, dict):
        iterable = value.values()
    elif isinstance(value, (list, tuple)):
        iterable = value
    else:
        iterable = []
        for key in ("body", "response", "response_metadata", "usage_metadata", "llm_output"):
            child = getattr(value, key, None)
            if child is not None:
                iterable = [*iterable, child]
    for item in iterable:
        usage = _usage_from_result(item, seen)
        if usage:
            return usage
    return None


def _model_from_call(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str | None:
    model = kwargs.get("model") or kwargs.get("model_id") or kwargs.get("modelId")
    if isinstance(model, str):
        return model
    if args and isinstance(args[0], dict):
        model = args[0].get("model") or args[0].get("model_id") or args[0].get("modelId")
        if isinstance(model, str):
            return model
    return None


def _model_from_result(value: Any, seen: set[int] | None = None) -> str | None:
    if value is None:
        return None
    seen = seen or set()
    if id(value) in seen:
        return None
    seen.add(id(value))
    model = _value_get(value, "model", "model_name", "modelName", "model_id", "modelId")
    if isinstance(model, str) and model:
        return model
    children: list[Any] = []
    if isinstance(value, dict):
        children.extend(value.values())
    elif isinstance(value, (list, tuple)):
        children.extend(value)
    else:
        for key in ("body", "response", "response_metadata", "usage_metadata", "llm_output"):
            child = getattr(value, key, None)
            if child is not None:
                children.append(child)
    for child in children:
        model = _model_from_result(child, seen)
        if model:
            return model
    return None


def _proxyable(value: Any) -> bool:
    return value is not None and not isinstance(value, (str, bytes, bytearray, int, float, bool, dict, list, tuple, set))


def _default_transport(endpoint: str, headers: dict[str, str], body: bytes) -> tuple[int, str]:
    request = urllib.request.Request(endpoint, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode("utf-8", errors="replace")


class _RunScope(AbstractContextManager[dict[str, Any]]):
    def __init__(self, papaya: "Papaya", options: dict[str, Any]):
        self._papaya = papaya
        self._options = options
        self._run: dict[str, Any] | None = None
        self._token: contextvars.Token[dict[str, Any] | None] | None = None

    def __enter__(self) -> dict[str, Any]:
        self._run = self._papaya.start_trace(self._options)
        self._token = _active_run.set(self._run)
        return self._run

    def __exit__(self, exc_type: Any, exc: BaseException | None, tb: Any) -> bool:
        if self._token is not None:
            _active_run.reset(self._token)
        if self._run is not None:
            self._papaya.finish_trace(self._run, "failed" if exc else "success", error=exc)
        return False


class _PapayaClientProxy:
    def __init__(
        self,
        papaya: "Papaya",
        provider: str,
        target: Any,
        path: list[str] | None = None,
        options: dict[str, Any] | None = None,
    ):
        object.__setattr__(self, "_papaya", papaya)
        object.__setattr__(self, "_provider", provider)
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "_path", path or [])
        object.__setattr__(self, "_options", options or {})

    def __getattr__(self, name: str) -> Any:
        value = getattr(object.__getattribute__(self, "_target"), name)
        path = [*object.__getattribute__(self, "_path"), name]
        if callable(value):
            def wrapped(*args: Any, **kwargs: Any) -> Any:
                return object.__getattribute__(self, "_papaya")._capture_provider_call(
                    object.__getattribute__(self, "_provider"),
                    path,
                    value,
                    args,
                    kwargs,
                    object.__getattribute__(self, "_options"),
                )
            return wrapped
        if _proxyable(value):
            return _PapayaClientProxy(
                object.__getattribute__(self, "_papaya"),
                object.__getattribute__(self, "_provider"),
                value,
                path,
                object.__getattribute__(self, "_options"),
            )
        return value

    def __repr__(self) -> str:
        return f"<PapayaClientProxy provider={object.__getattribute__(self, '_provider')!r} target={object.__getattribute__(self, '_target')!r}>"


class Papaya:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        endpoint: str | None = None,
        project: str | None = None,
        environment: str | None = None,
        capture: CaptureMode = "redacted",
        service_name: str | None = None,
        service_version: str | None = None,
        debug: bool = False,
        transport: Transport | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self.api_key = api_key or os.getenv("PAPAYA_API_KEY") or os.getenv("PAPAYA_INGEST_TOKEN")
        self.endpoint = endpoint or "https://papaya.fyi/api/v1/ingest/traces"
        self.project = project or "default"
        self.environment = environment or "development"
        self.capture = capture
        self.service_name = service_name
        self.service_version = service_version
        self.debug = debug
        self.transport = transport or _default_transport
        self.default_run_options = {"metadata": metadata} if metadata else {}
        self._completed: list[dict[str, Any]] = []

    @classmethod
    def init(cls, **options: Any) -> "Papaya":
        return cls(**options)

    def run(self, options: dict[str, Any] | None = None, **kwargs: Any) -> _RunScope:
        return _RunScope(self, _merge_options(self.default_run_options, options, kwargs))

    def wrap_client(self, provider: str, client: Any, options: dict[str, Any] | None = None, **kwargs: Any) -> Any:
        return _PapayaClientProxy(self, provider, client, options=_merge_options(options, kwargs))

    def openai(self, client: Any, options: dict[str, Any] | None = None, **kwargs: Any) -> Any:
        return self.wrap_client("openai", client, options, **kwargs)

    def claude(self, client: Any, options: dict[str, Any] | None = None, **kwargs: Any) -> Any:
        return self.wrap_client("claude", client, options, **kwargs)

    def anthropic(self, client: Any, options: dict[str, Any] | None = None, **kwargs: Any) -> Any:
        return self.claude(client, options, **kwargs)

    def gemini(self, client: Any, options: dict[str, Any] | None = None, **kwargs: Any) -> Any:
        return self.wrap_client("gemini", client, options, **kwargs)

    def bedrock(self, client: Any, options: dict[str, Any] | None = None, **kwargs: Any) -> Any:
        return self.wrap_client("bedrock", client, options, **kwargs)

    def start_trace(
        self,
        options: dict[str, Any] | None = None,
        *,
        root_span_id: str | None = None,
        root_name: str | None = None,
        root_kind: SpanKind = "workflow",
        input_value: Any | None = None,
        attributes: dict[str, Any] | None = None,
        started_at: str | None = None,
    ) -> dict[str, Any]:
        merged = _merge_options(self.default_run_options, options)
        trace_id = merged.get("traceId") or _id("trace")
        run_id = merged.get("runId") or _id("run")
        root_id = root_span_id or _id("span")
        root_span: dict[str, Any] = {
            "spanId": root_id,
            "name": root_name or merged.get("workflowLabel") or merged.get("workflowKey") or "papaya.run",
            "kind": root_kind,
            "startedAt": started_at or _iso(),
            "status": "unknown",
            "attributes": {
                "project": self.project,
                "environment": self.environment,
                "metadata": merged.get("metadata"),
                **(attributes or {}),
            },
        }
        if input_value is not None:
            root_span["inputPayload"] = _payload(input_value, self.capture)
        trace = {
            **merged,
            "traceId": trace_id,
            "runId": run_id,
            "rootSpanId": root_id,
            "spans": [root_span],
        }
        return trace

    def finish_trace(
        self,
        trace: dict[str, Any],
        status: SpanStatus,
        *,
        output_value: Any | None = None,
        error: BaseException | Any | None = None,
        ended_at: str | None = None,
    ) -> None:
        root = trace["spans"][0]
        root["endedAt"] = ended_at or _iso()
        root["status"] = status
        if output_value is not None:
            root["outputPayload"] = _payload(output_value, self.capture)
        if error is not None:
            root["error"] = _error_payload(error)
        if trace not in self._completed:
            self._completed.append(trace)

    def start_span(
        self,
        *,
        name: str,
        kind: SpanKind,
        trace: dict[str, Any] | None = None,
        parent_span_id: str | None = None,
        span_id: str | None = None,
        input_value: Any | None = None,
        model_ref: dict[str, Any] | None = None,
        attributes: dict[str, Any] | None = None,
        started_at: str | None = None,
    ) -> dict[str, Any]:
        target = trace or _active_run.get()
        if target is None:
            target = self.start_trace({"workflowKey": name})
            _active_run.set(target)
        span: dict[str, Any] = {
            "spanId": span_id or _id("span"),
            "parentSpanId": parent_span_id or target["rootSpanId"],
            "name": name,
            "kind": kind,
            "startedAt": started_at or _iso(),
            "status": "unknown",
        }
        if input_value is not None:
            span["inputPayload"] = _payload(input_value, self.capture)
        if model_ref:
            span["modelRef"] = model_ref
        if attributes:
            span["attributes"] = attributes
        target["spans"].append(span)
        return span

    def _capture_provider_call(
        self,
        provider: str,
        path: list[str],
        call: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        wrapper_options: dict[str, Any] | None = None,
    ) -> Any:
        provider_args, provider_kwargs, call_options = _provider_args_and_options(args, kwargs)
        active_trace = _active_run.get()
        boundary = _merge_options(self.default_run_options, wrapper_options, _run_options_from_trace(active_trace), call_options)

        if active_trace is not None:
            return self._capture_provider_call_in_trace(
                active_trace,
                provider,
                path,
                call,
                provider_args,
                provider_kwargs,
                boundary,
            )

        trace_options = {
            "workflowKey": boundary.get("workflowKey") or f"{provider}.{'.'.join(path)}",
            **boundary,
        }
        trace = self.start_trace(trace_options)
        return self._capture_provider_call_in_trace(
            trace,
            provider,
            path,
            call,
            provider_args,
            provider_kwargs,
            boundary,
            finish_trace=True,
        )

    def _capture_provider_call_in_trace(
        self,
        trace: dict[str, Any],
        provider: str,
        path: list[str],
        call: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        boundary: dict[str, Any],
        *,
        finish_trace: bool = False,
    ) -> Any:
        method = ".".join(path)
        model = _model_from_call(args, kwargs)
        span = self.start_span(
            trace=trace,
            name=f"{provider}.{method}",
            kind="llm",
            input_value=_call_input(args, kwargs),
            model_ref={"provider": provider, "requested": model},
            attributes={
                "provider": provider,
                "method": method,
                "workflowKey": boundary.get("workflowKey"),
                "workflowLabel": boundary.get("workflowLabel"),
                "sessionId": boundary.get("sessionId"),
                "conversationId": boundary.get("conversationId"),
                "userId": boundary.get("userId"),
                "organizationId": boundary.get("organizationId"),
                "metadata": boundary.get("metadata"),
            },
        )

        def finish(status: SpanStatus, result: Any | None = None, error: BaseException | Any | None = None) -> None:
            self.finish_span(
                span,
                status,
                output_value=result,
                usage=_usage_from_result(result),
                model_used=_model_from_result(result) or model,
                error=error,
            )
            if finish_trace:
                self.finish_trace(trace, status, output_value=result, error=error)

        try:
            result = call(*args, **kwargs)
        except Exception as error:
            finish("failed", error=error)
            raise

        if inspect.isawaitable(result):
            async def await_and_finish() -> Any:
                try:
                    value = await result
                except Exception as error:
                    finish("failed", error=error)
                    raise
                finish("success", result=value)
                return value
            return await_and_finish()

        finish("success", result=result)
        return result

    def finish_span(
        self,
        span: dict[str, Any],
        status: SpanStatus,
        *,
        output_value: Any | None = None,
        usage: dict[str, Any] | None = None,
        model_used: str | None = None,
        error: BaseException | Any | None = None,
        ended_at: str | None = None,
    ) -> None:
        span["endedAt"] = ended_at or _iso()
        span["status"] = status
        if output_value is not None:
            span["outputPayload"] = _payload(output_value, self.capture)
        if usage:
            span["usage"] = {key: value for key, value in usage.items() if value is not None}
        if model_used:
            span["modelRef"] = {**span.get("modelRef", {}), "used": model_used}
        if error is not None:
            span["error"] = _error_payload(error)

    def flush(self) -> dict[str, Any]:
        if not self._completed:
            return {"status": "skipped", "traceCount": 0, "reason": "empty"}
        traces = self._completed[:]
        self._completed.clear()
        batch = {
            "schemaVersion": "2026-06-05",
            "batchId": _id("batch"),
            "sentAt": _iso(),
            "sdk": {
                "name": "papaya-ai",
                "version": __version__,
                "language": "python",
                "runtime": f"python/{platform.python_version()}",
            },
            "resource": {
                "serviceName": self.service_name,
                "serviceVersion": self.service_version,
                "environment": self.environment,
            },
            "traces": traces,
        }
        if not self.api_key:
            self._completed[:0] = traces
            return {"status": "skipped", "traceCount": len(traces), "reason": "missing_api_key"}
        body = _json_bytes(batch)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": f"papaya-ai-python/{__version__}",
        }
        try:
            status, response_text = self.transport(self.endpoint, headers, body)
        except Exception as error:  # pragma: no cover - defensive transport boundary
            self._completed[:0] = traces
            if self.debug:
                print("[papaya] export failed", error, file=sys.stderr)
            return {
                "status": "failed",
                "traceCount": len(traces),
                "endpoint": self.endpoint,
                "error": str(error),
            }
        if status < 200 or status >= 300:
            self._completed[:0] = traces
            if self.debug:
                print(f"[papaya] export failed: {status} {response_text}", file=sys.stderr)
            return {
                "status": "failed",
                "traceCount": len(traces),
                "endpoint": self.endpoint,
                "httpStatus": status,
                "responseText": response_text,
            }
        return {
            "status": "sent",
            "traceCount": len(traces),
            "endpoint": self.endpoint,
            "httpStatus": status,
            "responseText": response_text,
        }


__all__ = ["Papaya", "CaptureMode", "SpanKind", "SpanStatus"]
