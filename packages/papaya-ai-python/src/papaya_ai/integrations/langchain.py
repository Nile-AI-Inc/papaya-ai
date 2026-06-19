from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from papaya_ai.core import Papaya, SpanKind

try:  # LangChain is optional for the base package and tests.
    from langchain_core.callbacks import BaseCallbackHandler
except Exception:  # pragma: no cover - exercised when langchain-core is absent
    BaseCallbackHandler = object  # type: ignore[assignment,misc]


def _id_text(value: Any) -> str:
    return str(value)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    for method in ("model_dump", "dict"):
        fn = getattr(value, method, None)
        if callable(fn):
            try:
                return _jsonable(fn())
            except TypeError:
                pass
    if hasattr(value, "content"):
        return {
            "role": _role_from_message(value),
            "content": _jsonable(getattr(value, "content")),
            "type": getattr(value, "type", value.__class__.__name__),
        }
    return str(value)


def _role_from_message(message: Any) -> str:
    raw = getattr(message, "role", None) or getattr(message, "type", None) or message.__class__.__name__.lower()
    text = str(raw).lower()
    if text in {"human", "user", "humanmessage"}:
        return "user"
    if text in {"ai", "assistant", "aimessage", "model"}:
        return "assistant"
    if text in {"system", "systemmessage", "developer"}:
        return "system"
    if text in {"tool", "toolmessage", "function"}:
        return "tool"
    return text


def _message_batch(messages: Any) -> Any:
    if isinstance(messages, (list, tuple)):
        return [_message_batch(item) for item in messages]
    if hasattr(messages, "content"):
        return {
            "role": _role_from_message(messages),
            "content": _jsonable(getattr(messages, "content")),
            "type": getattr(messages, "type", messages.__class__.__name__),
        }
    return _jsonable(messages)


def _serialized_name(serialized: Any, fallback: str) -> str:
    if isinstance(serialized, dict):
        name = serialized.get("name")
        if isinstance(name, str) and name:
            return name
        identifier = serialized.get("id")
        if isinstance(identifier, (list, tuple)) and identifier:
            return str(identifier[-1])
        if isinstance(identifier, str) and identifier:
            return identifier
    return fallback


def _usage_from_dict(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    usage = value.get("usage") or value.get("token_usage") or value.get("usage_metadata") or value.get("tokenUsage")
    if usage is None and any(key in value for key in ("input_tokens", "prompt_tokens", "inputTokens", "promptTokenCount", "output_tokens", "completion_tokens", "outputTokens", "candidatesTokenCount", "total_tokens", "totalTokens", "totalTokenCount")):
        usage = value
    if isinstance(usage, dict):
        input_tokens = usage.get("input_tokens") or usage.get("prompt_tokens") or usage.get("inputTokens") or usage.get("promptTokenCount")
        output_tokens = usage.get("output_tokens") or usage.get("completion_tokens") or usage.get("outputTokens") or usage.get("candidatesTokenCount")
        total_tokens = usage.get("total_tokens") or usage.get("totalTokens") or usage.get("totalTokenCount")
        return {
            "inputTokens": input_tokens,
            "outputTokens": output_tokens,
            "totalTokens": total_tokens if total_tokens is not None else (input_tokens or 0) + (output_tokens or 0),
            "cacheReadInputTokens": usage.get("cache_read_input_tokens") or usage.get("cached_input_tokens") or usage.get("cacheReadInputTokens"),
            "cacheCreationInputTokens": usage.get("cache_creation_input_tokens") or usage.get("cacheCreationInputTokens"),
            "reasoningTokens": usage.get("reasoning_tokens") or usage.get("reasoningTokens"),
            "costUsd": usage.get("cost_usd") or usage.get("costUsd"),
            "pricingSource": "provider" if usage.get("cost_usd") or usage.get("costUsd") else None,
        }
    return None


def _usage_from_nested(value: Any) -> dict[str, Any] | None:
    usage = _usage_from_dict(value)
    if usage:
        return usage
    if isinstance(value, dict):
        for item in value.values():
            usage = _usage_from_nested(item)
            if usage:
                return usage
    if isinstance(value, (list, tuple)):
        for item in value:
            usage = _usage_from_nested(item)
            if usage:
                return usage
    return None


def _usage_from_response(response: Any) -> dict[str, Any] | None:
    for attr in ("llm_output", "response_metadata", "usage_metadata"):
        usage = _usage_from_dict(getattr(response, attr, None))
        if usage:
            return usage
    return _usage_from_nested(_jsonable(response))


def _generations_from_response(response: Any) -> Any:
    generations = getattr(response, "generations", None)
    if generations is not None:
        return _jsonable(generations)
    return _jsonable(response)


@dataclass
class _RunState:
    trace: dict[str, Any]
    span: dict[str, Any]
    kind: SpanKind
    parent_run_id: str | None


class PapayaCallbackHandler(BaseCallbackHandler):  # type: ignore[misc,valid-type]
    """LangChain/LangGraph callback handler that emits Papaya native traces.

    The handler intentionally lives beside provider wrappers, not inside them:
    framework callbacks capture the agent tree, provider wrappers capture direct
    SDK calls. A customer should usually choose one path for a given execution.
    """

    def __init__(
        self,
        papaya: Papaya,
        *,
        workflow_key: str | None = None,
        workflow_label: str | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
        organization_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        capture_llm: bool = True,
    ):
        super().__init__()
        self.papaya = papaya
        self.workflow_key = workflow_key
        self.workflow_label = workflow_label
        self.session_id = session_id
        self.user_id = user_id
        self.organization_id = organization_id
        self.metadata = metadata or {}
        self.capture_llm = capture_llm
        self._runs: dict[str, _RunState] = {}

    def _start(
        self,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        name: str,
        kind: SpanKind,
        input_value: Any = None,
        metadata: dict[str, Any] | None = None,
        model_ref: dict[str, Any] | None = None,
    ) -> None:
        run_key = _id_text(run_id)
        parent_key = _id_text(parent_run_id) if parent_run_id is not None else None
        parent = self._runs.get(parent_key) if parent_key else None
        attributes = {
            "framework": "langchain",
            "langchainRunId": run_key,
            "langchainParentRunId": parent_key,
            "metadata": {**self.metadata, **(metadata or {})},
        }
        span_id = f"span_lc_{run_key}"
        if parent is None:
            trace = self.papaya.start_trace(
                {
                    "traceId": f"trace_lc_{run_key}",
                    "runId": f"run_lc_{run_key}",
                    "sessionId": self.session_id,
                    "userId": self.user_id,
                    "organizationId": self.organization_id,
                    "workflowKey": self.workflow_key or "langchain_callback",
                    "workflowLabel": self.workflow_label or "LangChain callback run",
                    "metadata": {**self.metadata, **(metadata or {}), "framework": "langchain"},
                },
                root_span_id=span_id,
                root_name=name,
                root_kind="workflow" if kind in {"agent", "workflow"} else kind,
                input_value=input_value,
                attributes=attributes,
            )
            span = trace["spans"][0]
        else:
            trace = parent.trace
            span = self.papaya.start_span(
                trace=trace,
                span_id=span_id,
                parent_span_id=parent.span["spanId"],
                name=name,
                kind=kind,
                input_value=input_value,
                model_ref=model_ref,
                attributes=attributes,
            )
        self._runs[run_key] = _RunState(trace=trace, span=span, kind=kind, parent_run_id=parent_key)

    def _finish(
        self,
        *,
        run_id: Any,
        status: str,
        output_value: Any = None,
        usage: dict[str, Any] | None = None,
        model_used: str | None = None,
        error: BaseException | Any | None = None,
    ) -> None:
        run_key = _id_text(run_id)
        state = self._runs.pop(run_key, None)
        if state is None:
            return
        if state.parent_run_id is None:
            self.papaya.finish_trace(state.trace, status, output_value=output_value, error=error)
        else:
            self.papaya.finish_span(
                state.span,
                status,
                output_value=output_value,
                usage=usage,
                model_used=model_used,
                error=error,
            )

    def on_chain_start(self, serialized: Any, inputs: Any, *, run_id: Any, parent_run_id: Any = None, tags: list[str] | None = None, metadata: dict[str, Any] | None = None, name: str | None = None, **kwargs: Any) -> None:
        self._start(
            run_id=run_id,
            parent_run_id=parent_run_id,
            name=name or _serialized_name(serialized, "langchain.chain"),
            kind="agent" if parent_run_id is None else "workflow",
            input_value=_jsonable(inputs),
            metadata={**(metadata or {}), "tags": tags or []},
        )

    def on_chain_end(self, outputs: Any, *, run_id: Any, **kwargs: Any) -> None:
        self._finish(run_id=run_id, status="success", output_value=_jsonable(outputs))

    def on_chain_error(self, error: BaseException, *, run_id: Any, **kwargs: Any) -> None:
        self._finish(run_id=run_id, status="failed", error=error)

    def on_llm_start(self, serialized: Any, prompts: list[str], *, run_id: Any, parent_run_id: Any = None, tags: list[str] | None = None, metadata: dict[str, Any] | None = None, invocation_params: dict[str, Any] | None = None, **kwargs: Any) -> None:
        if not self.capture_llm:
            return
        model = (invocation_params or {}).get("model") or (invocation_params or {}).get("model_name") or (metadata or {}).get("model")
        self._start(
            run_id=run_id,
            parent_run_id=parent_run_id,
            name=_serialized_name(serialized, "langchain.llm"),
            kind="llm",
            input_value=prompts,
            metadata={**(metadata or {}), "tags": tags or [], "invocationParams": invocation_params or {}},
            model_ref={"provider": "langchain", "requested": model} if model else {"provider": "langchain"},
        )

    def on_chat_model_start(self, serialized: Any, messages: list[list[Any]], *, run_id: Any, parent_run_id: Any = None, tags: list[str] | None = None, metadata: dict[str, Any] | None = None, invocation_params: dict[str, Any] | None = None, **kwargs: Any) -> None:
        if not self.capture_llm:
            return
        model = (invocation_params or {}).get("model") or (invocation_params or {}).get("model_name") or (metadata or {}).get("model")
        self._start(
            run_id=run_id,
            parent_run_id=parent_run_id,
            name=_serialized_name(serialized, "langchain.chat_model"),
            kind="llm",
            input_value=_message_batch(messages),
            metadata={**(metadata or {}), "tags": tags or [], "invocationParams": invocation_params or {}},
            model_ref={"provider": "langchain", "requested": model} if model else {"provider": "langchain"},
        )

    def on_llm_end(self, response: Any, *, run_id: Any, **kwargs: Any) -> None:
        self._finish(
            run_id=run_id,
            status="success",
            output_value=_generations_from_response(response),
            usage=_usage_from_response(response),
        )

    def on_llm_error(self, error: BaseException, *, run_id: Any, **kwargs: Any) -> None:
        self._finish(run_id=run_id, status="failed", error=error)

    def on_tool_start(self, serialized: Any, input_str: str, *, run_id: Any, parent_run_id: Any = None, tags: list[str] | None = None, metadata: dict[str, Any] | None = None, name: str | None = None, **kwargs: Any) -> None:
        tool_name = name or _serialized_name(serialized, "langchain.tool")
        self._start(
            run_id=run_id,
            parent_run_id=parent_run_id,
            name=tool_name,
            kind="tool",
            input_value=input_str,
            metadata={**(metadata or {}), "tags": tags or [], "tool": tool_name},
        )
        state = self._runs.get(_id_text(run_id))
        if state is not None:
            state.span["tool"] = {"name": tool_name}

    def on_tool_end(self, output: Any, *, run_id: Any, **kwargs: Any) -> None:
        self._finish(run_id=run_id, status="success", output_value=_jsonable(output))

    def on_tool_error(self, error: BaseException, *, run_id: Any, **kwargs: Any) -> None:
        self._finish(run_id=run_id, status="failed", error=error)

    def on_retriever_start(self, serialized: Any, query: str, *, run_id: Any, parent_run_id: Any = None, tags: list[str] | None = None, metadata: dict[str, Any] | None = None, name: str | None = None, **kwargs: Any) -> None:
        self._start(
            run_id=run_id,
            parent_run_id=parent_run_id,
            name=name or _serialized_name(serialized, "langchain.retriever"),
            kind="retrieval",
            input_value=query,
            metadata={**(metadata or {}), "tags": tags or []},
        )

    def on_retriever_end(self, documents: Any, *, run_id: Any, **kwargs: Any) -> None:
        self._finish(run_id=run_id, status="success", output_value=_jsonable(documents))

    def on_retriever_error(self, error: BaseException, *, run_id: Any, **kwargs: Any) -> None:
        self._finish(run_id=run_id, status="failed", error=error)


__all__ = ["PapayaCallbackHandler"]
