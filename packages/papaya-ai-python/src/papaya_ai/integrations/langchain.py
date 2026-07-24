from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from papaya_ai.core import Papaya, SpanKind, _byte_length, _jsonable

try:  # LangChain is optional for the base package and tests.
    from langchain_core.callbacks import BaseCallbackHandler
except Exception:  # pragma: no cover - exercised when langchain-core is absent
    BaseCallbackHandler = object  # type: ignore[assignment,misc]


def _id_text(value: Any) -> str:
    return str(value)


def _role_from_message(message: Any) -> str:
    if isinstance(message, dict):
        raw = message.get("role") or message.get("type") or "message"
    else:
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


def _present_text(value: Any) -> str | None:
    if not isinstance(value, str) or value == "" or value == "undefined":
        return None
    return value


def _message_value(message: Any, serialized: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if isinstance(message, dict) and key in message:
            return message[key]
        if not isinstance(message, dict) and hasattr(message, key):
            return getattr(message, key)
        if key in serialized:
            return serialized[key]
    return None


def _compact_tool_call(value: Any) -> Any:
    serialized = _jsonable(value)
    if not isinstance(serialized, dict):
        return serialized
    function = serialized.get("function") if isinstance(serialized.get("function"), dict) else {}
    name = _present_text(serialized.get("name")) or _present_text(function.get("name"))
    result: dict[str, Any] = {}
    call_id = _present_text(serialized.get("id"))
    if call_id:
        result["id"] = call_id
    if name:
        result["name"] = name
    if "args" in serialized:
        result["args"] = serialized["args"]
    elif "arguments" in function:
        result["args"] = function["arguments"]
    error = serialized.get("error")
    if error not in (None, ""):
        result["error"] = error
    return result or serialized


def _compact_message(message: Any) -> Any:
    serialized_value = _jsonable(message)
    if not isinstance(serialized_value, dict):
        return serialized_value
    serialized = serialized_value
    additional = _message_value(message, serialized, "additional_kwargs", "additionalKwargs")
    if not isinstance(additional, dict):
        additional = {}
    content = _message_value(message, serialized, "content")
    if content is None and "content" not in serialized:
        content = ""
    result: dict[str, Any] = {
        "role": _role_from_message(message),
        "content": _jsonable(content),
    }

    name = _present_text(_message_value(message, serialized, "name"))
    message_id = _present_text(_message_value(message, serialized, "id"))
    tool_calls = _message_value(message, serialized, "tool_calls", "toolCalls")
    if tool_calls is None:
        tool_calls = additional.get("tool_calls") or additional.get("toolCalls")
    invalid_tool_calls = _message_value(message, serialized, "invalid_tool_calls", "invalidToolCalls")
    tool_call_id = _present_text(_message_value(message, serialized, "tool_call_id", "toolCallId"))
    artifact = _message_value(message, serialized, "artifact")

    if name:
        result["name"] = name
    if message_id:
        result["id"] = message_id
    if isinstance(tool_calls, (list, tuple)) and tool_calls:
        result["toolCalls"] = [_compact_tool_call(call) for call in tool_calls]
    if isinstance(invalid_tool_calls, (list, tuple)) and invalid_tool_calls:
        result["invalidToolCalls"] = [_compact_tool_call(call) for call in invalid_tool_calls]
    if tool_call_id:
        result["toolCallId"] = tool_call_id
    if artifact is not None:
        result["artifact"] = _jsonable(artifact)
    return result


def _compact_message_batches(messages: Any) -> list[list[Any]]:
    if not isinstance(messages, (list, tuple)):
        return [[_compact_message(messages)]]
    if all(isinstance(item, (list, tuple)) for item in messages):
        return [[_compact_message(message) for message in batch] for batch in messages]
    return [[_compact_message(message) for message in messages]]


def _compact_generations(response: Any) -> Any:
    generations = getattr(response, "generations", None)
    if generations is None and isinstance(response, dict):
        generations = response.get("generations")
    if generations is None:
        return _jsonable(response)
    result: list[list[Any]] = []
    for batch in generations:
        items = batch if isinstance(batch, (list, tuple)) else [batch]
        compact_batch: list[Any] = []
        for generation in items:
            message = getattr(generation, "message", None)
            if message is None and isinstance(generation, dict):
                message = generation.get("message")
            if message is not None:
                compact_batch.append(_compact_message(message))
                continue
            text = getattr(generation, "text", None)
            if text is None and isinstance(generation, dict):
                text = generation.get("text")
            compact_batch.append({"role": "assistant", "content": text} if isinstance(text, str) else _jsonable(generation))
        result.append(compact_batch)
    return result


def _message_batches_from_payload(payload: dict[str, Any] | None) -> list[list[Any]] | None:
    value = (payload or {}).get("value")
    if not isinstance(value, list) or not all(isinstance(batch, list) for batch in value):
        return None
    return value


def _stable_value(value: Any) -> Any:
    if isinstance(value, list):
        return [_stable_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _stable_value(value[key]) for key in sorted(value)}
    return value


def _messages_equal(left: Any, right: Any, *, ignore_source_message_id: bool = False) -> bool:
    def comparable(value: Any) -> Any:
        if ignore_source_message_id and isinstance(value, dict):
            value = {key: item for key, item in value.items() if key != "id"}
        return _stable_value(value)

    return comparable(left) == comparable(right)


def _common_prefix_length(left: list[Any], right: list[Any]) -> int:
    index = 0
    while index < min(len(left), len(right)) and _messages_equal(left[index], right[index]):
        index += 1
    return index


def _boundary_overlap_length(previous_output: list[Any], suffix: list[Any]) -> int:
    for size in range(min(len(previous_output), len(suffix)), 0, -1):
        output_tail = previous_output[-size:]
        suffix_head = suffix[:size]
        if all(
            _messages_equal(message, suffix_head[index], ignore_source_message_id=True)
            for index, message in enumerate(output_tail)
        ):
            return size
    return 0


def _compact_repeated_history(
    previous: "_LaneSnapshot",
    current: list[list[Any]],
) -> list[list[Any]] | None:
    if len(previous.input) != len(current):
        return None
    retained: list[list[Any]] = []
    for index, batch in enumerate(current):
        previous_input = previous.input[index] if index < len(previous.input) else []
        previous_output = previous.output[index] if index < len(previous.output) else []
        repeated_input = _common_prefix_length(previous_input, batch)
        suffix = batch[repeated_input:]
        repeated_output = _boundary_overlap_length(previous_output, suffix)
        retained.append(suffix[repeated_output:])
    return retained


def _payload_with_value(payload: dict[str, Any], value: Any) -> dict[str, Any]:
    return {**payload, "value": value, "byteLength": _byte_length(value)}


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


def _first_present(value: Any, *keys: str) -> Any:
    if not isinstance(value, dict):
        return None
    for key in keys:
        item = value.get(key)
        if item is not None:
            return item
    return None


def _usage_from_dict(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    usage = _first_present(value, "usage", "token_usage", "usage_metadata", "usageMetadata", "tokenUsage")
    if usage is None and any(key in value for key in (
        "input_tokens", "prompt_tokens", "inputTokens", "promptTokenCount", "prompt_token_count",
        "output_tokens", "completion_tokens", "outputTokens", "candidatesTokenCount", "candidates_token_count",
        "total_tokens", "totalTokens", "totalTokenCount", "total_token_count",
    )):
        usage = value
    if isinstance(usage, dict):
        input_tokens = _first_present(
            usage,
            "input_tokens",
            "prompt_tokens",
            "inputTokens",
            "promptTokenCount",
            "prompt_token_count",
        )
        output_tokens = _first_present(
            usage,
            "output_tokens",
            "completion_tokens",
            "outputTokens",
            "candidatesTokenCount",
            "candidates_token_count",
        )
        total_tokens = _first_present(
            usage,
            "total_tokens",
            "totalTokens",
            "totalTokenCount",
            "total_token_count",
        )
        input_details = _first_present(usage, "input_token_details", "inputTokenDetails")
        output_details = _first_present(usage, "output_token_details", "outputTokenDetails")
        cache_read_input_tokens = _first_present(
            usage,
            "cache_read_input_tokens",
            "cached_input_tokens",
            "cacheReadInputTokens",
            "cached_content_token_count",
        )
        if cache_read_input_tokens is None:
            cache_read_input_tokens = _first_present(input_details, "cache_read", "cacheRead")
        cache_creation_input_tokens = _first_present(
            usage,
            "cache_creation_input_tokens",
            "cacheCreationInputTokens",
        )
        if cache_creation_input_tokens is None:
            cache_creation_input_tokens = _first_present(input_details, "cache_creation", "cacheCreation")
        reasoning_tokens = _first_present(usage, "reasoning_tokens", "reasoningTokens")
        if reasoning_tokens is None:
            reasoning_tokens = _first_present(output_details, "reasoning")
        cost_usd = _first_present(usage, "cost_usd", "costUsd")
        return {
            "inputTokens": input_tokens,
            "outputTokens": output_tokens,
            "totalTokens": total_tokens if total_tokens is not None else (input_tokens or 0) + (output_tokens or 0),
            "cacheReadInputTokens": cache_read_input_tokens,
            "cacheCreationInputTokens": cache_creation_input_tokens,
            "reasoningTokens": reasoning_tokens,
            "costUsd": cost_usd,
            "pricingSource": "provider" if cost_usd is not None else None,
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


def _model_from_response(value: Any) -> str | None:
    serialized = _jsonable(value)

    def visit(item: Any) -> str | None:
        if isinstance(item, dict):
            for key in ("model", "model_name", "modelName"):
                model = item.get(key)
                if isinstance(model, str) and model:
                    return model
            for child in item.values():
                model = visit(child)
                if model:
                    return model
        elif isinstance(item, list):
            for child in item:
                model = visit(child)
                if model:
                    return model
        return None

    return visit(serialized)


def _response_attributes_from_response(value: Any) -> dict[str, Any] | None:
    serialized = _jsonable(value)

    def find_metadata(item: Any) -> dict[str, Any] | None:
        if isinstance(item, dict):
            for key in ("response_metadata", "responseMetadata", "generation_info", "generationInfo"):
                metadata = item.get(key)
                if isinstance(metadata, dict):
                    return metadata
            for child in item.values():
                metadata = find_metadata(child)
                if metadata:
                    return metadata
        elif isinstance(item, list):
            for child in item:
                metadata = find_metadata(child)
                if metadata:
                    return metadata
        return None

    response_metadata = find_metadata(serialized)
    if not response_metadata:
        return None
    finish_reason = _present_text(_first_present(
        response_metadata,
        "finish_reason",
        "finishReason",
        "stop_reason",
        "stopReason",
    ))
    excluded = {
        "finish_reason", "finishReason", "stop_reason", "stopReason",
        "model", "model_name", "modelName",
        "usage", "usage_metadata", "usageMetadata", "token_usage", "tokenUsage",
    }
    provider_metadata = {
        key: item
        for key, item in response_metadata.items()
        if key not in excluded and item is not None
    }
    attributes: dict[str, Any] = {}
    if finish_reason:
        attributes["finishReason"] = finish_reason
    if provider_metadata:
        attributes["langchainResponseMetadata"] = provider_metadata
    return attributes or None


@dataclass
class _LaneSnapshot:
    input: list[list[Any]]
    output: list[list[Any]]


@dataclass
class _LaneState:
    active_run_ids: set[str] = field(default_factory=set)
    previous: _LaneSnapshot | None = None


@dataclass
class _LlmState:
    lane_key: str
    full_input: list[list[Any]] | None
    overlapped: bool = False


@dataclass
class _RunState:
    trace: dict[str, Any]
    span: dict[str, Any]
    kind: SpanKind
    parent_run_id: str | None
    llm: _LlmState | None = None


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
        self._lanes: dict[str, _LaneState] = {}

    def _start(
        self,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        name: str,
        kind: SpanKind,
        input_value: Any = None,
        input_payload: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        model_ref: dict[str, Any] | None = None,
        llm: _LlmState | None = None,
    ) -> None:
        run_key = _id_text(run_id)
        parent_key = _id_text(parent_run_id) if parent_run_id is not None else None
        parent = self._runs.get(parent_key) if parent_key else None
        attributes = {
            "framework": "langchain",
            "langchainRunId": run_key,
            "metadata": {**self.metadata, **(metadata or {})},
        }
        if parent_key:
            attributes["langchainParentRunId"] = parent_key
        if parent is None:
            trace = self.papaya.start_trace(
                {
                    "sessionId": self.session_id,
                    "userId": self.user_id,
                    "organizationId": self.organization_id,
                    "workflowKey": self.workflow_key or "langchain_callback",
                    "workflowLabel": self.workflow_label or "LangChain callback run",
                    "metadata": {**self.metadata, **(metadata or {}), "framework": "langchain"},
                },
                root_name=name,
                root_kind="workflow" if kind in {"agent", "workflow"} else kind,
                input_value=input_value,
                input_payload=input_payload,
                model_ref=model_ref,
                attributes=attributes,
            )
            span = trace["spans"][0]
        else:
            trace = parent.trace
            span = self.papaya.start_span(
                trace=trace,
                parent_span_id=parent.span["spanId"],
                name=name,
                kind=kind,
                input_value=input_value,
                input_payload=input_payload,
                model_ref=model_ref,
                attributes=attributes,
            )
        self._runs[run_key] = _RunState(
            trace=trace,
            span=span,
            kind=kind,
            parent_run_id=parent_key,
            llm=llm,
        )

    def _finish(
        self,
        *,
        run_id: Any,
        status: str,
        output_value: Any = None,
        output_payload: dict[str, Any] | None = None,
        output_messages: list[list[Any]] | None = None,
        usage: dict[str, Any] | None = None,
        model_used: str | None = None,
        response_attributes: dict[str, Any] | None = None,
        error: BaseException | Any | None = None,
    ) -> None:
        run_key = _id_text(run_id)
        state = self._runs.pop(run_key, None)
        if state is None:
            return
        if response_attributes:
            state.span["attributes"] = {
                **state.span.get("attributes", {}),
                **response_attributes,
            }
        if state.llm is not None:
            lane = self._lanes.get(state.llm.lane_key)
            if lane is not None:
                lane.active_run_ids.discard(run_key)
                if (
                    status == "success"
                    and not state.llm.overlapped
                    and not lane.active_run_ids
                    and state.llm.full_input is not None
                ):
                    lane.previous = _LaneSnapshot(
                        input=state.llm.full_input,
                        output=output_messages or [],
                    )
                else:
                    lane.previous = None
        child_lane_key = f"parent:{run_key}"
        child_lane = self._lanes.get(child_lane_key)
        if child_lane is not None and not child_lane.active_run_ids:
            self._lanes.pop(child_lane_key, None)
        root_lane_key = f"root:{run_key}"
        root_lane = self._lanes.get(root_lane_key)
        if root_lane is not None and not root_lane.active_run_ids:
            self._lanes.pop(root_lane_key, None)
        if state.parent_run_id is None:
            self.papaya.finish_trace(
                state.trace,
                status,
                output_value=output_value,
                output_payload=output_payload,
                usage=usage,
                model_used=model_used,
                error=error,
            )
        else:
            self.papaya.finish_span(
                state.span,
                status,
                output_value=output_value,
                output_payload=output_payload,
                usage=usage,
                model_used=model_used,
                error=error,
            )

    def _capture_chat_input(
        self,
        *,
        run_id: Any,
        parent_run_id: Any,
        messages: Any,
    ) -> tuple[dict[str, Any], _LlmState]:
        run_key = _id_text(run_id)
        parent_key = _id_text(parent_run_id) if parent_run_id is not None else None
        lane_key = f"parent:{parent_key}" if parent_key else f"root:{run_key}"
        lane = self._lanes.setdefault(lane_key, _LaneState())

        if lane.active_run_ids:
            for active_run_id in lane.active_run_ids:
                active = self._runs.get(active_run_id)
                if active is not None and active.llm is not None:
                    active.llm.overlapped = True
            lane.previous = None

        compact = _compact_message_batches(messages)
        full_payload = self.papaya.capture_payload(compact)
        full_input = _message_batches_from_payload(full_payload)
        retained = (
            _compact_repeated_history(lane.previous, full_input)
            if full_input is not None and not lane.active_run_ids and lane.previous is not None
            else None
        )
        input_payload = _payload_with_value(full_payload, retained) if retained is not None else full_payload
        lane.active_run_ids.add(run_key)
        return input_payload, _LlmState(
            lane_key=lane_key,
            full_input=full_input,
            overlapped=len(lane.active_run_ids) > 1,
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
        input_payload, llm = self._capture_chat_input(
            run_id=run_id,
            parent_run_id=parent_run_id,
            messages=messages,
        )
        self._start(
            run_id=run_id,
            parent_run_id=parent_run_id,
            name=_serialized_name(serialized, "langchain.chat_model"),
            kind="llm",
            input_payload=input_payload,
            metadata={**(metadata or {}), "tags": tags or [], "invocationParams": invocation_params or {}},
            model_ref={"provider": "langchain", "requested": model} if model else {"provider": "langchain"},
            llm=llm,
        )

    def on_llm_end(self, response: Any, *, run_id: Any, **kwargs: Any) -> None:
        usage = _usage_from_response(response)
        model_used = _model_from_response(response)
        response_attributes = _response_attributes_from_response(response)
        compact_output = _compact_generations(response)
        output_payload = self.papaya.capture_payload(compact_output)
        self._finish(
            run_id=run_id,
            status="success",
            output_payload=output_payload,
            output_messages=_message_batches_from_payload(output_payload),
            usage=usage,
            model_used=model_used,
            response_attributes=response_attributes,
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
