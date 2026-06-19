import { BaseCallbackHandler } from "@langchain/core/callbacks/base";

import {
  Papaya,
  type PapayaTrace,
  type PapayaTraceSpan,
  type RunOptions,
  type SpanKind,
  type SpanStatus,
} from "./index.js";

type Serialized = unknown;
type RunState = {
  trace: PapayaTrace;
  span: PapayaTraceSpan;
  kind: SpanKind;
  parentRunId?: string;
};

export type PapayaCallbackHandlerOptions = RunOptions & {
  metadata?: Record<string, unknown>;
  captureLLM?: boolean;
};

const idText = (value: unknown): string => String(value);

const isRecord = (value: unknown): value is Record<string, unknown> =>
  Boolean(value) && typeof value === "object" && !Array.isArray(value);

const jsonable = (value: unknown, seen = new WeakSet<object>()): unknown => {
  if (value === null || typeof value === "string" || typeof value === "number" || typeof value === "boolean") return value;
  if (typeof value === "bigint") return value.toString();
  if (Array.isArray(value)) return value.map((item) => jsonable(item, seen));
  if (!isRecord(value)) return String(value);
  if (seen.has(value)) return "[Circular]";
  seen.add(value);
  if ("content" in value) {
    return {
      role: roleFromMessage(value),
      content: jsonable(value.content, seen),
      type: typeof value.type === "string" ? value.type : value.constructor?.name,
    };
  }
  if (typeof value.toJSON === "function") {
    try {
      return jsonable(value.toJSON(), seen);
    } catch {
      // Fall through to enumerable fields.
    }
  }
  return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, jsonable(item, seen)]));
};

const roleFromMessage = (message: Record<string, unknown>): string => {
  const raw = typeof message.role === "string"
    ? message.role
    : typeof message.type === "string"
      ? message.type
      : typeof message._getType === "function"
        ? String(message._getType())
        : message.constructor?.name?.toLowerCase();
  const text = String(raw ?? "message").toLowerCase();
  if (["human", "user", "humanmessage"].includes(text)) return "user";
  if (["ai", "assistant", "aimessage", "model"].includes(text)) return "assistant";
  if (["system", "systemmessage", "developer"].includes(text)) return "system";
  if (["tool", "toolmessage", "function"].includes(text)) return "tool";
  return text;
};

const messageBatch = (messages: unknown): unknown => {
  if (Array.isArray(messages)) return messages.map((item) => messageBatch(item));
  if (isRecord(messages) && "content" in messages) {
    return {
      role: roleFromMessage(messages),
      content: jsonable(messages.content),
      type: typeof messages.type === "string" ? messages.type : messages.constructor?.name,
    };
  }
  return jsonable(messages);
};

const serializedName = (serialized: Serialized, fallback: string): string => {
  if (typeof serialized === "string" && serialized) return serialized;
  if (isRecord(serialized)) {
    if (typeof serialized.name === "string" && serialized.name) return serialized.name;
    if (Array.isArray(serialized.id) && serialized.id.length > 0) return String(serialized.id.at(-1));
    if (typeof serialized.id === "string" && serialized.id) return serialized.id;
  }
  return fallback;
};

const numberValue = (value: unknown): number | undefined =>
  typeof value === "number" && Number.isFinite(value) ? value : undefined;

const usageFromRecord = (value: unknown): Record<string, number | string | undefined> | undefined => {
  if (!isRecord(value)) return undefined;
  const usage = isRecord(value.usage) ? value.usage
    : isRecord(value.token_usage) ? value.token_usage
      : isRecord(value.usage_metadata) ? value.usage_metadata
        : isRecord(value.usageMetadata) ? value.usageMetadata
          : value;
  if (!isRecord(usage)) return undefined;
  const hasUsageKeys = [
    "input_tokens", "prompt_tokens", "inputTokens", "promptTokenCount",
    "output_tokens", "completion_tokens", "outputTokens", "candidatesTokenCount",
    "total_tokens", "totalTokens", "totalTokenCount",
  ].some((key) => key in usage);
  if (!hasUsageKeys) return undefined;
  const inputTokens = numberValue(usage.input_tokens ?? usage.prompt_tokens ?? usage.inputTokens ?? usage.promptTokenCount);
  const outputTokens = numberValue(usage.output_tokens ?? usage.completion_tokens ?? usage.outputTokens ?? usage.candidatesTokenCount);
  return {
    inputTokens,
    outputTokens,
    totalTokens: numberValue(usage.total_tokens ?? usage.totalTokens ?? usage.totalTokenCount) ?? (inputTokens ?? 0) + (outputTokens ?? 0),
    cacheReadInputTokens: numberValue(usage.cache_read_input_tokens ?? usage.cached_input_tokens ?? usage.cacheReadInputTokens),
    cacheCreationInputTokens: numberValue(usage.cache_creation_input_tokens ?? usage.cacheCreationInputTokens),
    reasoningTokens: numberValue(usage.reasoning_tokens ?? usage.reasoningTokens),
    costUsd: numberValue(usage.cost_usd ?? usage.costUsd),
    pricingSource: usage.cost_usd || usage.costUsd ? "provider" : undefined,
  };
};

const usageFromNested = (value: unknown): Record<string, number | string | undefined> | undefined => {
  const direct = usageFromRecord(value);
  if (direct) return direct;
  if (Array.isArray(value)) {
    for (const item of value) {
      const usage = usageFromNested(item);
      if (usage) return usage;
    }
  }
  if (isRecord(value)) {
    for (const item of Object.values(value)) {
      const usage = usageFromNested(item);
      if (usage) return usage;
    }
  }
  return undefined;
};

const modelLabel = (serialized: Serialized, extraParams?: Record<string, unknown>, metadata?: Record<string, unknown>): string | undefined => {
  const invocation = isRecord(extraParams?.invocation_params) ? extraParams.invocation_params : extraParams;
  const value = invocation?.model ?? invocation?.model_name ?? invocation?.modelName ?? metadata?.ls_model_name ?? metadata?.model;
  if (typeof value === "string" && value) return value;
  return serializedName(serialized, "");
};

const modelFromResponse = (value: unknown): string | undefined => {
  if (!isRecord(value)) return undefined;
  const candidates = [
    value.model,
    value.model_name,
    value.modelName,
    isRecord(value.llmOutput) ? value.llmOutput.model_name ?? value.llmOutput.model : undefined,
    isRecord(value.response_metadata) ? value.response_metadata.model_name ?? value.response_metadata.model : undefined,
    isRecord(value.responseMetadata) ? value.responseMetadata.model_name ?? value.responseMetadata.model : undefined,
  ];
  const direct = candidates.find((item) => typeof item === "string" && item);
  if (typeof direct === "string") return direct;
  for (const item of Object.values(value)) {
    const nested = modelFromResponse(item);
    if (nested) return nested;
  }
  return undefined;
};

const generationsFromResponse = (value: unknown): unknown =>
  isRecord(value) && "generations" in value ? jsonable(value.generations) : jsonable(value);

const knownChainRunTypes = new Set(["chain", "llm", "tool", "retriever", "parser", "prompt", "router"]);

export class PapayaCallbackHandler extends BaseCallbackHandler {
  name = "PapayaCallbackHandler";
  ignoreLLM = false;
  ignoreChain = false;
  ignoreAgent = false;
  ignoreRetriever = false;
  ignoreCustomEvent = true;
  raiseError = false;
  awaitHandlers = false;

  private readonly papaya: Papaya;
  private readonly options: PapayaCallbackHandlerOptions;
  private readonly runs = new Map<string, RunState>();

  constructor(papaya: Papaya, options: PapayaCallbackHandlerOptions = {}) {
    super();
    this.papaya = papaya;
    this.options = options;
  }

  copy(): PapayaCallbackHandler {
    return this;
  }

  private start(input: {
    runId: unknown;
    parentRunId?: unknown;
    name: string;
    kind: SpanKind;
    inputValue?: unknown;
    metadata?: Record<string, unknown>;
    modelRef?: { provider?: string; requested?: string };
  }): void {
    const runId = idText(input.runId);
    const parentRunId = input.parentRunId === undefined ? undefined : idText(input.parentRunId);
    const parent = parentRunId ? this.runs.get(parentRunId) : undefined;
    const metadata = { ...(this.options.metadata ?? {}), ...(input.metadata ?? {}) };
    const attributes = {
      framework: "langchain",
      langchainRunId: runId,
      langchainParentRunId: parentRunId,
      metadata,
    };
    const spanId = `span_lc_${runId}`;

    if (!parent) {
      const trace = this.papaya.startTrace({
        traceId: this.options.traceId ?? `trace_lc_${runId}`,
        runId: this.options.runId ?? `run_lc_${runId}`,
        sessionId: this.options.sessionId,
        conversationId: this.options.conversationId,
        userId: this.options.userId,
        organizationId: this.options.organizationId,
        workflowKey: this.options.workflowKey ?? "langchain_callback",
        workflowLabel: this.options.workflowLabel ?? "LangChain callback run",
        conversational: this.options.conversational,
        metadata: { ...metadata, framework: "langchain" },
      }, {
        rootSpanId: spanId,
        rootName: input.name,
        rootKind: input.kind === "agent" ? "workflow" : input.kind,
        inputValue: input.inputValue,
        modelRef: input.modelRef,
        attributes,
      });
      this.runs.set(runId, { trace, span: trace.spans[0]!, kind: input.kind, parentRunId });
      return;
    }

    const span = this.papaya.startSpan({
      trace: parent.trace,
      spanId,
      parentSpanId: parent.span.spanId,
      name: input.name,
      kind: input.kind,
      inputValue: input.inputValue,
      modelRef: input.modelRef,
      attributes,
    });
    this.runs.set(runId, { trace: parent.trace, span, kind: input.kind, parentRunId });
  }

  private finish(input: {
    runId: unknown;
    status: SpanStatus;
    outputValue?: unknown;
    usage?: Record<string, number | string | undefined>;
    modelUsed?: string;
    error?: unknown;
  }): void {
    const runId = idText(input.runId);
    const state = this.runs.get(runId);
    if (!state) return;
    this.runs.delete(runId);
    if (!state.parentRunId) {
      this.papaya.finishTrace(state.trace, input.status, {
        outputValue: input.outputValue,
        usage: input.usage as never,
        modelUsed: input.modelUsed,
        error: input.error,
      });
      return;
    }
    this.papaya.finishSpan(state.span, input.status, {
      outputValue: input.outputValue,
      usage: input.usage as never,
      modelUsed: input.modelUsed,
      error: input.error,
    });
  }

  handleChainStart(serialized: Serialized, inputs: unknown, runId: string, arg4?: string, _tags?: string[], metadata?: Record<string, unknown>, arg7?: string, arg8?: string): void {
    const declaredOrder = (arg8 !== undefined && this.runs.has(arg8)) || (arg4 !== undefined && knownChainRunTypes.has(arg4));
    const parentRunId = declaredOrder ? arg8 : arg4;
    const runType = declaredOrder ? arg4 : arg7;
    const runName = declaredOrder ? arg7 : arg8;
    this.start({
      runId,
      parentRunId,
      name: runName ?? serializedName(serialized, runType ? `langchain.${runType}` : "langchain.chain"),
      kind: parentRunId ? "workflow" : "agent",
      inputValue: jsonable(inputs),
      metadata,
    });
  }

  handleChainEnd(outputs: unknown, runId: string): void {
    this.finish({ runId, status: "success", outputValue: jsonable(outputs) });
  }

  handleChainError(error: Error, runId: string): void {
    this.finish({ runId, status: "failed", error });
  }

  handleLLMStart(serialized: Serialized, prompts: string[], runId: string, parentRunId?: string, extraParams?: Record<string, unknown>, _tags?: string[], metadata?: Record<string, unknown>, runName?: string): void {
    if (this.options.captureLLM === false) return;
    const model = modelLabel(serialized, extraParams, metadata);
    this.start({
      runId,
      parentRunId,
      name: runName ?? serializedName(serialized, "langchain.llm"),
      kind: "llm",
      inputValue: prompts,
      metadata,
      modelRef: { provider: "langchain", requested: model },
    });
  }

  handleChatModelStart(serialized: Serialized, messages: unknown[][], runId: string, parentRunId?: string, extraParams?: Record<string, unknown>, _tags?: string[], metadata?: Record<string, unknown>, runName?: string): void {
    if (this.options.captureLLM === false) return;
    const model = modelLabel(serialized, extraParams, metadata);
    this.start({
      runId,
      parentRunId,
      name: runName ?? serializedName(serialized, "langchain.chat_model"),
      kind: "llm",
      inputValue: messageBatch(messages),
      metadata,
      modelRef: { provider: "langchain", requested: model },
    });
  }

  handleLLMEnd(output: unknown, runId: string): void {
    if (this.options.captureLLM === false) return;
    this.finish({
      runId,
      status: "success",
      outputValue: generationsFromResponse(output),
      usage: usageFromNested(output),
      modelUsed: modelFromResponse(output),
    });
  }

  handleLLMError(error: Error, runId: string): void {
    if (this.options.captureLLM === false) return;
    this.finish({ runId, status: "failed", error });
  }

  handleToolStart(serialized: Serialized, input: string, runId: string, parentRunId?: string, _tags?: string[], metadata?: Record<string, unknown>, runName?: string, toolCallId?: string): void {
    this.start({
      runId,
      parentRunId,
      name: runName ?? serializedName(serialized, "langchain.tool"),
      kind: "tool",
      inputValue: input,
      metadata: { ...metadata, toolCallId },
    });
  }

  handleToolEnd(output: unknown, runId: string): void {
    this.finish({ runId, status: "success", outputValue: jsonable(output) });
  }

  handleToolError(error: Error, runId: string): void {
    this.finish({ runId, status: "failed", error });
  }

  handleRetrieverStart(serialized: Serialized, query: string, runId: string, parentRunId?: string, _tags?: string[], metadata?: Record<string, unknown>, runName?: string): void {
    this.start({
      runId,
      parentRunId,
      name: runName ?? serializedName(serialized, "langchain.retriever"),
      kind: "retrieval",
      inputValue: query,
      metadata,
    });
  }

  handleRetrieverEnd(documents: unknown, runId: string): void {
    this.finish({ runId, status: "success", outputValue: jsonable(documents) });
  }

  handleRetrieverError(error: Error, runId: string): void {
    this.finish({ runId, status: "failed", error });
  }
}

export { PapayaCallbackHandler as PapayaLangChainCallbackHandler };
