import { AsyncLocalStorage } from "node:async_hooks";
import { randomUUID } from "node:crypto";

export type CaptureMode = "metadata" | "redacted" | "full";
export type SpanStatus = "success" | "failed" | "partial" | "unknown";
export type SpanKind = "workflow" | "agent" | "llm" | "tool" | "retrieval" | "embedding" | "reranker" | "memory" | "state_transition" | "guardrail" | "router" | "human" | "handoff" | "evaluator" | "other";

export type PapayaOptions = RunOptions & {
  apiKey?: string;
  endpoint?: string;
  project?: string;
  environment?: string;
  capture?: CaptureMode;
  serviceName?: string;
  serviceVersion?: string;
  debug?: boolean;
};

export type PapayaFlushResult =
  | { status: "sent"; traceCount: number; endpoint: string; httpStatus: number; responseText?: string }
  | { status: "skipped"; traceCount: number; reason: "empty" | "missing_api_key" }
  | { status: "failed"; traceCount: number; endpoint: string; httpStatus?: number; responseText?: string; error?: string };

export type NativeFetch = typeof globalThis.fetch;
export type FetchInput = Parameters<NativeFetch>[0];
export type FetchInit = NonNullable<Parameters<NativeFetch>[1]>;

export type PapayaFetchCallOptions = RunOptions & {
  provider?: "openai" | "claude" | "anthropic" | "gemini" | "bedrock" | string;
  model?: string;
  spanName?: string;
};

export type PapayaFetchInit = FetchInit & {
  papaya?: PapayaFetchCallOptions;
};

export type PapayaFetchDefaults = RunOptions & {
  provider?: string;
  model?: string;
  spanName?: string;
};

export type PapayaFetch = (input: FetchInput, init?: PapayaFetchInit) => Promise<Response>;

export type RunOptions = {
  runId?: string;
  traceId?: string;
  sessionId?: string;
  conversationId?: string;
  userId?: string;
  organizationId?: string;
  workflowKey?: string;
  workflowLabel?: string;
  conversational?: boolean;
  metadata?: Record<string, unknown>;
};

type PayloadRef = {
  contentType: "text" | "json" | "messages" | "binary_ref";
  value?: unknown;
  contentRef?: string;
  redactionState: CaptureMode;
  byteLength?: number;
  sha256?: string;
};

type TraceSpan = {
  spanId: string;
  parentSpanId?: string;
  name: string;
  kind: SpanKind;
  startedAt: string;
  endedAt?: string;
  status: SpanStatus;
  error?: { type?: string; message?: string; code?: string };
  inputPayload?: PayloadRef;
  outputPayload?: PayloadRef;
  modelRef?: { provider?: string; requested?: string; used?: string };
  usage?: {
    inputTokens?: number;
    outputTokens?: number;
    totalTokens?: number;
    cacheReadInputTokens?: number;
    cacheCreationInputTokens?: number;
    reasoningTokens?: number;
    costUsd?: number;
    pricingSource?: "provider" | "papaya_catalog" | "customer" | "unknown";
  };
  tool?: { name: string; arguments?: PayloadRef; result?: PayloadRef };
  attributes?: Record<string, unknown>;
};

type ActiveRun = Required<Pick<RunOptions, "traceId" | "runId">> & RunOptions & {
  rootSpanId: string;
  spans: TraceSpan[];
};

type TraceBatch = {
  schemaVersion: "2026-06-05";
  batchId: string;
  sentAt: string;
  sdk: {
    name: "papaya-ai";
    version: string;
    language: "typescript";
    runtime?: string;
    framework?: string;
  };
  resource: {
    serviceName?: string;
    serviceVersion?: string;
    environment?: string;
  };
  traces: Array<ActiveRun & { rootSpanId: string }>;
};

const SDK_VERSION = "0.1.0";

const storage = new AsyncLocalStorage<ActiveRun>();

const iso = (): string => new Date().toISOString();

const id = (prefix: string): string => `${prefix}_${randomUUID()}`;

const redactString = (value: string): string =>
  value
    .replace(/[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}/gi, "[redacted-email]")
    .replace(/\b\d{3}-\d{2}-\d{4}\b/g, "[redacted-ssn]")
    .replace(/\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b/g, "[redacted-phone]")
    .replace(/\b(?:sk|pk|papaya|openai|anthropic|gemini|aws)[-_][A-Za-z0-9_-]{12,}\b/gi, "[redacted-secret]")
    .replace(/Bearer\s+[A-Za-z0-9._~+/-]+=*/gi, "Bearer [redacted-token]");

const redactValue = (value: unknown): unknown => {
  if (typeof value === "string") return redactString(value);
  if (Array.isArray(value)) return value.map((item) => redactValue(item));
  if (!value || typeof value !== "object") return value;
  return Object.fromEntries(Object.entries(value).map(([key, item]) => {
    if (/^(authorization|api[-_]?key|access[-_]?token|refresh[-_]?token|id[-_]?token|secret|password)$/i.test(key)) return [key, "[redacted-secret]"];
    return [key, redactValue(item)];
  }));
};

const contentTypeFor = (value: unknown): PayloadRef["contentType"] => {
  if (Array.isArray(value) && value.every((item) => item && typeof item === "object" && "role" in item)) return "messages";
  if (typeof value === "string") return "text";
  return "json";
};

const byteLength = (value: unknown): number => {
  const text = typeof value === "string" ? value : JSON.stringify(value ?? null);
  return new TextEncoder().encode(text).length;
};

const payload = (value: unknown, capture: CaptureMode): PayloadRef => {
  if (capture === "metadata") {
    return { contentType: contentTypeFor(value), redactionState: "metadata", byteLength: byteLength(value) };
  }
  const captured = capture === "redacted" ? redactValue(value) : value;
  return {
    contentType: contentTypeFor(captured),
    value: captured,
    redactionState: capture,
    byteLength: byteLength(captured),
  };
};

const errorPayload = (error: unknown): TraceSpan["error"] => {
  if (error instanceof Error) return { type: error.name, message: error.message };
  return { message: String(error) };
};

const modelFromArgs = (args: unknown[]): string | undefined => {
  const first = args[0];
  if (first && typeof first === "object" && "model" in first && typeof (first as { model?: unknown }).model === "string") {
    return (first as { model: string }).model;
  }
  return undefined;
};

const usageFrom = (result: unknown): TraceSpan["usage"] | undefined => {
  const usage = result && typeof result === "object" && "usage" in result ? (result as { usage?: Record<string, unknown> }).usage : undefined;
  if (!usage || typeof usage !== "object") return undefined;
  const inputTokens = numberValue(usage.input_tokens ?? usage.prompt_tokens ?? usage.inputTokens);
  const outputTokens = numberValue(usage.output_tokens ?? usage.completion_tokens ?? usage.outputTokens);
  return {
    inputTokens,
    outputTokens,
    totalTokens: numberValue(usage.total_tokens ?? usage.totalTokens) ?? (inputTokens ?? 0) + (outputTokens ?? 0),
    cacheReadInputTokens: numberValue(usage.cache_read_input_tokens ?? usage.cached_input_tokens),
    cacheCreationInputTokens: numberValue(usage.cache_creation_input_tokens),
    costUsd: numberValue(usage.cost_usd ?? usage.costUsd),
    pricingSource: usage.cost_usd || usage.costUsd ? "provider" : undefined,
  };
};

const numberValue = (value: unknown): number | undefined => typeof value === "number" && Number.isFinite(value) ? value : undefined;

const isPromiseLike = (value: unknown): value is Promise<unknown> =>
  Boolean(value) && typeof value === "object" && typeof (value as { then?: unknown }).then === "function";

const isRecord = (value: unknown): value is Record<string, unknown> =>
  Boolean(value) && typeof value === "object" && !Array.isArray(value);

const runOptionsFrom = (value: unknown): RunOptions | undefined => {
  if (!isRecord(value)) return undefined;
  const options: RunOptions = {};
  if (typeof value.runId === "string") options.runId = value.runId;
  if (typeof value.traceId === "string") options.traceId = value.traceId;
  if (typeof value.sessionId === "string") options.sessionId = value.sessionId;
  if (typeof value.conversationId === "string") options.conversationId = value.conversationId;
  if (typeof value.userId === "string") options.userId = value.userId;
  if (typeof value.organizationId === "string") options.organizationId = value.organizationId;
  if (typeof value.workflowKey === "string") options.workflowKey = value.workflowKey;
  if (typeof value.workflowLabel === "string") options.workflowLabel = value.workflowLabel;
  if (typeof value.conversational === "boolean") options.conversational = value.conversational;
  if (isRecord(value.metadata)) options.metadata = value.metadata;
  return options;
};

const mergeRunOptions = (...optionSets: Array<RunOptions | undefined>): RunOptions => {
  const result: RunOptions = {};
  for (const options of optionSets) {
    if (!options) continue;
    if (options.runId !== undefined) result.runId = options.runId;
    if (options.traceId !== undefined) result.traceId = options.traceId;
    if (options.sessionId !== undefined) result.sessionId = options.sessionId;
    if (options.conversationId !== undefined) result.conversationId = options.conversationId;
    if (options.userId !== undefined) result.userId = options.userId;
    if (options.organizationId !== undefined) result.organizationId = options.organizationId;
    if (options.workflowKey !== undefined) result.workflowKey = options.workflowKey;
    if (options.workflowLabel !== undefined) result.workflowLabel = options.workflowLabel;
    if (options.conversational !== undefined) result.conversational = options.conversational;
    if (options.metadata) result.metadata = { ...(result.metadata ?? {}), ...options.metadata };
  }
  return result;
};

const providerArgsAndPapayaOptions = (args: unknown[]): { providerArgs: unknown[]; callOptions?: RunOptions } => {
  const first = args[0];
  if (!isRecord(first) || !("papaya" in first)) return { providerArgs: args };
  const { papaya, ...providerRequest } = first;
  return {
    providerArgs: [providerRequest, ...args.slice(1)],
    callOptions: runOptionsFrom(papaya),
  };
};

const isRequestInput = (input: FetchInput): input is Request =>
  typeof Request !== "undefined" && input instanceof Request;

const fetchUrl = (input: FetchInput): string =>
  isRequestInput(input) ? input.url : String(input);

const requestMethod = (input: FetchInput): string =>
  isRequestInput(input) ? input.method : "GET";

const providerFromUrl = (url: string): string => {
  if (url.includes("generativelanguage.googleapis.com")) return "gemini";
  if (url.includes("api.anthropic.com")) return "claude";
  if (url.includes("api.openai.com")) return "openai";
  if (url.includes("bedrock-runtime")) return "bedrock";
  return "fetch";
};

const modelFromRest = (url: string, body: unknown): string | undefined => {
  if (body && typeof body === "object" && "model" in body) {
    return String((body as { model: unknown }).model);
  }
  return url.match(/\/models\/([^:/?]+)/)?.[1];
};

const captureableFetchBody = (body: BodyInit | null | undefined): unknown => {
  if (typeof body === "string") {
    try {
      return JSON.parse(body) as unknown;
    } catch {
      return body;
    }
  }
  if (typeof URLSearchParams !== "undefined" && body instanceof URLSearchParams) {
    return Object.fromEntries(body.entries());
  }
  if (!body) return undefined;
  return {
    contentType: (body as { constructor?: { name?: string } }).constructor?.name ?? "BodyInit",
    note: "body was not read by Papaya",
  };
};

const addHeaderNames = (names: Set<string>, headers: HeadersInit | undefined): void => {
  if (!headers) return;
  for (const name of new Headers(headers).keys()) names.add(name);
};

const fetchHeaderNames = (input: FetchInput, init: FetchInit): string[] => {
  const names = new Set<string>();
  if (isRequestInput(input)) addHeaderNames(names, input.headers);
  addHeaderNames(names, init.headers);
  return [...names].sort();
};

export class Papaya {
  private readonly options: Required<Pick<PapayaOptions, "endpoint" | "capture" | "project" | "environment">> & Omit<PapayaOptions, "endpoint" | "capture" | "project" | "environment">;
  private readonly defaultRunOptions: RunOptions;
  private readonly completed: ActiveRun[] = [];

  private constructor(options: PapayaOptions = {}) {
    const {
      apiKey,
      endpoint,
      capture,
      project,
      environment,
      serviceName,
      serviceVersion,
      debug,
      ...runDefaults
    } = options;
    this.options = {
      apiKey: apiKey ?? process.env.PAPAYA_API_KEY ?? process.env.PAPAYA_INGEST_TOKEN,
      endpoint: endpoint ?? "https://papaya.fyi/api/v1/ingest/traces",
      capture: capture ?? "redacted",
      project: project ?? "default",
      environment: environment ?? "development",
      serviceName,
      serviceVersion,
      debug,
    };
    this.defaultRunOptions = runOptionsFrom(runDefaults) ?? {};
  }

  static init(options: PapayaOptions = {}): Papaya {
    return new Papaya(options);
  }

  async run<T>(options: RunOptions, fn: () => T | Promise<T>): Promise<T> {
    const run = this.createRun(mergeRunOptions(this.defaultRunOptions, options));
    return storage.run(run, async () => {
      try {
        const result = await fn();
        this.finishRun(run, "success");
        return result;
      } catch (error) {
        this.finishRun(run, "failed", error);
        throw error;
      }
    });
  }

  openai<T extends object>(client: T, options?: RunOptions): T {
    return this.wrapClient("openai", client, [], options);
  }

  claude<T extends object>(client: T, options?: RunOptions): T {
    return this.wrapClient("claude", client, [], options);
  }

  anthropic<T extends object>(client: T, options?: RunOptions): T {
    return this.claude(client, options);
  }

  gemini<T extends object>(client: T, options?: RunOptions): T {
    return this.wrapClient("gemini", client, [], options);
  }

  bedrock<T extends object>(client: T, options?: RunOptions): T {
    return this.wrapClient("bedrock", client, [], options);
  }

  vercel<T extends object>(client: T, options?: RunOptions): T {
    return this.wrapClient("vercel", client, [], options);
  }

  fetch(fetchImpl: NativeFetch = globalThis.fetch, defaults: PapayaFetchDefaults = {}): PapayaFetch {
    return (input: FetchInput, init?: PapayaFetchInit) =>
      this.captureFetch(fetchImpl, input, init, defaults);
  }

  async flush(): Promise<PapayaFlushResult> {
    if (this.completed.length === 0) return { status: "skipped", traceCount: 0, reason: "empty" };
    const traces = this.completed.splice(0, this.completed.length);
    const traceCount = traces.length;
    const batch: TraceBatch = {
      schemaVersion: "2026-06-05",
      batchId: id("batch"),
      sentAt: iso(),
      sdk: {
        name: "papaya-ai",
        version: SDK_VERSION,
        language: "typescript",
        runtime: `node/${process.version}`,
      },
      resource: {
        serviceName: this.options.serviceName,
        serviceVersion: this.options.serviceVersion,
        environment: this.options.environment,
      },
      traces,
    };

    if (!this.options.apiKey) {
      if (this.options.debug) console.warn("[papaya] export skipped: missing PAPAYA_API_KEY");
      this.completed.unshift(...traces);
      return { status: "skipped", traceCount, reason: "missing_api_key" };
    }

    try {
      const response = await fetch(this.options.endpoint, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${this.options.apiKey}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify(batch),
      });
      const responseText = await response.text();
      if (!response.ok && this.options.debug) {
        console.warn(`[papaya] export failed: ${response.status} ${responseText}`);
      }
      if (!response.ok) {
        return { status: "failed", traceCount, endpoint: this.options.endpoint, httpStatus: response.status, responseText };
      }
      return { status: "sent", traceCount, endpoint: this.options.endpoint, httpStatus: response.status, responseText };
    } catch (error) {
      if (this.options.debug) console.warn("[papaya] export failed", error);
      return {
        status: "failed",
        traceCount,
        endpoint: this.options.endpoint,
        error: error instanceof Error ? error.message : String(error),
      };
    }
  }

  private wrapClient<T extends object>(provider: string, client: T, path: string[] = [], wrapperOptions?: RunOptions): T {
    const papaya = this;
    return new Proxy(client, {
      get(target, property, receiver) {
        const value = Reflect.get(target, property, receiver) as unknown;
        if (typeof property === "symbol") return value;
        if (typeof value === "function") {
          return function wrappedProviderCall(this: unknown, ...args: unknown[]) {
            return papaya.captureProviderCall(provider, [...path, property], (providerArgs) => value.apply(this === receiver ? target : this, providerArgs), args, wrapperOptions);
          };
        }
        if (value && typeof value === "object") return papaya.wrapClient(provider, value as object, [...path, property], wrapperOptions) as unknown;
        return value;
      },
    }) as T;
  }

  private async captureFetch(
    fetchImpl: NativeFetch,
    input: FetchInput,
    init: PapayaFetchInit = {},
    defaults: PapayaFetchDefaults,
  ): Promise<Response> {
    const { papaya: callPapaya, ...providerInit } = init;
    const url = fetchUrl(input);
    const requestBody = captureableFetchBody(providerInit.body);
    const provider = callPapaya?.provider ?? defaults.provider ?? providerFromUrl(url);
    const model = callPapaya?.model ?? defaults.model ?? modelFromRest(url, requestBody);
    const activeRun = storage.getStore();
    const callOptions = activeRun
      ? mergeRunOptions(this.defaultRunOptions, defaults, activeRun, runOptionsFrom(callPapaya))
      : mergeRunOptions(this.defaultRunOptions, defaults, runOptionsFrom(callPapaya));
    const spanName = callPapaya?.spanName ?? defaults.spanName;

    if (activeRun) {
      return this.captureFetchInRun(fetchImpl, input, providerInit, {
        provider,
        model,
        url,
        requestBody,
        callOptions,
        spanName,
      });
    }

    const implicitRun = this.createRun({
      workflowKey: callOptions.workflowKey ?? `${provider}.fetch`,
      ...callOptions,
    });

    return storage.run(implicitRun, async () => {
      try {
        const response = await this.captureFetchInRun(fetchImpl, input, providerInit, {
          provider,
          model,
          url,
          requestBody,
          callOptions,
          spanName,
        });
        this.finishRun(implicitRun, response.ok ? "success" : "partial");
        return response;
      } catch (error) {
        this.finishRun(implicitRun, "failed", error);
        throw error;
      }
    });
  }

  private async captureFetchInRun(
    fetchImpl: NativeFetch,
    input: FetchInput,
    init: FetchInit,
    info: {
      provider: string;
      model?: string;
      url: string;
      requestBody: unknown;
      callOptions: RunOptions;
      spanName?: string;
    },
  ): Promise<Response> {
    const run = storage.getStore();
    if (!run) return fetchImpl(input, init);

    const span: TraceSpan = {
      spanId: id("span"),
      parentSpanId: run.rootSpanId,
      name: info.spanName ?? `${info.provider}.fetch`,
      kind: "llm",
      startedAt: iso(),
      status: "unknown",
      inputPayload: payload({
        url: info.url,
        method: init.method ?? requestMethod(input),
        headerNames: fetchHeaderNames(input, init),
        body: info.requestBody,
      }, this.options.capture),
      modelRef: { provider: info.provider, requested: info.model },
      attributes: {
        provider: info.provider,
        method: "fetch",
        workflowKey: info.callOptions.workflowKey,
        workflowLabel: info.callOptions.workflowLabel,
        sessionId: info.callOptions.sessionId,
        conversationId: info.callOptions.conversationId,
        userId: info.callOptions.userId,
        organizationId: info.callOptions.organizationId,
        metadata: info.callOptions.metadata,
      },
    };
    run.spans.push(span);

    try {
      const response = await fetchImpl(input, init);
      span.endedAt = iso();
      span.status = response.ok ? "success" : "failed";
      span.outputPayload = payload({
        status: response.status,
        statusText: response.statusText,
        contentType: response.headers.get("content-type"),
      }, this.options.capture);
      span.modelRef = { ...span.modelRef, used: info.model };
      return response;
    } catch (error) {
      span.endedAt = iso();
      span.status = "failed";
      span.error = errorPayload(error);
      throw error;
    }
  }

  private createRun(options: RunOptions): ActiveRun {
    const rootSpanId = id("span");
    const traceId = options.traceId ?? id("trace");
    const runId = options.runId ?? id("run");
    return {
      ...options,
      traceId,
      runId,
      rootSpanId,
      spans: [{
        spanId: rootSpanId,
        name: options.workflowLabel ?? options.workflowKey ?? "papaya.run",
        kind: "workflow",
        startedAt: iso(),
        status: "unknown",
        attributes: {
          project: this.options.project,
          environment: this.options.environment,
          metadata: options.metadata,
        },
      }],
    };
  }

  private finishRun(run: ActiveRun, status: SpanStatus, error?: unknown): void {
    run.spans[0] = {
      ...run.spans[0]!,
      endedAt: iso(),
      status,
      ...(error !== undefined ? { error: errorPayload(error) } : {}),
    };
    this.completed.push(run);
  }

  private captureProviderCall<T>(provider: string, path: string[], call: (args: unknown[]) => T, args: unknown[], wrapperOptions?: RunOptions): T {
    const { providerArgs, callOptions } = providerArgsAndPapayaOptions(args);
    const run = storage.getStore();
    if (run) {
      return this.captureProviderCallInRun(provider, path, (nextArgs) => call(nextArgs), providerArgs, mergeRunOptions(this.defaultRunOptions, wrapperOptions, run, callOptions));
    }

    const implicitRunOptions = mergeRunOptions(this.defaultRunOptions, wrapperOptions, callOptions);
    const implicitRun = this.createRun({
      workflowKey: implicitRunOptions.workflowKey ?? `${provider}.${path.join(".")}`,
      ...implicitRunOptions,
    });
    return storage.run(implicitRun, () => {
      try {
        const result = this.captureProviderCallInRun(provider, path, (nextArgs) => call(nextArgs), providerArgs, implicitRunOptions);
        if (isPromiseLike(result)) {
          return result.then((value) => {
            this.finishRun(implicitRun, "success");
            return value;
          }, (error) => {
            this.finishRun(implicitRun, "failed", error);
            throw error;
          }) as T;
        }
        this.finishRun(implicitRun, "success");
        return result;
      } catch (error) {
        this.finishRun(implicitRun, "failed", error);
        throw error;
      }
    });
  }

  private captureProviderCallInRun<T>(provider: string, path: string[], call: (args: unknown[]) => T, args: unknown[], boundary: RunOptions): T {
    const run = storage.getStore();
    if (!run) return call(args);
    const spanId = id("span");
    const parentSpanId = run.rootSpanId;
    const model = modelFromArgs(args);
    const span: TraceSpan = {
      spanId,
      parentSpanId,
      name: `${provider}.${path.join(".")}`,
      kind: "llm",
      startedAt: iso(),
      status: "unknown",
      inputPayload: payload(args.length === 1 ? args[0] : args, this.options.capture),
      modelRef: { provider, requested: model },
      attributes: {
        provider,
        method: path.join("."),
        workflowKey: boundary.workflowKey,
        workflowLabel: boundary.workflowLabel,
        sessionId: boundary.sessionId,
        conversationId: boundary.conversationId,
        userId: boundary.userId,
        organizationId: boundary.organizationId,
        metadata: boundary.metadata,
      },
    };
    run.spans.push(span);

    const finish = (status: SpanStatus, result?: unknown, error?: unknown): void => {
      span.endedAt = iso();
      span.status = status;
      if (result !== undefined) {
        span.outputPayload = payload(result, this.options.capture);
        span.usage = usageFrom(result);
        span.modelRef = { ...span.modelRef, used: modelFromArgs([result]) ?? span.modelRef?.requested };
      }
      if (error !== undefined) span.error = errorPayload(error);
    };

    try {
      const result = call(args);
      if (isPromiseLike(result)) {
        return result.then((value) => {
          finish("success", value);
          return value;
        }, (error) => {
          finish("failed", undefined, error);
          throw error;
        }) as T;
      }
      finish("success", result);
      return result;
    } catch (error) {
      finish("failed", undefined, error);
      throw error;
    }
  }
}
