import assert from "node:assert/strict";

import { Papaya } from "../src/index.js";
import { PapayaCallbackHandler } from "../src/langchain.js";

type CapturedRequest = {
  url: string;
  init?: RequestInit;
  body: Record<string, unknown>;
};

const captured: CapturedRequest[] = [];
const originalFetch = globalThis.fetch;

globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
  const body = typeof init?.body === "string" ? JSON.parse(init.body) as Record<string, unknown> : {};
  captured.push({ url: String(input), init, body });
  return new Response(JSON.stringify({ accepted: 1, rejected: 0 }), {
    status: 202,
    headers: { "content-type": "application/json" },
  });
}) as typeof fetch;

try {
  const papaya = Papaya.init({
    apiKey: "papaya-test-token",
    endpoint: "https://papaya.example/api/v1/ingest/traces",
    capture: "redacted",
    project: "support",
    environment: "test",
  });
  const callback = new PapayaCallbackHandler(papaya, {
    workflowKey: "support_agent",
    workflowLabel: "Support agent",
    sessionId: "session-1",
    userId: "user-1",
    metadata: { route: "unit-test" },
  });

  callback.handleChainStart({ id: ["langchain", "RunnableLambda"] }, { message: "help ada@example.com" }, "root-run", "chain", [], { step: "root" }, "SupportAgent");
  callback.handleChatModelStart(
    { id: ["langchain", "FakeListChatModel"] },
    [[{ type: "human", content: "help ada@example.com" }]],
    "llm-run",
    "root-run",
    { invocation_params: { model: "fake-chat" } },
    [],
    { ls_model_name: "fake-chat" },
    "support_chat_model",
  );
  callback.handleLLMEnd({
    generations: [[{ text: "Use the refund policy.", message: { type: "ai", content: "Use the refund policy." } }]],
    llmOutput: {
      token_usage: { prompt_tokens: 11, completion_tokens: 7, total_tokens: 18 },
      model_name: "fake-chat-used",
    },
  }, "llm-run");
  callback.handleRetrieverStart({ name: "policy_retriever" }, "refund policy", "retriever-run", "root-run");
  callback.handleRetrieverEnd([{ pageContent: "Refunds above $500 need approval.", metadata: { source: "policy" } }], "retriever-run");
  callback.handleToolStart({ name: "lookup_account" }, "{\"customer\":\"team-demo\"}", "tool-run", "root-run");
  callback.handleToolEnd({ customerId: "team-demo", plan: "team" }, "tool-run");
  callback.handleChainEnd({ output: "Use the refund policy." }, "root-run");

  const flushResult = await papaya.flush();
  assert.equal(flushResult.status, "sent");
  assert.equal(flushResult.traceCount, 1);

  const trace = (captured[0]?.body.traces as Array<Record<string, unknown>>)[0];
  assert.equal(trace.workflowKey, "support_agent");
  assert.equal(trace.sessionId, "session-1");
  const spans = trace.spans as Array<Record<string, unknown>>;
  assert.equal(spans.length, 4);
  assert.equal(spans[0]?.name, "SupportAgent");
  assert.equal(spans[0]?.kind, "workflow");
  assert.equal(spans[0]?.status, "success");
  const llmSpan = spans.find((span) => span.name === "support_chat_model");
  assert.ok(llmSpan);
  assert.equal(llmSpan.kind, "llm");
  assert.deepEqual(llmSpan.modelRef, { provider: "langchain", requested: "fake-chat", used: "fake-chat-used" });
  assert.deepEqual(llmSpan.usage, { inputTokens: 11, outputTokens: 7, totalTokens: 18 });
  const exported = JSON.stringify(captured[0]?.body);
  assert.equal(exported.includes("ada@example.com"), false);
  assert.equal(exported.includes("[redacted-email]"), true);

  console.log("papaya-ai LangChain callback tests passed");
} finally {
  globalThis.fetch = originalFetch;
}
