import assert from "node:assert/strict";
import { AIMessage, HumanMessage } from "@langchain/core/messages";
import type { ChatGeneration, LLMResult } from "@langchain/core/outputs";

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
    { id: ["langchain", "ChatOpenAI"] },
    [[
      new HumanMessage("help ada@example.com"),
      new AIMessage({
        id: "history-message",
        content: "",
        tool_calls: [{ name: "lookup_policy", args: { account: 42 }, id: "call-history", type: "tool_call" }],
      }),
    ]],
    "llm-run",
    "root-run",
    { invocation_params: { model: "gpt-test" } },
    [],
    { ls_model_name: "gpt-test" },
    "support_chat_model",
  );
  const aiMessage = new AIMessage({
    id: "lc-message-1",
    content: "Use the refund policy.",
    response_metadata: { model_name: "gpt-test-used" },
    usage_metadata: {
      input_tokens: 11,
      output_tokens: 7,
      total_tokens: 18,
      input_token_details: { cache_read: 3 },
      output_token_details: { reasoning: 4 },
    },
  });
  const chatGeneration = { text: "Use the refund policy.", message: aiMessage } satisfies ChatGeneration;
  const llmResult = {
    generations: [[chatGeneration]],
    llmOutput: {
      token_usage: { prompt_tokens: 11, completion_tokens: 7, total_tokens: 18 },
      model_name: "gpt-test-used",
    },
  } satisfies LLMResult;
  callback.handleLLMEnd(llmResult, "llm-run");
  callback.handleChatModelStart(
    { id: ["langchain", "ChatOpenAI"] },
    [[
      new HumanMessage("help ada@example.com"),
      new AIMessage({
        id: "history-message",
        content: "",
        tool_calls: [{ name: "lookup_policy", args: { account: 42 }, id: "call-history", type: "tool_call" }],
      }),
      new AIMessage({ id: "history-copy", content: "Use the refund policy." }),
      new HumanMessage("What is the approval limit?"),
    ]],
    "llm-run-2",
    "root-run",
    { invocation_params: { model: "gpt-test" } },
  );
  callback.handleLLMEnd({
    generations: [[{
      text: "Approval is required above $500.",
      message: new AIMessage({ id: "lc-message-2", content: "Approval is required above $500." }),
    }]],
  }, "llm-run-2");
  callback.handleChatModelStart(
    { id: ["langchain", "ChatOpenAI"] },
    [[
      new HumanMessage("help ada@example.com"),
      new AIMessage({
        id: "history-message",
        content: "",
        tool_calls: [{ name: "lookup_policy", args: { account: 42 }, id: "call-history", type: "tool_call" }],
      }),
      new AIMessage({ id: "history-copy", content: "Use the refund policy." }),
      new HumanMessage("What is the approval limit?"),
    ]],
    "llm-run-3",
    "root-run",
  );
  callback.handleLLMEnd({
    generations: [[{
      text: "No change.",
      message: new AIMessage({ id: "lc-message-3", content: "No change." }),
    }]],
  }, "llm-run-3");
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
  assert.equal(spans.length, 6);
  assert.equal(spans[0]?.name, "SupportAgent");
  assert.equal(spans[0]?.kind, "workflow");
  assert.equal(spans[0]?.status, "success");
  const llmSpan = spans.find((span) => span.name === "support_chat_model");
  assert.ok(llmSpan);
  assert.equal(llmSpan.kind, "llm");
  assert.deepEqual(llmSpan.modelRef, { provider: "langchain", requested: "gpt-test", used: "gpt-test-used" });
  assert.deepEqual(llmSpan.usage, {
    inputTokens: 11,
    outputTokens: 7,
    totalTokens: 18,
    cacheReadInputTokens: 3,
    reasoningTokens: 4,
  });
  const llmInput = (llmSpan.inputPayload as { value?: Array<Array<{ id?: string; toolCalls?: Array<{ name?: string }> }>> }).value;
  assert.equal(llmInput?.[0]?.[1]?.id, "history-message");
  assert.equal(llmInput?.[0]?.[1]?.toolCalls?.[0]?.name, "lookup_policy");
  assert.deepEqual(Object.keys(llmInput?.[0]?.[1] ?? {}).sort(), ["content", "id", "role", "toolCalls"]);
  const llmOutput = (llmSpan.outputPayload as { value?: Array<Array<{ content?: string; id?: string; role?: string }>> }).value;
  assert.deepEqual(llmOutput?.[0]?.[0], {
    role: "assistant",
    content: "Use the refund policy.",
    id: "lc-message-1",
  });
  const secondLlm = spans.find((span) => span.attributes && (span.attributes as { langchainRunId?: string }).langchainRunId === "llm-run-2");
  const thirdLlm = spans.find((span) => span.attributes && (span.attributes as { langchainRunId?: string }).langchainRunId === "llm-run-3");
  assert.deepEqual(
    (secondLlm?.inputPayload as { value?: unknown }).value,
    [[{ role: "user", content: "What is the approval limit?" }]],
  );
  assert.deepEqual((thirdLlm?.inputPayload as { value?: unknown }).value, [[]]);
  for (const id of [captured[0]?.body.batchId, trace.traceId, trace.runId, ...spans.map((span) => span.spanId)]) {
    assert.match(String(id), /^(?:batch|trace|run|span)_[A-Za-z0-9_-]{22}$/);
  }
  assert.equal(String(trace.traceId).includes("root-run"), false);
  assert.equal((spans[0]?.attributes as { langchainRunId?: string }).langchainRunId, "root-run");
  const exported = JSON.stringify(captured[0]?.body);
  assert.equal(exported.includes("ada@example.com"), false);
  assert.equal(exported.includes("[redacted-email]"), true);
  assert.equal(exported.includes("lc_kwargs"), false);
  assert.equal(exported.includes("usage_metadata"), false);

  const branchPapaya = Papaya.init({
    apiKey: "papaya-test-token",
    endpoint: "https://papaya.example/api/v1/ingest/traces",
    capture: "full",
  });
  const branchCallback = new PapayaCallbackHandler(branchPapaya, { workflowKey: "branch_safety" });
  branchCallback.handleChainStart({ name: "BranchAgent" }, { messages: [] }, "branch-root");
  const multimodalUser = {
    type: "human",
    content: [
      { type: "text", text: "Inspect this receipt." },
      { type: "image_url", image_url: { url: "https://example.test/receipt.png" } },
    ],
    id: "undefined",
    name: "undefined",
    additional_kwargs: {},
    response_metadata: {},
    tool_calls: [],
    invalid_tool_calls: [],
  };
  branchCallback.handleChatModelStart({ name: "BranchA" }, [[multimodalUser]], "branch-a", "branch-root");
  branchCallback.handleChatModelStart({ name: "BranchB" }, [[multimodalUser]], "branch-b", "branch-root");
  branchCallback.handleLLMEnd({
    generations: [[{ message: new AIMessage({ id: "branch-a-output", content: "Branch A" }) }]],
  }, "branch-a");
  branchCallback.handleLLMEnd({
    generations: [[{ message: new AIMessage({ id: "branch-b-output", content: "Branch B" }) }]],
  }, "branch-b");
  branchCallback.handleChatModelStart(
    { name: "AfterBranch" },
    [[multimodalUser, new AIMessage({ id: "merged", content: "Merged result" })]],
    "after-branch",
    "branch-root",
  );
  branchCallback.handleLLMEnd({
    generations: [[{ message: new AIMessage({ id: "after-output", content: "Continue" }) }]],
  }, "after-branch");
  branchCallback.handleChatModelStart(
    { name: "AfterBranchSerial" },
    [[
      multimodalUser,
      new AIMessage({ id: "merged", content: "Merged result" }),
      new AIMessage({ id: "different-provider-id", content: "Continue" }),
      { type: "tool", content: "", tool_call_id: "call-42", artifact: { approved: true } },
    ]],
    "after-branch-serial",
    "branch-root",
  );
  branchCallback.handleLLMEnd({
    generations: [[{ message: new AIMessage({ content: "Done" }) }]],
  }, "after-branch-serial");
  branchCallback.handleChainEnd({ output: "Done" }, "branch-root");
  assert.equal((await branchPapaya.flush()).status, "sent");

  const branchTrace = (captured[1]?.body.traces as Array<Record<string, unknown>>)[0]!;
  const branchSpans = branchTrace.spans as Array<Record<string, unknown>>;
  const spanForRun = (runId: string): Record<string, unknown> | undefined =>
    branchSpans.find((span) =>
      (span.attributes as { langchainRunId?: string } | undefined)?.langchainRunId === runId);
  const branchAInput = (spanForRun("branch-a")?.inputPayload as { value?: unknown }).value;
  const branchBInput = (spanForRun("branch-b")?.inputPayload as { value?: unknown }).value;
  const afterBranchInput = (spanForRun("after-branch")?.inputPayload as { value?: unknown }).value;
  const serialInput = (spanForRun("after-branch-serial")?.inputPayload as { value?: unknown }).value;
  assert.deepEqual(branchAInput, branchBInput);
  assert.deepEqual(afterBranchInput, [[
    {
      role: "user",
      content: [
        { type: "text", text: "Inspect this receipt." },
        { type: "image_url", image_url: { url: "https://example.test/receipt.png" } },
      ],
    },
    { role: "assistant", content: "Merged result", id: "merged" },
  ]]);
  assert.deepEqual(serialInput, [[{
    role: "tool",
    content: "",
    toolCallId: "call-42",
    artifact: { approved: true },
  }]]);
  assert.equal(JSON.stringify(branchTrace).includes("additional_kwargs"), false);
  assert.equal(JSON.stringify(branchTrace).includes("response_metadata"), false);
  assert.equal(JSON.stringify(branchTrace).includes("invalid_tool_calls"), false);

  console.log("papaya-ai LangChain callback tests passed");
} finally {
  globalThis.fetch = originalFetch;
}
