# Papaya AI SDK

@papaya-ai/tracing is a lightweight TypeScript SDK that captures traces from your agentic workflows and sends them to Papaya for analysis.

Papaya runs 200+ analyses on every trace and shows you exactly where your workflows can improve — prompts, context, tool calls, sub-agents, all of it.

Integration takes just a few lines of instrumentation code, and the SDK adds no latency to your model calls. Keep using OpenAI, Anthropic/Claude, Gemini, Bedrock, Vercel AI SDK, or direct REST calls exactly as you do today.

## Install

```sh
npm install @papaya-ai/tracing
```

Set a Papaya ingest token for your service:

```sh
PAPAYA_API_KEY=papaya_...
```

## AI IDE Prompt

If you want your AI IDE to add tracing for you, paste this prompt:

```text
Add the @papaya-ai/tracing SDK to my existing project so my LLM and agent calls
show up as traces in Papaya. Keep changes minimal. Do not refactor anything else.
Reference: https://github.com/Nile-AI-Inc/papaya-ai

1. Install:
   npm i @papaya-ai/tracing

2. Initialize once, in a shared module. Set project/environment, and tie the
   build to a version so regressions are attributable:

   import { Papaya } from "@papaya-ai/tracing";

   export const papaya = Papaya.init({
     apiKey: process.env.PAPAYA_API_KEY,
     project: "<my-project>",
     environment: process.env.NODE_ENV ?? "development",
     serviceName: "<my-service>",
     serviceVersion: process.env.GIT_SHA,   // regression bisection
     capture: "redacted",                    // metadata | redacted (default) | full
   });

3. Capture the model calls. Apply whichever matches how I call the model:

   A. Provider SDK — wrap the client once, keep calling it as before:
        const openai = papaya.openai(new OpenAI());
        await openai.chat.completions.create({ ... });
      Use the matching wrapper for Anthropic/Claude, Bedrock, Gemini, or the
      Vercel AI SDK. Wrap EVERY model client I use, including secondary ones
      (embeddings, rerankers, a cheap "router" model).

   B. Raw HTTP — swap fetch for wrapped fetch and name the provider + model:
        const llmFetch = papaya.fetch(globalThis.fetch);
        await llmFetch(url, { method, headers, body,
          papaya: { provider, model, spanName, metadata } });

4. Group multi-step work into ONE trace. Anything that makes more than a single
   model call for one logical unit of work — an agent loop, a tool-use loop,
   retries, a RAG pipeline, multiple providers — must be wrapped in papaya.run()
   so the whole thing is one trace instead of N disconnected ones:

   await papaya.run(
     {
       workflowKey: "<stable_machine_key>",     // e.g. "refund_agent"
       workflowLabel: "<human label>",
       sessionId,                                // group a user's turns
       conversationId,                           // group a chat thread
       userId, organizationId,                   // who/which tenant
       conversational: true,                     // for chat UIs
       metadata: { route: "/api/agent", requestId, tenant },
     },
     async () => { /* all the model + tool calls */ },
   );

   Pass per-call context on individual provider calls when it varies:
     await openai.chat.completions.create({
       model, messages,
       papaya: { sessionId, userId, metadata: { step: "triage" } },
     });

5. Pass the real input each step. The SDK captures messages BY REFERENCE, so if I
   mutate a running messages[] array in an agent loop, pass a copy per call
   (e.g. messages: [...messages]) so each step freezes its actual input.

6. Make traces survive failures. Failures are the most important thing to capture:

   try {
     await handleRequest();
   } finally {
     await papaya.flush();          // always flush, even when my code throws
   }

   - In a long-running server, ALSO flush on an interval (e.g. every 5–10s) and
     once more on shutdown (SIGTERM, SIGINT, beforeExit) so in-flight traces
     aren't lost on deploy/restart.
   - Papaya swallows its own export errors (set debug:true to log them) and never
     changes my provider results, so the flush in finally is safe.
   - A workflow that throws is still exported, marked failed/partial — keep the
     model calls inside the run() so the failure is attached to the trace.

7. Watch-outs:
   - Streaming: streamed responses are currently captured with limited
     output/usage. If a call's output or token counts matter for analysis,
     prefer a non-streaming variant there, or confirm streaming capture is
     supported before relying on it.
   - Never put secrets in the `papaya` metadata field (it IS exported); the SDK
     records header NAMES only, not values, and strips the `papaya` field before
     the provider sees the request.
   - capture: "full" disables local redaction — only use it where policy allows.
   - Large payloads: avoid attaching huge blobs as metadata; keep prompts/outputs
     as the captured payload.

Find where I create each model client, where each request/job begins and ends, and
where my agent loop lives. Show me the exact lines to add for init, the wrappers,
the run() boundary, and the try/finally + shutdown flush. Leave the rest of my code
unchanged.
```

## Quick Start

```ts
import OpenAI from "openai";
import { Papaya } from "@papaya-ai/tracing";

const papaya = Papaya.init({
  apiKey: process.env.PAPAYA_API_KEY!,
});

const openai = papaya.openai(new OpenAI());

await openai.chat.completions.create({
  model: "gpt-4.1-mini",
  messages,
});

await papaya.flush();
```

## Fetch-Based Agent Loops

Use `papaya.fetch()` when your agent loop calls providers directly. The wrapper calls the provider URL with your original request, strips the Papaya-only `papaya` field before the provider sees it, records header names instead of header values, and preserves streaming responses.

```ts
import { Papaya, type PapayaFetchInit } from "@papaya-ai/tracing";

const papaya = Papaya.init({
  apiKey: process.env.PAPAYA_API_KEY!,
  project: "support-agent",
  environment: "production",
});

const llmFetch = papaya.fetch(globalThis.fetch, {
  workflowKey: "customer_support_agent",
});

await papaya.run({ sessionId, userId }, async () => {
  const response = await llmFetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      authorization: `Bearer ${process.env.OPENAI_API_KEY!}`,
      "content-type": "application/json",
    },
    body: JSON.stringify({
      model: "gpt-4.1-mini",
      stream: true,
      messages,
    }),
    papaya: {
      provider: "openai",
      model: "gpt-4.1-mini",
      spanName: "openai.chat",
      metadata: { route: "/api/agent/chat" },
    },
  } satisfies PapayaFetchInit);

  return response;
});

await papaya.flush();
```

Gemini REST calls work the same way:

```ts
await llmFetch("https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent", {
  method: "POST",
  headers: {
    "content-type": "application/json",
    "x-goog-api-key": process.env.GEMINI_API_KEY!,
  },
  body: JSON.stringify({
    contents: [{ parts: [{ text: "Summarize this customer issue." }] }],
  }),
  papaya: {
    provider: "gemini",
    model: "gemini-flash-latest",
  },
} satisfies PapayaFetchInit);
```

## SDK Wrappers

The SDK wraps clients by shape, so you do not need a Papaya-specific provider dependency.

```ts
const openai = papaya.openai(new OpenAI());
const anthropic = papaya.anthropic(new Anthropic());
const claude = papaya.claude(new Anthropic());
const gemini = papaya.gemini(genAI);
const bedrock = papaya.bedrock(bedrockRuntimeClient);
const vercel = papaya.vercel(aiSdkObject);
```

Pass run metadata either when you wrap the client or on a single provider call:

```ts
const openai = papaya.openai(new OpenAI(), {
  workflowKey: "claim_triage",
  metadata: { release: process.env.GIT_SHA },
});

await openai.chat.completions.create({
  model: "gpt-4.1-mini",
  messages,
  papaya: {
    sessionId,
    userId: customerUserId,
    metadata: { claimId },
  },
});
```

The `papaya` field is stripped before the provider SDK or REST endpoint receives the request.

## Workflow Boundaries

Papaya creates an implicit single-call run when no explicit run is active. Use `papaya.run()` when one business workflow spans several model calls, retries, retrievals, tools, guardrails, or provider SDKs.

```ts
await papaya.run({
  workflowKey: "refund_agent",
  workflowLabel: "Refund agent",
  sessionId,
  userId,
}, async () => {
  await openai.responses.create({ model: "gpt-4.1-mini", input });
  await llmFetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: {
      "anthropic-version": "2023-06-01",
      "x-api-key": process.env.ANTHROPIC_API_KEY!,
      "content-type": "application/json",
    },
    body: JSON.stringify({ model: "claude-3-5-sonnet-latest", messages }),
    papaya: { provider: "anthropic" },
  });
});

await papaya.flush();
```

## Capture Modes

Configure capture with `Papaya.init({ capture })`.

- `metadata`: records payload type and byte length only.
- `redacted`: records payloads after local redaction. This is the default.
- `full`: records payloads without local redaction. Use only when your Papaya token policy allows it.

Redaction runs locally before export. The hosted ingest API also enforces the token capture policy before raw trace landing.

## Flush Behavior

The first package slice exports traces when you call `await papaya.flush()`. Provider calls are awaited normally, and Papaya export failures do not change provider results. When `debug: true` is enabled, export failures are logged to `console.warn`.

Typical server usage:

```ts
try {
  await handleAgentRequest();
} finally {
  await papaya.flush();
}
```

## Configuration

```ts
const papaya = Papaya.init({
  apiKey: process.env.PAPAYA_API_KEY,
  endpoint: "https://papaya.fyi/api/v1/ingest/traces",
  project: "support-agent",
  environment: "production",
  serviceName: "agent-api",
  serviceVersion: process.env.GIT_SHA,
  capture: "redacted",
  debug: false,
});
```

`apiKey` defaults to `PAPAYA_API_KEY` and then `PAPAYA_INGEST_TOKEN`. `endpoint` defaults to `https://papaya.fyi/api/v1/ingest/traces`.

## Safety Defaults

- Capture defaults to `redacted`.
- Redaction runs locally before export.
- Provider API keys are not exported by the fetch wrapper; it records header names only.
- SDK errors are swallowed unless `debug: true` is enabled.
- Provider calls are awaited normally; Papaya export happens on `flush()`.
- Papaya-only call metadata is stripped before provider SDK and REST calls.
- Streaming fetch responses are returned without reading the response body.
- The hosted ingest API enforces capture policy again before raw landing.

## Development

```sh
npm install
npm run typecheck
npm test
npm pack --dry-run
```

Before publishing, follow `RELEASE.md`.
