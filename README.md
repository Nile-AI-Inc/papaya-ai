# Papaya AI SDK

Papaya is a TypeScript tracing SDK for production AI agents. It wraps your existing OpenAI, Anthropic/Claude, Gemini, Bedrock, Vercel AI SDK, or `fetch` calls without proxying them. It captures traces locally, redacts payloads by default, and sends them when you call `flush()`.

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
Add the @papaya-ai/tracing SDK to my existing project so my LLM calls show up as
traces in Papaya. Keep the changes minimal. Do not refactor anything else.
Reference: https://github.com/Nile-AI-Inc/papaya-ai

1. Install:
   npm i @papaya-ai/tracing

2. Initialize once, in a shared module:

   import { Papaya } from "@papaya-ai/tracing";

   export const papaya = Papaya.init({
     apiKey: process.env.PAPAYA_API_KEY,
     project: "<my-project>",
   });

3. Apply one of these, matching how I call the model:

   A. If I use a provider SDK, wrap the client once and keep calling it as before:

      const openai = papaya.openai(new OpenAI());
      await openai.chat.completions.create({ ... });

      Use the matching wrapper if I use Anthropic/Claude, Bedrock, Gemini, or Vercel AI SDK.

   B. If I call the model over raw HTTP, swap fetch for wrapped fetch:

      const llmFetch = papaya.fetch(globalThis.fetch);
      await llmFetch(url, { method, headers, body, papaya: { provider, model } });

4. Make sure traces are sent even when my code fails:

   try {
     // ... my model calls ...
   } finally {
     await papaya.flush();
   }

   In a long-running server, flush on an interval and once more on shutdown.

Find where I create my model client and where my request or job ends. Show me the
exact lines to add for init, the wrapper, and the try/finally flush. Leave the
rest of my code unchanged.
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
