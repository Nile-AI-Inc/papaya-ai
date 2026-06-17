# Examples

Small, self-contained scripts showing how to add `@papaya-ai/tracing` to an app.
Each one initializes Papaya, makes a model call, and flushes the trace.

| File | Shows |
|------|-------|
| [`gemini-sdk.ts`](gemini-sdk.ts) | Wrap a provider SDK client — `papaya.gemini(new GoogleGenAI(...))` |
| [`gemini-fetch.ts`](gemini-fetch.ts) | Trace a raw HTTP model call — `papaya.fetch(globalThis.fetch)` |
| [`gemini-fetch-raw.ts`](gemini-fetch-raw.ts) | The same call **without** the SDK, for comparison (no tracing) |
| [`error-handling.ts`](error-handling.ts) | Flush in `finally` so failed jobs still send their trace |
| [`agent.ts`](agent.ts) | A Bedrock tool-use agent loop captured as one workflow — `papaya.bedrock(...)` |

## Run

```sh
# from this folder
npm i @papaya-ai/tracing @aws-sdk/client-bedrock-runtime @google/genai tsx

cp .env.example .env   # then fill in your keys
npx tsx --env-file=.env gemini-sdk.ts
```

Each script sets a placeholder `apiKey: "ppy_live_XXXX..."` inline so you can see
where the Papaya key goes — replace it with your key, or switch to
`apiKey: process.env.PAPAYA_API_KEY`.

## Keys / env

- **Papaya** — set `apiKey` in each script (currently a `ppy_live_XXXX...` placeholder).
- **Gemini** (`gemini-*`, `error-handling`) — `GEMINI_API_KEY`, from https://aistudio.google.com/apikey
- **Bedrock** (`agent.ts`) — `AWS_PROFILE` + `AWS_REGION`, with access to the Claude inference profile.
