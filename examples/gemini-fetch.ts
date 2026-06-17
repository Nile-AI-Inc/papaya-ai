import { Papaya, type PapayaFetchInit } from "@papaya-ai/tracing";

const MODEL = "gemini-2.5-flash";

const papaya = Papaya.init({
  // Key is inline so you can see exactly where it goes. In a real app use:
  //   apiKey: process.env.PAPAYA_API_KEY
  apiKey: "ppy_live_XXXXXXXXXXXXXXXXXXXX",
  project: "papaya-demo",
  environment: "demo",
  capture: "full",
});

// A traced fetch — use it exactly like fetch(), but every call is captured.
// (No papaya.run needed: the fetch wrapper opens a trace on its own.)
const llmFetch = papaya.fetch(globalThis.fetch, { workflowKey: "gemini-fetch" });

const res = await llmFetch(
  `https://generativelanguage.googleapis.com/v1beta/models/${MODEL}:generateContent`,
  {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-goog-api-key": process.env.GEMINI_API_KEY!, // key lives in the header, never in the trace
    },
    body: JSON.stringify({
      contents: [{ parts: [{ text: "In one sentence, what should I pack for a sunny day in London?" }] }],
    }),
    // Tell Papaya what this call is (provider/model are also auto-detected from the URL).
    papaya: { provider: "gemini", model: MODEL, spanName: "gemini.generateContent" },
  } satisfies PapayaFetchInit,
);

const data: any = await res.json();
console.log("🤖 Gemini:", data.candidates?.[0]?.content?.parts?.[0]?.text ?? data);

const result = await papaya.flush();
console.log("\n📤 Papaya:", result.status, `· ${result.traceCount} trace(s)`);
