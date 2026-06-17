// gemini-fetch.ts with the Papaya wrapper removed — diff the two side by side.
// Removing the wrapper means: drop the import + Papaya.init, use plain `fetch`
// instead of papaya.fetch, drop the `papaya: {...}` field, drop papaya.flush().
// The call still works — but NOTHING is traced. That observability is exactly
// what the wrapper adds, for ~2 lines of code.

const MODEL = "gemini-2.5-flash";

const res = await fetch(
  `https://generativelanguage.googleapis.com/v1beta/models/${MODEL}:generateContent`,
  {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-goog-api-key": process.env.GEMINI_API_KEY!,
    },
    body: JSON.stringify({
      contents: [{ parts: [{ text: "In one sentence, what should I pack for a sunny day in London?" }] }],
    }),
  },
);

const data: any = await res.json();
console.log("🤖 Gemini:", data.candidates?.[0]?.content?.parts?.[0]?.text ?? data);
