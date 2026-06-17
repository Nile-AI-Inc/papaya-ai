import { Papaya } from "@papaya-ai/tracing";
import { GoogleGenAI } from "@google/genai";

const MODEL = "gemini-2.5-flash";

const papaya = Papaya.init({
  // Key is inline so you can see exactly where it goes. In a real app use:
  //   apiKey: process.env.PAPAYA_API_KEY
  apiKey: "ppy_live_XXXXXXXXXXXXXXXXXXXX",
  project: "papaya-demo",
  environment: "demo",
  capture: "full",
});

// Wrap the Google GenAI client — every call is traced automatically.
const gemini = papaya.gemini(new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY }));

// Group the call(s) into one trace.
const answer = await papaya.run(
  { workflowKey: "gemini-sdk", sessionId: "demo-session-2", userId: "demo-user" },
  async () => {
    const res: any = await gemini.models.generateContent({
      model: MODEL,
      contents: "In one sentence, what should I pack for a rainy day in Tokyo?",
    });
    return res.text;
  },
);

console.log("🤖 Gemini:", answer);

const result = await papaya.flush();
console.log("\n📤 Papaya:", result.status, `· ${result.traceCount} trace(s)`);
