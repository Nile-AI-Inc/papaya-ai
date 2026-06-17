import { Papaya } from "@papaya-ai/tracing";
import { GoogleGenAI } from "@google/genai";

const MODEL = "gemini-2.5-flash";

const papaya = Papaya.init({
  // Your Papaya API key. Inline here for clarity; in a real app use:
  //   apiKey: process.env.PAPAYA_API_KEY
  apiKey: "ppy_live_XXXXXXXXXXXXXXXXXXXX",
  project: "papaya-demo",
  environment: "demo",
  capture: "full",
});

const gemini = papaya.gemini(new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY }));

try {
  await papaya.run({ workflowKey: "error-handling", sessionId: "demo-session-4" }, async () => {
    const res: any = await gemini.models.generateContent({
      model: MODEL,
      contents: "Say hello in one word.",
    });
    console.log("🤖 Gemini:", res.text);

    // Simulate a downstream failure AFTER the model call succeeded.
    throw new Error("downstream service failed");
  });
} catch (err) {
  console.error("❌ Job failed:", (err as Error).message);
} finally {
  // flush() lives in finally, so the trace — including the failed workflow span —
  // still reaches Papaya even though the job threw.
  const result = await papaya.flush();
  console.log("📤 Papaya:", result.status, `· ${result.traceCount} trace(s)`);
}
