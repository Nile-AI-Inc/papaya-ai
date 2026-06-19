import { Papaya } from "@papaya-ai/tracing";
import { GoogleGenAI } from "@google/genai";

const MODEL = "gemini-2.5-flash";
const requireEnv = (name: string): string => {
  const value = process.env[name]?.trim();
  if (!value) throw new Error(`${name} is required to run this real LLM example.`);
  return value;
};

const papaya = Papaya.init({
  apiKey: requireEnv("PAPAYA_API_KEY"),
  project: "papaya-demo",
  environment: "demo",
  capture: "full",
});

const gemini = papaya.gemini(new GoogleGenAI({ apiKey: requireEnv("GEMINI_API_KEY") }));

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
