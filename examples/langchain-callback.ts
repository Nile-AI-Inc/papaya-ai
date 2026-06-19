import { ChatOpenAI } from "@langchain/openai";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { RunnableLambda, type RunnableConfig } from "@langchain/core/runnables";

import { Papaya } from "@papaya-ai/tracing";
import { PapayaCallbackHandler } from "@papaya-ai/tracing/langchain";

const requireEnv = (name: string): string => {
  const value = process.env[name]?.trim();
  if (!value) throw new Error(`${name} is required to run this real LLM example.`);
  return value;
};

requireEnv("OPENAI_API_KEY");

const MODEL = process.env.OPENAI_MODEL ?? "gpt-4.1-mini";
const messages = [
  new SystemMessage("You are a concise support operations assistant."),
  new HumanMessage("Can we refund this customer 700 dollars? Answer in one sentence."),
];

const papaya = Papaya.init({
  apiKey: process.env.PAPAYA_API_KEY,
  project: "papaya-demo",
  environment: "demo",
});
const callback = new PapayaCallbackHandler(papaya, {
  workflowKey: "typescript_langchain_callback",
  sessionId: "demo-session-ts",
  userId: "demo-user",
});
const chat = new ChatOpenAI({ model: MODEL, temperature: 0 }).withConfig({
  runName: "support_chat_model",
  metadata: { model: MODEL },
});
const agent = RunnableLambda.from(async (_input: Record<string, never>, config?: RunnableConfig) => {
  console.log("LangChain request:", JSON.stringify({
    model: MODEL,
    messages: messages.map((message) => ({
      role: message._getType(),
      content: message.content,
    })),
  }, null, 2));
  const response = await chat.invoke(messages, config);
  console.log("LangChain response:", JSON.stringify({
    content: response.content,
    responseMetadata: response.response_metadata,
    usageMetadata: response.usage_metadata,
  }, null, 2));
  return String(response.content);
}).withConfig({ runName: "SupportAgent" });

try {
  const answer = await agent.invoke(
    {},
    { callbacks: [callback], metadata: { route: "examples/langchain-callback.ts" } },
  );
  console.log("LangChain answer:", answer);
} finally {
  console.log("Papaya flush:", await papaya.flush());
}
