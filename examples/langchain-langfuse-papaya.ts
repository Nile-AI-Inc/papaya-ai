import { randomUUID } from "node:crypto";

import { CallbackHandler as LangfuseCallbackHandler } from "@langfuse/langchain";
import { LangfuseSpanProcessor } from "@langfuse/otel";
import { ChatOpenAI } from "@langchain/openai";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { RunnableLambda, type RunnableConfig } from "@langchain/core/runnables";
import { DynamicTool } from "@langchain/core/tools";
import { NodeSDK } from "@opentelemetry/sdk-node";

import { Papaya } from "@papaya-ai/tracing";
import { PapayaCallbackHandler } from "@papaya-ai/tracing/langchain";

const requireEnv = (name: string): string => {
  const value = process.env[name]?.trim();
  if (!value) throw new Error(`${name} is required.`);
  return value;
};

const openaiApiKey = requireEnv("OPENAI_API_KEY");
const papayaApiKey = requireEnv("PAPAYA_API_KEY");
requireEnv("LANGFUSE_PUBLIC_KEY");
requireEnv("LANGFUSE_SECRET_KEY");

const langfuse = new NodeSDK({ spanProcessors: [new LangfuseSpanProcessor()] });
langfuse.start();

const runId = randomUUID();
const sessionId = `session_${runId}`;
const userId = "demo-user";

const papaya = Papaya.init({
  apiKey: papayaApiKey,
  project: "papaya-demo",
  environment: "demo",
});
const papayaHandler = new PapayaCallbackHandler(papaya, {
  workflowKey: "langchain_langfuse_papaya_demo",
  sessionId,
  userId,
  metadata: { example: "langchain-langfuse-papaya", runId },
});
const langfuseHandler = new LangfuseCallbackHandler({
  sessionId,
  userId,
  tags: ["papaya-parallel"],
});

const model = new ChatOpenAI({
  apiKey: openaiApiKey,
  model: process.env.OPENAI_MODEL ?? "gpt-4.1-mini",
  temperature: 0,
}).withConfig({ runName: "write_customer_reply" });

const lookupOrder = new DynamicTool({
  name: "lookup_order",
  description: "Look up the current state of an order.",
  func: async (orderId) => `${orderId}: delayed in customs, refund eligible`,
});

const agent = RunnableLambda.from(async (input: { question: string }, config?: RunnableConfig) => {
  const order = await lookupOrder.invoke("order_123", config);
  const answer = await model.invoke([
    new SystemMessage("You are a concise support agent. Answer in one sentence."),
    new HumanMessage(`${input.question}\n\nOrder context: ${order}`),
  ], config);
  return String(answer.content);
}).withConfig({ runName: "TinySupportAgent" });

try {
  const answer = await agent.invoke(
    { question: "Should we refund this customer?" },
    {
      runId,
      callbacks: [langfuseHandler, papayaHandler],
      metadata: { runId, route: "examples/langchain-langfuse-papaya.ts" },
      tags: ["oss-example"],
    },
  );
  console.log({ answer, runId });
} finally {
  console.log("Papaya flush:", await papaya.flush());
  await langfuse.shutdown();
}
