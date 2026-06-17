import { Papaya } from "@papaya-ai/tracing";
import { BedrockRuntime } from "@aws-sdk/client-bedrock-runtime";

const MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0";

// 1. Start Papaya.
const papaya = Papaya.init({
  // Key is inline so you can see exactly where it goes. In a real app use:
  //   apiKey: process.env.PAPAYA_API_KEY
  apiKey: "ppy_live_XXXXXXXXXXXXXXXXXXXX",
  project: "papaya-demo",
  environment: "demo",
  capture: "full", // capture full prompts + responses for the demo
});

// 2. Wrap the Bedrock client — every model call is now traced automatically.
const bedrock = papaya.bedrock(new BedrockRuntime({ region: "us-east-1" }));

// The one tool our agent can use.
const tools = [{
  toolSpec: {
    name: "get_weather",
    description: "Get tomorrow's weather forecast for a city.",
    inputSchema: { json: {
      type: "object",
      properties: { city: { type: "string" } },
      required: ["city"],
    } },
  },
}];

const forecasts: Record<string, string> = {
  tokyo: "12°C, heavy rain all day",
  london: "19°C, sunny with light wind",
};
const getWeather = (city: string) =>
  forecasts[city.toLowerCase()] ?? "22°C, partly cloudy";

// The agent loop: call the model, run any tools it asks for, repeat.
async function agent(prompt: string) {
  const messages: any[] = [{ role: "user", content: [{ text: prompt }] }];

  for (let step = 0; step < 5; step++) {
    const res: any = await bedrock.converse({
      modelId: MODEL,
      system: [{ text: "You are a concise travel assistant. Always check the weather with your tool before advising what to pack." }],
      messages: [...messages], // snapshot: the SDK captures by reference, so pass a copy to freeze each step's real input
      toolConfig: { tools },
    });

    const reply = res.output.message;
    messages.push(reply);

    if (res.stopReason !== "tool_use") {
      return reply.content.map((c: any) => c.text).filter(Boolean).join("");
    }

    const toolResults = reply.content
      .filter((c: any) => c.toolUse)
      .map((c: any) => {
        const weather = getWeather(c.toolUse.input.city);
        console.log(`  🔧 get_weather(${c.toolUse.input.city}) → ${weather}`);
        return { toolResult: { toolUseId: c.toolUse.toolUseId, content: [{ text: weather }] } };
      });
    messages.push({ role: "user", content: toolResults });
  }
}

// 3. Wrap the run in a workflow so the whole agent loop is one trace.
const answer = await papaya.run(
  { workflowKey: "trip-assistant", sessionId: "demo-session-1", userId: "demo-user", metadata: { model: MODEL } },
  () => agent("I'm visiting Tokyo and London tomorrow. What should I pack?"),
);

console.log("\n🍈 Agent:", answer);

// 4. Flush traces to Papaya before exit.
const result = await papaya.flush();
console.log("\n📤 Papaya:", result.status, `· ${result.traceCount} trace(s)`);
