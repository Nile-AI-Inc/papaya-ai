# Examples

Small, self-contained scripts showing how to add Papaya tracing to an app.
Each one initializes Papaya, makes a provider or framework call, and flushes the trace.

| File | Shows |
|------|-------|
| [`gemini-sdk.ts`](gemini-sdk.ts) | Wrap a provider SDK client — `papaya.gemini(new GoogleGenAI(...))` |
| [`gemini-fetch.ts`](gemini-fetch.ts) | Trace a raw HTTP model call — `papaya.fetch(globalThis.fetch)` |
| [`gemini-fetch-raw.ts`](gemini-fetch-raw.ts) | The same call **without** the SDK, for comparison (no tracing) |
| [`error-handling.ts`](error-handling.ts) | Flush in `finally` so failed jobs still send their trace |
| [`agent.ts`](agent.ts) | A Bedrock tool-use agent loop captured as one workflow — `papaya.bedrock(...)` |
| [`openai-wrapper.py`](openai-wrapper.py) | Minimal Python provider SDK wrapper shape — `papaya.openai(...)` |
| [`langchain-callback.ts`](langchain-callback.ts) | Capture a real TypeScript LangChain runnable tree with `PapayaCallbackHandler` |
| [`langchain-callback-minimal.py`](langchain-callback-minimal.py) | Minimal Python LangChain runnable captured with `PapayaCallbackHandler` |
| [`langchain-callback.py`](langchain-callback.py) | Larger Python LangChain runnable tree with two real chat model calls |
| [`langchain-langfuse-papaya.ts`](langchain-langfuse-papaya.ts) | Send one LangChain agent run to both Langfuse and Papaya callback handlers |
| [`langchain-langfuse-papaya.py`](langchain-langfuse-papaya.py) | Python version of the Langfuse + Papaya callback handler example |

## Run TypeScript Examples

```sh
# from this folder
npm i @papaya-ai/tracing @aws-sdk/client-bedrock-runtime @google/genai @langchain/core @langchain/openai openai tsx

cp .env.example .env   # then fill in your keys
npx tsx --env-file=.env gemini-sdk.ts
```

Each script reads `PAPAYA_API_KEY` from the environment so you do not need to edit
source files or paste secrets into the examples.

The TypeScript LangChain callback example uses a real OpenAI-backed LangChain
chat model, so it requires `OPENAI_API_KEY`:

```sh
npx tsx --env-file=.env langchain-callback.ts
```

To test Papaya beside Langfuse on the same LangChain run:

```sh
npm i @papaya-ai/tracing @langchain/core @langchain/openai @langfuse/langchain @langfuse/otel @opentelemetry/sdk-node openai tsx
export LANGFUSE_BASE_URL=http://127.0.0.1:3000   # local Langfuse only
npx tsx --env-file=.env langchain-langfuse-papaya.ts
```

## Run The Python Callback Example

From this folder:

```sh
python3 -m pip install "openai>=1.0" "langchain-openai>=0.3"
python3 openai-wrapper.py
python3 langchain-callback-minimal.py
```

The Python wrapper and callback examples use real OpenAI calls, so they require
`OPENAI_API_KEY`. Without `PAPAYA_API_KEY`, the scripts still run the provider
call but report that Papaya export was skipped. To send the example trace, pass a
Papaya key:

```sh
PAPAYA_API_KEY=ppy_live_... python3 langchain-callback-minimal.py
```

Run `langchain-callback.py` when you want the fuller tree with two real
chat model calls in one LangChain runnable.

To test Papaya beside Langfuse on the same Python LangChain run:

```sh
python3 -m pip install "langfuse>=3" "langchain>=0.3" "langchain-openai>=0.3"
export LANGFUSE_HOST=http://127.0.0.1:3000   # local Langfuse only
python3 langchain-langfuse-papaya.py
```

When running from this monorepo, the example automatically imports the unpublished
local Python SDK from `packages/papaya-ai-python/src`. After the Python package is
published, install the callback extra instead:

```sh
pip install "papaya-ai[langchain]"
```

## LangChain Callback Model

Use the callback handler when you already run agents with LangChain or LangGraph
and want Papaya to capture the framework run tree. LangChain sends callback events
with run IDs and parent run IDs; `PapayaCallbackHandler` maps those IDs to a
native Papaya trace:

- the root chain or graph run becomes the workflow span
- child chat/LLM runs become `llm` spans with model and token usage
- tool calls become `tool` spans when your LangChain app uses tools
- retrievers become `retrieval` spans when your LangChain app uses retrievers
- inputs and outputs are stored as Papaya payloads, not as one serialized JSON string

Minimal real app shape:

```python
import os

from papaya_ai import Papaya
from papaya_ai.integrations.langchain import PapayaCallbackHandler

papaya = Papaya.init(
    api_key=os.environ["PAPAYA_API_KEY"],
    project="support-agent",
)

callback = PapayaCallbackHandler(
    papaya,
    workflow_key="support_agent",
    session_id=session_id,
    user_id=user_id,
)

try:
    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_message}]},
        config={"callbacks": [callback]},
    )
finally:
    papaya.flush()
```

For LangGraph, pass the same callback in the graph config:

```python
graph.invoke(state, config={"callbacks": [callback]})
```

Prefer one capture path for a given execution. The callback handler captures the
LangChain/LangGraph tree. Provider wrappers such as `papaya.openai(...)` or
`papaya.bedrock(...)` capture direct SDK calls. If you combine both around the
same model call, you may create duplicate LLM spans unless you intentionally
disable LLM capture in the callback handler.

## Keys / env

- **Papaya** — `PAPAYA_API_KEY` for trace export.
- **Gemini** (`gemini-*`, `error-handling`) — `GEMINI_API_KEY`, from https://aistudio.google.com/apikey
- **Bedrock** (`agent.ts`) — `AWS_PROFILE` + `AWS_REGION`, with access to the Claude inference profile.
- **TypeScript callback** (`langchain-callback.ts`) — `OPENAI_API_KEY` for the real model call, `PAPAYA_API_KEY` to export.
- **TypeScript Langfuse + Papaya callback** (`langchain-langfuse-papaya.ts`) — `OPENAI_API_KEY`, `PAPAYA_API_KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and optional `LANGFUSE_BASE_URL`.
- **Python provider wrapper** (`openai-wrapper.py`) — `OPENAI_API_KEY` for the real model call, `PAPAYA_API_KEY` to export.
- **Python callback** (`langchain-callback-minimal.py`, `langchain-callback.py`) — `OPENAI_API_KEY` for the real model call, `PAPAYA_API_KEY` to export, plus optional `PAPAYA_ENDPOINT` for local ingest testing.
- **Python Langfuse + Papaya callback** (`langchain-langfuse-papaya.py`) — `OPENAI_API_KEY`, `PAPAYA_API_KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and optional `LANGFUSE_HOST`.
