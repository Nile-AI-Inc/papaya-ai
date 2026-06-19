# Papaya AI Python SDK

Python tracing SDK for production AI agents. This package mirrors the native Papaya trace envelope used by `@papaya-ai/tracing` and adds a LangChain/LangGraph callback handler for framework-level trace trees.

## Quick Start

### LangChain / LangGraph

```python
import os

from papaya_ai import Papaya
from papaya_ai.integrations.langchain import PapayaCallbackHandler

papaya = Papaya.init(api_key=os.environ["PAPAYA_API_KEY"])
callback = PapayaCallbackHandler(papaya, workflow_key="support_agent")

result = agent.invoke(
    {"messages": [{"role": "user", "content": "Help this customer"}]},
    config={"callbacks": [callback]},
)

papaya.flush()
```

The LangChain dependency is optional:

```sh
pip install "papaya-ai[langchain]"
```

### Provider SDK Wrappers

Use provider wrappers when your app calls SDK clients directly instead of going
through LangChain callbacks.

```python
import os

from papaya_ai import Papaya

papaya = Papaya.init(api_key=os.environ["PAPAYA_API_KEY"])
openai = papaya.openai(OpenAI())

try:
    with papaya.run({"workflowKey": "support_agent", "sessionId": session_id}):
        result = openai.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": user_message}],
        )
finally:
    papaya.flush()
```

The same wrapper shape is available as `papaya.openai(...)`,
`papaya.anthropic(...)`, `papaya.claude(...)`, `papaya.gemini(...)`, and
`papaya.bedrock(...)`.
