#!/usr/bin/env python3
import json
import os
import pathlib
import sys

sdk_src = pathlib.Path(__file__).resolve().parents[2] / "papaya-ai-python" / "src"
if sdk_src.exists():
    sys.path.insert(0, str(sdk_src))

from papaya_ai import Papaya
from papaya_ai.integrations.langchain import PapayaCallbackHandler

try:
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_core.runnables import RunnableLambda
    from langchain_openai import ChatOpenAI
except ImportError as error:
    raise SystemExit(
        "langchain-openai is required for this real LLM example. "
        "Install it with: python3 -m pip install 'langchain-openai>=0.3'"
    ) from error


def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise SystemExit(f"{name} is required to run this real LLM example.")
    return value


require_env("OPENAI_API_KEY")
model = os.getenv("OPENAI_MODEL") or "gpt-4.1-mini"
messages = [
    SystemMessage(content="You are a concise support operations assistant."),
    HumanMessage(content="Can we refund a customer 700 dollars? Answer in one sentence."),
]


def run_agent(_inputs, config):
    chat = ChatOpenAI(model=model, temperature=0).with_config({"run_name": "support_chat_model"})
    print("LangChain request:")
    print(json.dumps({"model": model, "messages": [message.model_dump() for message in messages]}, indent=2, sort_keys=True))
    response = chat.invoke(messages, config=config)
    print("\nLangChain response:")
    print(json.dumps(response.model_dump(), indent=2, sort_keys=True, default=str))
    return response.content


papaya = Papaya.init(api_key=os.getenv("PAPAYA_API_KEY"), project="papaya-python-demo", environment="demo")
callback = PapayaCallbackHandler(papaya, workflow_key="minimal_langchain_callback", session_id="demo-session")
agent = RunnableLambda(run_agent).with_config({"run_name": "SupportAgent"})

try:
    print("\nLangChain answer:", agent.invoke({}, config={"callbacks": [callback]}))
finally:
    print("\nPapaya flush:", papaya.flush())
