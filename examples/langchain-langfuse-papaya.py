#!/usr/bin/env python3

import os
import pathlib
import sys
from uuid import uuid4

sdk_src = pathlib.Path(__file__).resolve().parents[2] / "papaya-ai-python" / "src"
if sdk_src.exists():
    sys.path.insert(0, str(sdk_src))

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langfuse import get_client
from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler

from papaya_ai import Papaya
from papaya_ai.integrations.langchain import PapayaCallbackHandler


openai_api_key = os.environ["OPENAI_API_KEY"]
papaya_api_key = os.environ["PAPAYA_API_KEY"]
os.environ["LANGFUSE_PUBLIC_KEY"]
os.environ["LANGFUSE_SECRET_KEY"]

run_id = str(uuid4())
session_id = f"session_{run_id}"
user_id = "demo-user"

papaya = Papaya.init(api_key=papaya_api_key, project="papaya-demo", environment="demo")
papaya_handler = PapayaCallbackHandler(
    papaya,
    workflow_key="langchain_langfuse_papaya_demo",
    session_id=session_id,
    user_id=user_id,
    metadata={"example": "langchain-langfuse-papaya", "runId": run_id},
)
langfuse = get_client()
langfuse_handler = LangfuseCallbackHandler()

model = ChatOpenAI(
    api_key=openai_api_key,
    model=os.getenv("OPENAI_MODEL") or "gpt-4.1-mini",
    temperature=0,
).with_config({"run_name": "write_customer_reply"})

lookup_order = Tool.from_function(
    name="lookup_order",
    description="Look up the current state of an order.",
    func=lambda order_id: f"{order_id}: delayed in customs, refund eligible",
)


def run_agent(inputs: dict[str, str], config: RunnableConfig) -> str:
    order = lookup_order.invoke("order_123", config=config)
    answer = model.invoke(
        [
            SystemMessage(content="You are a concise support agent. Answer in one sentence."),
            HumanMessage(content=f"{inputs['question']}\n\nOrder context: {order}"),
        ],
        config=config,
    )
    return str(answer.content)


agent = RunnableLambda(run_agent).with_config({"run_name": "TinySupportAgent"})

try:
    answer = agent.invoke(
        {"question": "Should we refund this customer?"},
        config={
            "callbacks": [langfuse_handler, papaya_handler],
            "metadata": {
                "runId": run_id,
                "route": "examples/langchain-langfuse-papaya.py",
                "langfuse_user_id": user_id,
                "langfuse_session_id": session_id,
                "langfuse_tags": ["oss-example", "papaya-parallel"],
            },
            "tags": ["oss-example"],
        },
    )
    print({"answer": answer, "runId": run_id, "sessionId": session_id})
finally:
    print("Papaya flush:", papaya.flush())
    langfuse.flush()
