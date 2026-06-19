#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import pathlib
import sys


def add_local_python_sdk_if_available() -> None:
    here = pathlib.Path(__file__).resolve()
    candidates = [
        here.parents[2] / "papaya-ai-python" / "src",
        here.parents[1] / "packages" / "papaya-ai-python" / "src",
    ]
    for candidate in candidates:
        if candidate.exists():
            sys.path.insert(0, str(candidate))
            return


add_local_python_sdk_if_available()

from papaya_ai import Papaya
from papaya_ai.integrations.langchain import PapayaCallbackHandler

try:
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    from langchain_core.runnables import RunnableConfig, RunnableLambda
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


def message_row(message: SystemMessage | HumanMessage | AIMessage) -> dict[str, object]:
    return message.model_dump()


def build_agent(model: str) -> RunnableLambda:
    planner = ChatOpenAI(model=model, temperature=0).with_config({"run_name": "planner_chat_model"})
    final = ChatOpenAI(model=model, temperature=0).with_config({"run_name": "final_chat_model"})

    def invoke_agent(inputs: dict[str, object], config: RunnableConfig) -> dict[str, object]:
        user_message = str(inputs["message"])
        planner_messages = [
            SystemMessage(content="You are a concise support operations assistant. Think through the next action."),
            HumanMessage(content=user_message),
        ]
        print("Planner request:")
        print(json.dumps({"model": model, "messages": [message_row(message) for message in planner_messages]}, indent=2, sort_keys=True))
        plan = planner.invoke(planner_messages, config=config)
        print("\nPlanner response:")
        print(json.dumps(plan.model_dump(), indent=2, sort_keys=True, default=str))

        final_messages = [
            SystemMessage(content="You are a concise support operations assistant. Answer in two sentences or fewer."),
            HumanMessage(content=user_message),
            AIMessage(content=f"Planning note: {plan.content}"),
        ]
        print("\nFinal request:")
        print(json.dumps({"model": model, "messages": [message_row(message) for message in final_messages]}, indent=2, sort_keys=True))
        response = final.invoke(final_messages, config=config)
        print("\nFinal response:")
        print(json.dumps(response.model_dump(), indent=2, sort_keys=True, default=str))
        return {"output": str(response.content), "plannerOutput": str(plan.content)}

    return RunnableLambda(invoke_agent).with_config({"run_name": "RefundAgent"})


def main() -> int:
    require_env("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL") or "gpt-4.1-mini"
    papaya = Papaya.init(
        api_key=os.getenv("PAPAYA_API_KEY"),
        endpoint=os.getenv("PAPAYA_ENDPOINT") or "https://papaya.fyi/api/v1/ingest/traces",
        project=os.getenv("PAPAYA_PROJECT") or "papaya-python-demo",
        environment=os.getenv("PAPAYA_ENVIRONMENT") or "demo",
        service_name="python-langchain-callback-example",
        capture=os.getenv("PAPAYA_CAPTURE") or "redacted",
        debug=os.getenv("PAPAYA_DEBUG") == "1",
        metadata={"example": "python-langchain-callback"},
    )
    callback = PapayaCallbackHandler(
        papaya,
        workflow_key="refund_agent_callback_example",
        workflow_label="Refund agent callback example",
        session_id="demo-session-1",
        user_id="demo-user",
        metadata={"framework": "langchain", "mode": "callback-handler"},
    )

    try:
        result = build_agent(model).invoke(
            {"message": "Can we refund this customer after failed onboarding?"},
            config={"callbacks": [callback], "metadata": {"route": "examples/python-langchain-callback.py", "model": model}},
        )
        print("\nAgent answer:")
        print(result["output"])
    finally:
        flush_result = papaya.flush()

    print("\nPapaya flush:")
    print(json.dumps(flush_result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
