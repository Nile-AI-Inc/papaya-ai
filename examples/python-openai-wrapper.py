#!/usr/bin/env python3
import json
import os
import pathlib
import sys

sdk_src = pathlib.Path(__file__).resolve().parents[2] / "papaya-ai-python" / "src"
if sdk_src.exists():
    sys.path.insert(0, str(sdk_src))

from papaya_ai import Papaya

try:
    from openai import OpenAI
except ImportError as error:
    raise SystemExit(
        "openai is required for this real LLM example. "
        "Install it with: python3 -m pip install 'openai>=1.0'"
    ) from error


def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise SystemExit(f"{name} is required to run this real LLM example.")
    return value


openai_api_key = require_env("OPENAI_API_KEY")
model = os.getenv("OPENAI_MODEL") or "gpt-4.1-mini"
request = {
    "model": model,
    "messages": [
        {"role": "system", "content": "You are a concise support operations assistant."},
        {"role": "user", "content": "Can we refund this customer 700 dollars? Answer in one sentence."},
    ],
    "temperature": 0,
}

papaya = Papaya.init(
    api_key=os.getenv("PAPAYA_API_KEY"),
    project="papaya-python-demo",
    environment="demo",
)
openai = papaya.openai(OpenAI(api_key=openai_api_key), {"workflowKey": "python_openai_wrapper"})

try:
    print("OpenAI request:")
    print(json.dumps(request, indent=2, sort_keys=True))
    with papaya.run({"sessionId": "demo-session-py", "userId": "demo-user"}):
        response = openai.chat.completions.create(
            **request,
            papaya={"metadata": {"example": "python-openai-wrapper"}},
        )
    response_json = response.model_dump()
    print("\nOpenAI response:")
    print(json.dumps(response_json, indent=2, sort_keys=True))
    print("\nOpenAI wrapper answer:", response.choices[0].message.content)
finally:
    print("\nPapaya flush:", papaya.flush())
