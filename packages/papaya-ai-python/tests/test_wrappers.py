import asyncio
import json
import pathlib
import sys
import unittest

from openai.types.chat import ChatCompletion

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from papaya_ai import Papaya


def openai_chat_completion(*, response_id, model, input_tokens, output_tokens, content):
    return ChatCompletion.model_validate(
        {
            "id": response_id,
            "object": "chat.completion",
            "created": 1_700_000_000,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "logprobs": None,
                    "message": {"role": "assistant", "content": content},
                }
            ],
            "usage": {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
        }
    )


class OpenAICompletionsHarness:
    def __init__(self):
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return openai_chat_completion(
            response_id="chatcmpl-test",
            model="gpt-test-used",
            input_tokens=9,
            output_tokens=4,
            content="hello back",
        )

    async def acreate(self, **kwargs):
        self.requests.append(kwargs)
        await asyncio.sleep(0)
        return openai_chat_completion(
            response_id="chatcmpl-async-test",
            model=kwargs["model"],
            input_tokens=3,
            output_tokens=2,
            content="async hello",
        )


class OpenAIChatHarness:
    def __init__(self):
        self.completions = OpenAICompletionsHarness()


class OpenAIClientHarness:
    def __init__(self):
        self.chat = OpenAIChatHarness()


class PapayaWrapperTest(unittest.TestCase):
    def test_provider_wrapper_captures_nested_sync_call_and_strips_papaya_option(self):
        captured = []

        def transport(endpoint, headers, body):
            captured.append(json.loads(body.decode("utf-8")))
            return 202, '{"accepted":1,"rejected":0}'

        provider = OpenAIClientHarness()
        papaya = Papaya.init(
            api_key="papaya-test-token",
            endpoint="https://papaya.example/api/v1/ingest/traces",
            capture="redacted",
            transport=transport,
            metadata={"release": "sha-test"},
        )
        client = papaya.openai(provider, {"workflowKey": "support_agent", "metadata": {"route": "wrapper-test"}})

        with papaya.run({"sessionId": "session-1", "userId": "user-1"}):
            result = client.chat.completions.create(
                model="gpt-test",
                messages=[{"role": "user", "content": "email ada@example.com token Bearer abc.secret.token"}],
                papaya={"metadata": {"attempt": 1}},
            )

        self.assertIsInstance(result, ChatCompletion)
        self.assertEqual(result.model, "gpt-test-used")
        self.assertEqual(len(provider.chat.completions.requests), 1)
        self.assertNotIn("papaya", provider.chat.completions.requests[0])

        flush = papaya.flush()
        self.assertEqual(flush["status"], "sent")
        trace = captured[0]["traces"][0]
        self.assertEqual(trace["sessionId"], "session-1")
        span = next(span for span in trace["spans"] if span["kind"] == "llm")
        self.assertEqual(span["name"], "openai.chat.completions.create")
        self.assertEqual(span["modelRef"], {"provider": "openai", "requested": "gpt-test", "used": "gpt-test-used"})
        self.assertEqual(span["usage"]["inputTokens"], 9)
        self.assertIsInstance(span["outputPayload"]["value"], dict)
        self.assertEqual(span["outputPayload"]["value"]["choices"][0]["message"]["content"], "hello back")
        self.assertEqual(span["attributes"]["metadata"], {"release": "sha-test", "route": "wrapper-test", "attempt": 1})
        exported = json.dumps(captured[0])
        self.assertNotIn("ada@example.com", exported)
        self.assertIn("[redacted-email]", exported)

    def test_provider_wrapper_creates_implicit_trace_for_async_call(self):
        captured = []

        def transport(endpoint, headers, body):
            captured.append(json.loads(body.decode("utf-8")))
            return 202, '{"accepted":1,"rejected":0}'

        provider = OpenAIClientHarness()
        papaya = Papaya.init(
            api_key="papaya-test-token",
            endpoint="https://papaya.example/api/v1/ingest/traces",
            transport=transport,
        )
        client = papaya.openai(provider)

        async def run():
            return await client.chat.completions.acreate(model="gpt-async", messages=[])

        result = asyncio.run(run())
        self.assertIsInstance(result, ChatCompletion)
        self.assertEqual(result.choices[0].message.content, "async hello")
        flush = papaya.flush()
        self.assertEqual(flush["status"], "sent")
        trace = captured[0]["traces"][0]
        self.assertEqual(trace["workflowKey"], "openai.chat.completions.acreate")
        span = next(span for span in trace["spans"] if span["kind"] == "llm")
        self.assertEqual(span["usage"]["totalTokens"], 5)


if __name__ == "__main__":
    unittest.main()
