import json
import pathlib
import sys
import unittest
from typing import TypedDict

import boto3
from anthropic.types import Message as AnthropicMessage
from botocore.stub import Stubber
from google.genai import types as google_types
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from langgraph.graph import END, START, StateGraph
from openai.types.chat import ChatCompletion

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from papaya_ai import Papaya
from papaya_ai.integrations.langchain import PapayaCallbackHandler


SUPPORTED_PYTHON_COMBINATIONS = frozenset(
    {"openai", "anthropic", "gemini", "bedrock", "langchain", "langgraph"}
)


class _MethodHarness:
    def __init__(self, response):
        self.response = response

    def create(self, **_kwargs):
        return self.response

    def generate_content(self, **_kwargs):
        return self.response


class _OpenAIHarness:
    def __init__(self, response):
        self.chat = type("ChatResource", (), {"completions": _MethodHarness(response)})()


class _AnthropicHarness:
    def __init__(self, response):
        self.messages = _MethodHarness(response)


class _GeminiHarness:
    def __init__(self, response):
        self.models = _MethodHarness(response)


class _GraphState(TypedDict):
    value: str


class PapayaProviderMatrixTest(unittest.TestCase):
    def _papaya(self):
        captured = []

        def transport(_endpoint, _headers, body):
            captured.append(json.loads(body.decode("utf-8")))
            return 202, '{"accepted":1,"rejected":0}'

        return (
            Papaya.init(
                api_key="papaya-test-token",
                endpoint="https://papaya.example/api/v1/ingest/traces",
                capture="full",
                transport=transport,
            ),
            captured,
        )

    def _assert_provider_span(self, captured, provider, usage, raw_marker):
        trace = captured[0]["traces"][0]
        span = next(item for item in trace["spans"] if item["kind"] == "llm")
        self.assertEqual(span["modelRef"]["provider"], provider)
        self.assertEqual(
            {key: span["usage"].get(key) for key in ("inputTokens", "outputTokens", "totalTokens")},
            usage,
        )
        self.assertIsInstance(span["outputPayload"]["value"], dict)
        self.assertIn(raw_marker, json.dumps(span["outputPayload"]["value"]))

    def test_all_supported_python_combinations_use_real_library_types(self):
        covered = set()

        openai_response = ChatCompletion.model_validate(
            {
                "id": "chatcmpl-matrix",
                "object": "chat.completion",
                "created": 1_700_000_000,
                "model": "gpt-5.4-mini",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "logprobs": None,
                        "message": {"role": "assistant", "content": "openai-runtime-response"},
                    }
                ],
                "usage": {"prompt_tokens": 9, "completion_tokens": 4, "total_tokens": 13},
            }
        )
        self.assertIsInstance(openai_response, ChatCompletion)
        papaya, captured = self._papaya()
        response = papaya.openai(_OpenAIHarness(openai_response)).chat.completions.create(
            model="gpt-5.4-mini", messages=[]
        )
        self.assertIs(response, openai_response)
        self.assertEqual(papaya.flush()["status"], "sent")
        self._assert_provider_span(
            captured,
            "openai",
            {"inputTokens": 9, "outputTokens": 4, "totalTokens": 13},
            "openai-runtime-response",
        )
        covered.add("openai")

        anthropic_response = AnthropicMessage.model_validate(
            {
                "id": "msg_matrix",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-5",
                "content": [{"type": "text", "text": "anthropic-runtime-response"}],
                "container": None,
                "stop_details": None,
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 2,
                },
            }
        )
        self.assertIsInstance(anthropic_response, AnthropicMessage)
        papaya, captured = self._papaya()
        response = papaya.anthropic(_AnthropicHarness(anthropic_response)).messages.create(
            model="claude-sonnet-4-5", max_tokens=32, messages=[]
        )
        self.assertIs(response, anthropic_response)
        self.assertEqual(papaya.flush()["status"], "sent")
        self._assert_provider_span(
            captured,
            "claude",
            {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
            "anthropic-runtime-response",
        )
        papaya, alias_captured = self._papaya()
        self.assertIs(
            papaya.claude(_AnthropicHarness(anthropic_response)).messages.create(
                model="claude-sonnet-4-5", max_tokens=32, messages=[]
            ),
            anthropic_response,
        )
        self.assertEqual(papaya.flush()["status"], "sent")
        self.assertEqual(
            next(item for item in alias_captured[0]["traces"][0]["spans"] if item["kind"] == "llm")["modelRef"]["provider"],
            "claude",
        )
        covered.add("anthropic")

        gemini_response = google_types.GenerateContentResponse(
            modelVersion="gemini-3.5-flash",
            responseId="gemini-matrix",
            candidates=[
                google_types.Candidate(
                    content=google_types.Content(
                        role="model", parts=[google_types.Part(text="gemini-runtime-response")]
                    ),
                    finishReason="STOP",
                )
            ],
            usageMetadata=google_types.GenerateContentResponseUsageMetadata(
                promptTokenCount=11,
                candidatesTokenCount=6,
                totalTokenCount=17,
                cachedContentTokenCount=2,
                thoughtsTokenCount=3,
            ),
        )
        self.assertIsInstance(gemini_response, google_types.GenerateContentResponse)
        papaya, captured = self._papaya()
        response = papaya.gemini(_GeminiHarness(gemini_response)).models.generate_content(
            model="gemini-3.5-flash", contents="hello"
        )
        self.assertIs(response, gemini_response)
        self.assertEqual(papaya.flush()["status"], "sent")
        self._assert_provider_span(
            captured,
            "gemini",
            {"inputTokens": 11, "outputTokens": 6, "totalTokens": 17},
            "gemini-runtime-response",
        )
        covered.add("gemini")

        bedrock_request = {
            "modelId": "anthropic.claude-3-5-sonnet-20241022-v2:0",
            "messages": [{"role": "user", "content": [{"text": "hello"}]}],
        }
        bedrock_response = {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{"text": "bedrock-runtime-response"}],
                }
            },
            "stopReason": "end_turn",
            "usage": {"inputTokens": 12, "outputTokens": 7, "totalTokens": 19},
            "metrics": {"latencyMs": 20},
        }
        bedrock_client = boto3.client(
            "bedrock-runtime",
            region_name="us-east-1",
            aws_access_key_id="test",
            aws_secret_access_key="test",
        )
        with Stubber(bedrock_client) as stubber:
            stubber.add_response("converse", bedrock_response, bedrock_request)
            papaya, captured = self._papaya()
            response = papaya.bedrock(bedrock_client).converse(**bedrock_request)
        self.assertIsInstance(response, dict)
        self.assertEqual(papaya.flush()["status"], "sent")
        self._assert_provider_span(
            captured,
            "bedrock",
            {"inputTokens": 12, "outputTokens": 7, "totalTokens": 19},
            "bedrock-runtime-response",
        )
        covered.add("bedrock")

        papaya, captured = self._papaya()
        callback = PapayaCallbackHandler(papaya, workflow_key="langchain_matrix")
        langchain_message = AIMessage(
            id="lc-matrix",
            content="langchain-runtime-response",
            response_metadata={"model_name": "gpt-5.4-mini"},
            usage_metadata={
                "input_tokens": 13,
                "output_tokens": 8,
                "total_tokens": 21,
                "input_token_details": {"cache_read": 3},
                "output_token_details": {"reasoning": 4},
            },
        )
        langchain_result = LLMResult(
            generations=[[ChatGeneration(message=langchain_message)]],
            llm_output={"model_name": "gpt-5.4-mini"},
        )
        callback.on_chat_model_start(
            {"name": "ChatOpenAI"},
            [[HumanMessage(content="hello")]],
            run_id="langchain-matrix",
            invocation_params={"model": "gpt-5.4-mini"},
        )
        callback.on_llm_end(langchain_result, run_id="langchain-matrix")
        self.assertEqual(papaya.flush()["status"], "sent")
        root = captured[0]["traces"][0]["spans"][0]
        self.assertEqual(root["usage"]["totalTokens"], 21)
        self.assertEqual(root["usage"]["cacheReadInputTokens"], 3)
        self.assertEqual(root["usage"]["reasoningTokens"], 4)
        self.assertEqual(
            root["outputPayload"]["value"][0][0]["content"],
            "langchain-runtime-response",
        )
        self.assertNotIn("usage_metadata", root["outputPayload"]["value"][0][0])
        covered.add("langchain")

        papaya, captured = self._papaya()
        callback = PapayaCallbackHandler(papaya, workflow_key="langgraph_matrix")
        graph = (
            StateGraph(_GraphState)
            .add_node("append_result", lambda state: {"value": state["value"] + "-langgraph-runtime-response"})
            .add_edge(START, "append_result")
            .add_edge("append_result", END)
            .compile()
        )
        graph_result = graph.invoke({"value": "start"}, config={"callbacks": [callback]})
        self.assertEqual(graph_result["value"], "start-langgraph-runtime-response")
        self.assertEqual(papaya.flush()["status"], "sent")
        self.assertIn("langgraph-runtime-response", json.dumps(captured[0]))
        covered.add("langgraph")

        self.assertSetEqual(covered, SUPPORTED_PYTHON_COMBINATIONS)


if __name__ == "__main__":
    unittest.main()
