import json
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from papaya_ai import Papaya
from papaya_ai.integrations.langchain import PapayaCallbackHandler


class FakeMessage:
    def __init__(self, role, content):
        self.type = role
        self.content = content


class FakeLlmResult:
    generations = [[{"text": "The refund needs manager approval."}]]
    llm_output = {
        "token_usage": {
            "prompt_tokens": 12,
            "completion_tokens": 8,
            "total_tokens": 20,
        },
        "model_name": "gpt-test",
    }


class PapayaLangChainCallbackTest(unittest.TestCase):
    def test_callback_handler_exports_langchain_run_tree(self):
        captured = []

        def transport(endpoint, headers, body):
            captured.append(json.loads(body.decode("utf-8")))
            return 202, '{"accepted":1,"rejected":0}'

        papaya = Papaya.init(
            api_key="papaya-test-token",
            endpoint="https://papaya.example/api/v1/ingest/traces",
            capture="full",
            transport=transport,
        )
        handler = PapayaCallbackHandler(
            papaya,
            workflow_key="sales_manager_chat",
            session_id="session-1",
            user_id="user-1",
            metadata={"route": "/api/chat"},
        )

        handler.on_chain_start(
            {"name": "SalesManagerAgent"},
            {"messages": [{"role": "user", "content": "Can we refund account 42?"}]},
            run_id="root",
            metadata={"tenant": "demo"},
        )
        handler.on_chat_model_start(
            {"name": "ChatOpenAI"},
            [[FakeMessage("human", "Can we refund account 42?")]],
            run_id="llm-1",
            parent_run_id="root",
            invocation_params={"model": "gpt-test"},
        )
        handler.on_llm_end(FakeLlmResult(), run_id="llm-1")
        handler.on_tool_start({"name": "lookup_policy"}, "refund policy", run_id="tool-1", parent_run_id="root")
        handler.on_tool_end({"approval_required": True}, run_id="tool-1")
        handler.on_chain_end({"output": "The refund needs manager approval."}, run_id="root")

        result = papaya.flush()

        self.assertEqual(result["status"], "sent")
        batch = captured[0]
        trace = batch["traces"][0]
        self.assertEqual(trace["workflowKey"], "sales_manager_chat")
        self.assertEqual(trace["sessionId"], "session-1")
        self.assertEqual(trace["userId"], "user-1")
        spans = trace["spans"]
        self.assertEqual([span["kind"] for span in spans], ["workflow", "llm", "tool"])
        root_span = spans[0]
        llm_span = spans[1]
        tool_span = spans[2]
        self.assertEqual(llm_span["parentSpanId"], root_span["spanId"])
        self.assertEqual(tool_span["parentSpanId"], root_span["spanId"])
        self.assertEqual(llm_span["modelRef"]["requested"], "gpt-test")
        self.assertEqual(llm_span["usage"]["inputTokens"], 12)
        self.assertEqual(llm_span["usage"]["outputTokens"], 8)
        self.assertEqual(tool_span["tool"]["name"], "lookup_policy")
        exported = json.dumps(batch)
        self.assertIn("The refund needs manager approval.", exported)


if __name__ == "__main__":
    unittest.main()
