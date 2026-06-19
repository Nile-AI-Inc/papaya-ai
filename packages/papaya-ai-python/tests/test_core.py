import json
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from papaya_ai import Papaya


class PapayaCoreTest(unittest.TestCase):
    def test_flush_exports_native_python_batch_and_redacts_payloads(self):
        captured = []

        def transport(endpoint, headers, body):
            captured.append((endpoint, headers, json.loads(body.decode("utf-8"))))
            return 202, '{"accepted":1,"rejected":0}'

        papaya = Papaya.init(
            api_key="papaya-test-token",
            endpoint="https://papaya.example/api/v1/ingest/traces",
            project="checkout",
            environment="test",
            capture="redacted",
            service_name="python-test",
            metadata={"release": "sha-test"},
            transport=transport,
        )

        with papaya.run({"workflowKey": "checkout", "sessionId": "session-1"}) as trace:
            span = papaya.start_span(
                trace=trace,
                name="openai.chat",
                kind="llm",
                input_value=[{"role": "user", "content": "email ada@example.com token Bearer abc.secret.token"}],
                model_ref={"provider": "openai", "requested": "gpt-test"},
            )
            papaya.finish_span(
                span,
                "success",
                output_value={"choices": [{"message": {"role": "assistant", "content": "hello"}}]},
                usage={"inputTokens": 5, "outputTokens": 2, "totalTokens": 7},
                model_used="gpt-test",
            )

        result = papaya.flush()

        self.assertEqual(result["status"], "sent")
        self.assertEqual(len(captured), 1)
        endpoint, headers, body = captured[0]
        self.assertEqual(endpoint, "https://papaya.example/api/v1/ingest/traces")
        self.assertEqual(headers["Authorization"], "Bearer papaya-test-token")
        self.assertEqual(body["schemaVersion"], "2026-06-05")
        self.assertEqual(body["sdk"]["name"], "papaya-ai")
        self.assertEqual(body["sdk"]["language"], "python")
        exported = json.dumps(body)
        self.assertNotIn("ada@example.com", exported)
        self.assertNotIn("Bearer abc.secret.token", exported)
        self.assertIn("[redacted-email]", exported)
        self.assertIn("Bearer [redacted-token]", exported)
        self.assertEqual(body["traces"][0]["workflowKey"], "checkout")
        llm_span = next(span for span in body["traces"][0]["spans"] if span["kind"] == "llm")
        self.assertEqual(llm_span["usage"]["inputTokens"], 5)

    def test_missing_api_key_keeps_completed_trace_for_later_flush(self):
        papaya = Papaya.init(endpoint="https://papaya.example/api/v1/ingest/traces")

        with papaya.run({"workflowKey": "missing-key"}):
            pass

        first = papaya.flush()
        second = papaya.flush()

        self.assertEqual(first, {"status": "skipped", "traceCount": 1, "reason": "missing_api_key"})
        self.assertEqual(second, {"status": "skipped", "traceCount": 1, "reason": "missing_api_key"})

    def test_failed_flush_keeps_completed_trace_for_retry(self):
        calls = []

        def transport(endpoint, headers, body):
            calls.append(json.loads(body.decode("utf-8")))
            if len(calls) == 1:
                return 503, '{"error":"temporarily unavailable"}'
            return 202, '{"accepted":1,"rejected":0}'

        papaya = Papaya.init(
            api_key="papaya-test-token",
            endpoint="https://papaya.example/api/v1/ingest/traces",
            transport=transport,
        )

        with papaya.run({"workflowKey": "retryable"}):
            pass

        first = papaya.flush()
        second = papaya.flush()

        self.assertEqual(first["status"], "failed")
        self.assertEqual(first["traceCount"], 1)
        self.assertEqual(second["status"], "sent")
        self.assertEqual(second["traceCount"], 1)
        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
