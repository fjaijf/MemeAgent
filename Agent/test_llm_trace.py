from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from memeagent.llm import ChatResponse
from memeagent.llm_trace import LLMTraceRecorder, TracedLLM


class FakeLLM:
    model = "fake-model"

    def invoke(self, messages: list[object]) -> ChatResponse:
        return ChatResponse(
            content="""
ITERATION_CONFIDENCE:
- 0.92

SHOULD_FINALIZE:
- yes

CONFIDENCE_REASON:
- Evidence is sufficient.
""".strip()
        )

    def stream(self, messages: list[object]):
        yield "hello"
        yield " world"


class LLMTraceTests(unittest.TestCase):
    def test_invoke_records_structured_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.json"
            recorder = LLMTraceRecorder(trace_path, metadata={"topic": "test"})
            llm = TracedLLM(FakeLLM(), recorder=recorder, name="controller")

            response = llm.invoke(
                [
                    SystemMessage(content="system"),
                    HumanMessage(content="You are the controller model for MemeAgent."),
                ]
            )

            self.assertIn("ITERATION_CONFIDENCE", response.content)
            data = json.loads(trace_path.read_text(encoding="utf-8"))
            self.assertEqual(1, len(data["calls"]))
            call = data["calls"][0]
            self.assertEqual("controller", call["llm_name"])
            self.assertEqual("invoke", call["operation"])
            self.assertEqual("controller_iteration_planning", call["call_kind"])
            self.assertEqual("completed", call["status"])
            self.assertEqual("fake-model", call["model"])
            self.assertEqual("0.92", call["visible_reasoning_summary"]["iteration_confidence"].lstrip("- "))

    def test_stream_records_combined_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.json"
            recorder = LLMTraceRecorder(trace_path)
            llm = TracedLLM(FakeLLM(), recorder=recorder, name="primary")

            chunks = list(llm.stream([HumanMessage(content="final analysis")]))

            self.assertEqual(["hello", " world"], chunks)
            data = json.loads(trace_path.read_text(encoding="utf-8"))
            call = data["calls"][0]
            self.assertEqual("stream", call["operation"])
            self.assertEqual("hello world", call["output"])
            self.assertEqual(2, call["stream_chunk_count"])

    def test_image_data_url_is_redacted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.json"
            recorder = LLMTraceRecorder(trace_path)
            llm = TracedLLM(FakeLLM(), recorder=recorder, name="primary")

            llm.invoke(
                [
                    HumanMessage(
                        content=[
                            {"type": "text", "text": "describe"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "data:image/png;base64,ZmFrZS1pbWFnZS1ieXRlcw=="
                                },
                            },
                        ]
                    )
                ]
            )

            raw = trace_path.read_text(encoding="utf-8")
            self.assertNotIn("ZmFrZS1pbWFnZS1ieXRlcw==", raw)
            data = json.loads(raw)
            image = data["calls"][0]["messages"][0]["content"][1]["image_url"]
            self.assertTrue(image["redacted"])
            self.assertEqual("image/png", image["mime_type"])
            self.assertEqual(16, image["byte_length"])


if __name__ == "__main__":
    unittest.main()
