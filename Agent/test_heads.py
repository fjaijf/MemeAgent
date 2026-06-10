from __future__ import annotations

import unittest

from memeagent.heads import (
    HEADS,
    MemeAnalysisHeadRunner,
    format_head_results,
    normalize_head_names,
)


class FakeResponse:
    content = "fake head output"


class FakeLLM:
    def __init__(self) -> None:
        self.messages = []

    def invoke(self, messages):
        self.messages.append(messages)
        return FakeResponse()


class HeadTests(unittest.TestCase):
    def test_normalize_head_names_supports_commas_and_all(self) -> None:
        self.assertEqual(
            normalize_head_names(["harmfulness,sentiment", "intent"]),
            ["harmfulness", "sentiment", "intent"],
        )
        self.assertEqual(normalize_head_names(["all"]), list(HEADS.keys()))

    def test_unknown_head_raises_clear_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown task head"):
            normalize_head_names(["unknown"])

    def test_runner_uses_shared_evidence_context(self) -> None:
        llm = FakeLLM()
        runner = MemeAnalysisHeadRunner(llm=llm, system_prompt="system")

        results = runner.run_heads(
            head_names=["harmfulness"],
            topic="test topic",
            evidence_context="shared evidence [Image]",
            input_mode="text_only",
        )

        self.assertEqual(results[0].name, "harmfulness")
        self.assertEqual(results[0].output, "fake head output")
        user_prompt = llm.messages[0][1].content
        self.assertIn("shared evidence [Image]", user_prompt)
        self.assertIn("test topic", user_prompt)

    def test_format_head_results_keeps_outputs_separate(self) -> None:
        llm = FakeLLM()
        runner = MemeAnalysisHeadRunner(llm=llm, system_prompt="system")
        results = runner.run_heads(
            head_names=["harmfulness", "sentiment"],
            topic="topic",
            evidence_context="evidence",
            input_mode="text_only",
        )

        formatted = format_head_results(results)

        self.assertIn("## Harmfulness Analysis", formatted)
        self.assertIn("## Sentiment Analysis", formatted)
        self.assertEqual(formatted.count("fake head output"), 2)


if __name__ == "__main__":
    unittest.main()
