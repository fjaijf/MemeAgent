from __future__ import annotations

import unittest

from memeagent.heads import (
    HEADS,
    MemeAnalysisHeadRunner,
    format_head_results,
    normalize_head_names,
)


class FakeResponse:
    def __init__(self, content: str = "fake head output") -> None:
        self.content = content


class FakeLLM:
    def __init__(self) -> None:
        self.messages = []

    def invoke(self, messages):
        self.messages.append(messages)
        return FakeResponse()


class FakeHarmfulnessLLM:
    def __init__(self) -> None:
        self.messages = []

    def invoke(self, messages):
        self.messages.append(messages)
        if len(self.messages) <= 6:
            return FakeResponse(
                """
{
  "harmful_probability": 0.72,
  "not_harmful_probability": 0.18,
  "unclear_probability": 0.10,
  "label_probabilities": {
    "Discrimination": 0.05,
    "Offensive": 0.60,
    "Violence": 0.00,
    "Vulgar": 0.00,
    "Antagonism": 0.42
  },
  "primary_label": "Offensive",
  "severity": "medium",
  "confidence": 0.70,
  "key_evidence": ["[Image] OCR frames a sensitive topic as ridicule."],
  "reasoning_summary": "该视角认为主要风险来自冒犯性和敌意表达。",
  "uncertainties": ["缺少原始发布语境。"]
}
""".strip()
            )
        return FakeResponse(
            """
{
  "counterfactual_tests": [
    {
      "condition": "If the text is in-group self-deprecation",
      "expected_change": "harmful probability becomes slightly lower but remains borderline harmful.",
      "affected_labels": ["Offensive"],
      "reason": "The public reading may still treat the wording as ridicule.",
      "evidence_needed": "Original post community context."
    }
  ],
  "robustness": "medium",
  "recommended_adjustment": {
    "harmful_probability_delta": -0.05,
    "confidence_delta": -0.04,
    "label_probability_deltas": {
      "Discrimination": 0.00,
      "Offensive": -0.04,
      "Violence": 0.00,
      "Vulgar": 0.00,
      "Antagonism": 0.00
    }
  },
  "summary": "反事实分析显示结论会受发布语境影响，但不会完全反转。"
}
""".strip()
        )


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

    def test_runner_uses_shared_evidence_context_for_standard_heads(self) -> None:
        llm = FakeLLM()
        runner = MemeAnalysisHeadRunner(llm=llm, system_prompt="system")

        results = runner.run_heads(
            head_names=["sentiment"],
            topic="test topic",
            evidence_context="shared evidence [Image]",
            input_mode="text_only",
        )

        self.assertEqual(results[0].name, "sentiment")
        self.assertEqual(results[0].output, "fake head output")
        user_prompt = llm.messages[0][1].content
        self.assertIn("shared evidence [Image]", user_prompt)
        self.assertIn("test topic", user_prompt)

    def test_harmfulness_head_runs_perspective_ensemble(self) -> None:
        llm = FakeHarmfulnessLLM()
        runner = MemeAnalysisHeadRunner(llm=llm, system_prompt="system")

        results = runner.run_heads(
            head_names=["harmfulness"],
            topic="test topic",
            evidence_context="shared evidence [Image]",
            input_mode="image_only",
        )

        self.assertEqual(results[0].name, "harmfulness")
        self.assertIn("Ensemble label", results[0].output)
        self.assertIn("Soft-vote breakdown", results[0].output)
        self.assertIn("Counterfactual reasoning", results[0].output)
        self.assertIn("Offensive", results[0].output)
        self.assertEqual(7, len(llm.messages))
        self.assertIn("shared evidence [Image]", llm.messages[0][1].content)

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
        self.assertIn("Ensemble label", formatted)
        self.assertEqual(formatted.count("fake head output"), 1)


if __name__ == "__main__":
    unittest.main()
