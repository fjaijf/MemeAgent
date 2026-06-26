from __future__ import annotations

import unittest
from unittest.mock import patch

from memeagent.workflow import MemeResearchWorkflow


class FakeMemeAgent:
    def __init__(self) -> None:
        self.analysis_context = ""
        self.plan_calls = 0
        self.controller_calls = 0

    def plan_retrieval(self, **_: object) -> str:
        self.plan_calls += 1
        return """
EVIDENCE_QUESTIONS:
- What public context exists for this meme?

SUPPLEMENTAL_WEB_QUERIES:
- forced meme context

SUPPLEMENTAL_NEWS_QUERIES:
- None

QUERY_CAUTIONS:
- Do not assume the first result is authoritative.
""".strip()

    def run(self, topic: str, context: str, **_: object) -> str:
        self.analysis_context = context
        return f"analysis for {topic}"

    def plan_analysis_iteration(self, **_: object) -> str:
        self.controller_calls += 1
        return """
ITERATION_CONFIDENCE:
- 0.85

SHOULD_FINALIZE:
- yes

CONFIDENCE_REASON:
- Enough evidence for final output.

KEY_FINDINGS_SO_FAR:
- Search evidence exists.

FOCUS_QUESTIONS:
- None

MULTIMODAL_ANALYSIS_REQUESTS:
- None

SUPPLEMENTAL_WEB_QUERIES:
- None

SUPPLEMENTAL_NEWS_QUERIES:
- None

FINAL_OUTPUT_NOTES:
- Keep evidence grounded.
""".strip()


class FakeHeadMemeAgent(FakeMemeAgent):
    llm = object()
    system_prompt = "test"


class LowThenHighControllerMemeAgent(FakeMemeAgent):
    def __init__(self) -> None:
        super().__init__()
        self.image_followups = 0

    def plan_analysis_iteration(self, **_: object) -> str:
        self.controller_calls += 1
        if self.controller_calls == 1:
            return """
ITERATION_CONFIDENCE:
- 0.40

SHOULD_FINALIZE:
- no

CONFIDENCE_REASON:
- Need closer visual evidence and retrieval.

KEY_FINDINGS_SO_FAR:
- Initial evidence is incomplete.

FOCUS_QUESTIONS:
- Does the image include offensive sensitive-event references?

MULTIMODAL_ANALYSIS_REQUESTS:
- Re-check OCR and harm cues.

SUPPLEMENTAL_WEB_QUERIES:
- exact phrase source

SUPPLEMENTAL_NEWS_QUERIES:
- None

FINAL_OUTPUT_NOTES:
- Keep uncertainty explicit.
""".strip()
        return """
ITERATION_CONFIDENCE:
- 0.90

SHOULD_FINALIZE:
- yes

CONFIDENCE_REASON:
- Follow-up evidence is sufficient.

KEY_FINDINGS_SO_FAR:
- Follow-up visual and retrieval evidence exist.

FOCUS_QUESTIONS:
- None

MULTIMODAL_ANALYSIS_REQUESTS:
- None

SUPPLEMENTAL_WEB_QUERIES:
- None

SUPPLEMENTAL_NEWS_QUERIES:
- None

FINAL_OUTPUT_NOTES:
- Finalize under rubric.
""".strip()

    def describe_images_for_search(self, **_: object) -> str:
        return "Initial visual report with \"exact phrase\"."

    def analyze_images_for_plan(self, **_: object) -> str:
        self.image_followups += 1
        return "Follow-up image analysis confirms OCR and harm cues."


class HighConfidenceNoFinalizeMemeAgent(FakeMemeAgent):
    def plan_analysis_iteration(self, **_: object) -> str:
        self.controller_calls += 1
        return """
ITERATION_CONFIDENCE:
- 0.82

SHOULD_FINALIZE:
- no

CONFIDENCE_REASON:
- Confidence is above the configured threshold.

KEY_FINDINGS_SO_FAR:
- Evidence is sufficient by score.

FOCUS_QUESTIONS:
- More questions are optional.

MULTIMODAL_ANALYSIS_REQUESTS:
- None

SUPPLEMENTAL_WEB_QUERIES:
- None

SUPPLEMENTAL_NEWS_QUERIES:
- None

FINAL_OUTPUT_NOTES:
- Finalize because threshold is reached.
""".strip()


class FakeSearchAgent:
    def __init__(self) -> None:
        self.calls = 0
        self.context = ""

    def run(self, **kwargs: object) -> str:
        self.calls += 1
        self.context = str(kwargs.get("context", ""))
        return "[W1] Search evidence"


class ForcedRetrievalWorkflowTests(unittest.TestCase):
    def test_run_always_plans_and_searches_even_when_search_flags_are_false(self) -> None:
        meme_agent = FakeMemeAgent()
        search_agent = FakeSearchAgent()
        workflow = MemeResearchWorkflow(meme_agent=meme_agent, search_agent=search_agent)

        result = workflow.run(
            topic="simple meme",
            use_search=False,
            force_search=False,
        )

        self.assertEqual(1, meme_agent.plan_calls)
        self.assertEqual(1, meme_agent.controller_calls)
        self.assertEqual(1, search_agent.calls)
        self.assertIn("Forced retrieval plan", result.combined_context)
        self.assertIn("Controller planning and confidence report", result.combined_context)
        self.assertIn("Cumulative internet search findings", result.combined_context)
        self.assertIn("[W1] Search evidence", result.search_report)
        self.assertIn("forced meme context", search_agent.context)

    def test_run_heads_always_plans_and_searches_even_when_search_flags_are_false(self) -> None:
        meme_agent = FakeHeadMemeAgent()
        search_agent = FakeSearchAgent()
        workflow = MemeResearchWorkflow(meme_agent=meme_agent, search_agent=search_agent)

        with patch(
            "memeagent.workflow.MemeAnalysisHeadRunner.run_heads",
            return_value=[],
        ):
            result = workflow.run_heads(
                topic="simple meme",
                task_heads=["harmfulness"],
                use_search=False,
                force_search=False,
            )

        self.assertEqual(1, meme_agent.plan_calls)
        self.assertEqual(1, meme_agent.controller_calls)
        self.assertEqual(1, search_agent.calls)
        self.assertIn("Forced retrieval plan", result.combined_context)
        self.assertIn("Controller planning and confidence report", result.combined_context)
        self.assertIn("Cumulative internet search findings", result.combined_context)
        self.assertIn("[W1] Search evidence", result.search_report)

    def test_low_controller_confidence_triggers_followup_image_analysis_and_search(self) -> None:
        meme_agent = LowThenHighControllerMemeAgent()
        search_agent = FakeSearchAgent()
        workflow = MemeResearchWorkflow(meme_agent=meme_agent, search_agent=search_agent)

        result = workflow.run(
            topic="image meme",
            image_paths=["test.png"],
            controller_max_rounds=3,
            controller_confidence_threshold=0.8,
        )

        self.assertEqual(2, meme_agent.controller_calls)
        self.assertEqual(1, meme_agent.image_followups)
        self.assertEqual(2, search_agent.calls)
        self.assertIn("Parsed confidence: 0.40", result.controller_report)
        self.assertIn("Parsed confidence: 0.90", result.controller_report)
        self.assertIn("Follow-up image analysis confirms OCR", result.visual_report)
        self.assertIn("Controller-Directed Retrieval Round 2", result.search_report)

    def test_controller_stops_when_confidence_reaches_threshold(self) -> None:
        meme_agent = HighConfidenceNoFinalizeMemeAgent()
        search_agent = FakeSearchAgent()
        workflow = MemeResearchWorkflow(meme_agent=meme_agent, search_agent=search_agent)

        result = workflow.run(
            topic="simple meme",
            controller_max_rounds=3,
            controller_confidence_threshold=0.8,
        )

        self.assertEqual(1, meme_agent.controller_calls)
        self.assertEqual(1, search_agent.calls)
        self.assertIn("Parsed confidence: 0.82", result.controller_report)


if __name__ == "__main__":
    unittest.main()
