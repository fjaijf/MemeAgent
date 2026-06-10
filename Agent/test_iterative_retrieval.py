from __future__ import annotations

import unittest

from memeagent.workflow import MemeResearchWorkflow


class FakeReflectingMemeAgent:
    def __init__(self) -> None:
        self.reflections = 0

    def reflect_retrieval(self, **kwargs) -> str:
        self.reflections += 1
        return """
RETRIEVAL_SCORE:
- 4

SHOULD_CONTINUE:
- yes

STOP_REASON:
- Need one more concrete search.

EVIDENCE_GAPS:
- Missing original post context.

SUPPLEMENTAL_WEB_QUERIES:
- "exact meme phrase" original post

SUPPLEMENTAL_NEWS_QUERIES:
- None

QUERY_CAUTIONS:
- Do not assume intent without source context.
""".strip()


class FakeStoppingMemeAgent:
    def reflect_retrieval(self, **kwargs) -> str:
        return """
RETRIEVAL_SCORE:
- 8

SHOULD_CONTINUE:
- no

STOP_REASON:
- Current evidence is sufficient.

EVIDENCE_GAPS:
- None

SUPPLEMENTAL_WEB_QUERIES:
- None

SUPPLEMENTAL_NEWS_QUERIES:
- None

QUERY_CAUTIONS:
- Keep inference separate from evidence.
""".strip()


class FakeSearchAgent:
    def __init__(self) -> None:
        self.contexts: list[str] = []

    def run(self, topic: str, context: str = "") -> str:
        self.contexts.append(context)
        return """
Search provider: fake

## Web Search Results
[W1] Fake result
   Candidate type: background_or_related_result
   Snippet: fake snippet
   URL: https://example.com/fake
""".strip()


class IterativeRetrievalTests(unittest.TestCase):
    def test_iterative_search_adds_reflection_round_and_unique_labels(self) -> None:
        search_agent = FakeSearchAgent()
        workflow = MemeResearchWorkflow(
            meme_agent=FakeReflectingMemeAgent(),
            search_agent=search_agent,
        )

        report = workflow._run_search_with_reflection(
            topic="topic",
            context="context",
            visual_report="visual",
            retrieval_plan="plan",
            input_mode="text_only",
            iterative_search=True,
            search_max_rounds=2,
            progress=None,
        )

        self.assertIn("## Retrieval Round 1", report)
        self.assertIn("## Retrieval Reflection after Round 1", report)
        self.assertIn("## Retrieval Round 2", report)
        self.assertIn("[R2-W1] Fake result", report)
        self.assertEqual(len(search_agent.contexts), 2)
        self.assertIn('"exact meme phrase" original post', search_agent.contexts[1])

    def test_iterative_search_stops_when_reflection_says_no(self) -> None:
        search_agent = FakeSearchAgent()
        workflow = MemeResearchWorkflow(
            meme_agent=FakeStoppingMemeAgent(),
            search_agent=search_agent,
        )

        report = workflow._run_search_with_reflection(
            topic="topic",
            context="context",
            visual_report="visual",
            retrieval_plan="plan",
            input_mode="text_only",
            iterative_search=True,
            search_max_rounds=3,
            progress=None,
        )

        self.assertIn("## Retrieval Round 1", report)
        self.assertIn("## Retrieval Reflection after Round 1", report)
        self.assertNotIn("## Retrieval Round 2", report)
        self.assertEqual(len(search_agent.contexts), 1)


if __name__ == "__main__":
    unittest.main()
