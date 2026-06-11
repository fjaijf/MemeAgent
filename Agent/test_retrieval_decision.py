from __future__ import annotations

import unittest

from memeagent.workflow import MemeResearchWorkflow


NO_RETRIEVAL_DECISION = """
RETRIEVAL_NEEDED:
- no

DECISION_REASON:
- Local evidence is sufficient for this analysis.

EVIDENCE_QUESTIONS:
- None

SUPPLEMENTAL_WEB_QUERIES:
- None

SUPPLEMENTAL_NEWS_QUERIES:
- None

QUERY_CAUTIONS:
- Do not assume source context beyond the provided prompt.
""".strip()


YES_RETRIEVAL_DECISION = """
RETRIEVAL_NEEDED:
- yes

DECISION_REASON:
- The exact quoted phrase needs source context.

EVIDENCE_QUESTIONS:
- Where did the quoted phrase appear?

SUPPLEMENTAL_WEB_QUERIES:
- "this is fine" meme origin

SUPPLEMENTAL_NEWS_QUERIES:
- None

QUERY_CAUTIONS:
- Do not assume the image is the original post.
""".strip()


class FakeMemeAgent:
    def __init__(self, retrieval_decision: str) -> None:
        self.retrieval_decision = retrieval_decision
        self.analysis_context = ""
        self.decide_calls = 0
        self.plan_calls = 0

    def decide_retrieval(self, **_: object) -> str:
        self.decide_calls += 1
        return self.retrieval_decision

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


class FakeSearchAgent:
    def __init__(self) -> None:
        self.calls = 0

    def run(self, **_: object) -> str:
        self.calls += 1
        return "[W1] Search evidence"


class RetrievalDecisionWorkflowTests(unittest.TestCase):
    def test_skips_search_when_llm_decides_retrieval_is_not_needed(self) -> None:
        meme_agent = FakeMemeAgent(NO_RETRIEVAL_DECISION)
        search_agent = FakeSearchAgent()
        workflow = MemeResearchWorkflow(meme_agent=meme_agent, search_agent=search_agent)

        result = workflow.run(topic="simple meme", use_search=True)

        self.assertEqual(0, search_agent.calls)
        self.assertEqual(1, meme_agent.decide_calls)
        self.assertEqual(0, meme_agent.plan_calls)
        self.assertEqual("", result.search_report)
        self.assertIn("RETRIEVAL_NEEDED", result.retrieval_plan)
        self.assertIn("Retrieval decision and query plan", result.combined_context)
        self.assertNotIn("Internet search findings", result.combined_context)

    def test_runs_search_when_llm_decides_retrieval_is_needed(self) -> None:
        meme_agent = FakeMemeAgent(YES_RETRIEVAL_DECISION)
        search_agent = FakeSearchAgent()
        workflow = MemeResearchWorkflow(meme_agent=meme_agent, search_agent=search_agent)

        result = workflow.run(topic="this is fine", use_search=True)

        self.assertEqual(1, search_agent.calls)
        self.assertEqual(1, meme_agent.decide_calls)
        self.assertEqual(0, meme_agent.plan_calls)
        self.assertIn("[W1] Search evidence", result.search_report)
        self.assertIn("Internet search findings", result.combined_context)

    def test_force_search_bypasses_retrieval_decision(self) -> None:
        meme_agent = FakeMemeAgent(NO_RETRIEVAL_DECISION)
        search_agent = FakeSearchAgent()
        workflow = MemeResearchWorkflow(meme_agent=meme_agent, search_agent=search_agent)

        result = workflow.run(
            topic="simple meme",
            use_search=False,
            force_search=True,
        )

        self.assertEqual(1, search_agent.calls)
        self.assertEqual(0, meme_agent.decide_calls)
        self.assertEqual(1, meme_agent.plan_calls)
        self.assertIn("[W1] Search evidence", result.search_report)
        self.assertIn("Forced retrieval plan", result.combined_context)


if __name__ == "__main__":
    unittest.main()
