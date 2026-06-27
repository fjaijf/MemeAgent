from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from memeagent.trajectory import MemeTrajectoryCache
from memeagent.workflow import MemeResearchWorkflow


class FakeMemeAgent:
    llm = object()
    system_prompt = "test"

    def __init__(self) -> None:
        self.final_context = ""

    def plan_retrieval(self, **_: object) -> str:
        return "SUPPLEMENTAL_WEB_QUERIES:\n- example meme context"

    def plan_analysis_iteration(self, **_: object) -> str:
        return """
ITERATION_CONFIDENCE:
- 0.95

SHOULD_FINALIZE:
- yes

CONFIDENCE_REASON:
- Enough evidence.

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
- Finalize.
""".strip()

    def run(self, topic: str, context: str, **_: object) -> str:
        self.final_context = context
        return f"final analysis for {topic}"


class FailingMemeAgent(FakeMemeAgent):
    def run(self, topic: str, context: str, **_: object) -> str:
        raise ValueError("final call failed")


class FakeSearchAgent:
    def run(self, topic: str, context: str = "") -> str:
        return "[W1] Search evidence"


class TrajectoryCacheWorkflowTests(unittest.TestCase):
    def test_workflow_records_completed_trajectory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trajectory_cache = MemeTrajectoryCache(Path(tmp) / "trajectory.sqlite3")
            workflow = MemeResearchWorkflow(
                meme_agent=FakeMemeAgent(),
                search_agent=FakeSearchAgent(),
                trajectory_cache=trajectory_cache,
            )

            result = workflow.run(topic="test meme", context="user context")

            self.assertTrue(result.trajectory_run_id)
            run = trajectory_cache.get_run(result.trajectory_run_id)
            self.assertIsNotNone(run)
            assert run is not None
            self.assertEqual("completed", run.status)
            self.assertEqual("analysis", run.workflow_kind)
            self.assertEqual("text_only", run.input_mode)
            self.assertEqual("final analysis for test meme", run.output_json["analysis"])
            self.assertIn("[W1] Search evidence", run.output_json["search_report"])

            events = trajectory_cache.list_events(result.trajectory_run_id)
            names = [event.name for event in events]
            self.assertIn("run_started", names)
            self.assertIn("input_detected", names)
            self.assertIn("retrieval_plan_ready", names)
            self.assertIn("search_report_ready", names)
            self.assertIn("controller_loop_ready", names)
            self.assertIn("combined_context_ready", names)
            self.assertIn("final_analysis_ready", names)
            self.assertIn("run_finished", names)

    def test_workflow_records_failed_trajectory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trajectory_cache = MemeTrajectoryCache(Path(tmp) / "trajectory.sqlite3")
            workflow = MemeResearchWorkflow(
                meme_agent=FailingMemeAgent(),
                search_agent=FakeSearchAgent(),
                trajectory_cache=trajectory_cache,
            )

            with self.assertRaises(RuntimeError):
                workflow.run(topic="test meme")

            runs = trajectory_cache.list_runs()
            self.assertEqual(1, len(runs))
            self.assertEqual("failed", runs[0].status)
            self.assertIn("Final analysis LLM call failed", runs[0].error)
            self.assertIn("[W1] Search evidence", runs[0].output_json["search_report"])

            events = trajectory_cache.list_events(runs[0].run_id)
            self.assertEqual("run_failed", events[-1].name)


if __name__ == "__main__":
    unittest.main()
