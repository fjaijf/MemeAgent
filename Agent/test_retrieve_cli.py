from __future__ import annotations

import unittest

from memeagent.retrieve_cli import (
    _build_context_with_visual_report,
    _format_context_results,
    format_retrieval_report,
    run_retrieval,
)
from memeagent.context_fetcher import ContextFetchResult
from memeagent.search_agent import SearchAgentConfig, WebSearchAgent


class FakeSearchAgent(WebSearchAgent):
    def __init__(self) -> None:
        super().__init__(
            SearchAgentConfig(
                search_provider="ddgs",
                search_max_results=3,
                news_max_results=2,
                cache_enabled=False,
            )
        )
        self.web_queries: list[str] = []
        self.news_queries: list[str] = []

    def _search_text(self, query: str) -> list[dict[str, object]]:
        self.web_queries.append(query)
        return [
            {
                "title": f"Web result for {query}",
                "body": "web snippet",
                "href": f"https://example.com/web/{len(self.web_queries)}",
                "source": "Fake",
            }
        ]

    def _search_text_provider(
        self,
        provider: str,
        query: str,
    ) -> list[dict[str, object]]:
        return self._search_text(query)

    def _search_news(self, query: str) -> list[dict[str, object]]:
        self.news_queries.append(query)
        return [
            {
                "title": f"News result for {query}",
                "body": "news snippet",
                "href": f"https://example.com/news/{len(self.news_queries)}",
                "source": "Fake News",
                "date": "2026-01-01",
            }
        ]

    def _search_news_provider(
        self,
        provider: str,
        query: str,
    ) -> list[dict[str, object]]:
        return self._search_news(query)


class RetrievalCliTests(unittest.TestCase):
    def test_plan_mode_builds_queries_without_searching(self) -> None:
        agent = FakeSearchAgent()

        result = run_retrieval(
            agent=agent,
            topic="test meme",
            context='OCR/text visible in the image: "this is fine"',
            mode="plan",
        )

        planned = {(item.category, item.query) for item in result.query_plan}
        self.assertIn(("exact_anchor", '"this is fine"'), planned)
        self.assertTrue(result.news_queries)
        self.assertEqual([], result.web_results)
        self.assertEqual([], result.news_results)
        self.assertEqual([], agent.web_queries)
        self.assertEqual([], agent.news_queries)

    def test_direct_query_runs_exact_web_and_news_query(self) -> None:
        agent = FakeSearchAgent()

        result = run_retrieval(
            agent=agent,
            topic="",
            direct_queries=['"this is fine"'],
            mode="both",
        )

        self.assertEqual(['"this is fine"'], agent.web_queries)
        self.assertEqual(['"this is fine"'], agent.news_queries)
        self.assertEqual("direct", result.query_plan[0].category)
        self.assertEqual(1, len(result.web_results))
        self.assertEqual(1, len(result.news_results))

    def test_news_can_use_separate_agent_and_query_limit(self) -> None:
        web_agent = FakeSearchAgent()
        news_agent = FakeSearchAgent()

        result = run_retrieval(
            agent=web_agent,
            topic="PEPE",
            mode="both",
            news_agent=news_agent,
            news_max_queries=1,
        )

        self.assertTrue(web_agent.web_queries)
        self.assertEqual([], web_agent.news_queries)
        self.assertEqual(["PEPE news"], news_agent.news_queries)
        self.assertEqual(1, len(result.news_results))

    def test_report_includes_mode_specific_sections(self) -> None:
        agent = FakeSearchAgent()
        result = run_retrieval(
            agent=agent,
            topic="",
            direct_queries=["sample query"],
            mode="web",
        )

        report = format_retrieval_report(agent, result, mode="web")

        self.assertIn("Search query plan:", report)
        self.assertIn("## Web Search Results", report)
        self.assertIn("## Thread/Page Context", report)
        self.assertNotIn("News query plan:", report)

    def test_report_includes_result_provider(self) -> None:
        agent = FakeSearchAgent()
        report = format_retrieval_report(
            agent,
            result=type(
                "Result",
                (),
                {
                    "query_plan": [],
                    "news_queries": [],
                    "web_results": [
                        {
                            "title": "Anspire hit",
                            "body": "snippet",
                            "href": "https://example.com/hit",
                            "provider": "anspire",
                        }
                    ],
                    "news_results": [],
                    "context_results": [],
                    "errors": [],
                    "cache_stats": "Search cache: disabled",
                },
            )(),
            mode="web",
        )

        self.assertIn("Provider: anspire", report)

    def test_context_formatter_includes_comments_and_failure_diagnostics(self) -> None:
        report = _format_context_results(
            [
                ContextFetchResult(
                    source_id="W1",
                    url="https://www.reddit.com/r/test/comments/abc/example/",
                    site="reddit",
                    title="Example thread",
                    post_text="Original post text.",
                    comments=["user1: first comment", "user2: second comment"],
                    metadata={"subreddit": "test", "score": "12"},
                ),
                ContextFetchResult(
                    source_id="W2",
                    url="https://x.com/example/status/1",
                    site="x.com",
                    error="403 forbidden",
                ),
            ]
        )

        self.assertIn("Context for [W1]", report)
        self.assertIn("Original post text.", report)
        self.assertIn("user1: first comment", report)
        self.assertIn("Fetch diagnostics:", report)
        self.assertIn("W2 x.com: 403 forbidden", report)
        self.assertNotIn("Context for [W2]", report)

    def test_context_formatter_summarizes_all_fetch_failures(self) -> None:
        report = _format_context_results(
            [
                ContextFetchResult(
                    source_id="W13",
                    url="https://www.tiktok.com/@example/video/1",
                    site="tiktok.com",
                    error="ConnectionError: reset",
                ),
                ContextFetchResult(
                    source_id="W22",
                    url="https://www.zhihu.com/question/1/answer/2",
                    site="zhihu.com",
                    error="HTTPError: 403 Client Error",
                ),
            ]
        )

        self.assertIn("No readable public context was extracted", report)
        self.assertIn("Fetch diagnostics:", report)
        self.assertIn("W13 tiktok.com: ConnectionError: reset", report)
        self.assertNotIn("[C1] Context for", report)

    def test_visual_report_is_usable_as_search_context(self) -> None:
        agent = FakeSearchAgent()
        context = _build_context_with_visual_report(
            context="user note",
            visual_report=(
                'OCR/text visible in the image: "feels good man"\n'
                "Suggested retrieval queries:\n"
                '- "feels good man" meme origin'
            ),
        )

        result = run_retrieval(
            agent=agent,
            topic="",
            context=context,
            mode="plan",
        )

        planned = {(item.category, item.query) for item in result.query_plan}
        planned_queries = {item.query for item in result.query_plan}
        self.assertIn(("exact_anchor", '"feels good man"'), planned)
        self.assertTrue(any("feels good man origin" in query for query in planned_queries))


if __name__ == "__main__":
    unittest.main()
