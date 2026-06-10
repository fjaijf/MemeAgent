from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from memeagent.cache import SearchResultCache
from memeagent.search_agent import SearchAgentConfig, WebSearchAgent


class CountingSearchAgent(WebSearchAgent):
    def __init__(self, config: SearchAgentConfig) -> None:
        super().__init__(config)
        self.text_calls = 0

    def _search_ddgs_text(self, query: str):
        self.text_calls += 1
        return [
            {
                "title": f"Result for {query}",
                "body": "cached body",
                "href": "https://example.test/result",
            }
        ]


class SearchResultCacheTest(unittest.TestCase):
    def test_cache_returns_unexpired_value(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = SearchResultCache(Path(tmp) / "search.sqlite3")
            value = [{"title": "Example", "href": "https://example.test"}]

            cache.set("key", value, ttl_seconds=60)

            entry = cache.get("key")
            self.assertIsNotNone(entry)
            self.assertEqual(value, entry.value)

    def test_cache_drops_expired_value(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = SearchResultCache(Path(tmp) / "search.sqlite3")
            cache.set("key", [{"title": "Expired"}], ttl_seconds=0)

            time.sleep(0.01)

            self.assertIsNone(cache.get("key"))

    def test_web_search_agent_reuses_cached_provider_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = CountingSearchAgent(
                SearchAgentConfig(
                    search_provider="ddgs",
                    search_cache_path=str(Path(tmp) / "search.sqlite3"),
                    search_cache_ttl_seconds=60,
                )
            )

            first = agent._search_text("same query")
            second = agent._search_text("same query")

            self.assertEqual(first, second)
            self.assertEqual(1, agent.text_calls)


if __name__ == "__main__":
    unittest.main()
