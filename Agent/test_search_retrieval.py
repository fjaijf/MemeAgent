from __future__ import annotations

import json
import unittest
from urllib.parse import parse_qs, urlparse

import memeagent.search_agent as search_agent_module
from memeagent.search_agent import SearchAgentConfig, WebSearchAgent


class SearchRetrievalTests(unittest.TestCase):
    def test_query_plan_adds_exact_and_site_context_queries(self) -> None:
        agent = WebSearchAgent(
            SearchAgentConfig(
                search_context_sites="reddit.com,x.com",
                cache_enabled=False,
            )
        )

        plan = agent._build_query_plan(
            topic="test meme",
            context='OCR/text visible in the image: "this is fine"',
        )
        queries_by_category = {(item.category, item.query) for item in plan}

        self.assertIn(("exact_anchor", '"this is fine"'), queries_by_category)
        self.assertIn(("site_context", 'site:reddit.com "this is fine"'), queries_by_category)
        self.assertIn(("site_context", 'site:x.com "this is fine"'), queries_by_category)

    def test_news_queries_skip_site_context_queries(self) -> None:
        agent = WebSearchAgent(SearchAgentConfig(cache_enabled=False))

        queries = agent._build_news_queries(
            [
                'site:reddit.com "this is fine"',
                '"this is fine"',
            ],
            context="",
        )

        self.assertTrue(queries)
        self.assertFalse(any(query.startswith("site:") for query in queries))

    def test_result_type_classification_marks_social_context(self) -> None:
        agent = WebSearchAgent(SearchAgentConfig(cache_enabled=False))

        reddit_type = agent._classify_result_type(
            {
                "title": "A meme thread",
                "body": "comments about the image",
                "href": "https://www.reddit.com/r/memes/comments/abc123/example/",
            }
        )
        x_type = agent._classify_result_type(
            {
                "title": "Post",
                "body": "original caption",
                "href": "https://x.com/example/status/123",
            }
        )

        self.assertEqual(reddit_type, "post_or_comment_context_candidate")
        self.assertEqual(x_type, "social_post_candidate")

    def test_google_search_results_are_normalized(self) -> None:
        class FakeGoogleResult:
            title = "This Is Fine"
            url = "https://example.com/this-is-fine"
            description = "A result snippet from Google."

        calls: list[tuple[str, dict[str, object]]] = []

        def fake_google_search(query: str, **kwargs: object) -> list[FakeGoogleResult]:
            calls.append((query, kwargs))
            return [FakeGoogleResult()]

        original_google_search = search_agent_module.google_search
        search_agent_module.google_search = fake_google_search
        try:
            agent = WebSearchAgent(
                SearchAgentConfig(
                    search_provider="google",
                    search_max_results=3,
                    search_lang="zh-cn",
                    search_country="cn",
                    cache_enabled=False,
                )
            )

            results = agent._search_text_provider_uncached("google", "this is fine")
        finally:
            search_agent_module.google_search = original_google_search

        self.assertEqual(1, len(results))
        self.assertEqual("this is fine", calls[0][0])
        self.assertEqual(3, calls[0][1]["num_results"])
        self.assertEqual("zh", calls[0][1]["lang"])
        self.assertEqual("cn", calls[0][1]["region"])
        self.assertTrue(calls[0][1]["advanced"])
        self.assertEqual("This Is Fine", results[0]["title"])
        self.assertEqual("A result snippet from Google.", results[0]["body"])
        self.assertEqual("https://example.com/this-is-fine", results[0]["href"])
        self.assertEqual("Google", results[0]["source"])

    def test_searxng_search_results_are_requested_and_normalized(self) -> None:
        class FakeResponse:
            def __init__(self, payload: dict[str, object]) -> None:
                self.payload = payload

            def __enter__(self) -> "FakeResponse":
                return self

            def __exit__(self, *args: object) -> None:
                return None

            def read(self) -> bytes:
                return json.dumps(self.payload).encode("utf-8")

        class FakeSearxngAgent(WebSearchAgent):
            def __init__(self, config: SearchAgentConfig) -> None:
                super().__init__(config)
                self.requests: list[object] = []

            def _open_url(self, req: object, data: bytes | None = None) -> FakeResponse:
                self.requests.append(req)
                return FakeResponse(
                    {
                        "results": [
                            {
                                "title": "This Is Fine",
                                "content": "A SearXNG result snippet.",
                                "url": "https://example.com/this-is-fine",
                                "engines": ["google", "bing"],
                                "publishedDate": "2026-01-01",
                            },
                            {
                                "title": "Second Result",
                                "content": "Should be trimmed by max_results.",
                                "url": "https://example.com/second",
                            },
                        ]
                    }
                )

        agent = FakeSearxngAgent(
            SearchAgentConfig(
                search_provider="searxng",
                searxng_url="http://searx.local",
                searxng_engines="google,bing",
                searxng_web_categories="general",
                searxng_news_categories="news",
                search_lang="zh-CN",
                search_max_results=1,
                news_max_results=1,
                cache_enabled=False,
            )
        )

        web_results = agent._search_text_provider_uncached("searxng", "this is fine")
        news_results = agent._search_news_provider_uncached("searxng", "this is fine")

        web_url = urlparse(agent.requests[0].full_url)
        web_params = parse_qs(web_url.query)
        news_url = urlparse(agent.requests[1].full_url)
        news_params = parse_qs(news_url.query)

        self.assertEqual("/search", web_url.path)
        self.assertEqual(["this is fine"], web_params["q"])
        self.assertEqual(["json"], web_params["format"])
        self.assertEqual(["general"], web_params["categories"])
        self.assertEqual(["zh-CN"], web_params["language"])
        self.assertEqual(["google,bing"], web_params["engines"])
        self.assertEqual(["news"], news_params["categories"])
        self.assertEqual(1, len(web_results))
        self.assertEqual(1, len(news_results))
        self.assertEqual("This Is Fine", web_results[0]["title"])
        self.assertEqual("A SearXNG result snippet.", web_results[0]["body"])
        self.assertEqual("https://example.com/this-is-fine", web_results[0]["href"])
        self.assertEqual("google, bing", web_results[0]["source"])
        self.assertEqual("2026-01-01", web_results[0]["date"])

    def test_zhihu_payload_shape_is_extracted_and_normalized(self) -> None:
        agent = WebSearchAgent(SearchAgentConfig(cache_enabled=False))
        payload = {
            "Code": 0,
            "Message": "success",
            "Data": {
                "Items": [
                    {
                        "Title": "AI之镜 - 知乎",
                        "ContentType": "Article",
                        "ContentID": "123",
                        "ContentText": "这是一段知乎搜索结果正文。",
                        "Url": "https://www.zhihu.com/example",
                        "AuthorName": "知乎作者",
                        "EditTime": "2026-01-01",
                    }
                ]
            },
        }

        items = agent._extract_result_list(payload)
        normalized = agent._normalize_zhihu_item(items[0])

        self.assertEqual(len(items), 1)
        self.assertEqual(normalized["title"], "AI之镜 - 知乎")
        self.assertEqual(normalized["body"], "这是一段知乎搜索结果正文。")
        self.assertEqual(normalized["href"], "https://www.zhihu.com/example")
        self.assertEqual(normalized["source"], "Zhihu / 知乎作者")
        self.assertEqual(normalized["date"], "2026-01-01")
        self.assertEqual(normalized["content_type"], "Article")
        self.assertEqual(normalized["content_id"], "123")


if __name__ == "__main__":
    unittest.main()
