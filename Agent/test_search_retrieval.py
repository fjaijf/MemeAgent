from __future__ import annotations

import json
import time
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

    def test_query_plan_does_not_append_meme_to_topic_or_anchors(self) -> None:
        agent = WebSearchAgent(SearchAgentConfig(cache_enabled=False))

        plan = agent._build_query_plan(
            topic="PEPE",
            context='OCR/text visible in the image: "Obama"',
        )
        queries = [item.query for item in plan]

        self.assertIn("PEPE", queries)
        self.assertIn('"Obama"', queries)
        self.assertNotIn("PEPE meme", queries)
        self.assertNotIn("Obama meme", queries)

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

    def test_visual_suggested_queries_are_prioritized_over_noisy_ocr(self) -> None:
        agent = WebSearchAgent(SearchAgentConfig(cache_enabled=False))

        plan = agent._build_query_plan(
            topic="",
            context="""
1. OCR/text visible in the image:
- "Obama: Don't discuss Titanic with Joe DiCaprio"
- "Obama: Why?"

7. Search keywords:
- "Titanic door meme"
- "Could Jack fit on the door"

8. Suggested retrieval queries:
- "Obama Leonardo DiCaprio White House meeting photo"
- "Titanic door meme Jack fit on board"
- "Leonardo DiCaprio Jimmy Kimmel Titanic door"
""",
        )

        queries = [item.query for item in plan]
        categories = {(item.category, item.query) for item in plan}

        self.assertEqual(
            "Obama Leonardo DiCaprio White House meeting photo",
            queries[0],
        )
        self.assertIn(
            (
                "visual_suggested",
                "Titanic door Jack fit on board",
            ),
            categories,
        )
        self.assertNotIn(
            "Obama: Don't discuss Titanic with Joe DiCaprio meme",
            queries,
        )

    def test_visual_suggested_queries_remove_generic_meme_noise(self) -> None:
        agent = WebSearchAgent(SearchAgentConfig(cache_enabled=False))

        plan = agent._build_query_plan(
            topic="",
            context="""
8. Suggested retrieval queries:
- Pingu" meme "妈的老子不干了
- 你应该多说话" meme "被打断" "被忽略
- Pingu angry meme Chinese text "无人注意
- 企鹅家族" Pingu 表情包 愤怒
- 妈的老子不干了" meme template
- 我： *说话* *被打断*" meme
""",
        )
        queries = [item.query for item in plan if item.category == "visual_suggested"]

        self.assertIn("Pingu 妈的老子不干了", queries)
        self.assertIn("Pingu angry 无人注意", queries)
        self.assertIn("企鹅家族 Pingu 愤怒", queries)
        self.assertFalse(any('"' in query for query in queries))
        self.assertFalse(any("meme" in query.lower() for query in queries))
        self.assertFalse(any("template" in query.lower() for query in queries))
        self.assertFalse(any("表情包" in query for query in queries))

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

    def test_searxng_url_preserves_base_path_and_query_params(self) -> None:
        class FakeResponse:
            def __enter__(self) -> "FakeResponse":
                return self

            def __exit__(self, *args: object) -> None:
                return None

            def read(self) -> bytes:
                return json.dumps({"results": []}).encode("utf-8")

        class FakeSearxngAgent(WebSearchAgent):
            def __init__(self, config: SearchAgentConfig) -> None:
                super().__init__(config)
                self.requests: list[object] = []

            def _open_url(self, req: object, data: bytes | None = None) -> FakeResponse:
                self.requests.append(req)
                return FakeResponse()

        agent = FakeSearxngAgent(
            SearchAgentConfig(
                search_provider="searxng",
                searxng_url="http://localhost:8888/searxng?theme=simple",
                cache_enabled=False,
            )
        )

        agent._search_text_provider_uncached("searxng", "this is fine")

        request_url = urlparse(agent.requests[0].full_url)
        params = parse_qs(request_url.query)

        self.assertEqual("/searxng/search", request_url.path)
        self.assertEqual(["simple"], params["theme"])
        self.assertEqual(["this is fine"], params["q"])
        self.assertEqual(["json"], params["format"])

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

    def test_qwen_search_uses_enable_search_and_normalizes_content(self) -> None:
        class FakeResponse:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict[str, object]:
                return {
                    "choices": [
                        {
                            "message": {
                                "content": "杭州明天天气摘要",
                            }
                        }
                    ]
                }

        calls: list[dict[str, object]] = []

        def fake_post(*args: object, **kwargs: object) -> FakeResponse:
            calls.append({"args": args, "kwargs": kwargs})
            return FakeResponse()

        original_post = search_agent_module.requests.post
        search_agent_module.requests.post = fake_post
        try:
            agent = WebSearchAgent(
                SearchAgentConfig(
                    search_provider="qwen",
                    qwen_search_api_key="dashscope-key",
                    qwen_search_model="qwen-plus",
                    cache_enabled=False,
                )
            )
            results = agent._search_text_provider_uncached("qwen", "杭州明天天气如何")
        finally:
            search_agent_module.requests.post = original_post

        request_kwargs = calls[0]["kwargs"]
        payload = request_kwargs["json"]

        self.assertEqual(1, len(results))
        self.assertEqual("Qwen search answer: 杭州明天天气如何", results[0]["title"])
        self.assertEqual("杭州明天天气摘要", results[0]["body"])
        self.assertEqual("Qwen Search", results[0]["source"])
        self.assertEqual("qwen-plus", payload["model"])
        self.assertTrue(payload["enable_search"])

    def test_glm_search_invokes_zai_client_and_normalizes_results(self) -> None:
        class FakeWebSearch:
            def __init__(self) -> None:
                self.calls: list[dict[str, object]] = []

            def web_search(self, **kwargs: object) -> dict[str, object]:
                self.calls.append(kwargs)
                return {
                    "search_result": [
                        {
                            "title": "财经新闻",
                            "content": "新闻摘要",
                            "url": "https://example.com/news",
                            "media": "Example News",
                            "publish_time": "2025-04-01",
                        }
                    ]
                }

        fake_web_search = FakeWebSearch()

        class FakeZhipuAiClient:
            def __init__(self, api_key: str) -> None:
                self.api_key = api_key
                self.web_search = fake_web_search

        original_client = search_agent_module.ZhipuAiClient
        search_agent_module.ZhipuAiClient = FakeZhipuAiClient
        try:
            agent = WebSearchAgent(
                SearchAgentConfig(
                    search_provider="glm",
                    glm_search_api_key="glm-key",
                    glm_search_engine="search_pro",
                    glm_search_domain_filter="www.sohu.com",
                    cache_enabled=False,
                )
            )
            results = agent._search_text_provider_uncached("glm", "搜索2025年4月的财经新闻")
        finally:
            search_agent_module.ZhipuAiClient = original_client

        self.assertEqual(1, len(results))
        self.assertEqual("财经新闻", results[0]["title"])
        self.assertEqual("新闻摘要", results[0]["body"])
        self.assertEqual("https://example.com/news", results[0]["href"])
        self.assertEqual("Example News", results[0]["source"])
        self.assertEqual("2025-04-01", results[0]["date"])
        self.assertEqual("search_pro", fake_web_search.calls[0]["search_engine"])
        self.assertEqual("www.sohu.com", fake_web_search.calls[0]["search_domain_filter"])

    def test_run_with_timeout_returns_after_configured_timeout(self) -> None:
        agent = WebSearchAgent(
            SearchAgentConfig(search_timeout=0.1, cache_enabled=False)
        )

        started = time.perf_counter()
        results, error = agent._run_with_timeout(
            lambda _query: (time.sleep(2), [])[1],
            "slow query",
            "Web search",
        )
        elapsed = time.perf_counter() - started

        self.assertEqual([], results)
        self.assertIsNotNone(error)
        self.assertIn("timed out", error or "")
        self.assertLess(elapsed, 1.0)

    def test_text_query_provider_tasks_run_in_parallel(self) -> None:
        class SlowProviderAgent(WebSearchAgent):
            def _search_text_provider(
                self,
                provider: str,
                query: str,
            ) -> list[dict[str, object]]:
                time.sleep(0.2)
                return [
                    {
                        "title": f"{provider} {query}",
                        "body": "snippet",
                        "href": f"https://example.com/{provider}",
                    }
                ]

        agent = SlowProviderAgent(
            SearchAgentConfig(
                search_provider="ddgs,qwen",
                search_timeout=2,
                cache_enabled=False,
            )
        )

        started = time.perf_counter()
        results = agent._search_text_queries(["parallel query"])
        elapsed = time.perf_counter() - started

        self.assertEqual(2, len(results))
        self.assertLess(elapsed, 0.35)


if __name__ == "__main__":
    unittest.main()
