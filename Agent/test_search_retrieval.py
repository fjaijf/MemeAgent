from __future__ import annotations

import unittest

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
