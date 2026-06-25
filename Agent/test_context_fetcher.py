from __future__ import annotations

import unittest
from unittest.mock import patch

import requests

from memeagent.context_fetcher import (
    _is_context_host,
    _parse_html_context,
    _reddit_comment_text,
    fetch_contexts_for_results,
)


class ContextFetcherTests(unittest.TestCase):
    def test_parse_html_context_extracts_title_description_and_text(self) -> None:
        title, description, text = _parse_html_context(
            """
            <html>
              <head>
                <title>Example Page</title>
                <meta name="description" content="A useful public summary.">
              </head>
              <body>
                <script>ignore me</script>
                <article>This is readable context from the page.</article>
              </body>
            </html>
            """
        )

        self.assertEqual("Example Page", title)
        self.assertEqual("A useful public summary.", description)
        self.assertIn("readable context", text)
        self.assertNotIn("ignore me", text)

    def test_reddit_comment_text_flattens_nested_comments(self) -> None:
        comments: list[str] = []
        _reddit_comment_text(
            {
                "data": {
                    "author": "parent",
                    "score": 5,
                    "body": "Parent comment",
                    "replies": {
                        "data": {
                            "children": [
                                {
                                    "kind": "t1",
                                    "data": {
                                        "author": "child",
                                        "body": "Child comment",
                                    },
                                }
                            ]
                        }
                    },
                }
            },
            comments,
        )

        self.assertEqual(2, len(comments))
        self.assertIn("parent | score=5: Parent comment", comments)
        self.assertIn("child: Child comment", comments)

    def test_fetch_contexts_filters_non_context_hosts(self) -> None:
        contexts = fetch_contexts_for_results(
            [
                {
                    "title": "Example",
                    "href": "https://example.com/not-a-platform",
                }
            ]
        )

        self.assertEqual([], contexts)

    def test_context_hosts_can_be_overridden(self) -> None:
        self.assertFalse(_is_context_host("https://www.tiktok.com/@user/video/1", ("reddit.com",)))
        self.assertTrue(_is_context_host("https://old.reddit.com/r/test/comments/1", ("reddit.com",)))

    def test_fetch_contexts_uses_configured_proxy(self) -> None:
        class FakeResponse:
            headers = {"content-type": "text/html"}
            encoding = "utf-8"
            apparent_encoding = "utf-8"
            text = "<html><head><title>Thread</title></head><body>Readable body</body></html>"

            def raise_for_status(self) -> None:
                return None

        captured: dict[str, object] = {}

        def fake_get(url: str, **kwargs: object) -> FakeResponse:
            captured["url"] = url
            captured["kwargs"] = kwargs
            return FakeResponse()

        with patch("memeagent.context_fetcher.requests.get", side_effect=fake_get):
            contexts = fetch_contexts_for_results(
                [{"href": "https://x.com/example/status/1"}],
                context_sites="x.com",
                proxy="http://127.0.0.1:7890",
            )

        self.assertEqual(1, len(contexts))
        self.assertEqual("Thread", contexts[0].title)
        self.assertEqual(
            {
                "http": "http://127.0.0.1:7890",
                "https": "http://127.0.0.1:7890",
            },
            captured["kwargs"]["proxies"],
        )

    def test_fetch_contexts_retries_with_fallback_proxy_after_403(self) -> None:
        class FakeResponse:
            def __init__(self, status_code: int, text: str = "") -> None:
                self.status_code = status_code
                self.text = text
                self.headers = {"content-type": "text/html"}
                self.encoding = "utf-8"
                self.apparent_encoding = "utf-8"

            def raise_for_status(self) -> None:
                if self.status_code >= 400:
                    error = requests.HTTPError(f"{self.status_code} error")
                    error.response = self
                    raise error

        calls: list[dict[str, object]] = []

        def fake_get(url: str, **kwargs: object) -> FakeResponse:
            calls.append(kwargs)
            if len(calls) == 1:
                return FakeResponse(403)
            return FakeResponse(
                200,
                "<html><head><title>Retried</title></head><body>Readable body</body></html>",
            )

        with patch("memeagent.context_fetcher.requests.get", side_effect=fake_get):
            contexts = fetch_contexts_for_results(
                [{"href": "https://www.zhihu.com/question/1/answer/2"}],
                context_sites="zhihu.com",
                fallback_proxy="http://127.0.0.1:7890",
            )

        self.assertEqual(2, len(calls))
        self.assertNotIn("proxies", calls[0])
        self.assertEqual(
            {
                "http": "http://127.0.0.1:7890",
                "https": "http://127.0.0.1:7890",
            },
            calls[1]["proxies"],
        )
        self.assertEqual("Retried", contexts[0].title)


if __name__ == "__main__":
    unittest.main()
