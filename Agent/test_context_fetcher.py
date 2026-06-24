from __future__ import annotations

import unittest

from memeagent.context_fetcher import (
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


if __name__ == "__main__":
    unittest.main()
