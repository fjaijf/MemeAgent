from __future__ import annotations

import unittest

from langchain_core.messages import HumanMessage, SystemMessage

from memeagent.llm import LocalTransformersChatClient


class LocalTransformersChatClientTests(unittest.TestCase):
    def test_text_only_conversion_drops_image_blocks_for_text_model(self) -> None:
        client = LocalTransformersChatClient(
            model_path="/tmp/nonexistent-model",
            temperature=0.2,
            max_tokens=32,
            timeout=5,
        )

        messages, images = client._convert_messages(
            [
                SystemMessage(content="system"),
                HumanMessage(
                    content=[
                        {"type": "text", "text": "describe this"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,"}},
                    ]
                ),
            ],
            include_images=False,
        )

        self.assertEqual([], images)
        self.assertEqual("system", messages[0]["content"])
        self.assertEqual("describe this", messages[1]["content"])

    def test_postprocess_strips_thinking_block(self) -> None:
        client = LocalTransformersChatClient(
            model_path="/tmp/nonexistent-model",
            temperature=0.2,
            max_tokens=32,
            timeout=5,
        )

        content = client._postprocess_content("<think>\nprivate\n</think>\n\nfinal answer")

        self.assertEqual("final answer", content)


if __name__ == "__main__":
    unittest.main()
