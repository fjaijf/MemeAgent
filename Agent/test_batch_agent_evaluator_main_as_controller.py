from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from batch_agent_evaluator_main_as_controller import NoThinkingChatEndpoint


class MainAsControllerTests(unittest.TestCase):
    def test_endpoint_forces_thinking_off(self) -> None:
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "choices": [{"message": {"content": "result"}}]
        }
        endpoint = NoThinkingChatEndpoint(
            base_url="http://localhost/v1",
            api_key="test-key",
            timeout=1,
            retries=0,
            strip_thinking=True,
        )

        with patch("batch_agent_evaluator.requests.post", return_value=response) as post:
            output = endpoint.chat(
                model="memeagent-main",
                messages=[{"role": "user", "content": "test"}],
                temperature=0,
                max_tokens=100,
                enable_thinking=True,
            )

        self.assertEqual("result", output)
        self.assertFalse(
            post.call_args.kwargs["json"]["chat_template_kwargs"]["enable_thinking"]
        )


if __name__ == "__main__":
    unittest.main()
