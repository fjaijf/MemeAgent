from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from local_inference_server import (
    ServerSettings,
    VLLMModelSettings,
    create_app,
)


def _model(name: str, port: int) -> VLLMModelSettings:
    return VLLMModelSettings(
        name=name,
        model_path=f"/models/{name}",
        host="127.0.0.1",
        port=port,
        backend_url=f"http://127.0.0.1:{port}/v1",
        tensor_parallel_size=None,
        gpu_memory_utilization=None,
        max_model_len=None,
        max_num_seqs=None,
        max_num_batched_tokens=None,
        dtype="auto",
        quantization=None,
        limit_mm_per_prompt=None,
        cuda_visible_devices=None,
        extra_args=(),
    )


class LocalInferenceServerTests(unittest.TestCase):
    def test_router_forwards_to_model_backend(self) -> None:
        settings = ServerSettings(
            main=_model("memeagent-main", 8009),
            controller=_model("memeagent-controller", 8010),
            host="127.0.0.1",
            port=8008,
            command="vllm",
            spawn_vllm=False,
            startup_timeout_seconds=1,
            startup_poll_seconds=0.01,
            trust_remote_code=True,
            enable_prefix_caching=True,
            api_key=None,
            backend_api_key=None,
        )
        response_mock = Mock()
        response_mock.status_code = 200
        response_mock.json.return_value = {
            "id": "chatcmpl-test",
            "choices": [{"message": {"content": "ok"}}],
        }

        with patch("local_inference_server._wait_for_backend"), patch(
            "local_inference_server.requests.post",
            return_value=response_mock,
        ) as post:
            with TestClient(create_app(settings)) as client:
                response = client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "memeagent-controller",
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                )

        self.assertEqual(200, response.status_code)
        post.assert_called_once()
        self.assertEqual(
            "http://127.0.0.1:8010/v1/chat/completions",
            post.call_args.args[0],
        )
        self.assertEqual("memeagent-controller", post.call_args.kwargs["json"]["model"])


if __name__ == "__main__":
    unittest.main()
