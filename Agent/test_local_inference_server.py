from __future__ import annotations

import signal
import subprocess
import unittest
from dataclasses import replace
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from local_inference_server import (
    ManagedVLLMProcess,
    ServerSettings,
    VLLMModelSettings,
    _assign_launch_cuda_devices,
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
    def test_stop_force_kills_process_group_after_grace_period(self) -> None:
        settings = ServerSettings(
            main=_model("memeagent-main", 8009),
            controller=_model("memeagent-controller", 8010),
            host="127.0.0.1",
            port=8008,
            command="vllm",
            spawn_vllm=True,
            startup_timeout_seconds=3600,
            startup_poll_seconds=2,
            backend_timeout_seconds=60,
            trust_remote_code=True,
            enable_prefix_caching=True,
            api_key=None,
            backend_api_key=None,
        )
        managed = ManagedVLLMProcess(settings.main, settings)
        process = Mock()
        process.pid = 12345
        process.poll.return_value = None
        process.wait.side_effect = [
            subprocess.TimeoutExpired("vllm", 30),
            None,
        ]
        managed.process = process

        with patch.object(managed, "_terminate_process_group") as terminate:
            managed.stop()

        self.assertEqual(2, terminate.call_count)
        self.assertEqual(signal.SIGTERM, terminate.call_args_list[0].args[0])
        self.assertEqual(signal.SIGKILL, terminate.call_args_list[1].args[0])

    def test_launch_cuda_devices_are_split_by_tensor_parallel_size(self) -> None:
        main = replace(_model("memeagent-main", 8009), tensor_parallel_size=2)
        controller = replace(
            _model("memeagent-controller", 8010),
            tensor_parallel_size=2,
        )

        main, controller = _assign_launch_cuda_devices(
            main,
            controller,
            "4,5,6,7",
        )

        self.assertEqual("4,5", main.cuda_visible_devices)
        self.assertEqual("6,7", controller.cuda_visible_devices)

    def test_launch_cuda_devices_require_enough_devices(self) -> None:
        main = replace(_model("memeagent-main", 8009), tensor_parallel_size=2)
        controller = replace(
            _model("memeagent-controller", 8010),
            tensor_parallel_size=2,
        )

        with self.assertRaisesRegex(ValueError, "requires at least 4"):
            _assign_launch_cuda_devices(main, controller, "4,5")

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
            backend_timeout_seconds=60,
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
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                )

        self.assertEqual(200, response.status_code)
        post.assert_called_once()
        self.assertEqual(
            "http://127.0.0.1:8010/v1/chat/completions",
            post.call_args.args[0],
        )
        self.assertEqual("memeagent-controller", post.call_args.kwargs["json"]["model"])
        self.assertEqual(
            {"enable_thinking": False},
            post.call_args.kwargs["json"]["chat_template_kwargs"],
        )


if __name__ == "__main__":
    unittest.main()
