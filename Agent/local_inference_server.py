from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
from dataclasses import dataclass
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any, Iterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

try:
    from pydantic import ConfigDict
except ImportError:  # pragma: no cover - pydantic v1 compatibility
    ConfigDict = None
import requests

from memeagent.config import MemeAgentConfig, load_project_env


@dataclass(frozen=True)
class VLLMModelSettings:
    name: str
    model_path: str
    host: str
    port: int
    backend_url: str
    tensor_parallel_size: int | None
    gpu_memory_utilization: float | None
    max_model_len: int | None
    max_num_seqs: int | None
    max_num_batched_tokens: int | None
    dtype: str | None
    quantization: str | None
    limit_mm_per_prompt: str | None
    cuda_visible_devices: str | None
    extra_args: tuple[str, ...]


@dataclass(frozen=True)
class ServerSettings:
    main: VLLMModelSettings
    controller: VLLMModelSettings
    host: str
    port: int
    command: str
    spawn_vllm: bool
    startup_timeout_seconds: float
    startup_poll_seconds: float
    backend_timeout_seconds: float | None
    trust_remote_code: bool
    enable_prefix_caching: bool
    api_key: str | None
    backend_api_key: str | None


class ChatCompletionRequest(BaseModel):
    if ConfigDict is not None:
        model_config = ConfigDict(extra="allow")
    else:  # pragma: no cover - pydantic v1 compatibility
        class Config:
            extra = "allow"

    model: str
    messages: list[dict[str, Any]] = Field(default_factory=list)
    temperature: float | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    stream: bool = False


class ManagedVLLMProcess:
    def __init__(self, settings: VLLMModelSettings, server: ServerSettings) -> None:
        self.settings = settings
        self.server = server
        self.process: subprocess.Popen[str] | None = None

    def command(self) -> list[str]:
        cmd = [
            self.server.command,
            "serve",
            self.settings.model_path,
            "--host",
            self.settings.host,
            "--port",
            str(self.settings.port),
            "--served-model-name",
            self.settings.name,
        ]
        if self.server.trust_remote_code:
            cmd.append("--trust-remote-code")
        if self.server.enable_prefix_caching:
            cmd.append("--enable-prefix-caching")
        if self.server.backend_api_key:
            cmd.extend(["--api-key", self.server.backend_api_key])
        if self.settings.tensor_parallel_size is not None:
            cmd.extend(["--tensor-parallel-size", str(self.settings.tensor_parallel_size)])
        if self.settings.gpu_memory_utilization is not None:
            cmd.extend(
                ["--gpu-memory-utilization", str(self.settings.gpu_memory_utilization)]
            )
        if self.settings.max_model_len is not None:
            cmd.extend(["--max-model-len", str(self.settings.max_model_len)])
        if self.settings.max_num_seqs is not None:
            cmd.extend(["--max-num-seqs", str(self.settings.max_num_seqs)])
        if self.settings.max_num_batched_tokens is not None:
            cmd.extend(
                ["--max-num-batched-tokens", str(self.settings.max_num_batched_tokens)]
            )
        if self.settings.dtype:
            cmd.extend(["--dtype", self.settings.dtype])
        if self.settings.quantization:
            cmd.extend(["--quantization", self.settings.quantization])
        if self.settings.limit_mm_per_prompt:
            cmd.extend(["--limit-mm-per-prompt", self.settings.limit_mm_per_prompt])
        cmd.extend(self.settings.extra_args)
        return cmd

    def start(self) -> None:
        env = os.environ.copy()
        if self.settings.cuda_visible_devices:
            env["CUDA_VISIBLE_DEVICES"] = self.settings.cuda_visible_devices
        cmd = self.command()
        print(f"[vLLM:{self.settings.name}] starting: {' '.join(shlex.quote(x) for x in cmd)}")
        self.process = subprocess.Popen(
            cmd,
            env=env,
            text=True,
        )

    def stop(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        print(f"[vLLM:{self.settings.name}] stopping pid={self.process.pid}", flush=True)
        self.process.terminate()
        try:
            self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=10)

    def assert_running(self) -> None:
        if self.process is not None and self.process.poll() is not None:
            raise RuntimeError(
                f"vLLM process for {self.settings.name!r} exited with code "
                f"{self.process.returncode}."
            )


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip().strip('"').strip("'")


def _env_bool(name: str, default: bool) -> bool:
    value = _env(name)
    if not value:
        return default
    return value.lower() not in {"0", "false", "no", "off", "disabled"}


def _env_int(name: str, default: int | None = None) -> int | None:
    value = _env(name)
    return int(value) if value else default


def _env_float(name: str, default: float | None = None) -> float | None:
    value = _env(name)
    return float(value) if value else default


def _split_extra_args(value: str) -> tuple[str, ...]:
    return tuple(shlex.split(value)) if value else ()


def _first_env(names: tuple[str, ...], default: str = "") -> str:
    for name in names:
        value = _env(name)
        if value:
            return value
    return default


def _resolve_model_path(env_name: str, configured_model: str) -> str:
    value = _env(env_name)
    return value or configured_model


def _backend_base_url(host: str, port: int) -> str:
    return f"http://{host}:{port}/v1"


def _model_settings(
    *,
    role: str,
    name: str,
    model_path: str,
    host: str,
    port: int,
) -> VLLMModelSettings:
    prefix = f"MEMEAGENT_VLLM_{role.upper()}_"
    common_prefix = "MEMEAGENT_VLLM_"
    resolved_host = _first_env((prefix + "HOST",), host)
    resolved_port = _env_int(prefix + "PORT", port) or port
    backend_url = _first_env(
        (prefix + "BACKEND_URL",),
        _backend_base_url(resolved_host, resolved_port),
    )
    return VLLMModelSettings(
        name=name,
        model_path=model_path,
        host=resolved_host,
        port=resolved_port,
        backend_url=backend_url.rstrip("/"),
        tensor_parallel_size=_env_int(
            prefix + "TENSOR_PARALLEL_SIZE",
            _env_int(common_prefix + "TENSOR_PARALLEL_SIZE"),
        ),
        gpu_memory_utilization=_env_float(
            prefix + "GPU_MEMORY_UTILIZATION",
            _env_float(common_prefix + "GPU_MEMORY_UTILIZATION"),
        ),
        max_model_len=_env_int(
            prefix + "MAX_MODEL_LEN",
            _env_int(common_prefix + "MAX_MODEL_LEN"),
        ),
        max_num_seqs=_env_int(
            prefix + "MAX_NUM_SEQS",
            _env_int(common_prefix + "MAX_NUM_SEQS"),
        ),
        max_num_batched_tokens=_env_int(
            prefix + "MAX_NUM_BATCHED_TOKENS",
            _env_int(common_prefix + "MAX_NUM_BATCHED_TOKENS"),
        ),
        dtype=_first_env((prefix + "DTYPE", common_prefix + "DTYPE"), "auto") or None,
        quantization=_first_env((prefix + "QUANTIZATION", common_prefix + "QUANTIZATION")),
        limit_mm_per_prompt=_first_env(
            (prefix + "LIMIT_MM_PER_PROMPT", common_prefix + "LIMIT_MM_PER_PROMPT")
        ),
        cuda_visible_devices=_first_env((prefix + "CUDA_VISIBLE_DEVICES",)),
        extra_args=(
            _split_extra_args(_env(common_prefix + "EXTRA_ARGS"))
            + _split_extra_args(_env(prefix + "EXTRA_ARGS"))
        ),
    )


def _build_settings(project_root: Path) -> ServerSettings:
    load_project_env(project_root)
    config = MemeAgentConfig.from_env()
    main_model = _env("MEMEAGENT_SERVICE_MAIN_MODEL", config.model or "memeagent-main")
    controller_model = _env(
        "MEMEAGENT_SERVICE_CONTROLLER_MODEL",
        config.controller_model or "memeagent-controller",
    )
    service_host = _env("MEMEAGENT_SERVICE_HOST", "127.0.0.1")
    service_port = int(_env("MEMEAGENT_SERVICE_PORT", "8008"))
    vllm_host = _env("MEMEAGENT_VLLM_HOST", "127.0.0.1")
    main_port = int(_env("MEMEAGENT_VLLM_MAIN_PORT", "8009"))
    controller_port = int(_env("MEMEAGENT_VLLM_CONTROLLER_PORT", "8010"))
    backend_timeout = _env_float("MEMEAGENT_VLLM_BACKEND_TIMEOUT", config.timeout)
    if backend_timeout is not None and backend_timeout <= 0:
        backend_timeout = None
    return ServerSettings(
        main=_model_settings(
            role="main",
            name=main_model,
            model_path=_resolve_model_path(
                "MEMEAGENT_SERVICE_MAIN_MODEL_PATH",
                config.model,
            ),
            host=vllm_host,
            port=main_port,
        ),
        controller=_model_settings(
            role="controller",
            name=controller_model,
            model_path=_resolve_model_path(
                "MEMEAGENT_SERVICE_CONTROLLER_MODEL_PATH",
                config.controller_model,
            ),
            host=vllm_host,
            port=controller_port,
        ),
        host=service_host,
        port=service_port,
        command=_env("MEMEAGENT_VLLM_COMMAND", "vllm"),
        spawn_vllm=_env_bool("MEMEAGENT_VLLM_SPAWN", True),
        startup_timeout_seconds=float(_env("MEMEAGENT_VLLM_STARTUP_TIMEOUT", "900")),
        startup_poll_seconds=float(_env("MEMEAGENT_VLLM_STARTUP_POLL_SECONDS", "2")),
        backend_timeout_seconds=backend_timeout,
        trust_remote_code=_env_bool("MEMEAGENT_VLLM_TRUST_REMOTE_CODE", True),
        enable_prefix_caching=_env_bool("MEMEAGENT_VLLM_ENABLE_PREFIX_CACHING", True),
        api_key=_env("MEMEAGENT_SERVICE_API_KEY") or None,
        backend_api_key=(
            _env("MEMEAGENT_VLLM_API_KEY")
            or _env("MEMEAGENT_SERVICE_BACKEND_API_KEY")
            or None
        ),
    )


def _auth_headers(settings: ServerSettings) -> dict[str, str]:
    if not settings.backend_api_key:
        return {}
    return {"Authorization": f"Bearer {settings.backend_api_key}"}


def _check_frontend_auth(request: Request, settings: ServerSettings) -> None:
    if not settings.api_key:
        return
    auth = request.headers.get("authorization", "")
    if auth != f"Bearer {settings.api_key}":
        raise HTTPException(status_code=401, detail="Invalid API key.")


def _wait_for_backend(
    model: VLLMModelSettings,
    *,
    settings: ServerSettings,
    process: ManagedVLLMProcess | None = None,
) -> None:
    deadline = time.time() + settings.startup_timeout_seconds
    models_url = model.backend_url + "/models"
    while time.time() < deadline:
        if process is not None:
            process.assert_running()
        try:
            response = requests.get(
                models_url,
                headers=_auth_headers(settings),
                timeout=5,
            )
            if response.status_code < 500:
                print(f"[vLLM:{model.name}] ready at {model.backend_url}", flush=True)
                return
        except requests.RequestException:
            pass
        time.sleep(settings.startup_poll_seconds)
    raise RuntimeError(
        f"Timed out waiting for vLLM backend {model.name!r} at {model.backend_url}."
    )


def _target_url(model: VLLMModelSettings, endpoint: str) -> str:
    return model.backend_url.rstrip("/") + endpoint


def _forward_json(
    *,
    model: VLLMModelSettings,
    payload: dict[str, Any],
    settings: ServerSettings,
) -> JSONResponse:
    try:
        response = requests.post(
            _target_url(model, "/chat/completions"),
            json=payload,
            headers=_auth_headers(settings),
            timeout=settings.backend_timeout_seconds,
        )
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    try:
        body = response.json()
    except ValueError:
        body = {"error": response.text}
    return JSONResponse(content=body, status_code=response.status_code)


def _forward_stream(
    *,
    model: VLLMModelSettings,
    payload: dict[str, Any],
    settings: ServerSettings,
) -> Iterator[bytes]:
    try:
        with requests.post(
            _target_url(model, "/chat/completions"),
            json=payload,
            headers=_auth_headers(settings),
            timeout=settings.backend_timeout_seconds,
            stream=True,
        ) as response:
            response.raise_for_status()
            for chunk in response.iter_content(chunk_size=None):
                if chunk:
                    yield chunk
    except requests.RequestException as exc:
        yield f"data: {{\"error\": {str(exc)!r}}}\n\n".encode("utf-8")
        yield b"data: [DONE]\n\n"


def _request_payload(body: ChatCompletionRequest) -> dict[str, Any]:
    if hasattr(body, "model_dump"):
        return body.model_dump(exclude_none=True)
    return body.dict(exclude_none=True)


def create_app(settings: ServerSettings) -> FastAPI:
    model_routes = {
        settings.main.name: settings.main,
        settings.controller.name: settings.controller,
    }
    processes: list[ManagedVLLMProcess] = []

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if settings.spawn_vllm:
            for model in dict.fromkeys(model_routes.values()):
                process = ManagedVLLMProcess(model, settings)
                process.start()
                processes.append(process)
            for process in processes:
                _wait_for_backend(process.settings, settings=settings, process=process)
        else:
            for model in dict.fromkeys(model_routes.values()):
                _wait_for_backend(model, settings=settings)
        print(
            "MemeAgent vLLM router is ready. "
            f"Models: {', '.join(sorted(model_routes))}",
            flush=True,
        )
        try:
            yield
        finally:
            for process in reversed(processes):
                process.stop()

    app = FastAPI(title="MemeAgent vLLM Router", lifespan=lifespan)

    @app.get("/health")
    def health(request: Request) -> dict[str, Any]:
        _check_frontend_auth(request, settings)
        backends = {}
        for name, model in model_routes.items():
            try:
                response = requests.get(
                    _target_url(model, "/models"),
                    headers=_auth_headers(settings),
                    timeout=5,
                )
                backends[name] = {
                    "status": "ok" if response.ok else "error",
                    "status_code": response.status_code,
                    "backend_url": model.backend_url,
                }
            except requests.RequestException as exc:
                backends[name] = {
                    "status": "error",
                    "error": str(exc),
                    "backend_url": model.backend_url,
                }
        return {"status": "ok", "models": sorted(model_routes), "backends": backends}

    @app.get("/v1/models")
    def models(request: Request) -> dict[str, Any]:
        _check_frontend_auth(request, settings)
        now = int(time.time())
        return {
            "object": "list",
            "data": [
                {"id": name, "object": "model", "created": now, "owned_by": "vllm"}
                for name in sorted(model_routes)
            ],
        }

    @app.post("/v1/chat/completions")
    def chat_completions(request: Request, body: ChatCompletionRequest):
        _check_frontend_auth(request, settings)
        model = model_routes.get(body.model)
        if model is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Unknown model {body.model!r}. "
                    f"Available: {', '.join(sorted(model_routes))}"
                ),
            )
        payload = _request_payload(body)
        if body.max_completion_tokens is not None and body.max_tokens is None:
            payload["max_tokens"] = body.max_completion_tokens
        if body.stream:
            return StreamingResponse(
                _forward_stream(model=model, payload=payload, settings=settings),
                media_type="text/event-stream",
            )
        return _forward_json(model=model, payload=payload, settings=settings)

    return app


def _print_model_settings(label: str, settings: VLLMModelSettings) -> None:
    print(f"  {label}: {settings.name}", flush=True)
    print(f"    path: {settings.model_path}", flush=True)
    print(f"    backend: {settings.backend_url}", flush=True)
    if settings.cuda_visible_devices:
        print(f"    CUDA_VISIBLE_DEVICES: {settings.cuda_visible_devices}", flush=True)
    cmd_settings = {
        "tensor_parallel_size": settings.tensor_parallel_size,
        "gpu_memory_utilization": settings.gpu_memory_utilization,
        "max_model_len": settings.max_model_len,
        "max_num_seqs": settings.max_num_seqs,
        "max_num_batched_tokens": settings.max_num_batched_tokens,
        "dtype": settings.dtype,
        "quantization": settings.quantization,
        "limit_mm_per_prompt": settings.limit_mm_per_prompt,
        "extra_args": " ".join(settings.extra_args),
    }
    for key, value in cmd_settings.items():
        if value not in {None, ""}:
            print(f"    {key}: {value}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Serve MemeAgent local models through vLLM OpenAI-compatible backends "
            "and a small model-name router."
        )
    )
    parser.add_argument("--host", default=None, help="Override MEMEAGENT_SERVICE_HOST.")
    parser.add_argument("--port", type=int, default=None, help="Override MEMEAGENT_SERVICE_PORT.")
    parser.add_argument(
        "--no-spawn",
        action="store_true",
        help=(
            "Do not launch vLLM processes. Proxy to MEMEAGENT_VLLM_*_BACKEND_URL "
            "instead."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved vLLM/router settings without starting servers.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    settings = _build_settings(project_root)
    if args.host:
        settings = ServerSettings(**{**settings.__dict__, "host": args.host})
    if args.port:
        settings = ServerSettings(**{**settings.__dict__, "port": args.port})
    if args.no_spawn:
        settings = ServerSettings(**{**settings.__dict__, "spawn_vllm": False})

    print("Resolved vLLM service settings:", flush=True)
    print(f"  router: {settings.host}:{settings.port}", flush=True)
    print(f"  command: {settings.command}", flush=True)
    print(f"  spawn_vllm: {settings.spawn_vllm}", flush=True)
    print(f"  trust_remote_code: {settings.trust_remote_code}", flush=True)
    print(f"  enable_prefix_caching: {settings.enable_prefix_caching}", flush=True)
    _print_model_settings("main", settings.main)
    _print_model_settings("controller", settings.controller)
    if args.dry_run:
        return 0

    import uvicorn

    try:
        uvicorn.run(
            create_app(settings),
            host=settings.host,
            port=settings.port,
            log_level="info",
        )
    except FileNotFoundError as exc:
        if settings.command in str(exc):
            print(
                "Could not find the vLLM command. Install vLLM in this environment "
                "or set MEMEAGENT_VLLM_COMMAND.",
                file=sys.stderr,
            )
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
