from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
from types import SimpleNamespace
import time
from threading import Lock
from typing import Any
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from memeagent.config import MemeAgentConfig, load_project_env
from memeagent.llm import ChatResponse, LocalTransformersChatClient


_LOCAL_PROVIDERS = {"local", "local-transformers", "transformers", "hf", "huggingface"}


@dataclass(frozen=True)
class ServerSettings:
    main_model: str
    main_model_path: str
    controller_model: str
    controller_model_path: str
    host: str
    port: int
    temperature: float
    max_tokens: int | None
    timeout: float
    controller_temperature: float
    controller_max_tokens: int | None
    controller_timeout: float
    controller_thinking_enabled: bool


@dataclass
class ModelRuntime:
    name: str
    client: LocalTransformersChatClient
    lock: Lock


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[dict[str, Any]] = Field(default_factory=list)
    temperature: float | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    stream: bool = False


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip().strip('"').strip("'")


def _resolve_model_path(
    *,
    env_name: str,
    provider: str | None,
    configured_model: str,
    label: str,
) -> str:
    value = _env(env_name)
    if value:
        return value
    if provider in _LOCAL_PROVIDERS:
        return configured_model
    raise RuntimeError(
        f"{env_name} is required because {label} is configured as an API/client model."
    )


def _build_settings(project_root: Path) -> ServerSettings:
    load_project_env(project_root)
    config = MemeAgentConfig.from_env()
    main_model = _env("MEMEAGENT_SERVICE_MAIN_MODEL", config.model or "memeagent-main")
    controller_model = _env(
        "MEMEAGENT_SERVICE_CONTROLLER_MODEL",
        config.controller_model or "memeagent-controller",
    )
    return ServerSettings(
        main_model=main_model,
        main_model_path=_resolve_model_path(
            env_name="MEMEAGENT_SERVICE_MAIN_MODEL_PATH",
            provider=config.provider,
            configured_model=config.model,
            label="main model",
        ),
        controller_model=controller_model,
        controller_model_path=_resolve_model_path(
            env_name="MEMEAGENT_SERVICE_CONTROLLER_MODEL_PATH",
            provider=config.controller_provider,
            configured_model=config.controller_model,
            label="controller model",
        ),
        host=_env("MEMEAGENT_SERVICE_HOST", "127.0.0.1"),
        port=int(_env("MEMEAGENT_SERVICE_PORT", "8008")),
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        timeout=config.timeout,
        controller_temperature=config.controller_temperature,
        controller_max_tokens=config.controller_max_tokens,
        controller_timeout=config.controller_timeout,
        controller_thinking_enabled=config.controller_thinking_enabled,
    )


def _to_langchain_message(message: dict[str, Any]) -> Any:
    role = str(message.get("role") or "user").strip().lower()
    content = message.get("content", "")
    if role == "system":
        if isinstance(content, str):
            return SystemMessage(content=content)
        return SystemMessage(content=_text_only_content(content))
    if role == "user":
        return HumanMessage(content=content)
    if role == "assistant":
        return SimpleNamespace(role="assistant", content=content)
    return SimpleNamespace(role=role or "user", content=content)


def _text_only_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and (item.get("type") == "text" or "text" in item):
                parts.append(str(item.get("text", "")))
        return "\n".join(part for part in parts if part)
    return str(content)


def _load_runtime(
    *,
    name: str,
    model_path: str,
    temperature: float,
    max_tokens: int | None,
    timeout: float,
    enable_thinking: bool | None,
) -> ModelRuntime:
    client = LocalTransformersChatClient(
        model_path=model_path,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        enable_thinking=enable_thinking,
    )
    print(f"[{name}] loading {model_path}", flush=True)
    client._ensure_loaded()
    model = getattr(client, "_model", None)
    device_map = getattr(model, "hf_device_map", None)
    if device_map:
        print(f"[{name}] loaded with hf_device_map={device_map}", flush=True)
    else:
        print(f"[{name}] loaded on device={getattr(model, 'device', 'unknown')}", flush=True)
    return ModelRuntime(name=name, client=client, lock=Lock())


def _invoke_runtime(
    runtime: ModelRuntime,
    messages: list[dict[str, Any]],
    *,
    temperature: float | None,
    max_tokens: int | None,
) -> ChatResponse:
    converted = [_to_langchain_message(message) for message in messages]
    with runtime.lock:
        old_temperature = runtime.client.temperature
        old_max_tokens = runtime.client.max_tokens
        try:
            if temperature is not None:
                runtime.client.temperature = temperature
            if max_tokens is not None:
                runtime.client.max_tokens = max_tokens
            return runtime.client.invoke(converted)
        finally:
            runtime.client.temperature = old_temperature
            runtime.client.max_tokens = old_max_tokens


def _completion_payload(
    *,
    completion_id: str,
    model: str,
    content: str,
) -> dict[str, Any]:
    created = int(time.time())
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _stream_payloads(*, completion_id: str, model: str, content: str):
    created = int(time.time())
    chunks = [
        {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        },
        {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
        },
        {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        },
    ]
    import json

    for chunk in chunks:
        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


def create_app(settings: ServerSettings) -> FastAPI:
    app = FastAPI(title="MemeAgent Local Inference Server")
    runtimes: dict[str, ModelRuntime] = {}

    @app.on_event("startup")
    def startup() -> None:
        runtimes[settings.main_model] = _load_runtime(
            name=settings.main_model,
            model_path=settings.main_model_path,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            timeout=settings.timeout,
            enable_thinking=False,
        )
        runtimes[settings.controller_model] = _load_runtime(
            name=settings.controller_model,
            model_path=settings.controller_model_path,
            temperature=settings.controller_temperature,
            max_tokens=settings.controller_max_tokens,
            timeout=settings.controller_timeout,
            enable_thinking=settings.controller_thinking_enabled,
        )
        print(
            "Local inference server is ready. "
            f"Models: {', '.join(runtimes)}",
            flush=True,
        )

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "models": sorted(runtimes)}

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        now = int(time.time())
        return {
            "object": "list",
            "data": [
                {"id": name, "object": "model", "created": now, "owned_by": "memeagent"}
                for name in sorted(runtimes)
            ],
        }

    @app.post("/v1/chat/completions")
    def chat_completions(request: ChatCompletionRequest):
        runtime = runtimes.get(request.model)
        if runtime is None:
            raise HTTPException(
                status_code=404,
                detail=f"Unknown model {request.model!r}. Available: {', '.join(sorted(runtimes))}",
            )
        max_tokens = request.max_tokens or request.max_completion_tokens
        response = _invoke_runtime(
            runtime,
            request.messages,
            temperature=request.temperature,
            max_tokens=max_tokens,
        )
        content = response.content
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        if request.stream:
            return StreamingResponse(
                _stream_payloads(
                    completion_id=completion_id,
                    model=request.model,
                    content=content,
                ),
                media_type="text/event-stream",
            )
        return _completion_payload(
            completion_id=completion_id,
            model=request.model,
            content=content,
        )

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve MemeAgent local models.")
    parser.add_argument("--host", default=None, help="Override MEMEAGENT_SERVICE_HOST.")
    parser.add_argument("--port", type=int, default=None, help="Override MEMEAGENT_SERVICE_PORT.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved service settings without loading model weights.",
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

    print("Resolved service settings:", flush=True)
    print(f"  {settings.main_model}: {settings.main_model_path}", flush=True)
    print(f"  {settings.controller_model}: {settings.controller_model_path}", flush=True)
    print(f"  listen: {settings.host}:{settings.port}", flush=True)
    if args.dry_run:
        return 0

    import uvicorn

    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
