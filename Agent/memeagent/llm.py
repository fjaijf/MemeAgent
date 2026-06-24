from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from openai import OpenAI
import requests

from .config import MemeAgentConfig


@dataclass(frozen=True)
class ChatResponse:
    content: str


class OpenAICompatibleChatClient:
    """Small OpenAI SDK wrapper with the invoke() shape used by MemeAgent."""

    def __init__(
        self,
        config: MemeAgentConfig,
        api_key: str,
        *,
        model: str | None = None,
        temperature: float | None = None,
        timeout: float | None = None,
        max_tokens: int | None = None,
        max_retries: int | None = None,
    ) -> None:
        client_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "timeout": config.timeout if timeout is None else timeout,
            "max_retries": config.max_retries if max_retries is None else max_retries,
        }
        if config.base_url:
            client_kwargs["base_url"] = config.base_url

        self.client = OpenAI(**client_kwargs)
        self.model = model or config.model
        self.temperature = config.temperature if temperature is None else temperature
        self.max_tokens = config.max_tokens if max_tokens is None else max_tokens

    def invoke(self, messages: list[Any]) -> ChatResponse:
        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [self._convert_message(message) for message in messages],
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            request_kwargs["max_tokens"] = self.max_tokens

        response = self.client.chat.completions.create(**request_kwargs)
        content = response.choices[0].message.content or ""
        return ChatResponse(content=content)

    def stream(self, messages: list[Any]):
        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [self._convert_message(message) for message in messages],
            "temperature": self.temperature,
            "stream": True,
        }
        if self.max_tokens is not None:
            request_kwargs["max_tokens"] = self.max_tokens

        for chunk in self.client.chat.completions.create(**request_kwargs):
            if not getattr(chunk, "choices", None):
                continue
            delta = getattr(chunk.choices[0], "delta", None)
            if delta is None:
                continue
            content = getattr(delta, "content", None)
            if content:
                yield content

    def _convert_message(self, message: Any) -> dict[str, Any]:
        if isinstance(message, SystemMessage):
            return {"role": "system", "content": message.content}
        if isinstance(message, HumanMessage):
            return {"role": "user", "content": message.content}

        role = getattr(message, "type", None) or getattr(message, "role", "user")
        content = getattr(message, "content", message)
        if role == "human":
            role = "user"
        if role == "ai":
            role = "assistant"
        return {"role": role, "content": content}


class ZhipuAIChatClient:
    """Small Zhipu GLM HTTP wrapper with the invoke() shape used by MemeAgent."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        temperature: float,
        max_tokens: int | None,
        timeout: float,
        thinking_enabled: bool = True,
        base_url: str | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.thinking_enabled = thinking_enabled
        self.base_url = (
            base_url
            or os.getenv("ZAI_BASE_URL")
            or os.getenv("ZHIPUAI_BASE_URL")
            or "https://open.bigmodel.cn/api/paas/v4"
        ).rstrip("/")

    def invoke(self, messages: list[Any]) -> ChatResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [self._convert_message(message) for message in messages],
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        if self.thinking_enabled:
            payload["thinking"] = {"type": "enabled"}

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            return ChatResponse(content="")
        message = choices[0].get("message") or {}
        return ChatResponse(content=str(message.get("content") or "").strip())

    def _convert_message(self, message: Any) -> dict[str, Any]:
        if isinstance(message, SystemMessage):
            return {"role": "system", "content": message.content}
        if isinstance(message, HumanMessage):
            return {"role": "user", "content": self._text_only_content(message.content)}

        role = getattr(message, "type", None) or getattr(message, "role", "user")
        content = getattr(message, "content", message)
        if role == "human":
            role = "user"
        if role == "ai":
            role = "assistant"
        return {"role": role, "content": self._text_only_content(content)}

    def _text_only_content(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
            return "\n".join(part for part in parts if part)
        return str(content)


def _openai_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        raise ValueError(
            "OPENAI_API_KEY is not set. Create D:\\自研Agent\\MemeAgent\\Agent\\.env "
            "from .env.example and set OPENAI_API_KEY=your_real_key, or set "
            "OPENAI_API_KEY in your shell before running."
        )
    return api_key


def _zai_api_key() -> str:
    api_key = (
        os.getenv("ZAI_API_KEY")
        or os.getenv("GLM_API_KEY")
        or os.getenv("ZHIPUAI_API_KEY")
    )
    if not api_key or api_key == "your-api-key":
        raise ValueError(
            "ZAI_API_KEY is not set. Add ZAI_API_KEY=your_real_key to .env "
            "when MEMEAGENT_CONTROLLER_PROVIDER=glm is enabled."
        )
    return api_key


def create_llm(config: MemeAgentConfig) -> Any:
    """Create the primary model client used for vision and final analysis."""

    if config.provider in {"openai", "openai-compatible"}:
        return OpenAICompatibleChatClient(config=config, api_key=_openai_api_key())

    if config.provider in {"glm", "zai", "zhipu"}:
        return ZhipuAIChatClient(
            api_key=_zai_api_key(),
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout=config.timeout,
            thinking_enabled=config.controller_thinking_enabled,
        )

    raise ValueError(f"Unsupported provider: {config.provider}")


def create_controller_llm(config: MemeAgentConfig) -> Any | None:
    """Create a separate controller model when configured."""

    provider = config.controller_provider
    if not provider:
        return None

    if provider in {"openai", "openai-compatible"}:
        return OpenAICompatibleChatClient(
            config=config,
            api_key=_openai_api_key(),
            model=config.controller_model,
            temperature=config.controller_temperature,
            timeout=config.controller_timeout,
            max_tokens=config.controller_max_tokens,
            max_retries=config.controller_max_retries,
        )

    if provider in {"glm", "zai", "zhipu"}:
        return ZhipuAIChatClient(
            api_key=_zai_api_key(),
            model=config.controller_model,
            temperature=config.controller_temperature,
            max_tokens=config.controller_max_tokens,
            timeout=config.controller_timeout,
            thinking_enabled=config.controller_thinking_enabled,
        )

    raise ValueError(f"Unsupported controller provider: {provider}")
