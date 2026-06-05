from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from openai import OpenAI

from .config import MemeAgentConfig


@dataclass(frozen=True)
class ChatResponse:
    content: str


class OpenAICompatibleChatClient:
    """Small OpenAI SDK wrapper with the invoke() shape used by MemeAgent."""

    def __init__(self, config: MemeAgentConfig, api_key: str) -> None:
        client_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "timeout": config.timeout,
            "max_retries": config.max_retries,
        }
        if config.base_url:
            client_kwargs["base_url"] = config.base_url

        self.client = OpenAI(**client_kwargs)
        self.model = config.model
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens

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


def create_llm(config: MemeAgentConfig) -> OpenAICompatibleChatClient:
    """Create the model client with the same OpenAI SDK path as test_llm_api.py."""

    if config.provider != "openai":
        raise ValueError(f"Unsupported provider for initial scaffold: {config.provider}")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        raise ValueError(
            "OPENAI_API_KEY is not set. Create D:\\自研Agent\\MemeAgent\\Agent\\.env "
            "from .env.example and set OPENAI_API_KEY=your_real_key, or set "
            "OPENAI_API_KEY in your shell before running."
        )

    return OpenAICompatibleChatClient(config=config, api_key=api_key)
