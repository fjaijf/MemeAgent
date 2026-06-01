from __future__ import annotations

import os

from langchain_openai import ChatOpenAI

from .config import MemeAgentConfig


def create_llm(config: MemeAgentConfig) -> ChatOpenAI:
    """Create the LLM object.

    This mirrors the TradingAgents pattern:
    1. build config
    2. create client
    3. hand the LLM object to the agent
    """
    if config.provider != "openai":
        raise ValueError(f"Unsupported provider for initial scaffold: {config.provider}")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        raise ValueError(
            "OPENAI_API_KEY is not set. Create D:\\自研Agent\\MemeAgent\\.env "
            "from .env.example and set OPENAI_API_KEY=your_real_key, or set "
            "OPENAI_API_KEY in your shell before running."
        )

    kwargs = {
        "model": config.model,
        "api_key": api_key,
        "temperature": config.temperature,
    }
    if config.base_url:
        kwargs["base_url"] = config.base_url

    return ChatOpenAI(**kwargs)
