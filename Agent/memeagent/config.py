from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class MemeAgentConfig:
    provider: str
    model: str
    base_url: str | None
    temperature: float
    timeout: float
    max_retries: int
    system_prompt: str

    @classmethod
    def from_env(cls) -> "MemeAgentConfig":
        base_url = (os.getenv("OPENAI_BASE_URL") or "").strip().strip('"').strip("'")
        model = (os.getenv("MEMEAGENT_MODEL") or "gpt-4o-mini").strip()
        return cls(
            provider="openai",
            model=model,
            base_url=base_url or None,
            temperature=float(os.getenv("MEMEAGENT_TEMPERATURE", "0.2")),
            timeout=float(os.getenv("MEMEAGENT_TIMEOUT", "60")),
            max_retries=int(os.getenv("MEMEAGENT_MAX_RETRIES", "0")),
            system_prompt=(
                "You are MemeAgent, a crypto-native research assistant. "
                "Analyze meme-driven narratives, community momentum, risks, and "
                "near-term speculation clearly and without hype."
            ),
        )
