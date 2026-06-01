from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class MemeAgentConfig:
    provider: str
    model: str
    base_url: str | None
    temperature: float
    system_prompt: str

    @classmethod
    def from_env(cls) -> "MemeAgentConfig":
        base_url = os.getenv("OPENAI_BASE_URL") or None
        return cls(
            provider="openai",
            model=os.getenv("MEMEAGENT_MODEL", "gpt-4o-mini"),
            base_url=base_url,
            temperature=float(os.getenv("MEMEAGENT_TEMPERATURE", "0.2")),
            system_prompt=(
                "You are MemeAgent, a crypto-native research assistant. "
                "Analyze meme-driven narratives, community momentum, risks, and "
                "near-term speculation clearly and without hype."
            ),
        )
