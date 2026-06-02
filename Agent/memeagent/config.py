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
    search_provider: str
    search_api_key: str | None
    search_proxy: str | None
    search_max_results: int
    news_max_results: int
    search_timeout: float
    search_country: str
    search_lang: str
    tavily_search_depth: str
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
            search_provider=os.getenv("MEMEAGENT_SEARCH_PROVIDER", "ddgs").strip(),
            search_api_key=(
                os.getenv("MEMEAGENT_SEARCH_API_KEY", "").strip().strip('"').strip("'")
                or None
            ),
            search_proxy=(
                os.getenv("MEMEAGENT_SEARCH_PROXY", "").strip().strip('"').strip("'")
                or None
            ),
            search_max_results=int(os.getenv("MEMEAGENT_SEARCH_MAX_RESULTS", "5")),
            news_max_results=int(os.getenv("MEMEAGENT_NEWS_MAX_RESULTS", "5")),
            search_timeout=float(os.getenv("MEMEAGENT_SEARCH_TIMEOUT", "12")),
            search_country=os.getenv("MEMEAGENT_SEARCH_COUNTRY", "us").strip(),
            search_lang=os.getenv("MEMEAGENT_SEARCH_LANG", "en").strip(),
            tavily_search_depth=os.getenv("MEMEAGENT_TAVILY_SEARCH_DEPTH", "basic").strip(),
            system_prompt=(
                "You are MemeAgent, a crypto-native research assistant. "
                "Analyze meme-driven narratives, community momentum, risks, and "
                "near-term speculation clearly and without hype."
            ),
        )
