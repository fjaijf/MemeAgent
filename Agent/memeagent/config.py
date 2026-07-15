from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv


def load_project_env(project_root: str | Path) -> None:
    root = Path(project_root)
    load_dotenv(root / ".env")
    load_dotenv(root / ".env.local", override=True)


@dataclass(frozen=True)
class MemeAgentConfig:
    provider: str
    model: str
    base_url: str | None
    temperature: float
    timeout: float
    max_tokens: int | None
    max_retries: int
    controller_provider: str | None
    controller_model: str
    controller_temperature: float
    controller_timeout: float
    controller_max_tokens: int | None
    controller_max_retries: int
    controller_thinking_enabled: bool
    search_enabled: bool
    search_provider: str
    search_api_key: str | None
    tavily_api_key: str | None
    zhihu_api_key: str | None
    anspire_api_key: str | None
    glm_search_api_key: str | None
    glm_search_engine: str
    glm_search_recency_filter: str
    glm_search_content_size: str
    glm_search_domain_filter: str | None
    search_proxy: str | None
    context_proxy: str | None
    search_max_results: int
    news_max_results: int
    search_timeout: float
    search_country: str
    search_lang: str
    search_context_sites: str
    tavily_search_depth: str
    cache_enabled: bool
    cache_dir: str
    search_cache_ttl_seconds: int
    news_cache_ttl_seconds: int
    trajectory_cache_enabled: bool
    trajectory_cache_dir: str
    memory_enabled: bool
    memory_dir: str
    memory_recall_limit: int
    system_prompt: str

    @classmethod
    def from_env(cls) -> "MemeAgentConfig":
        base_url = (os.getenv("OPENAI_BASE_URL") or "").strip().strip('"').strip("'")
        provider = os.getenv("MEMEAGENT_PROVIDER", "openai").strip().lower()
        model = (os.getenv("MEMEAGENT_MODEL") or "gpt-4o-mini").strip()
        raw_max_tokens = (os.getenv("MEMEAGENT_MAX_TOKENS") or "0").strip()
        max_tokens_value = int(raw_max_tokens) if raw_max_tokens else 0
        controller_provider = (
            os.getenv("MEMEAGENT_CONTROLLER_PROVIDER", "").strip().lower() or None
        )
        controller_model = (
            os.getenv("MEMEAGENT_CONTROLLER_MODEL")
            or ("glm-5.1" if controller_provider in {"glm", "zai", "zhipu"} else model)
        ).strip()
        raw_controller_max_tokens = (
            os.getenv("MEMEAGENT_CONTROLLER_MAX_TOKENS") or "0"
        ).strip()
        controller_max_tokens_value = (
            int(raw_controller_max_tokens) if raw_controller_max_tokens else 0
        )
        controller_thinking = (
            os.getenv("MEMEAGENT_CONTROLLER_THINKING", "true").strip().lower()
        )
        cache_enabled = os.getenv("MEMEAGENT_CACHE_ENABLED", "true").strip().lower()
        search_enabled = os.getenv("MEMEAGENT_SEARCH_ENABLED", "true").strip().lower()
        trajectory_cache_enabled = (
            os.getenv("MEMEAGENT_TRAJECTORY_CACHE_ENABLED", "true").strip().lower()
        )
        memory_enabled = os.getenv("MEMEAGENT_MEMORY_ENABLED", "true").strip().lower()
        return cls(
            provider=provider,
            model=model,
            base_url=base_url or None,
            temperature=float(os.getenv("MEMEAGENT_TEMPERATURE", "0.2")),
            timeout=float(os.getenv("MEMEAGENT_TIMEOUT", "60")),
            max_tokens=max_tokens_value if max_tokens_value > 0 else None,
            max_retries=int(os.getenv("MEMEAGENT_MAX_RETRIES", "0")),
            controller_provider=controller_provider,
            controller_model=controller_model,
            controller_temperature=float(
                os.getenv("MEMEAGENT_CONTROLLER_TEMPERATURE", "0.2")
            ),
            controller_timeout=float(
                os.getenv("MEMEAGENT_CONTROLLER_TIMEOUT", os.getenv("MEMEAGENT_TIMEOUT", "60"))
            ),
            controller_max_tokens=(
                controller_max_tokens_value if controller_max_tokens_value > 0 else None
            ),
            controller_max_retries=int(
                os.getenv(
                    "MEMEAGENT_CONTROLLER_MAX_RETRIES",
                    os.getenv("MEMEAGENT_MAX_RETRIES", "0"),
                )
            ),
            controller_thinking_enabled=controller_thinking not in {
                "0",
                "false",
                "no",
                "off",
                "disabled",
            },
            search_enabled=search_enabled
            not in {"0", "false", "no", "off", "disabled"},
            search_provider=os.getenv("MEMEAGENT_SEARCH_PROVIDER", "ddgs").strip(),
            search_api_key=(
                os.getenv("MEMEAGENT_SEARCH_API_KEY", "").strip().strip('"').strip("'")
                or None
            ),
            tavily_api_key=(
                os.getenv("MEMEAGENT_TAVILY_API_KEY", "").strip().strip('"').strip("'")
                or os.getenv("TAVILY_API_KEY", "").strip().strip('"').strip("'")
                or None
            ),
            zhihu_api_key=(
                os.getenv("MEMEAGENT_ZHIHU_API_KEY", "").strip().strip('"').strip("'")
                or None
            ),
            anspire_api_key=(
                os.getenv("MEMEAGENT_ANSPIRE_API_KEY", "").strip().strip('"').strip("'")
                or os.getenv("ANSPIRE_API_KEY", "").strip().strip('"').strip("'")
                or None
            ),
            glm_search_api_key=(
                os.getenv("MEMEAGENT_GLM_SEARCH_API_KEY", "").strip().strip('"').strip("'")
                or os.getenv("MEMEAGENT_GLM_API_KEY", "").strip().strip('"').strip("'")
                or os.getenv("MEMEAGENT_ZAI_API_KEY", "").strip().strip('"').strip("'")
                or None
            ),
            glm_search_engine=os.getenv("MEMEAGENT_GLM_SEARCH_ENGINE", "search_pro").strip(),
            glm_search_recency_filter=os.getenv(
                "MEMEAGENT_GLM_SEARCH_RECENCY_FILTER",
                "noLimit",
            ).strip(),
            glm_search_content_size=os.getenv(
                "MEMEAGENT_GLM_SEARCH_CONTENT_SIZE",
                "medium",
            ).strip(),
            glm_search_domain_filter=(
                os.getenv("MEMEAGENT_GLM_SEARCH_DOMAIN_FILTER", "")
                .strip()
                .strip('"')
                .strip("'")
                or None
            ),
            search_proxy=(
                os.getenv("MEMEAGENT_SEARCH_PROXY", "").strip().strip('"').strip("'")
                or None
            ),
            context_proxy=(
                os.getenv("MEMEAGENT_CONTEXT_PROXY", "").strip().strip('"').strip("'")
                or os.getenv("MEMEAGENT_SEARCH_PROXY", "").strip().strip('"').strip("'")
                or None
            ),
            search_max_results=int(os.getenv("MEMEAGENT_SEARCH_MAX_RESULTS", "5")),
            news_max_results=int(os.getenv("MEMEAGENT_NEWS_MAX_RESULTS", "5")),
            search_timeout=float(os.getenv("MEMEAGENT_SEARCH_TIMEOUT", "12")),
            search_country=os.getenv("MEMEAGENT_SEARCH_COUNTRY", "us").strip(),
            search_lang=os.getenv("MEMEAGENT_SEARCH_LANG", "en").strip(),
            search_context_sites=os.getenv(
                "MEMEAGENT_SEARCH_CONTEXT_SITES",
                (
                    "reddit.com,x.com,twitter.com,weibo.com,zhihu.com,"
                    "tieba.baidu.com,bilibili.com,tiktok.com"
                ),
            ).strip(),
            tavily_search_depth=os.getenv("MEMEAGENT_TAVILY_SEARCH_DEPTH", "basic").strip(),
            cache_enabled=cache_enabled not in {"0", "false", "no", "off"},
            cache_dir=os.getenv("MEMEAGENT_CACHE_DIR", ".memeagent_cache").strip(),
            search_cache_ttl_seconds=int(
                os.getenv("MEMEAGENT_SEARCH_CACHE_TTL_SECONDS", str(7 * 24 * 60 * 60))
            ),
            news_cache_ttl_seconds=int(
                os.getenv("MEMEAGENT_NEWS_CACHE_TTL_SECONDS", str(6 * 60 * 60))
            ),
            trajectory_cache_enabled=trajectory_cache_enabled
            not in {"0", "false", "no", "off"},
            trajectory_cache_dir=(
                os.getenv("MEMEAGENT_TRAJECTORY_CACHE_DIR")
                or os.getenv("MEMEAGENT_CACHE_DIR")
                or ".memeagent_cache"
            ).strip(),
            memory_enabled=memory_enabled not in {"0", "false", "no", "off"},
            memory_dir=os.getenv("MEMEAGENT_MEMORY_DIR", ".memeagent_memory").strip(),
            memory_recall_limit=int(os.getenv("MEMEAGENT_MEMORY_RECALL_LIMIT", "3")),
            system_prompt=(
                "You are MemeAgent, a rigorous research assistant for meme studies "
                "and online discourse analysis. Analyze memes as multimodal cultural "
                "objects by grounding every important claim in visible image/OCR "
                "evidence, user-provided context, or explicit "
                "inference."
            ),
        )
