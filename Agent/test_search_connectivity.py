from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import time
from typing import Callable

from memeagent.config import MemeAgentConfig, load_project_env
from memeagent.search_agent import SearchAgentConfig, WebSearchAgent


ProviderCheck = Callable[[WebSearchAgent, str], list[dict[str, object]]]
DEFAULT_PROVIDERS = ("zhihu", "anspire", "glm", "ddgs", "tavily")


def _clip(value: object, max_chars: int = 100) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def _build_agent(config: MemeAgentConfig, provider: str, max_results: int) -> WebSearchAgent:
    return WebSearchAgent(
        SearchAgentConfig(
            search_provider=provider,
            search_api_key=config.search_api_key,
            tavily_api_key=config.tavily_api_key,
            zhihu_api_key=config.zhihu_api_key,
            anspire_api_key=config.anspire_api_key,
            glm_search_api_key=config.glm_search_api_key,
            glm_search_engine=config.glm_search_engine,
            glm_search_recency_filter=config.glm_search_recency_filter,
            glm_search_content_size=config.glm_search_content_size,
            glm_search_domain_filter=config.glm_search_domain_filter,
            search_proxy=config.search_proxy,
            search_max_results=max_results,
            news_max_results=max_results,
            search_timeout=config.search_timeout,
            search_country=config.search_country,
            search_lang=config.search_lang,
            tavily_search_depth=config.tavily_search_depth,
            cache_enabled=False,
        )
    )


def _check_ddgs(agent: WebSearchAgent, query: str) -> list[dict[str, object]]:
    return agent._search_ddgs_text(query)


def _check_ddgs_news(agent: WebSearchAgent, query: str) -> list[dict[str, object]]:
    return agent._search_ddgs_news(query)


def _check_zhihu(agent: WebSearchAgent, query: str) -> list[dict[str, object]]:
    return agent._search_zhihu(query, agent.config.search_max_results)


def _check_tavily(agent: WebSearchAgent, query: str) -> list[dict[str, object]]:
    return agent._search_tavily(
        query,
        agent.config.search_max_results,
        topic="general",
    )


def _check_tavily_news(agent: WebSearchAgent, query: str) -> list[dict[str, object]]:
    return agent._search_tavily(
        query,
        agent.config.news_max_results,
        topic="news",
    )


def _check_anspire(agent: WebSearchAgent, query: str) -> list[dict[str, object]]:
    return agent._search_anspire(query, agent.config.search_max_results)


def _check_anspire_news(agent: WebSearchAgent, query: str) -> list[dict[str, object]]:
    return agent._search_anspire(query + " news", agent.config.news_max_results)


def _check_glm(agent: WebSearchAgent, query: str) -> list[dict[str, object]]:
    return agent._search_glm(query, agent.config.search_max_results)


def _check_glm_news(agent: WebSearchAgent, query: str) -> list[dict[str, object]]:
    return agent._search_glm(query, agent.config.news_max_results, news=True)


def _print_config(config: MemeAgentConfig) -> None:
    print("=== Search connectivity config ===")
    print(f"proxy: {config.search_proxy or '(none)'}")
    print(f"search_lang: {config.search_lang}")
    print(f"search_country: {config.search_country}")
    print(f"timeout: {config.search_timeout}s")
    print(f"tavily_api_key: {'set' if config.tavily_api_key or config.search_api_key else 'missing'}")
    print(f"zhihu_api_key: {'set' if config.zhihu_api_key or config.search_api_key else 'missing'}")
    print(f"anspire_api_key: {'set' if config.anspire_api_key or config.search_api_key else 'missing'}")
    print(
        "glm_search_key: "
        f"{'set' if config.glm_search_api_key or config.search_api_key else 'env/default'}"
    )


def _run_check(
    config: MemeAgentConfig,
    provider: str,
    label: str,
    query: str,
    max_results: int,
    check: ProviderCheck,
) -> bool:
    agent = _build_agent(config, provider, max_results)
    start = time.perf_counter()
    results, error = agent._run_with_timeout(
        lambda current_query: check(agent, current_query),
        query,
        label,
    )
    if error:
        elapsed = time.perf_counter() - start
        print(f"[FAILED] {label:<13} {elapsed:5.2f}s  {error}", flush=True)
        return False

    elapsed = time.perf_counter() - start
    if not results:
        print(
            f"[FAILED] {label:<13} {elapsed:5.2f}s  connected but returned 0 results",
            flush=True,
        )
        return False

    first = results[0]
    title = _clip(first.get("title") or first.get("body") or first)
    href = _clip(first.get("href") or first.get("url") or "", max_chars=140)
    print(f"[OK]     {label:<13} {elapsed:5.2f}s  {len(results)} result(s)", flush=True)
    print(f"         title: {title}", flush=True)
    if href:
        print(f"         url:   {href}", flush=True)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test MemeAgent search provider connectivity."
    )
    parser.add_argument(
        "--query",
        default="meme sentiment analysis",
        help="Query used by all provider checks.",
    )
    parser.add_argument(
        "--providers",
        default=",".join(DEFAULT_PROVIDERS),
        help=("Comma-separated providers to test, or 'all'. Supported: zhihu, anspire, glm, ddgs, tavily."),
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=2,
        help="Maximum results requested from each provider.",
    )
    parser.add_argument(
        "--kind",
        choices=("web", "news", "both"),
        default="web",
        help="Search path to test.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Override MEMEAGENT_SEARCH_TIMEOUT for this connectivity run.",
    )
    return parser.parse_args()


def main() -> int:
    project_root = Path(__file__).resolve().parent
    load_project_env(project_root)

    args = parse_args()
    config = MemeAgentConfig.from_env()
    if args.timeout is not None:
        config = replace(config, search_timeout=args.timeout)

    web_checks: dict[str, ProviderCheck] = {
        "ddgs": _check_ddgs,
        "zhihu": _check_zhihu,
        "tavily": _check_tavily,
        "anspire": _check_anspire,
        "glm": _check_glm,
        "zai": _check_glm,
        "zhipu": _check_glm,
    }
    news_checks: dict[str, ProviderCheck] = {
        "ddgs": _check_ddgs_news,
        "tavily": _check_tavily_news,
        "anspire": _check_anspire_news,
        "glm": _check_glm_news,
        "zai": _check_glm_news,
        "zhipu": _check_glm_news,
    }
    requested_providers = args.providers.strip().lower()
    providers = (
        list(DEFAULT_PROVIDERS)
        if requested_providers == "all"
        else [
            provider.strip().lower()
            for provider in args.providers.split(",")
            if provider.strip()
        ]
    )

    _print_config(config)
    print(f"query: {args.query}")
    print()

    failures: list[str] = []
    kinds = ("web", "news") if args.kind == "both" else (args.kind,)
    for provider in providers:
        for kind in kinds:
            checks = web_checks if kind == "web" else news_checks
            check = checks.get(provider)
            label = provider if args.kind != "both" else f"{provider}/{kind}"
            if check is None:
                print(f"[SKIP]   {label:<13} no {kind} search check")
                continue
            ok = _run_check(
                config,
                provider,
                label,
                args.query,
                args.max_results,
                check,
            )
            if not ok:
                failures.append(label)

    print()
    if failures:
        print("Summary: FAILED providers: " + ", ".join(failures))
        return 1

    print("Summary: all requested providers are reachable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
