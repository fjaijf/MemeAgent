from __future__ import annotations

import argparse
from pathlib import Path
import sys

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from memeagent.config import MemeAgentConfig
from memeagent.llm import create_llm
from memeagent.search_agent import SearchAgentConfig, WebSearchAgent


def _print_section(title: str) -> None:
    print(f"\n=== {title} ===")


def _clip(value: str, max_chars: int = 500) -> str:
    value = " ".join(value.split())
    if len(value) <= max_chars:
        return value
    return value[:max_chars].rstrip() + "..."


def test_llm(config: MemeAgentConfig) -> None:
    _print_section("LLM")
    llm = create_llm(config)
    response = llm.invoke(
        [
            HumanMessage(
                content=(
                    "Reply with exactly one short sentence: "
                    "MemeAgent LLM smoke test passed."
                )
            )
        ]
    )
    content = getattr(response, "content", response)
    print(_clip(str(content)))


def test_search(config: MemeAgentConfig, query: str) -> None:
    _print_section("Search")
    search_agent = WebSearchAgent(
        SearchAgentConfig(
            search_provider=config.search_provider,
            search_api_key=config.search_api_key,
            zhihu_api_key=config.zhihu_api_key,
            search_proxy=config.search_proxy,
            search_max_results=min(config.search_max_results, 2),
            news_max_results=min(config.news_max_results, 1),
            search_timeout=config.search_timeout,
            search_country=config.search_country,
            search_lang=config.search_lang,
            tavily_search_depth=config.tavily_search_depth,
        )
    )
    report = search_agent.run(topic=query)
    print(_clip(report, max_chars=1200))
    if "No web or news results found." in report:
        raise RuntimeError("Search completed but returned no results.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MemeAgent smoke tests.")
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip the LLM API call.",
    )
    parser.add_argument(
        "--skip-search",
        action="store_true",
        help="Skip the search provider call.",
    )
    parser.add_argument(
        "--query",
        default="meme sentiment analysis",
        help="Small query used for the search smoke test.",
    )
    return parser.parse_args()


def main() -> int:
    project_root = Path(__file__).resolve().parent
    load_dotenv(project_root / ".env")
    args = parse_args()
    config = MemeAgentConfig.from_env()

    failures: list[str] = []

    if not args.skip_llm:
        try:
            test_llm(config)
            print("LLM smoke test: OK")
        except Exception as exc:
            failures.append(f"LLM smoke test failed: {exc}")
            print(f"LLM smoke test: FAILED - {exc}")

    if not args.skip_search:
        try:
            test_search(config, args.query)
            print("Search smoke test: OK")
        except Exception as exc:
            failures.append(f"Search smoke test failed: {exc}")
            print(f"Search smoke test: FAILED - {exc}")

    if failures:
        _print_section("Summary")
        for failure in failures:
            print(f"- {failure}")
        return 1

    _print_section("Summary")
    print("All requested smoke tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
