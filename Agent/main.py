from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from memeagent.agent import MemeAgent
from memeagent.config import MemeAgentConfig
from memeagent.llm import create_llm
from memeagent.search_agent import SearchAgentConfig, WebSearchAgent
from memeagent.workflow import MemeResearchWorkflow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MemeAgent on a topic.")
    parser.add_argument("--topic", default="", help="Topic, token, or meme coin to analyze.")
    parser.add_argument(
        "--context",
        default="",
        help="Optional extra context, such as recent posts, headlines, or notes.",
    )
    parser.add_argument(
        "--image",
        action="append",
        default=[],
        help="Optional local image path. Pass multiple times to attach multiple images.",
    )
    parser.add_argument(
        "--image-url",
        action="append",
        default=[],
        help="Optional remote image URL. Pass multiple times to attach multiple images.",
    )
    parser.add_argument(
        "--search",
        action="store_true",
        help="Run a web-searching agent before meme analysis.",
    )
    parser.add_argument(
        "--show-search",
        action="store_true",
        help="Print the search report before the final analysis result.",
    )
    return parser.parse_args()


def main() -> None:
    project_root = Path(__file__).resolve().parent
    load_dotenv(project_root / ".env")
    args = parse_args()

    config = MemeAgentConfig.from_env()
    llm = create_llm(config)
    agent = MemeAgent(llm=llm, system_prompt=config.system_prompt)
    search_agent = WebSearchAgent(
        SearchAgentConfig(
            search_provider=config.search_provider,
            search_api_key=config.search_api_key,
            search_proxy=config.search_proxy,
            search_max_results=config.search_max_results,
            news_max_results=config.news_max_results,
            search_timeout=config.search_timeout,
            search_country=config.search_country,
            search_lang=config.search_lang,
            tavily_search_depth=config.tavily_search_depth,
        )
    )
    workflow = MemeResearchWorkflow(meme_agent=agent, search_agent=search_agent)

    try:
        print(
            f"Starting MemeAgent request with model={config.model}, "
            f"images={len(args.image) + len(args.image_url)}, timeout={config.timeout}s..."
        )
        if args.search:
            print(
                f"Web search enabled with provider={config.search_provider}: "
                f"up to {config.search_max_results} web results, "
                f"{config.news_max_results} news results, search timeout={config.search_timeout}s."
            )
        result = workflow.run(
            topic=args.topic,
            context=args.context,
            image_paths=args.image,
            image_urls=args.image_url,
            use_search=args.search,
        )
        if args.search and args.show_search and result.search_report:
            print("\n=== Search Report ===")
            print(result.search_report)
            print("\n=== Final Analysis ===")
        print(result.analysis)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        raise SystemExit(130)


if __name__ == "__main__":
    main()
