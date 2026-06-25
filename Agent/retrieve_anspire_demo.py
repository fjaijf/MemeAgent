from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from memeagent.config import MemeAgentConfig, load_project_env
from memeagent.retrieve_cli import (
    build_search_config,
    format_retrieval_report,
    run_retrieval,
)
from memeagent.search_agent import WebSearchAgent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a small MemeAgent retrieval demo with Anspire Search."
    )
    parser.add_argument(
        "--topic",
        default="meme sentiment analysis",
        help="Topic or direct query to retrieve with Anspire Search.",
    )
    parser.add_argument(
        "--mode",
        choices=("web", "news", "both"),
        default="web",
        help="Retrieval mode for the demo.",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=3,
        help="Maximum Anspire results to request.",
    )
    parser.add_argument(
        "--query",
        action="append",
        default=[],
        help="Optional direct query. Pass multiple times to bypass planning.",
    )
    return parser.parse_args()


def main() -> int:
    project_root = Path(__file__).resolve().parent
    load_project_env(project_root)
    args = parse_args()

    config = MemeAgentConfig.from_env()
    search_config = replace(
        build_search_config(config, project_root=project_root),
        search_provider="anspire",
        search_max_results=args.max_results,
        news_max_results=args.max_results,
        cache_enabled=False,
    )
    agent = WebSearchAgent(search_config)
    result = run_retrieval(
        agent=agent,
        topic=args.topic,
        direct_queries=args.query,
        mode=args.mode,
        news_agent=agent,
    )
    print(format_retrieval_report(agent, result, mode=args.mode, news_agent=agent))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
