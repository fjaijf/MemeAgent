from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from memeagent.agent import MemeAgent
from memeagent.config import MemeAgentConfig
from memeagent.llm import create_llm


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
    return parser.parse_args()


def main() -> None:
    project_root = Path(__file__).resolve().parent
    load_dotenv(project_root / ".env")
    args = parse_args()

    config = MemeAgentConfig.from_env()
    llm = create_llm(config)
    agent = MemeAgent(llm=llm, system_prompt=config.system_prompt)

    try:
        print(
            f"Starting MemeAgent request with model={config.model}, "
            f"images={len(args.image) + len(args.image_url)}, timeout={config.timeout}s..."
        )
        result = agent.run(
            topic=args.topic,
            context=args.context,
            image_paths=args.image,
            image_urls=args.image_url,
        )
        print(result)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        raise SystemExit(130)


if __name__ == "__main__":
    main()
