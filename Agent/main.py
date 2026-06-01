from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from memeagent.agent import MemeAgent
from memeagent.config import MemeAgentConfig
from memeagent.llm import create_llm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MemeAgent on a topic.")
    parser.add_argument("--topic", required=True, help="Topic, token, or meme coin to analyze.")
    parser.add_argument(
        "--context",
        default="",
        help="Optional extra context, such as recent posts, headlines, or notes.",
    )
    return parser.parse_args()


def main() -> None:
    project_root = Path(__file__).resolve().parent
    load_dotenv(project_root / ".env")
    args = parse_args()

    config = MemeAgentConfig.from_env()
    llm = create_llm(config)
    agent = MemeAgent(llm=llm, system_prompt=config.system_prompt)

    result = agent.run(topic=args.topic, context=args.context)
    print(result)


if __name__ == "__main__":
    main()
