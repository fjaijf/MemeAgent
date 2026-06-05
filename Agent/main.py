from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from memeagent.agent import MemeAgent
from memeagent.cli_ui import MemeAgentCLI, RunSummary
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
    parser.add_argument(
        "--plain",
        action="store_true",
        help="Disable rich terminal visuals and print plain text output.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream the final analysis as it is generated.",
    )
    parser.add_argument(
        "--stream-markdown",
        action="store_true",
        help="Render streamed analysis as a live Markdown panel. Plain streaming is steadier.",
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
            zhihu_api_key=config.zhihu_api_key,
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
    ui = MemeAgentCLI(enabled=not args.plain, stream_markdown=args.stream_markdown)

    try:
        ui.print_start(
            RunSummary(
                model=config.model,
                image_count=len(args.image) + len(args.image_url),
                timeout=config.timeout,
                search_enabled=args.search,
                search_provider=config.search_provider,
                web_results=config.search_max_results,
                news_results=config.news_max_results,
                search_timeout=config.search_timeout,
            )
        )
        ui.start_activity()
        stream_started = False
        search_shown = False

        def handle_progress(stage: str, message: str) -> None:
            nonlocal stream_started
            ui.update(stage, message)
            if args.stream and stage == "analysis" and not stream_started:
                stream_started = True
                ui.stop_activity()
                ui.start_stream()

        def handle_search_ready(
            input_mode: str,
            search_report: str,
            visual_report: str,
            retrieval_plan: str,
        ) -> None:
            nonlocal search_shown
            if not (args.stream and args.search and args.show_search):
                return
            search_shown = True
            ui.stop_activity()
            ui.print_result(
                analysis="",
                input_mode=input_mode,
                search_report=search_report,
                visual_report=visual_report,
                retrieval_plan=retrieval_plan,
                show_search=True,
            )

        result = workflow.run(
            topic=args.topic,
            context=args.context,
            image_paths=args.image,
            image_urls=args.image_url,
            use_search=args.search,
            progress=handle_progress if args.stream else ui.update,
            stream_analysis=args.stream,
            analysis_delta=ui.stream_delta if args.stream else None,
            search_ready=handle_search_ready if args.stream else None,
        )
        if args.stream:
            ui.stop_stream()
            if args.search and args.show_search and not search_shown:
                ui.print_result(
                    analysis="",
                    input_mode=result.input_mode,
                    search_report=result.search_report,
                    visual_report=result.visual_report,
                    retrieval_plan=result.retrieval_plan,
                    show_search=True,
                )
        else:
            ui.stop_activity()
            ui.print_result(
                analysis=result.analysis,
                input_mode=result.input_mode,
                search_report=result.search_report,
                visual_report=result.visual_report,
                retrieval_plan=result.retrieval_plan,
                show_search=args.search and args.show_search,
            )
    except KeyboardInterrupt:
        ui.stop_activity()
        ui.print_error("\nInterrupted by user.")
        raise SystemExit(130)
    except Exception as exc:
        ui.stop_activity()
        ui.print_error(f"MemeAgent failed: {exc}")
        raise


if __name__ == "__main__":
    main()
