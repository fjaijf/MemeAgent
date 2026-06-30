from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from memeagent.agent import MemeAgent
from memeagent.cli_ui import MemeAgentCLI, RunSummary
from memeagent.config import MemeAgentConfig, load_project_env
from memeagent.heads import HEADS
from memeagent.llm import create_controller_llm, create_llm
from memeagent.memory import MemeMemoryStore
from memeagent.search_agent import SearchAgentConfig, WebSearchAgent
from memeagent.trajectory import MemeTrajectoryCache
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
        help="Enable external retrieval for this run.",
    )
    parser.add_argument(
        "--no-search",
        action="store_true",
        help="Disable external retrieval; analyze only with the agent and local context.",
    )
    parser.add_argument(
        "--force-search",
        action="store_true",
        help="Force external retrieval planning when retrieval is enabled.",
    )
    parser.add_argument(
        "--show-search",
        action="store_true",
        help="Print the search report before the final analysis result.",
    )
    parser.add_argument(
        "--iterative-search",
        action="store_true",
        help="Use LLM reflection to run multiple retrieval rounds before analysis.",
    )
    parser.add_argument(
        "--search-max-rounds",
        type=int,
        default=5,
        help="Maximum retrieval rounds when --iterative-search is enabled.",
    )
    parser.add_argument(
        "--controller-max-rounds",
        type=int,
        default=3,
        help="Maximum controller-planned analysis rounds before final output.",
    )
    parser.add_argument(
        "--controller-confidence-threshold",
        type=float,
        default=0.8,
        help="Controller confidence threshold for final output.",
    )
    parser.add_argument(
        "--search-provider",
        default=None,
        help="Override search provider. Supported values include ddgs, tavily, zhihu, anspire, glm, or comma-separated combinations.",
    )
    parser.add_argument(
        "--search-api-key",
        default=None,
        help="Override search API key for providers such as Tavily.",
    )
    parser.add_argument(
        "--tavily-api-key",
        default=None,
        help="Override Tavily search API key.",
    )
    parser.add_argument(
        "--zhihu-api-key",
        default=None,
        help="Override Zhihu search API key.",
    )
    parser.add_argument(
        "--anspire-api-key",
        default=None,
        help="Override Anspire search API key.",
    )
    parser.add_argument(
        "--glm-search-engine",
        default=None,
        help="Override GLM web search engine, for example search_pro.",
    )
    parser.add_argument(
        "--glm-search-domain-filter",
        default=None,
        help="Restrict GLM web search to one domain, for example www.sohu.com.",
    )
    parser.add_argument(
        "--search-proxy",
        default=None,
        help="Override search proxy URL, for example http://127.0.0.1:7890.",
    )
    parser.add_argument(
        "--context-proxy",
        default=None,
        help="Override page/context fetch proxy URL, for example http://127.0.0.1:7890.",
    )
    parser.add_argument(
        "--search-max-results",
        type=int,
        default=None,
        help="Override maximum web results per provider.",
    )
    parser.add_argument(
        "--news-max-results",
        type=int,
        default=None,
        help="Override maximum news results.",
    )
    parser.add_argument(
        "--search-timeout",
        type=float,
        default=None,
        help="Override search timeout in seconds.",
    )
    parser.add_argument(
        "--search-country",
        default=None,
        help="Override search country code, for example us or cn.",
    )
    parser.add_argument(
        "--search-lang",
        default=None,
        help="Override search language, for example en or zh-cn.",
    )
    parser.add_argument(
        "--search-context-sites",
        default=None,
        help=(
            "Comma-separated sites for original-post/context queries, for example "
            "reddit.com,x.com,weibo.com,zhihu.com."
        ),
    )
    parser.add_argument(
        "--tavily-search-depth",
        choices=["basic", "advanced"],
        default=None,
        help="Override Tavily search depth.",
    )
    parser.add_argument(
        "--no-search-cache",
        action="store_true",
        help="Disable search result cache for this run.",
    )
    parser.add_argument(
        "--search-cache-ttl-seconds",
        type=int,
        default=None,
        help="Override web search cache TTL in seconds.",
    )
    parser.add_argument(
        "--news-cache-ttl-seconds",
        type=int,
        default=None,
        help="Override news search cache TTL in seconds.",
    )
    parser.add_argument(
        "--no-trajectory-cache",
        action="store_true",
        help="Disable end-to-end workflow trajectory recording for this run.",
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
    parser.add_argument(
        "--trace-mode",
        choices=["live", "off"],
        default="live",
        help="Show a live workflow trace while running, then clear it before final output.",
    )
    parser.add_argument(
        "--task",
        action="append",
        default=[],
        help=(
            "Run one or more independent analysis heads instead of final synthesis. "
            "Choices: harmfulness, sentiment, intent, evolution, audience, "
            "evidence-audit, all. Can be passed multiple times or comma-separated."
        ),
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="List available multi-head task names and exit.",
    )
    return parser.parse_args()


def main() -> None:
    project_root = Path(__file__).resolve().parent
    load_project_env(project_root)
    args = parse_args()

    if args.list_tasks:
        print("Available MemeAgent task heads:")
        for name, head in HEADS.items():
            print(f"- {name}: {head.description}")
        print("- all: run every task head above")
        return

    config = MemeAgentConfig.from_env()
    search_requested = config.search_enabled
    if args.search:
        search_requested = True
    if args.no_search:
        search_requested = False
    search_overrides = {}
    if args.search_provider is not None:
        search_overrides["search_provider"] = args.search_provider.strip()
    if args.search_api_key is not None:
        search_overrides["search_api_key"] = args.search_api_key.strip() or None
    if args.tavily_api_key is not None:
        search_overrides["tavily_api_key"] = args.tavily_api_key.strip() or None
    if args.zhihu_api_key is not None:
        search_overrides["zhihu_api_key"] = args.zhihu_api_key.strip() or None
    if args.anspire_api_key is not None:
        search_overrides["anspire_api_key"] = args.anspire_api_key.strip() or None
    if args.glm_search_engine is not None:
        search_overrides["glm_search_engine"] = args.glm_search_engine.strip()
    if args.glm_search_domain_filter is not None:
        search_overrides["glm_search_domain_filter"] = (
            args.glm_search_domain_filter.strip() or None
        )
    if args.search_proxy is not None:
        search_proxy = args.search_proxy.strip() or None
        search_overrides["search_proxy"] = search_proxy
        if args.context_proxy is None:
            search_overrides["context_proxy"] = search_proxy
    if args.context_proxy is not None:
        search_overrides["context_proxy"] = args.context_proxy.strip() or None
    if args.search_max_results is not None:
        search_overrides["search_max_results"] = args.search_max_results
    if args.news_max_results is not None:
        search_overrides["news_max_results"] = args.news_max_results
    if args.search_timeout is not None:
        search_overrides["search_timeout"] = args.search_timeout
    if args.search_country is not None:
        search_overrides["search_country"] = args.search_country.strip()
    if args.search_lang is not None:
        search_overrides["search_lang"] = args.search_lang.strip()
    if args.search_context_sites is not None:
        search_overrides["search_context_sites"] = args.search_context_sites.strip()
    if args.tavily_search_depth is not None:
        search_overrides["tavily_search_depth"] = args.tavily_search_depth
    if args.no_search_cache:
        search_overrides["cache_enabled"] = False
    if args.search_cache_ttl_seconds is not None:
        search_overrides["search_cache_ttl_seconds"] = args.search_cache_ttl_seconds
    if args.news_cache_ttl_seconds is not None:
        search_overrides["news_cache_ttl_seconds"] = args.news_cache_ttl_seconds
    if args.no_trajectory_cache:
        search_overrides["trajectory_cache_enabled"] = False
    if search_overrides:
        config = replace(config, **search_overrides)
    cache_dir = Path(config.cache_dir).expanduser()
    if not cache_dir.is_absolute():
        cache_dir = project_root / cache_dir
    search_cache_path = str(cache_dir / "search.sqlite3")
    trajectory_cache_dir = Path(config.trajectory_cache_dir).expanduser()
    if not trajectory_cache_dir.is_absolute():
        trajectory_cache_dir = project_root / trajectory_cache_dir
    trajectory_cache = (
        MemeTrajectoryCache(trajectory_cache_dir / "trajectory.sqlite3")
        if config.trajectory_cache_enabled
        else None
    )
    memory_dir = Path(config.memory_dir).expanduser()
    if not memory_dir.is_absolute():
        memory_dir = project_root / memory_dir
    memory_store = (
        MemeMemoryStore(memory_dir / "memory.sqlite3")
        if config.memory_enabled
        else None
    )
    llm = create_llm(config)
    agent = MemeAgent(llm=llm, system_prompt=config.system_prompt)
    controller_llm = create_controller_llm(config)
    controller_agent = (
        MemeAgent(llm=controller_llm, system_prompt=config.system_prompt)
        if controller_llm is not None
        else None
    )
    search_agent = WebSearchAgent(
        SearchAgentConfig(
            search_provider=config.search_provider,
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
            context_proxy=config.context_proxy,
            search_max_results=config.search_max_results,
            news_max_results=config.news_max_results,
            search_timeout=config.search_timeout,
            search_country=config.search_country,
            search_lang=config.search_lang,
            search_context_sites=config.search_context_sites,
            tavily_search_depth=config.tavily_search_depth,
            cache_enabled=config.cache_enabled,
            search_cache_path=search_cache_path,
            search_cache_ttl_seconds=config.search_cache_ttl_seconds,
            news_cache_ttl_seconds=config.news_cache_ttl_seconds,
        )
    )
    workflow = MemeResearchWorkflow(
        meme_agent=agent,
        controller_agent=controller_agent,
        search_agent=search_agent,
        memory_store=memory_store,
        memory_recall_limit=config.memory_recall_limit,
        trajectory_cache=trajectory_cache,
    )
    ui = MemeAgentCLI(
        enabled=not args.plain,
        stream_markdown=args.stream_markdown,
        trace_mode=args.trace_mode,
    )

    try:
        ui.print_start(
            RunSummary(
                model=config.model,
                image_count=len(args.image) + len(args.image_url),
                timeout=config.timeout,
                search_enabled=search_requested,
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
            controller_report: str,
        ) -> None:
            nonlocal search_shown
            if not (search_requested and args.show_search):
                return
            search_shown = True
            ui.stop_activity()
            ui.print_result(
                analysis="",
                input_mode=input_mode,
                search_report=search_report,
                visual_report=visual_report,
                retrieval_plan=retrieval_plan,
                controller_report=controller_report,
                show_search=True,
            )

        if args.task:
            result = workflow.run_heads(
                topic=args.topic,
                context=args.context,
                image_paths=args.image,
                image_urls=args.image_url,
                task_heads=args.task,
                use_search=search_requested,
                force_search=args.force_search,
                progress=ui.update,
                search_ready=(
                    handle_search_ready
                    if search_requested and args.show_search
                    else None
                ),
                iterative_search=args.iterative_search,
                search_max_rounds=args.search_max_rounds,
                controller_max_rounds=args.controller_max_rounds,
                controller_confidence_threshold=args.controller_confidence_threshold,
            )
            ui.stop_activity()
            ui.print_result(
                analysis=result.formatted_output,
                input_mode=result.input_mode,
                search_report=result.search_report,
                visual_report=result.visual_report,
                retrieval_plan=result.retrieval_plan,
                controller_report=result.controller_report,
                show_search=search_requested and args.show_search and not search_shown,
                analysis_title="Task Heads",
            )
            return

        result = workflow.run(
            topic=args.topic,
            context=args.context,
            image_paths=args.image,
            image_urls=args.image_url,
            use_search=search_requested,
            force_search=args.force_search,
            progress=handle_progress if args.stream else ui.update,
            stream_analysis=args.stream,
            analysis_delta=ui.stream_delta if args.stream else None,
            search_ready=handle_search_ready if search_requested and args.show_search else None,
            iterative_search=args.iterative_search,
            search_max_rounds=args.search_max_rounds,
            controller_max_rounds=args.controller_max_rounds,
            controller_confidence_threshold=args.controller_confidence_threshold,
        )
        if args.stream:
            ui.stop_stream()
            if search_requested and args.show_search and not search_shown:
                ui.print_result(
                    analysis="",
                    input_mode=result.input_mode,
                    search_report=result.search_report,
                    visual_report=result.visual_report,
                    retrieval_plan=result.retrieval_plan,
                    controller_report=result.controller_report,
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
                controller_report=result.controller_report,
                show_search=search_requested and args.show_search and not search_shown,
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
