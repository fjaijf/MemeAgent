from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
from pathlib import Path
import sys
from typing import Any

from .agent import MemeAgent
from .config import MemeAgentConfig, load_project_env
from .context_fetcher import ContextFetchResult, fetch_contexts_for_results
from .llm import create_llm
from .search_agent import PlannedSearchQuery, SearchAgentConfig, WebSearchAgent


_RETRIEVE_NEWS_PROVIDER = "ddgs"
_RETRIEVE_NEWS_TIMEOUT = 20.0
_RETRIEVE_NEWS_MAX_QUERIES = 1


@dataclass(frozen=True)
class RetrievalRunResult:
    query_plan: list[PlannedSearchQuery]
    news_queries: list[str]
    web_results: list[dict[str, Any]]
    news_results: list[dict[str, Any]]
    context_results: list[ContextFetchResult]
    errors: list[str]
    cache_stats: str


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_project_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path


def _read_text_input(value: str) -> str:
    path = Path(value)
    if value == "-":
        return sys.stdin.read()
    return path.read_text(encoding="utf-8")


def _build_context_with_visual_report(context: str, visual_report: str) -> str:
    parts = []
    if context.strip():
        parts.append(context.strip())
    if visual_report.strip():
        parts.append(
            "Image-derived meme description for retrieval:\n" + visual_report.strip()
        )
    return "\n\n".join(parts)


def describe_images_for_retrieval(
    config: MemeAgentConfig,
    topic: str,
    context: str,
    image_paths: list[str],
    image_urls: list[str],
) -> str:
    if not image_paths and not image_urls:
        return ""

    llm = create_llm(config)
    agent = MemeAgent(llm=llm, system_prompt=config.system_prompt)
    return agent.describe_images_for_search(
        topic=topic,
        context=context,
        image_paths=image_paths,
        image_urls=image_urls,
    )


def build_search_config(
    config: MemeAgentConfig,
    project_root: Path | None = None,
) -> SearchAgentConfig:
    project_root = project_root or _project_root()
    cache_dir = _resolve_project_path(project_root, config.cache_dir)
    search_cache_path = str(cache_dir / "search.sqlite3")
    return SearchAgentConfig(
        search_provider=config.search_provider,
        search_api_key=config.search_api_key,
        tavily_api_key=config.tavily_api_key,
        zhihu_api_key=config.zhihu_api_key,
        qwen_search_api_key=config.qwen_search_api_key,
        qwen_search_base_url=config.qwen_search_base_url,
        qwen_search_model=config.qwen_search_model,
        glm_search_api_key=config.glm_search_api_key,
        glm_search_engine=config.glm_search_engine,
        glm_search_recency_filter=config.glm_search_recency_filter,
        glm_search_content_size=config.glm_search_content_size,
        glm_search_domain_filter=config.glm_search_domain_filter,
        search_proxy=config.search_proxy,
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


def _result_relevance_terms(
    agent: WebSearchAgent,
    topic: str,
    context: str,
    queries: list[str],
) -> set[str]:
    return agent._build_relevance_terms(topic, context, queries)


def _search_news_direct(
    agent: WebSearchAgent,
    queries: list[str],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    errors: list[str] = []
    for query in queries:
        try:
            results.extend(agent._search_news(query))
        except Exception as exc:
            errors.append(f"{query}: {exc}")

    if not results and errors:
        raise ValueError("; ".join(errors))

    relevance_terms = set()
    for query in queries:
        relevance_terms.update(agent._build_relevance_terms(query, "", [query]))
    return agent._dedupe_results(
        results,
        max_results=agent.config.news_max_results,
        relevance_terms=relevance_terms,
    )


def run_retrieval(
    agent: WebSearchAgent,
    topic: str,
    context: str = "",
    direct_queries: list[str] | None = None,
    mode: str = "both",
    news_agent: WebSearchAgent | None = None,
    news_max_queries: int = _RETRIEVE_NEWS_MAX_QUERIES,
) -> RetrievalRunResult:
    if mode not in {"plan", "web", "news", "both"}:
        raise ValueError("mode must be one of: plan, web, news, both")

    agent._reset_cache_stats()
    direct_queries = [
        " ".join(query.split()).strip()
        for query in (direct_queries or [])
        if " ".join(query.split()).strip()
    ]
    if direct_queries:
        query_plan = [
            PlannedSearchQuery(query=query, category="direct")
            for query in dict.fromkeys(direct_queries)
        ]
    else:
        query_plan = agent._build_query_plan(topic, context)

    queries = [item.query for item in query_plan]
    news_max_queries = max(1, news_max_queries)
    news_queries = queries if direct_queries else agent._build_news_queries(
        queries,
        context=context,
    )[:news_max_queries]
    if mode == "plan":
        return RetrievalRunResult(
            query_plan=query_plan,
            news_queries=news_queries,
            web_results=[],
            news_results=[],
            context_results=[],
            errors=[],
            cache_stats=agent._format_cache_stats(),
        )

    errors: list[str] = []
    web_results: list[dict[str, Any]] = []
    news_results: list[dict[str, Any]] = []

    if mode in {"web", "both"}:
        relevance_terms = _result_relevance_terms(agent, topic, context, queries)
        web_results, web_error = agent._run_with_timeout(
            agent._search_text_queries,
            (queries, relevance_terms),
            "Web search",
        )
        if web_error:
            errors.append(web_error)

    if mode in {"news", "both"}:
        news_runner = news_agent or agent
        news_results, news_error = news_runner._run_with_timeout(
            lambda payload: _search_news_direct(*payload),
            (news_runner, news_queries),
            "News search",
        )
        if news_error:
            errors.append(news_error)

    context_results = (
        fetch_contexts_for_results(web_results)
        if web_results and mode in {"web", "both"}
        else []
    )

    return RetrievalRunResult(
        query_plan=query_plan,
        news_queries=news_queries,
        web_results=web_results,
        news_results=news_results,
        context_results=context_results,
        errors=errors,
        cache_stats=agent._format_cache_stats(),
    )


def _format_query_plan(query_plan: list[PlannedSearchQuery]) -> str:
    lines = ["Search query plan:"]
    for item in query_plan:
        lines.append(f"- [{item.category}] {item.query}")
    return "\n".join(lines)


def _format_news_query_plan(news_queries: list[str]) -> str:
    lines = ["News query plan:"]
    if not news_queries:
        lines.append("- None")
    for query in news_queries:
        lines.append(f"- {query}")
    return "\n".join(lines)


def _format_results(
    agent: WebSearchAgent,
    heading: str,
    results: list[dict[str, Any]],
    news: bool = False,
) -> str:
    if not results:
        return f"## {heading}\nNo results."

    lines = [f"## {heading}"]
    for idx, item in enumerate(results, start=1):
        source_id = f"{'N' if news else 'W'}{idx}"
        title = _clean_text(item.get("title"))
        body = _clean_text(item.get("body"))
        href = _clean_text(item.get("href") or item.get("url"))
        result_type = agent._classify_result_type(item, news=news)
        lines.append(
            f"[{source_id}] {title}\n"
            f"   Candidate type: {result_type}\n"
            f"   Snippet: {body or 'N/A'}\n"
            f"   URL: {href or 'N/A'}"
        )
    return "\n".join(lines)


def _format_context_results(results: list[ContextFetchResult]) -> str:
    if not results:
        return "## Thread/Page Context\nNo high-value platform URLs were fetched."

    lines = ["## Thread/Page Context"]
    for idx, item in enumerate(results, start=1):
        label = f"C{idx}"
        lines.append(
            f"[{label}] Context for [{item.source_id}] ({item.site})\n"
            f"   URL: {item.url}"
        )
        if item.error:
            lines.append(f"   Error: {item.error}")
            continue
        if item.title:
            lines.append(f"   Title: {item.title}")
        if item.metadata:
            metadata = " | ".join(
                f"{key}={value}" for key, value in item.metadata.items() if value
            )
            if metadata:
                lines.append(f"   Metadata: {metadata}")
        if item.post_text:
            lines.append(f"   Post/Page text: {item.post_text}")
        comments = item.comments or []
        if comments:
            lines.append("   Comments/context:")
            for comment_index, comment in enumerate(comments, start=1):
                lines.append(f"   - c{comment_index}: {comment}")
        if not item.title and not item.post_text and not comments:
            lines.append("   No readable public context extracted.")
    return "\n".join(lines)


def format_retrieval_report(
    agent: WebSearchAgent,
    result: RetrievalRunResult,
    mode: str,
    news_agent: WebSearchAgent | None = None,
    visual_report: str = "",
) -> str:
    news_agent = news_agent or agent
    sections = [
        f"Search provider: {agent.config.search_provider}",
        (
            f"News provider: {news_agent.config.search_provider}"
            if mode in {"news", "both"}
            and news_agent.config.search_provider != agent.config.search_provider
            else ""
        ),
        result.cache_stats,
        _format_query_plan(result.query_plan),
    ]
    if visual_report.strip():
        sections.append("## Image-Derived Search Context\n" + visual_report.strip())
    if mode in {"plan", "news", "both"}:
        sections.append(_format_news_query_plan(result.news_queries))
    sections.extend(result.errors)
    if mode in {"web", "both"}:
        sections.append(_format_results(agent, "Web Search Results", result.web_results))
        sections.append(_format_context_results(result.context_results))
    if mode in {"news", "both"}:
        sections.append(
            _format_results(agent, "News Results", result.news_results, news=True)
        )
    return "\n\n".join(section for section in sections if section)


def result_to_jsonable(result: RetrievalRunResult) -> dict[str, Any]:
    return {
        "query_plan": [
            {"query": item.query, "category": item.category}
            for item in result.query_plan
        ],
        "news_queries": result.news_queries,
        "web_results": result.web_results,
        "news_results": result.news_results,
        "context_results": [
            {
                "source_id": item.source_id,
                "url": item.url,
                "site": item.site,
                "title": item.title,
                "post_text": item.post_text,
                "comments": item.comments or [],
                "metadata": item.metadata or {},
                "error": item.error,
            }
            for item in result.context_results
        ],
        "errors": result.errors,
        "cache_stats": result.cache_stats,
    }


def _apply_overrides(config: MemeAgentConfig, args: argparse.Namespace) -> MemeAgentConfig:
    overrides: dict[str, Any] = {}
    if args.search_provider is not None:
        overrides["search_provider"] = args.search_provider.strip()
    if args.search_api_key is not None:
        overrides["search_api_key"] = args.search_api_key.strip() or None
    if args.tavily_api_key is not None:
        overrides["tavily_api_key"] = args.tavily_api_key.strip() or None
    if args.zhihu_api_key is not None:
        overrides["zhihu_api_key"] = args.zhihu_api_key.strip() or None
    if args.qwen_search_api_key is not None:
        overrides["qwen_search_api_key"] = args.qwen_search_api_key.strip() or None
    if args.qwen_search_model is not None:
        overrides["qwen_search_model"] = args.qwen_search_model.strip()
    if args.qwen_search_base_url is not None:
        overrides["qwen_search_base_url"] = args.qwen_search_base_url.strip()
    if args.glm_search_api_key is not None:
        overrides["glm_search_api_key"] = args.glm_search_api_key.strip() or None
    if args.glm_search_engine is not None:
        overrides["glm_search_engine"] = args.glm_search_engine.strip()
    if args.glm_search_recency_filter is not None:
        overrides["glm_search_recency_filter"] = args.glm_search_recency_filter.strip()
    if args.glm_search_content_size is not None:
        overrides["glm_search_content_size"] = args.glm_search_content_size.strip()
    if args.glm_search_domain_filter is not None:
        overrides["glm_search_domain_filter"] = (
            args.glm_search_domain_filter.strip() or None
        )
    if args.search_proxy is not None:
        overrides["search_proxy"] = args.search_proxy.strip() or None
    if args.search_max_results is not None:
        overrides["search_max_results"] = args.search_max_results
    if args.news_max_results is not None:
        overrides["news_max_results"] = args.news_max_results
    if args.search_timeout is not None:
        overrides["search_timeout"] = args.search_timeout
    if args.search_country is not None:
        overrides["search_country"] = args.search_country.strip()
    if args.search_lang is not None:
        overrides["search_lang"] = args.search_lang.strip()
    if args.search_context_sites is not None:
        overrides["search_context_sites"] = args.search_context_sites.strip()
    if args.tavily_search_depth is not None:
        overrides["tavily_search_depth"] = args.tavily_search_depth
    if args.no_search_cache:
        overrides["cache_enabled"] = False
    if args.search_cache_ttl_seconds is not None:
        overrides["search_cache_ttl_seconds"] = args.search_cache_ttl_seconds
    if args.news_cache_ttl_seconds is not None:
        overrides["news_cache_ttl_seconds"] = args.news_cache_ttl_seconds
    return replace(config, **overrides) if overrides else config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MemeAgent retrieval without any LLM analysis calls."
    )
    parser.add_argument("--topic", default="", help="Topic, meme phrase, or entity.")
    parser.add_argument(
        "--context",
        default="",
        help="Optional OCR, caption, notes, or retrieval-plan text.",
    )
    parser.add_argument(
        "--context-file",
        default=None,
        help="Read extra context from a UTF-8 file. Use '-' to read stdin.",
    )
    parser.add_argument(
        "--query",
        action="append",
        default=[],
        help="Direct search query. Pass multiple times to bypass query planning.",
    )
    parser.add_argument(
        "--image",
        action="append",
        default=[],
        help="Local meme image path. Pass multiple times to attach multiple images.",
    )
    parser.add_argument(
        "--image-url",
        action="append",
        default=[],
        help="Remote meme image URL. Pass multiple times to attach multiple images.",
    )
    parser.add_argument(
        "--mode",
        choices=("plan", "web", "news", "both"),
        default="both",
        help="Run only query planning, web search, news search, or both.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional UTF-8 file path to write the report or JSON output.",
    )
    parser.add_argument(
        "--search-provider",
        "--provider",
        dest="search_provider",
        default=None,
        help="Search provider: ddgs, tavily, zhihu, qwen, glm, or comma-separated combinations.",
    )
    parser.add_argument("--search-api-key", default=None)
    parser.add_argument("--tavily-api-key", default=None)
    parser.add_argument("--zhihu-api-key", default=None)
    parser.add_argument("--qwen-search-api-key", default=None)
    parser.add_argument("--qwen-search-model", default=None)
    parser.add_argument("--qwen-search-base-url", default=None)
    parser.add_argument("--glm-search-api-key", default=None)
    parser.add_argument("--glm-search-engine", default=None)
    parser.add_argument("--glm-search-recency-filter", default=None)
    parser.add_argument("--glm-search-content-size", default=None)
    parser.add_argument("--glm-search-domain-filter", default=None)
    parser.add_argument("--search-proxy", default=None)
    parser.add_argument("--search-max-results", type=int, default=None)
    parser.add_argument("--news-max-results", type=int, default=None)
    parser.add_argument("--search-timeout", type=float, default=None)
    parser.add_argument("--search-country", default=None)
    parser.add_argument("--search-lang", default=None)
    parser.add_argument("--search-context-sites", default=None)
    parser.add_argument(
        "--tavily-search-depth",
        choices=("basic", "advanced"),
        default=None,
    )
    parser.add_argument(
        "--no-search-cache",
        action="store_true",
        help="Disable the SQLite search cache for this retrieval run.",
    )
    parser.add_argument("--search-cache-ttl-seconds", type=int, default=None)
    parser.add_argument("--news-cache-ttl-seconds", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    project_root = _project_root()
    load_project_env(project_root)
    args = parse_args()
    context = args.context
    if args.context_file:
        file_context = _read_text_input(args.context_file)
        context = "\n\n".join(part for part in (context, file_context) if part.strip())
    if not (
        args.topic.strip()
        or context.strip()
        or args.query
        or args.image
        or args.image_url
    ):
        raise SystemExit(
            "Provide --topic, --context, --context-file, --query, --image, or --image-url."
        )

    config = _apply_overrides(MemeAgentConfig.from_env(), args)
    visual_report = describe_images_for_retrieval(
        config=config,
        topic=args.topic,
        context=context,
        image_paths=args.image,
        image_urls=args.image_url,
    )
    search_context = _build_context_with_visual_report(context, visual_report)
    search_config = build_search_config(config, project_root=project_root)
    agent = WebSearchAgent(search_config)
    news_agent = WebSearchAgent(
        replace(
            search_config,
            search_provider=_RETRIEVE_NEWS_PROVIDER,
            search_timeout=_RETRIEVE_NEWS_TIMEOUT,
        )
    )
    result = run_retrieval(
        agent=agent,
        topic=args.topic,
        context=search_context,
        direct_queries=args.query,
        mode=args.mode,
        news_agent=news_agent,
    )
    if args.json:
        payload = {
            "search_provider": agent.config.search_provider,
            "news_provider": (news_agent or agent).config.search_provider,
            "mode": args.mode,
            "visual_report": visual_report,
            **result_to_jsonable(result),
        }
        output = json.dumps(payload, ensure_ascii=False, indent=2)
    else:
        output = format_retrieval_report(
            agent,
            result,
            mode=args.mode,
            news_agent=news_agent,
            visual_report=visual_report,
        )

    if args.output:
        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
