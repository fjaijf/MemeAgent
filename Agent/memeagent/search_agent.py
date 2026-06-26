from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
import hashlib
import json
import logging
import os
from queue import Empty, Queue
import re
from threading import Lock, Thread
import time
from typing import Any
from urllib.parse import urlencode
from urllib.request import ProxyHandler, Request, build_opener, urlopen

import requests

try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

try:
    from zai import ZhipuAiClient as ZaiZhipuAiClient
except (ImportError, AttributeError):
    ZhipuAiClient = None
else:
    ZhipuAiClient = ZaiZhipuAiClient

try:
    from zhipuai import ZhipuAI
except ImportError:
    ZhipuAI = None

from .cache import SearchResultCache
from .context_fetcher import ContextFetchResult, fetch_contexts_for_results


logger = logging.getLogger(__name__)

_TAVILY_SEARCH_API = "https://api.tavily.com/search"
_ZHIHU_SEARCH_API = "https://developer.zhihu.com/api/v1/content/zhihu_search"
_ANSPIRE_SEARCH_API = "https://plugin.anspire.cn/api/ntsearch/search"
_SEARCH_PARALLEL_WORKERS = 8
_DEFAULT_CONTEXT_SITES = (
    "reddit.com",
    "x.com",
    "twitter.com",
    "weibo.com",
    "zhihu.com",
    "tieba.baidu.com",
    "bilibili.com",
    "tiktok.com",
)

_GENERIC_RELEVANCE_TERMS = {
    "and",
    "for",
    "from",
    "image",
    "meme",
    "memes",
    "news",
    "origin",
    "reaction",
    "social",
    "the",
    "with",
}


@dataclass(frozen=True)
class SearchAgentConfig:
    search_provider: str = "ddgs"
    search_api_key: str | None = None
    tavily_api_key: str | None = None
    zhihu_api_key: str | None = None
    anspire_api_key: str | None = None
    glm_search_api_key: str | None = None
    glm_search_engine: str = "search_pro"
    glm_search_recency_filter: str = "noLimit"
    glm_search_content_size: str = "medium"
    glm_search_domain_filter: str | None = None
    search_proxy: str | None = None
    context_proxy: str | None = None
    search_max_results: int = 5
    news_max_results: int = 5
    search_timeout: float = 12.0
    search_country: str = "us"
    search_lang: str = "en"
    search_context_sites: str = ",".join(_DEFAULT_CONTEXT_SITES)
    tavily_search_depth: str = "basic"
    cache_enabled: bool = True
    search_cache_path: str | None = None
    search_cache_ttl_seconds: int = 7 * 24 * 60 * 60
    news_cache_ttl_seconds: int = 6 * 60 * 60


@dataclass(frozen=True)
class PlannedSearchQuery:
    query: str
    category: str


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _format_context_results(results: list[ContextFetchResult]) -> str:
    if not results:
        return "## Thread/Page Context\nNo high-value platform URLs were fetched."

    lines = ["## Thread/Page Context"]
    readable_results = [
        item
        for item in results
        if not item.error and (item.title or item.post_text or (item.comments or []))
    ]
    empty_results = [
        item
        for item in results
        if not item.error and not (item.title or item.post_text or (item.comments or []))
    ]
    failed_results = [item for item in results if item.error]
    if not readable_results and failed_results:
        lines.append(
            "No readable public context was extracted from the selected platform URLs."
        )
        lines.append(
            "Fetch diagnostics: "
            + "; ".join(
                f"{item.source_id} {item.site}: {item.error}"
                for item in failed_results
            )
        )
        return "\n".join(lines)

    for idx, item in enumerate([*readable_results, *empty_results], start=1):
        label = f"C{idx}"
        lines.append(
            f"[{label}] Context for [{item.source_id}] ({item.site})\n"
            f"   URL: {item.url}"
        )
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
    if failed_results:
        lines.append(
            "Fetch diagnostics: "
            + "; ".join(
                f"{item.source_id} {item.site}: {item.error}"
                for item in failed_results
            )
        )
    return "\n".join(lines)


def _with_provider_metadata(
    results: list[dict[str, Any]],
    provider: str,
) -> list[dict[str, Any]]:
    return [{**item, "provider": provider} for item in results]


def _compact_query_text(value: str, max_chars: int = 220) -> str:
    return " ".join(value.split())[:max_chars].strip()


def _dedupe_strings(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = _compact_query_text(value, max_chars=160)
        key = cleaned.lower()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        deduped.append(cleaned)
    return deduped


def _strip_query_noise(value: str) -> str:
    value = _compact_query_text(value, max_chars=120)
    value = value.strip(" \"'`")
    uncertain_markers = "uncertain|\\u4e0d\\u786e|\\u7591\\u4f3c|\\u53ef\\u80fd"
    value = re.sub(rf"\s+\((?:{uncertain_markers})\)\s*$", "", value, flags=re.I)
    value = re.sub(rf"\s*[-:]\s*(?:{uncertain_markers})\s*$", "", value, flags=re.I)
    return value.strip(" -:")


def _is_low_confidence_anchor(value: str) -> bool:
    lowered = value.lower()
    return any(
        marker in lowered
        for marker in (
            "uncertain",
            "possibly",
            "maybe",
            "not sure",
            "\u4e0d\u786e\u5b9a",
            "\u7591\u4f3c",
            "\u53ef\u80fd\u662f",
        )
    )


def _is_useful_query(value: str) -> bool:
    if not value or _is_none_query(value) or _is_low_confidence_anchor(value):
        return False
    if len(value) < 2:
        return False
    if value.lower().startswith("site:"):
        return len(value) <= 180
    if value.startswith('"') and value.endswith('"'):
        return len(value) <= 180
    words = re.findall(r"[A-Za-z0-9_]+", value)
    if len(words) > 12:
        return False
    return True


def _query_terms(value: str) -> set[str]:
    value = value.lower()
    terms = {
        term
        for term in re.findall(r"[a-z0-9][a-z0-9_-]{2,}", value)
        if term not in _GENERIC_RELEVANCE_TERMS
    }
    terms.update(re.findall(r"[\u4e00-\u9fff]{2,}", value))
    return terms


def _contains_word(value: str, word: str) -> bool:
    if not word:
        return False
    return bool(re.search(rf"(?<!\w){re.escape(word)}(?!\w)", value, flags=re.I))


def _is_none_query(value: str) -> bool:
    normalized = value.strip().strip("-*").strip().lower()
    return normalized in {
        "none",
        "n/a",
        "na",
        "\u65e0",
        "\u6ca1\u6709",
        "\u65e0\u9700",
        "\u65e0\u8865\u5145",
    }


def _normalize_section_heading(value: str) -> str:
    heading = re.sub(r"^#+\s*", "", value.strip())
    heading = re.sub(r"^\d+[.)\u3001]\s*", "", heading)
    return heading.rstrip(":\uff1a").strip().lower()


def _quote_query(value: str) -> str:
    value = _strip_query_noise(value).replace('"', " ")
    value = _compact_query_text(value, max_chars=140)
    return f'"{value}"' if value else ""


def _is_site_query(value: str) -> bool:
    return value.strip().lower().startswith("site:")


def _with_news_intent(value: str) -> str:
    query = _compact_query_text(value, max_chars=180)
    lowered = query.lower()
    if any(term in lowered for term in ("news", "controversy", "\u65b0\u95fb", "\u4e89\u8bae")):
        return query
    return f"{query} news" if query else "news"


def _strip_list_marker(value: str) -> str:
    return re.sub(r"^(\d+[.)]\s*|[-*]\s+)", "", value).strip()


def _clean_visual_query(value: str) -> str:
    value = _strip_query_noise(value)
    value = value.replace('"', " ")
    value = value.replace("*", " ")
    value = re.sub(r"\b(?:meme|memes|template|templates)\b", " ", value, flags=re.I)
    value = re.sub(r"\bchinese\s+text\b", " ", value, flags=re.I)
    value = re.sub(r"\btext\b", " ", value, flags=re.I)
    value = re.sub(r"(?:表情包|梗图|模板)", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip(" -:：")


def _is_markdown_section_heading(value: str) -> bool:
    line = value.strip()
    if not line:
        return False
    if re.match(r"^(\d+[.)]\s*)?\*{0,2}[^:：]{2,120}\*{0,2}\s*[:：]\s*$", line):
        return True
    return bool(re.match(r"^[A-Z_ ]+[:：]\s*$", line))


def _url_host(value: str) -> str:
    lowered = value.lower()
    lowered = re.sub(r"^https?://", "", lowered)
    lowered = lowered.split("/", 1)[0]
    return lowered.removeprefix("www.")


def _parse_context_sites(value: str) -> list[str]:
    sites = []
    seen = set()
    for raw_site in value.split(","):
        site = raw_site.strip().lower()
        site = re.sub(r"^https?://", "", site).split("/", 1)[0]
        site = site.removeprefix("www.")
        if not site or site in seen:
            continue
        seen.add(site)
        sites.append(site)
    return sites


class WebSearchAgent:
    """Small retrieval agent that gathers public web and news context."""

    def __init__(self, config: SearchAgentConfig) -> None:
        self.config = config
        self._cache = (
            SearchResultCache(config.search_cache_path)
            if config.cache_enabled and config.search_cache_path
            else None
        )
        self._cache_stats_lock = Lock()
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_stores = 0

    def _reset_cache_stats(self) -> None:
        with self._cache_stats_lock:
            self._cache_hits = 0
            self._cache_misses = 0
            self._cache_stores = 0

    def _record_cache_stat(self, field: str) -> None:
        with self._cache_stats_lock:
            if field == "hit":
                self._cache_hits += 1
            elif field == "miss":
                self._cache_misses += 1
            elif field == "store":
                self._cache_stores += 1

    def _format_cache_stats(self) -> str:
        if not self._cache:
            return "Search cache: disabled"

        with self._cache_stats_lock:
            return (
                "Search cache: enabled "
                f"(hits={self._cache_hits}, misses={self._cache_misses}, "
                f"stored={self._cache_stores})"
            )

    def _extract_supplemental_queries(
        self,
        context: str,
        section_name: str,
        limit: int,
    ) -> list[str]:
        queries: list[str] = []
        in_section = False

        for raw_line in context.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            heading = _normalize_section_heading(line)
            if heading == section_name.lower():
                in_section = True
                continue

            if in_section and re.match(r"^[A-Z_ ]+[:：]$", line):
                in_section = False
                continue

            if in_section and re.match(r"^(\d+\.|[-*]\s+)", line):
                query = re.sub(r"^(\d+\.\s*|[-*]\s+)", "", line).strip()
                if query and not _is_none_query(query):
                    queries.append(query)

        return _dedupe_strings(queries)[:limit]

    def _extract_visual_search_anchors(self, context: str) -> list[str]:
        anchors: list[str] = []
        anchors.extend(re.findall(r'"([^"]{2,120})"', context))

        anchors.extend(self._extract_visual_suggested_queries(context, limit=8))

        cleaned = [_strip_query_noise(anchor) for anchor in anchors]
        return _dedupe_strings([anchor for anchor in cleaned if _is_useful_query(anchor)])[:8]

    def _extract_visual_suggested_queries(self, context: str, limit: int) -> list[str]:
        queries: list[str] = []
        in_suggested_queries = False

        for raw_line in context.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            lower_line = line.lower()
            if (
                "suggested retrieval queries" in lower_line
                or "suggested search queries" in lower_line
            ):
                in_suggested_queries = True
                continue

            if not in_suggested_queries:
                continue

            if re.match(r"^(\d+[.)]\s*|[-*]\s+)", line):
                query = _strip_list_marker(line).strip()
                if query and not _is_none_query(query):
                    queries.append(query)
                continue

            if _is_markdown_section_heading(line):
                in_suggested_queries = False

        cleaned = [_clean_visual_query(query) for query in queries]
        return _dedupe_strings([query for query in cleaned if _is_useful_query(query)])[:limit]

    def _build_relevance_terms(
        self,
        topic: str,
        context: str,
        queries: list[str],
    ) -> set[str]:
        terms: set[str] = set()
        for value in [topic, *self._extract_visual_search_anchors(context), *queries]:
            terms.update(_query_terms(value))
        return terms

    def _add_planned_query(
        self,
        plan: list[PlannedSearchQuery],
        seen: set[str],
        query: str,
        category: str,
    ) -> None:
        query = _compact_query_text(query, max_chars=180)
        key = query.lower()
        if not _is_useful_query(query) or key in seen:
            return
        seen.add(key)
        plan.append(PlannedSearchQuery(query=query, category=category))

    def _build_query_plan(self, topic: str, context: str = "") -> list[PlannedSearchQuery]:
        topic = _strip_query_noise(_clean_text(topic))
        visual_suggested_queries = self._extract_visual_suggested_queries(
            context,
            limit=6,
        )
        anchors = self._extract_visual_search_anchors(context)
        supplemental_web_queries = self._extract_supplemental_queries(
            context,
            "supplemental_web_queries",
            limit=3,
        )
        supplemental_web_queries = [
            _strip_query_noise(query) for query in supplemental_web_queries
        ]
        context_sites = _parse_context_sites(self.config.search_context_sites)

        plan: list[PlannedSearchQuery] = []
        seen: set[str] = set()
        if _is_useful_query(topic):
            self._add_planned_query(plan, seen, topic, "topic")

        for query in visual_suggested_queries:
            self._add_planned_query(plan, seen, query, "visual_suggested")

        for anchor in anchors[:5]:
            quoted_anchor = _quote_query(anchor)
            if quoted_anchor:
                self._add_planned_query(plan, seen, quoted_anchor, "exact_anchor")
            self._add_planned_query(plan, seen, anchor, "anchor_context")
            if topic and not _contains_word(anchor, topic):
                self._add_planned_query(plan, seen, f"{topic} {anchor}", "topic_anchor")

        for anchor in anchors[:3]:
            if anchor in visual_suggested_queries:
                continue
            quoted_anchor = _quote_query(anchor)
            if not quoted_anchor:
                continue
            for site in context_sites[:8]:
                self._add_planned_query(
                    plan,
                    seen,
                    f"site:{site} {quoted_anchor}",
                    "site_context",
                )

        for query in supplemental_web_queries:
            self._add_planned_query(plan, seen, query, "supplemental")

        if not plan:
            self._add_planned_query(plan, seen, "meme", "fallback")

        return plan[:18]

    def _build_queries(self, topic: str, context: str = "") -> list[str]:
        return [item.query for item in self._build_query_plan(topic, context)]

    def _build_news_queries(self, queries: list[str], context: str = "") -> list[str]:
        anchors = self._extract_visual_search_anchors(context)
        supplemental_news_queries = self._extract_supplemental_queries(
            context,
            "supplemental_news_queries",
            limit=2,
        )
        seed_queries = [query for query in queries if not _is_site_query(query)]
        news_queries: list[str] = []
        for anchor in anchors[:4] or seed_queries[:4]:
            anchor = anchor.strip('"')
            news_queries.append(f"{anchor} news")
            news_queries.append(f"{anchor} controversy")
        news_queries.extend(supplemental_news_queries)
        return _dedupe_strings([query for query in news_queries if _is_useful_query(query)])[:6]

    def _open_url(self, req: Request, data: bytes | None = None):
        if self.config.search_proxy:
            opener = build_opener(
                ProxyHandler(
                    {
                        "http": self.config.search_proxy,
                        "https": self.config.search_proxy,
                    }
                )
            )
            return opener.open(req, data=data, timeout=self.config.search_timeout)

        return urlopen(req, data=data, timeout=self.config.search_timeout)

    def _search_providers(self) -> list[str]:
        providers = [
            provider.strip().lower()
            for provider in self.config.search_provider.split(",")
            if provider.strip()
        ]
        return providers or ["ddgs"]

    def _search_text(self, query: str) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        errors: list[str] = []

        for provider in self._search_providers():
            try:
                results.extend(self._search_text_provider(provider, query))
            except Exception as exc:
                logger.debug(
                    "Web search provider %s failed for query=%s: %s",
                    provider,
                    query,
                    exc,
                )
                errors.append(f"{provider}: {exc}")

        if not results and errors:
            raise ValueError("; ".join(errors))
        return results

    def _search_text_provider(
        self,
        provider: str,
        query: str,
    ) -> list[dict[str, Any]]:
        return self._cached_provider_search(
            kind="web",
            provider=provider,
            query=query,
            search_fn=lambda: _with_provider_metadata(
                self._search_text_provider_uncached(provider, query),
                provider,
            ),
        )

    def _search_text_provider_uncached(
        self,
        provider: str,
        query: str,
    ) -> list[dict[str, Any]]:
        if provider == "ddgs":
            return self._search_ddgs_text(query)
        if provider == "tavily":
            return self._search_tavily(
                query,
                max_results=self.config.search_max_results,
                topic="general",
            )
        if provider == "zhihu":
            return self._search_zhihu(
                query,
                max_results=self.config.search_max_results,
            )
        if provider == "anspire":
            return self._search_anspire(
                query,
                max_results=self.config.search_max_results,
            )
        if provider in {"glm", "zai", "zhipu"}:
            return self._search_glm(
                query,
                max_results=self.config.search_max_results,
            )
        raise ValueError(
            f"Unsupported search provider '{provider}'. "
            "Use ddgs, tavily, zhihu, anspire, or glm."
        )

    def _search_news(self, query: str) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        errors: list[str] = []

        for provider in self._search_providers():
            try:
                results.extend(self._search_news_provider(provider, query))
            except Exception as exc:
                logger.debug(
                    "News search provider %s failed for query=%s: %s",
                    provider,
                    query,
                    exc,
                )
                errors.append(f"{provider}: {exc}")

        if not results and errors:
            raise ValueError("; ".join(errors))
        return results

    def _search_news_provider(
        self,
        provider: str,
        query: str,
    ) -> list[dict[str, Any]]:
        return self._cached_provider_search(
            kind="news",
            provider=provider,
            query=query,
            search_fn=lambda: _with_provider_metadata(
                self._search_news_provider_uncached(provider, query),
                provider,
            ),
        )

    def _search_news_provider_uncached(
        self,
        provider: str,
        query: str,
    ) -> list[dict[str, Any]]:
        if provider == "ddgs":
            return self._search_ddgs_news(query)
        if provider == "tavily":
            return self._search_tavily(
                query,
                max_results=self.config.news_max_results,
                topic="news",
            )
        if provider == "zhihu":
            return []
        if provider == "anspire":
            return self._search_anspire(
                _with_news_intent(query),
                max_results=self.config.news_max_results,
            )
        if provider in {"glm", "zai", "zhipu"}:
            return self._search_glm(
                query,
                max_results=self.config.news_max_results,
                news=True,
            )
        raise ValueError(
            f"Unsupported search provider '{provider}'. "
            "Use ddgs, tavily, zhihu, anspire, or glm."
        )

    def _cached_provider_search(
        self,
        kind: str,
        provider: str,
        query: str,
        search_fn,
    ) -> list[dict[str, Any]]:
        if not self._cache:
            return search_fn()

        cache_key = self._build_cache_key(kind=kind, provider=provider, query=query)
        cached = self._cache.get(cache_key)
        if cached is not None:
            self._record_cache_stat("hit")
            logger.debug(
                "Search cache hit for kind=%s provider=%s query=%s",
                kind,
                provider,
                query,
            )
            return _with_provider_metadata(cached.value, provider)

        self._record_cache_stat("miss")
        results = search_fn()
        ttl_seconds = (
            self.config.news_cache_ttl_seconds
            if kind == "news"
            else self.config.search_cache_ttl_seconds
        )
        self._cache.set(cache_key, results, ttl_seconds=ttl_seconds)
        self._record_cache_stat("store")
        logger.debug(
            "Search cache stored for kind=%s provider=%s query=%s",
            kind,
            provider,
            query,
        )
        return results

    def _build_cache_key(self, kind: str, provider: str, query: str) -> str:
        payload = {
            "version": 3,
            "kind": kind,
            "provider": provider,
            "query": query,
            "search_max_results": self.config.search_max_results,
            "news_max_results": self.config.news_max_results,
            "search_country": self.config.search_country,
            "search_lang": self.config.search_lang,
            "search_context_sites": self.config.search_context_sites,
            "tavily_search_depth": self.config.tavily_search_depth,
            "glm_search_engine": self.config.glm_search_engine,
            "glm_search_recency_filter": self.config.glm_search_recency_filter,
            "glm_search_content_size": self.config.glm_search_content_size,
            "glm_search_domain_filter": self.config.glm_search_domain_filter,
        }
        raw_key = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    def _search_ddgs_text(self, query: str) -> list[dict[str, Any]]:
        with self._open_ddgs() as ddgs:
            return list(ddgs.text(query, max_results=self.config.search_max_results))

    def _search_ddgs_news(self, query: str) -> list[dict[str, Any]]:
        with self._open_ddgs() as ddgs:
            return list(ddgs.news(query, max_results=self.config.news_max_results))

    def _open_ddgs(self):
        candidates: list[dict[str, Any]] = []
        if self.config.search_proxy:
            candidates.extend(
                [
                    {
                        "proxy": self.config.search_proxy,
                        "timeout": self.config.search_timeout,
                    },
                    {
                        "proxies": self.config.search_proxy,
                        "timeout": self.config.search_timeout,
                    },
                    {"proxy": self.config.search_proxy},
                    {"proxies": self.config.search_proxy},
                ]
            )
        candidates.extend([{"timeout": self.config.search_timeout}, {}])

        for kwargs in candidates:
            try:
                return DDGS(**kwargs)
            except TypeError:
                continue

        return DDGS()

    def _search_tavily(
        self,
        query: str,
        max_results: int,
        topic: str,
    ) -> list[dict[str, Any]]:
        api_key = (
            self.config.tavily_api_key
            or os.getenv("MEMEAGENT_TAVILY_API_KEY")
            or os.getenv("TAVILY_API_KEY")
            or self.config.search_api_key
        )
        if not api_key:
            raise ValueError(
                "MEMEAGENT_TAVILY_API_KEY, TAVILY_API_KEY, or "
                "MEMEAGENT_SEARCH_API_KEY is required for Tavily search"
            )

        data = json.dumps(
            {
                "query": query,
                "max_results": max_results,
                "topic": topic,
                "search_depth": self.config.tavily_search_depth,
                "include_answer": False,
                "include_raw_content": False,
            }
        ).encode("utf-8")
        req = Request(
            _TAVILY_SEARCH_API,
            data=data,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with self._open_url(req) as resp:
            payload = json.loads(resp.read())

        raw_results = payload.get("results") or []
        return [
            {
                "title": item.get("title"),
                "body": item.get("content"),
                "href": item.get("url"),
                "source": item.get("source"),
                "date": item.get("published_date"),
            }
            for item in raw_results
            if isinstance(item, dict)
        ]

    def _search_anspire(
        self,
        query: str,
        max_results: int,
    ) -> list[dict[str, Any]]:
        api_key = (
            self.config.anspire_api_key
            or self.config.search_api_key
            or os.getenv("MEMEAGENT_ANSPIRE_API_KEY")
            or os.getenv("ANSPIRE_API_KEY")
        )
        if not api_key:
            raise ValueError(
                "MEMEAGENT_ANSPIRE_API_KEY, ANSPIRE_API_KEY, or "
                "MEMEAGENT_SEARCH_API_KEY is required for Anspire search"
            )

        qs = urlencode({"query": query, "top_k": max_results})
        req = Request(
            f"{_ANSPIRE_SEARCH_API}?{qs}",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json",
            },
        )
        with self._open_url(req) as resp:
            payload = json.loads(resp.read())

        return self._normalize_anspire_payload(payload, max_results=max_results)

    def _normalize_anspire_payload(
        self,
        payload: Any,
        max_results: int,
    ) -> list[dict[str, Any]]:
        payload = self._object_to_plain_data(payload)
        raw_results = self._extract_result_list(payload)
        return [
            self._normalize_vendor_search_item(item, default_source="Anspire Search")
            for item in raw_results[:max_results]
            if isinstance(item, dict)
        ]

    def _search_glm(
        self,
        query: str,
        max_results: int,
        news: bool = False,
    ) -> list[dict[str, Any]]:
        if ZhipuAiClient is None:
            if ZhipuAI is None:
                raise ValueError(
                    "zhipuai or zai is required for GLM search. "
                    "Install it with: pip install zhipuai"
                )

        api_key = (
            self.config.glm_search_api_key
            or self.config.search_api_key
            or os.getenv("MEMEAGENT_GLM_API_KEY")
            or os.getenv("MEMEAGENT_ZAI_API_KEY")
            or os.getenv("ZAI_API_KEY")
            or os.getenv("GLM_API_KEY")
            or os.getenv("ZHIPUAI_API_KEY")
        )
        if not api_key:
            raise ValueError(
                "ZAI_API_KEY, GLM_API_KEY, ZHIPUAI_API_KEY, "
                "MEMEAGENT_GLM_API_KEY, MEMEAGENT_GLM_SEARCH_API_KEY, or "
                "MEMEAGENT_SEARCH_API_KEY is required for GLM search"
            )

        client = (
            ZhipuAiClient(api_key=api_key)
            if ZhipuAiClient is not None
            else ZhipuAI(api_key=api_key)
        )
        kwargs: dict[str, Any] = {
            "search_engine": self.config.glm_search_engine,
            "search_query": query if not news else _with_news_intent(query),
            "count": max_results,
            "search_recency_filter": self.config.glm_search_recency_filter,
            "content_size": self.config.glm_search_content_size,
        }
        if self.config.glm_search_domain_filter:
            kwargs["search_domain_filter"] = self.config.glm_search_domain_filter

        response = client.web_search.web_search(**kwargs)
        return self._normalize_glm_payload(response, max_results=max_results)

    def _normalize_glm_payload(self, payload: Any, max_results: int) -> list[dict[str, Any]]:
        payload = self._object_to_plain_data(payload)
        raw_results = self._extract_result_list(payload)
        return [
            self._normalize_vendor_search_item(item, default_source="GLM Search")
            for item in raw_results[:max_results]
            if isinstance(item, dict)
        ]

    def _search_zhihu(self, query: str, max_results: int) -> list[dict[str, Any]]:
        api_key = self.config.zhihu_api_key or self.config.search_api_key
        if not api_key:
            raise ValueError(
                "MEMEAGENT_ZHIHU_API_KEY or MEMEAGENT_SEARCH_API_KEY "
                "is required for Zhihu search"
            )

        qs = urlencode({"Query": query})
        req = Request(
            f"{_ZHIHU_SEARCH_API}?{qs}",
            headers={
                "Authorization": f"Bearer {api_key}",
                "X-Request-Timestamp": str(int(time.time())),
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        with self._open_url(req) as resp:
            payload = json.loads(resp.read())

        raw_results = self._extract_result_list(payload)
        return [
            self._normalize_zhihu_item(item)
            for item in raw_results[:max_results]
            if isinstance(item, dict)
        ]

    def _extract_result_list(self, payload: Any) -> list[Any]:
        payload = self._object_to_plain_data(payload)
        if isinstance(payload, list):
            return payload
        if not isinstance(payload, dict):
            return []

        for key in (
            "data",
            "Data",
            "results",
            "Results",
            "items",
            "Items",
            "list",
            "List",
            "contents",
            "Contents",
            "search_result",
            "search_results",
            "SearchResult",
            "SearchResults",
            "records",
            "Records",
        ):
            value = payload.get(key)
            if isinstance(value, list):
                return value
            if isinstance(value, dict):
                nested = self._extract_result_list(value)
                if nested:
                    return nested

        content_keys = {
            "title",
            "Title",
            "url",
            "Url",
            "content",
            "Content",
            "content_text",
            "ContentText",
        }
        return [payload] if any(key in payload for key in content_keys) else []

    def _object_to_plain_data(self, value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, list):
            return [self._object_to_plain_data(item) for item in value]
        if isinstance(value, tuple):
            return [self._object_to_plain_data(item) for item in value]
        if isinstance(value, dict):
            return {key: self._object_to_plain_data(item) for key, item in value.items()}
        pydantic_data = self._pydantic_model_to_plain_data(value)
        if pydantic_data is not None:
            return pydantic_data
        for method_name in ("model_dump", "dict", "to_dict"):
            method = getattr(value, method_name, None)
            if callable(method):
                try:
                    return self._object_to_plain_data(method())
                except TypeError:
                    continue
        if hasattr(value, "__dict__"):
            return {
                key: self._object_to_plain_data(item)
                for key, item in vars(value).items()
                if isinstance(key, str) and not key.startswith("_")
            }
        return value

    def _pydantic_model_to_plain_data(self, value: Any) -> dict[str, Any] | None:
        model_cls = type(value)
        fields = getattr(model_cls, "model_fields", None)
        if fields is None:
            fields = getattr(model_cls, "__pydantic_fields__", None)
        if fields is None:
            fields = getattr(model_cls, "__fields__", None)
        if not isinstance(fields, Mapping):
            return None

        data: dict[str, Any] = {}
        for field_name in fields:
            if not isinstance(field_name, str):
                continue
            try:
                field_value = getattr(value, field_name)
            except AttributeError:
                continue
            data[field_name] = self._object_to_plain_data(field_value)

        for extra_attr in ("model_extra", "__pydantic_extra__"):
            try:
                extra = getattr(value, extra_attr, None)
            except Exception:
                continue
            if not isinstance(extra, Mapping):
                continue
            for key, item in extra.items():
                if isinstance(key, str) and key not in data:
                    data[key] = self._object_to_plain_data(item)

        return data

    def _normalize_vendor_search_item(
        self,
        item: dict[str, Any],
        default_source: str,
    ) -> dict[str, Any]:
        title = _clean_text(
            item.get("title")
            or item.get("Title")
            or item.get("name")
            or item.get("Name")
            or item.get("site_name")
            or item.get("SiteName")
        )
        body = _clean_text(
            item.get("content")
            or item.get("Content")
            or item.get("summary")
            or item.get("Summary")
            or item.get("snippet")
            or item.get("Snippet")
            or item.get("description")
            or item.get("Description")
            or item.get("body")
            or item.get("Body")
            or item.get("text")
            or item.get("Text")
        )
        href = _clean_text(
            item.get("url")
            or item.get("Url")
            or item.get("href")
            or item.get("Href")
            or item.get("link")
            or item.get("Link")
            or item.get("source_url")
            or item.get("SourceUrl")
        )
        source = _clean_text(
            item.get("source")
            or item.get("Source")
            or item.get("media")
            or item.get("Media")
            or item.get("site")
            or item.get("Site")
        )
        date = _clean_text(
            item.get("date")
            or item.get("Date")
            or item.get("publish_time")
            or item.get("PublishTime")
            or item.get("published_time")
            or item.get("PublishedTime")
            or item.get("published_date")
            or item.get("PublishedDate")
        )
        return {
            "title": title or href or f"{default_source} result",
            "body": body,
            "href": href,
            "source": source or default_source,
            "date": date,
        }

    def _normalize_zhihu_item(self, item: dict[str, Any]) -> dict[str, Any]:
        title = _clean_text(
            item.get("title")
            or item.get("Title")
            or item.get("question_title")
            or item.get("QuestionTitle")
            or item.get("name")
            or item.get("Name")
            or item.get("headline")
            or item.get("Headline")
        )
        body = _clean_text(
            item.get("summary")
            or item.get("Summary")
            or item.get("excerpt")
            or item.get("Excerpt")
            or item.get("snippet")
            or item.get("Snippet")
            or item.get("content")
            or item.get("Content")
            or item.get("content_text")
            or item.get("ContentText")
            or item.get("description")
            or item.get("Description")
        )
        href = _clean_text(
            item.get("url")
            or item.get("Url")
            or item.get("link")
            or item.get("Link")
            or item.get("target_url")
            or item.get("TargetUrl")
            or item.get("web_url")
            or item.get("WebUrl")
        )
        date = _clean_text(
            item.get("created_time")
            or item.get("CreatedTime")
            or item.get("updated_time")
            or item.get("UpdatedTime")
            or item.get("published_time")
            or item.get("PublishedTime")
            or item.get("edit_time")
            or item.get("EditTime")
        )
        author = _clean_text(item.get("author_name") or item.get("AuthorName"))
        content_type = _clean_text(item.get("content_type") or item.get("ContentType"))
        content_id = _clean_text(item.get("content_id") or item.get("ContentID"))

        if not body:
            body = _clean_text(json.dumps(item, ensure_ascii=False))[:500]

        return {
            "title": title or "Zhihu result",
            "body": body,
            "href": href,
            "source": f"Zhihu / {author}" if author else "Zhihu",
            "date": date,
            "content_type": content_type,
            "content_id": content_id,
        }

    def _search_text_queries(
        self,
        query_input: list[str] | tuple[list[str], set[str]],
    ) -> list[dict[str, Any]]:
        if isinstance(query_input, tuple):
            queries, relevance_terms = query_input
        else:
            queries = query_input
            relevance_terms = set()

        results, errors = self._search_queries_in_parallel(
            queries=queries,
            search_fn=self._search_text_provider,
            relevance_terms=relevance_terms,
            max_results=self.config.search_max_results * len(self._search_providers()),
            kind="web",
        )
        if not results and errors:
            raise ValueError("; ".join(errors))
        return results

    def _search_news_queries(
        self,
        query_input: list[str] | tuple[list[str], str],
    ) -> list[dict[str, Any]]:
        context = ""
        if isinstance(query_input, tuple):
            queries, context = query_input
        else:
            queries = query_input

        news_queries = self._build_news_queries(queries, context=context)[:6]
        relevance_terms = set().union(*(_query_terms(query) for query in news_queries))
        results, errors = self._search_queries_in_parallel(
            queries=news_queries,
            search_fn=self._search_news_provider,
            relevance_terms=relevance_terms,
            max_results=self.config.news_max_results,
            kind="news",
        )
        if not results and errors:
            raise ValueError("; ".join(errors))
        return results

    def _search_queries_in_parallel(
        self,
        queries: list[str],
        search_fn,
        relevance_terms: set[str],
        max_results: int,
        kind: str,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        providers = self._search_providers()
        tasks: list[tuple[int, str, str]] = [
            (query_index * len(providers) + provider_index, provider, query)
            for query_index, query in enumerate(queries)
            for provider_index, provider in enumerate(providers)
        ]
        if not tasks:
            return [], []

        results: list[dict[str, Any]] = []
        errors: list[str] = []
        deadline = time.monotonic() + self.config.search_timeout

        def run_task(provider: str, query: str) -> list[dict[str, Any]]:
            return search_fn(provider, query)

        executor = ThreadPoolExecutor(
            max_workers=min(_SEARCH_PARALLEL_WORKERS, len(tasks)),
            thread_name_prefix=f"memeagent-{kind}-search",
        )
        futures = {
            executor.submit(run_task, provider, query): (order, provider, query)
            for order, provider, query in tasks
        }
        pending = set(futures)

        try:
            while pending:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break

                done, pending = wait(
                    pending,
                    timeout=remaining,
                    return_when=FIRST_COMPLETED,
                )
                if not done:
                    break

                for future in done:
                    order, provider, query = futures[future]
                    try:
                        task_results = future.result()
                    except Exception as exc:
                        logger.debug(
                            "%s search provider %s failed for query=%s: %s",
                            kind,
                            provider,
                            query,
                            exc,
                        )
                        errors.append(f"{query} / {provider}: {exc}")
                        continue

                    for item in task_results:
                        results.append(item)
        finally:
            if pending:
                errors.append(
                    f"{kind.capitalize()} search timed out with {len(pending)} task(s) unfinished"
                )
                for future in pending:
                    future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)

        return self._dedupe_results(
            results,
            max_results=max_results,
            relevance_terms=relevance_terms,
        ), errors

    def _dedupe_results(
        self,
        results: list[dict[str, Any]],
        max_results: int,
        relevance_terms: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        relevance_terms = relevance_terms or set()
        scored: list[tuple[int, int, dict[str, Any]]] = []
        fallback: list[tuple[int, dict[str, Any]]] = []
        seen: set[str] = set()

        for order, item in enumerate(results):
            href = _clean_text(item.get("href") or item.get("url"))
            title = _clean_text(item.get("title")).lower()
            key = href or title
            if not key or key in seen:
                continue

            seen.add(key)

            score = self._result_relevance_score(item, relevance_terms)
            if score > 0 or not relevance_terms:
                scored.append((score, order, item))
            else:
                fallback.append((order, item))

        scored.sort(key=lambda row: (-row[0], row[1]))
        ranked = [item for _, _, item in scored]
        if not ranked:
            ranked = [item for _, item in fallback]

        return ranked[:max_results]

    def _result_relevance_score(
        self,
        item: dict[str, Any],
        relevance_terms: set[str],
    ) -> int:
        if not relevance_terms:
            return 1

        title = _clean_text(item.get("title")).lower()
        body = _clean_text(item.get("body")).lower()
        href = _clean_text(item.get("href") or item.get("url")).lower()
        haystack = f"{title}\n{body}\n{href}"

        score = 0
        for term in relevance_terms:
            if term in title:
                score += 3
            elif term in body:
                score += 2
            elif term in href:
                score += 1
        return score

    def _classify_result_type(self, item: dict[str, Any], news: bool = False) -> str:
        if news:
            return "news_background"

        href = _clean_text(item.get("href") or item.get("url"))
        title = _clean_text(item.get("title")).lower()
        body = _clean_text(item.get("body")).lower()
        host = _url_host(href)
        haystack = f"{title}\n{body}\n{href.lower()}"

        if "reddit.com" in host:
            if "/comments/" in href.lower() or "/r/" in href.lower():
                return "post_or_comment_context_candidate"
            return "platform_context_candidate"
        if "x.com" in host or "twitter.com" in host:
            if "/status/" in href.lower():
                return "social_post_candidate"
            return "platform_context_candidate"
        if "weibo.com" in host:
            return "social_post_candidate"
        if "tieba.baidu.com" in host:
            return "discussion_thread_candidate"
        if "zhihu.com" in host:
            return "discussion_context_candidate"
        if any(site in host for site in ("bilibili.com", "tiktok.com", "douyin.com")):
            return "platform_context_candidate"
        if "knowyourmeme.com" in host or "meme" in host and "wiki" in host:
            return "meme_template_reference"
        if any(term in haystack for term in ("origin", "template", "meme explained")):
            return "meme_template_reference"
        if any(term in haystack for term in ("comment", "thread", "discussion", "reply")):
            return "comment_context_candidate"
        return "background_or_related_result"

    def _format_query_plan(self, query_plan: list[PlannedSearchQuery]) -> str:
        lines = ["Search query plan:"]
        for item in query_plan:
            lines.append(f"- [{item.category}] {item.query}")
        return "\n".join(lines)

    def _run_with_timeout(
        self,
        fn,
        query,
        label: str,
    ) -> tuple[list[dict[str, Any]], str | None]:
        output: Queue[tuple[list[dict[str, Any]], Exception | None]] = Queue(maxsize=1)

        def target() -> None:
            try:
                output.put((fn(query), None), block=False)
            except Exception as exc:
                output.put(([], exc), block=False)

        thread = Thread(target=target, daemon=True)
        thread.start()
        thread.join(timeout=self.config.search_timeout)
        if thread.is_alive():
            return [], f"{label} timed out after {self.config.search_timeout:.0f}s"

        try:
            results, exc = output.get_nowait()
        except Empty:
            return [], f"{label} failed without returning results"
        if exc is not None:
            logger.debug("%s failed for query=%s: %s", label, query, exc)
            return [], f"{label} failed: {exc}"
        return results, None

    def run(self, topic: str, context: str = "") -> str:
        self._reset_cache_stats()
        query_plan = self._build_query_plan(topic, context)
        queries = [item.query for item in query_plan]
        relevance_terms = self._build_relevance_terms(topic, context, queries)

        sections: list[str] = [
            f"Search provider: {self.config.search_provider}",
            self._format_query_plan(query_plan),
        ]

        text_results, text_error = self._run_with_timeout(
            self._search_text_queries,
            (queries, relevance_terms),
            "Web search",
        )
        if text_error:
            sections.append(text_error)

        news_results, news_error = self._run_with_timeout(
            self._search_news_queries,
            (queries, context),
            "News search",
        )
        if news_error:
            sections.append(news_error)

        sections.insert(1, self._format_cache_stats())

        if text_results:
            web_lines = ["## Web Search Results"]
            for idx, item in enumerate(text_results, start=1):
                source_id = f"W{idx}"
                title = _clean_text(item.get("title"))
                body = _clean_text(item.get("body"))
                href = _clean_text(item.get("href") or item.get("url"))
                provider = _clean_text(item.get("provider"))
                result_type = self._classify_result_type(item)
                web_lines.append(
                    f"[{source_id}] {title}\n"
                    f"   Provider: {provider or 'N/A'}\n"
                    f"   Candidate type: {result_type}\n"
                    f"   Snippet: {body or 'N/A'}\n"
                    f"   URL: {href or 'N/A'}"
                )
            sections.append("\n".join(web_lines))
            context_results = fetch_contexts_for_results(
                text_results,
                context_sites=self.config.search_context_sites,
                fallback_proxy=self.config.context_proxy,
                timeout=min(self.config.search_timeout, 8.0),
                total_timeout=max(self.config.search_timeout, 12.0),
            )
            sections.append(_format_context_results(context_results))

        if news_results:
            news_lines = ["## News Results"]
            for idx, item in enumerate(news_results, start=1):
                source_id = f"N{idx}"
                title = _clean_text(item.get("title"))
                body = _clean_text(item.get("body"))
                source = _clean_text(item.get("source"))
                date = _clean_text(item.get("date"))
                href = _clean_text(item.get("href") or item.get("url"))
                provider = _clean_text(item.get("provider"))
                result_type = self._classify_result_type(item, news=True)
                news_lines.append(
                    f"[{source_id}] {title}\n"
                    f"   Provider: {provider or 'N/A'}\n"
                    f"   Candidate type: {result_type}\n"
                    f"   Source: {source or 'N/A'} | Date: {date or 'N/A'}\n"
                    f"   Snippet: {body or 'N/A'}\n"
                    f"   URL: {href or 'N/A'}"
                )
            sections.append("\n".join(news_lines))

        if len(sections) == 2:
            sections.append("No web or news results found.")

        return "\n\n".join(sections)
