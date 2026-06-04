from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
import json
import logging
import time
from typing import Any
from urllib.parse import urlencode
from urllib.request import ProxyHandler, Request, build_opener, urlopen

try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS


logger = logging.getLogger(__name__)

_BRAVE_WEB_API = "https://api.search.brave.com/res/v1/web/search"
_BRAVE_NEWS_API = "https://api.search.brave.com/res/v1/news/search"
_TAVILY_SEARCH_API = "https://api.tavily.com/search"
_ZHIHU_SEARCH_API = "https://developer.zhihu.com/api/v1/content/zhihu_search"

_RESEARCH_QUERY_SUFFIXES = (
    "meme harmfulness sentiment audience intent evolution discourse analysis",
    "meme risk emotional reaction audience interpretation intent",
    "meme controversy social impact reception meaning transformation",
)


@dataclass(frozen=True)
class SearchAgentConfig:
    search_provider: str = "ddgs"
    search_api_key: str | None = None
    zhihu_api_key: str | None = None
    search_proxy: str | None = None
    search_max_results: int = 5
    news_max_results: int = 5
    search_timeout: float = 12.0
    search_country: str = "us"
    search_lang: str = "en"
    tavily_search_depth: str = "basic"


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _compact_query_text(value: str, max_chars: int = 220) -> str:
    return " ".join(value.split())[:max_chars].strip()


class WebSearchAgent:
    """Small retrieval agent that gathers public web and news context."""

    def __init__(self, config: SearchAgentConfig) -> None:
        self.config = config

    def _build_queries(self, topic: str, context: str = "") -> list[str]:
        topic = _compact_query_text(_clean_text(topic), max_chars=120)
        context = _compact_query_text(_clean_text(context), max_chars=220)
        base = " ".join(part for part in (topic, context) if part) or "meme"

        queries = [
            f"{base} {_RESEARCH_QUERY_SUFFIXES[0]}",
            f"{base} {_RESEARCH_QUERY_SUFFIXES[1]}",
            f"{base} {_RESEARCH_QUERY_SUFFIXES[2]}",
        ]
        return list(dict.fromkeys(query for query in queries if query.strip()))

    def _build_news_queries(self, queries: list[str]) -> list[str]:
        bases: list[str] = []
        for query in queries:
            base = _clean_text(query)
            for suffix in _RESEARCH_QUERY_SUFFIXES:
                marker = f" {suffix}"
                if base.endswith(marker):
                    base = base[: -len(marker)].strip()
                    break
            if base:
                bases.append(base)

        base = bases[0] if bases else "meme"
        news_queries = [
            f"{base} meme news",
            f"{base} meme controversy",
            f"{base} social media reaction",
        ]
        return list(dict.fromkeys(query for query in news_queries if query.strip()))

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
        if provider == "ddgs":
            return self._search_ddgs_text(query)
        if provider == "brave":
            return self._search_brave(
                query,
                max_results=self.config.search_max_results,
            )
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
        raise ValueError(
            f"Unsupported search provider '{provider}'. "
            "Use ddgs, brave, tavily, or zhihu."
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
        if provider == "ddgs":
            return self._search_ddgs_news(query)
        if provider == "brave":
            return self._search_brave(
                query,
                max_results=self.config.news_max_results,
                news=True,
            )
        if provider == "tavily":
            return self._search_tavily(
                query,
                max_results=self.config.news_max_results,
                topic="news",
            )
        if provider == "zhihu":
            return []
        raise ValueError(
            f"Unsupported search provider '{provider}'. "
            "Use ddgs, brave, tavily, or zhihu."
        )

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

    def _search_brave(
        self,
        query: str,
        max_results: int,
        news: bool = False,
    ) -> list[dict[str, Any]]:
        if not self.config.search_api_key:
            raise ValueError("MEMEAGENT_SEARCH_API_KEY is required for Brave search")

        qs = urlencode(
            {
                "q": query,
                "count": max_results,
                "country": self.config.search_country,
                "search_lang": self.config.search_lang,
            }
        )
        endpoint = _BRAVE_NEWS_API if news else _BRAVE_WEB_API
        req = Request(
            f"{endpoint}?{qs}",
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": self.config.search_api_key,
            },
        )
        with self._open_url(req) as resp:
            payload = json.loads(resp.read())

        raw_results = (
            (payload.get("web") or {}).get("results")
            or (payload.get("news") or {}).get("results")
            or payload.get("results")
            or []
        )
        return [
            {
                "title": item.get("title"),
                "body": item.get("description") or item.get("snippet"),
                "href": item.get("url"),
                "source": item.get("source"),
                "date": item.get("age") or item.get("page_age"),
            }
            for item in raw_results
            if isinstance(item, dict)
        ]

    def _search_tavily(
        self,
        query: str,
        max_results: int,
        topic: str,
    ) -> list[dict[str, Any]]:
        if not self.config.search_api_key:
            raise ValueError("MEMEAGENT_SEARCH_API_KEY is required for Tavily search")

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
                "Authorization": f"Bearer {self.config.search_api_key}",
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
        if isinstance(payload, list):
            return payload
        if not isinstance(payload, dict):
            return []

        for key in ("data", "results", "items", "list", "contents"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
            if isinstance(value, dict):
                nested = self._extract_result_list(value)
                if nested:
                    return nested

        return [payload] if any(key in payload for key in ("title", "url", "content")) else []

    def _normalize_zhihu_item(self, item: dict[str, Any]) -> dict[str, Any]:
        title = _clean_text(
            item.get("title")
            or item.get("question_title")
            or item.get("name")
            or item.get("headline")
        )
        body = _clean_text(
            item.get("summary")
            or item.get("excerpt")
            or item.get("snippet")
            or item.get("content")
            or item.get("description")
        )
        href = _clean_text(
            item.get("url")
            or item.get("link")
            or item.get("target_url")
            or item.get("web_url")
        )
        date = _clean_text(
            item.get("created_time")
            or item.get("updated_time")
            or item.get("published_time")
        )

        if not body:
            body = _clean_text(json.dumps(item, ensure_ascii=False))[:500]

        return {
            "title": title or "Zhihu result",
            "body": body,
            "href": href,
            "source": "Zhihu",
            "date": date,
        }

    def _search_text_queries(self, queries: list[str]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        errors: list[str] = []

        for query in queries:
            try:
                results.extend(self._search_text(query))
            except Exception as exc:
                logger.debug("Web search failed for query=%s: %s", query, exc)
                errors.append(f"{query}: {exc}")

        if not results and errors:
            raise ValueError("; ".join(errors))
        return self._dedupe_results(
            results,
            max_results=self.config.search_max_results * len(self._search_providers()),
        )

    def _search_news_queries(self, queries: list[str]) -> list[dict[str, Any]]:
        news_queries = self._build_news_queries(queries)[:2]
        results: list[dict[str, Any]] = []
        errors: list[str] = []

        for query in news_queries:
            try:
                results.extend(self._search_news(query))
            except Exception as exc:
                logger.debug("News search failed for query=%s: %s", query, exc)
                errors.append(f"{query}: {exc}")

        if not results and errors:
            raise ValueError("; ".join(errors))
        return self._dedupe_results(results, max_results=self.config.news_max_results)

    def _dedupe_results(
        self,
        results: list[dict[str, Any]],
        max_results: int,
    ) -> list[dict[str, Any]]:
        deduped: list[dict[str, Any]] = []
        seen: set[str] = set()

        for item in results:
            href = _clean_text(item.get("href") or item.get("url"))
            title = _clean_text(item.get("title")).lower()
            key = href or title
            if not key or key in seen:
                continue

            seen.add(key)
            deduped.append(item)
            if len(deduped) >= max_results:
                break

        return deduped

    def _run_with_timeout(
        self,
        fn,
        query,
        label: str,
    ) -> tuple[list[dict[str, Any]], str | None]:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(fn, query)
            try:
                return future.result(timeout=self.config.search_timeout), None
            except FuturesTimeoutError:
                future.cancel()
                return [], f"{label} timed out after {self.config.search_timeout:.0f}s"
            except Exception as exc:
                logger.debug("%s failed for query=%s: %s", label, query, exc)
                return [], f"{label} failed: {exc}"

    def run(self, topic: str, context: str = "") -> str:
        queries = self._build_queries(topic, context)

        sections: list[str] = [
            f"Search provider: {self.config.search_provider}",
            "Search queries:\n" + "\n".join(f"- {query}" for query in queries),
        ]

        text_results, text_error = self._run_with_timeout(
            self._search_text_queries,
            queries,
            "Web search",
        )
        if text_error:
            sections.append(text_error)

        news_results, news_error = self._run_with_timeout(
            self._search_news_queries,
            queries,
            "News search",
        )
        if news_error:
            sections.append(news_error)

        if text_results:
            web_lines = ["## Web Search Results"]
            for idx, item in enumerate(text_results, start=1):
                title = _clean_text(item.get("title"))
                body = _clean_text(item.get("body"))
                href = _clean_text(item.get("href") or item.get("url"))
                web_lines.append(
                    f"{idx}. {title}\n"
                    f"   Snippet: {body or 'N/A'}\n"
                    f"   URL: {href or 'N/A'}"
                )
            sections.append("\n".join(web_lines))

        if news_results:
            news_lines = ["## News Results"]
            for idx, item in enumerate(news_results, start=1):
                title = _clean_text(item.get("title"))
                body = _clean_text(item.get("body"))
                source = _clean_text(item.get("source"))
                date = _clean_text(item.get("date"))
                href = _clean_text(item.get("href") or item.get("url"))
                news_lines.append(
                    f"{idx}. {title}\n"
                    f"   Source: {source or 'N/A'} | Date: {date or 'N/A'}\n"
                    f"   Snippet: {body or 'N/A'}\n"
                    f"   URL: {href or 'N/A'}"
                )
            sections.append("\n".join(news_lines))

        if len(sections) == 2:
            sections.append("No web or news results found.")

        return "\n\n".join(sections)
