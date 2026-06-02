from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
import json
import logging
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


@dataclass(frozen=True)
class SearchAgentConfig:
    search_provider: str = "ddgs"
    search_api_key: str | None = None
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


class WebSearchAgent:
    """Small retrieval agent that gathers public web and news context."""

    def __init__(self, config: SearchAgentConfig) -> None:
        self.config = config

    def _build_query(self, topic: str, context: str = "") -> str:
        topic = _clean_text(topic)
        context = _clean_text(context)
        if topic and context:
            return f"{topic} {context} meme coin crypto community sentiment"
        if topic:
            return f"{topic} meme coin crypto community sentiment"
        if context:
            return context
        return "crypto meme coin sentiment"

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

    def _search_text(self, query: str) -> list[dict[str, Any]]:
        provider = self.config.search_provider.lower()
        if provider == "ddgs":
            return self._search_ddgs_text(query)
        if provider == "brave":
            return self._search_brave(query, max_results=self.config.search_max_results)
        if provider == "tavily":
            return self._search_tavily(
                query,
                max_results=self.config.search_max_results,
                topic="general",
            )
        raise ValueError(
            f"Unsupported search provider '{self.config.search_provider}'. "
            "Use ddgs, brave, or tavily."
        )

    def _search_news(self, query: str) -> list[dict[str, Any]]:
        provider = self.config.search_provider.lower()
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
        raise ValueError(
            f"Unsupported search provider '{self.config.search_provider}'. "
            "Use ddgs, brave, or tavily."
        )

    def _search_ddgs_text(self, query: str) -> list[dict[str, Any]]:
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=self.config.search_max_results))

    def _search_ddgs_news(self, query: str) -> list[dict[str, Any]]:
        with DDGS() as ddgs:
            return list(ddgs.news(query, max_results=self.config.news_max_results))

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

    def _run_with_timeout(self, fn, query: str, label: str) -> tuple[list[dict[str, Any]], str | None]:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(fn, query)
            try:
                return future.result(timeout=self.config.search_timeout), None
            except FuturesTimeoutError:
                future.cancel()
                return [], f"{label} timed out after {self.config.search_timeout:.0f}s"
            except Exception as exc:
                logger.warning("%s failed for query=%s: %s", label, query, exc)
                return [], f"{label} failed: {exc}"

    def run(self, topic: str, context: str = "") -> str:
        query = self._build_query(topic, context)

        sections: list[str] = [
            f"Search provider: {self.config.search_provider}",
            f"Search query: {query}",
        ]

        text_results, text_error = self._run_with_timeout(
            self._search_text, query, "Web search"
        )
        if text_error:
            sections.append(text_error)

        news_results, news_error = self._run_with_timeout(
            self._search_news, query, "News search"
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
