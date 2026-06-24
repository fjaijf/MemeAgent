from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
import json
import re
import time
from typing import Any
from urllib.parse import urlsplit

import requests


_CONTEXT_FETCH_WORKERS = 6
_CONTEXT_FETCH_TIMEOUT = 8.0
_CONTEXT_TOTAL_TIMEOUT = 35.0
_MAX_PAGE_TEXT_CHARS = 2600
_MAX_REDDIT_COMMENTS = 24
_MAX_COMMENT_CHARS = 420

_CONTEXT_HOSTS = (
    "reddit.com",
    "old.reddit.com",
    "weibo.com",
    "m.weibo.cn",
    "zhihu.com",
    "tieba.baidu.com",
    "bilibili.com",
    "x.com",
    "twitter.com",
    "tiktok.com",
    "douyin.com",
)


@dataclass(frozen=True)
class ContextFetchResult:
    source_id: str
    url: str
    site: str
    title: str = ""
    post_text: str = ""
    comments: list[str] | None = None
    metadata: dict[str, str] | None = None
    error: str = ""


def _clean_text(value: Any, max_chars: int | None = None) -> str:
    text = " ".join(str(value or "").split())
    if max_chars is not None and len(text) > max_chars:
        return text[:max_chars].rstrip() + "..."
    return text


def _url_host(value: str) -> str:
    parts = urlsplit(value)
    host = (parts.netloc or "").lower().removeprefix("www.")
    return host


def _is_context_host(url: str) -> bool:
    host = _url_host(url)
    return any(host == site or host.endswith("." + site) for site in _CONTEXT_HOSTS)


def _request_headers() -> dict[str, str]:
    return {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/126.0 Safari/537.36 MemeAgent/0.1"
        ),
        "Accept": "text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.8,zh-CN;q=0.7,zh;q=0.6",
    }


class _HTMLContextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title = ""
        self.description = ""
        self._in_title = False
        self._skip_depth = 0
        self._chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        attrs_dict = {key.lower(): value or "" for key, value in attrs}
        if tag in {"script", "style", "noscript", "svg"}:
            self._skip_depth += 1
            return
        if tag == "title":
            self._in_title = True
            return
        if tag == "meta":
            name = attrs_dict.get("name", "").lower()
            prop = attrs_dict.get("property", "").lower()
            content = attrs_dict.get("content", "")
            if name == "description" or prop in {"og:description", "twitter:description"}:
                if not self.description:
                    self.description = _clean_text(content, max_chars=700)
            if prop in {"og:title", "twitter:title"} and not self.title:
                self.title = _clean_text(content, max_chars=250)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "noscript", "svg"} and self._skip_depth:
            self._skip_depth -= 1
            return
        if tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        text = _clean_text(data)
        if not text:
            return
        if self._in_title:
            self.title = _clean_text(f"{self.title} {text}", max_chars=250)
            return
        if len(text) >= 2:
            self._chunks.append(text)

    def text(self) -> str:
        joined = _clean_text(" ".join(self._chunks))
        return joined[:_MAX_PAGE_TEXT_CHARS].rstrip()


def _parse_html_context(html: str) -> tuple[str, str, str]:
    parser = _HTMLContextParser()
    parser.feed(html)
    return (
        unescape(_clean_text(parser.title, max_chars=250)),
        unescape(_clean_text(parser.description, max_chars=700)),
        unescape(parser.text()),
    )


def _reddit_json_url(url: str) -> str:
    url = url.split("?", 1)[0].rstrip("/")
    if url.endswith(".json"):
        return url
    return url + ".json"


def _reddit_comment_text(node: dict[str, Any], comments: list[str]) -> None:
    if len(comments) >= _MAX_REDDIT_COMMENTS:
        return
    data = node.get("data") if isinstance(node, dict) else {}
    if not isinstance(data, dict):
        return
    body = _clean_text(data.get("body"), max_chars=_MAX_COMMENT_CHARS)
    author = _clean_text(data.get("author"), max_chars=80)
    score = data.get("score")
    if body:
        prefix = f"{author}"
        if isinstance(score, int):
            prefix += f" | score={score}"
        comments.append(f"{prefix}: {body}" if prefix else body)
    replies = data.get("replies")
    if isinstance(replies, dict):
        children = ((replies.get("data") or {}).get("children") or [])
        for child in children:
            if isinstance(child, dict) and child.get("kind") == "t1":
                _reddit_comment_text(child, comments)


def _fetch_reddit_context(source_id: str, url: str) -> ContextFetchResult:
    response = requests.get(
        _reddit_json_url(url),
        headers=_request_headers(),
        timeout=_CONTEXT_FETCH_TIMEOUT,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list) or not payload:
        raise ValueError("Reddit JSON payload did not contain listings")

    post_listing = payload[0]
    post_children = ((post_listing.get("data") or {}).get("children") or [])
    post_data = {}
    if post_children and isinstance(post_children[0], dict):
        post_data = post_children[0].get("data") or {}

    title = _clean_text(post_data.get("title"), max_chars=250)
    selftext = _clean_text(post_data.get("selftext"), max_chars=1600)
    author = _clean_text(post_data.get("author"), max_chars=80)
    subreddit = _clean_text(post_data.get("subreddit"), max_chars=80)
    score = post_data.get("score")
    num_comments = post_data.get("num_comments")
    metadata = {
        "platform": "reddit",
        "author": author,
        "subreddit": subreddit,
        "score": str(score) if score is not None else "",
        "num_comments": str(num_comments) if num_comments is not None else "",
    }

    comments: list[str] = []
    if len(payload) > 1:
        comment_children = ((payload[1].get("data") or {}).get("children") or [])
        for child in comment_children:
            if isinstance(child, dict) and child.get("kind") == "t1":
                _reddit_comment_text(child, comments)
            if len(comments) >= _MAX_REDDIT_COMMENTS:
                break

    return ContextFetchResult(
        source_id=source_id,
        url=url,
        site="reddit",
        title=title,
        post_text=selftext,
        comments=comments,
        metadata={key: value for key, value in metadata.items() if value},
    )


def _fetch_generic_context(source_id: str, url: str) -> ContextFetchResult:
    response = requests.get(
        url,
        headers=_request_headers(),
        timeout=_CONTEXT_FETCH_TIMEOUT,
        allow_redirects=True,
    )
    response.raise_for_status()
    content_type = response.headers.get("content-type", "")
    if "json" in content_type.lower():
        text = _clean_text(json.dumps(response.json(), ensure_ascii=False), max_chars=2200)
        return ContextFetchResult(
            source_id=source_id,
            url=url,
            site=_url_host(url),
            post_text=text,
            metadata={"content_type": content_type},
        )

    response.encoding = response.encoding or response.apparent_encoding
    title, description, page_text = _parse_html_context(response.text)
    post_text = description or page_text
    if description and page_text and description not in page_text:
        post_text = _clean_text(f"{description}\n{page_text}", max_chars=_MAX_PAGE_TEXT_CHARS)

    return ContextFetchResult(
        source_id=source_id,
        url=url,
        site=_url_host(url),
        title=title,
        post_text=post_text,
        comments=[],
        metadata={"content_type": content_type},
    )


def fetch_context_for_url(source_id: str, url: str) -> ContextFetchResult:
    host = _url_host(url)
    try:
        if "reddit.com" in host and "/comments/" in url:
            return _fetch_reddit_context(source_id, url)
        return _fetch_generic_context(source_id, url)
    except Exception as exc:
        return ContextFetchResult(
            source_id=source_id,
            url=url,
            site=host,
            error=f"{type(exc).__name__}: {exc}",
        )


def fetch_contexts_for_results(
    web_results: list[dict[str, Any]],
) -> list[ContextFetchResult]:
    targets: list[tuple[str, str]] = []
    seen_urls: set[str] = set()
    for index, item in enumerate(web_results, start=1):
        url = _clean_text(item.get("href") or item.get("url"))
        if not url or url in seen_urls or not _is_context_host(url):
            continue
        seen_urls.add(url)
        targets.append((f"W{index}", url))

    if not targets:
        return []

    deadline = time.monotonic() + _CONTEXT_TOTAL_TIMEOUT
    executor = ThreadPoolExecutor(
        max_workers=min(_CONTEXT_FETCH_WORKERS, len(targets)),
        thread_name_prefix="memeagent-context",
    )
    futures = {
        executor.submit(fetch_context_for_url, source_id, url): (source_id, url)
        for source_id, url in targets
    }
    pending = set(futures)
    results: list[ContextFetchResult] = []

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
                results.append(future.result())
    finally:
        for future in pending:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)

    if pending:
        for future in pending:
            source_id, url = futures[future]
            results.append(
                ContextFetchResult(
                    source_id=source_id,
                    url=url,
                    site=_url_host(url),
                    error="Context fetch timed out before this URL completed",
                )
            )

    source_order = {source_id: index for index, (source_id, _url) in enumerate(targets)}
    return sorted(results, key=lambda item: source_order.get(item.source_id, 10**9))
