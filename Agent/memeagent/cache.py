from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
import time
from typing import Any


@dataclass(frozen=True)
class CacheEntry:
    value: list[dict[str, Any]]
    expires_at: float


class SearchResultCache:
    """Tiny SQLite cache for provider search results."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=5)

    def _init_db(self) -> None:
        with closing(self._connect()) as conn:
            with conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS search_cache (
                        cache_key TEXT PRIMARY KEY,
                        value_json TEXT NOT NULL,
                        created_at REAL NOT NULL,
                        expires_at REAL NOT NULL
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_search_cache_expires_at "
                    "ON search_cache(expires_at)"
                )

    def get(self, cache_key: str) -> CacheEntry | None:
        now = time.time()
        with closing(self._connect()) as conn:
            with conn:
                row = conn.execute(
                    "SELECT value_json, expires_at FROM search_cache WHERE cache_key = ?",
                    (cache_key,),
                ).fetchone()
                if not row:
                    return None

                value_json, expires_at = row
                if expires_at <= now:
                    conn.execute(
                        "DELETE FROM search_cache WHERE cache_key = ?",
                        (cache_key,),
                    )
                    return None

        try:
            value = json.loads(value_json)
        except json.JSONDecodeError:
            self.delete(cache_key)
            return None

        if not isinstance(value, list):
            self.delete(cache_key)
            return None

        return CacheEntry(value=value, expires_at=float(expires_at))

    def set(
        self,
        cache_key: str,
        value: list[dict[str, Any]],
        ttl_seconds: int,
    ) -> None:
        now = time.time()
        expires_at = now + max(0, ttl_seconds)
        value_json = json.dumps(value, ensure_ascii=False, sort_keys=True)
        with closing(self._connect()) as conn:
            with conn:
                conn.execute(
                    """
                    INSERT INTO search_cache(cache_key, value_json, created_at, expires_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(cache_key) DO UPDATE SET
                        value_json = excluded.value_json,
                        created_at = excluded.created_at,
                        expires_at = excluded.expires_at
                    """,
                    (cache_key, value_json, now, expires_at),
                )

    def delete(self, cache_key: str) -> None:
        with closing(self._connect()) as conn:
            with conn:
                conn.execute("DELETE FROM search_cache WHERE cache_key = ?", (cache_key,))

    def delete_expired(self) -> int:
        now = time.time()
        with closing(self._connect()) as conn:
            with conn:
                cursor = conn.execute(
                    "DELETE FROM search_cache WHERE expires_at <= ?",
                    (now,),
                )
                return cursor.rowcount
