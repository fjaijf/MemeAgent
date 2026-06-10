from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import time
from typing import Any


def _normalize_topic(topic: str) -> str:
    return " ".join(topic.lower().split()).strip()


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _hash_file(path: str) -> str:
    file_path = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_memory_keys(topic: str, image_paths: list[str], image_urls: list[str]) -> tuple[str, str]:
    topic_key = _hash_text(_normalize_topic(topic)) if topic.strip() else ""
    image_parts: list[str] = []

    for image_path in image_paths:
        try:
            image_parts.append(f"file:{_hash_file(image_path)}")
        except OSError:
            image_parts.append(f"path:{_hash_text(image_path)}")

    for image_url in image_urls:
        image_parts.append(f"url:{_hash_text(image_url.strip())}")

    image_key = _hash_text("\n".join(sorted(image_parts))) if image_parts else ""
    return topic_key, image_key


@dataclass(frozen=True)
class MemoryRecord:
    topic: str
    input_mode: str
    analysis: str
    visual_report: str
    retrieval_plan: str
    search_report: str
    created_at: float


@dataclass(frozen=True)
class MemoryCard:
    topic: str
    artifact_count: int
    object_context: str
    visual_traits: str
    sentiment_notes: str
    harmfulness_notes: str
    intent_notes: str
    evolution_notes: str
    evidence_gaps: str
    updated_at: float


class MemeMemoryStore:
    """Persistent exact-match memory for previous MemeAgent analyses."""

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
                    CREATE TABLE IF NOT EXISTS meme_memory (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        topic_key TEXT NOT NULL,
                        image_key TEXT NOT NULL,
                        topic TEXT NOT NULL,
                        input_mode TEXT NOT NULL,
                        analysis TEXT NOT NULL,
                        visual_report TEXT NOT NULL,
                        retrieval_plan TEXT NOT NULL,
                        search_report TEXT NOT NULL,
                        metadata_json TEXT NOT NULL,
                        created_at REAL NOT NULL
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_meme_memory_topic_key "
                    "ON meme_memory(topic_key, created_at)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_meme_memory_image_key "
                    "ON meme_memory(image_key, created_at)"
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS meme_memory_cards (
                        topic_key TEXT PRIMARY KEY,
                        topic TEXT NOT NULL,
                        artifact_count INTEGER NOT NULL,
                        object_context TEXT NOT NULL,
                        visual_traits TEXT NOT NULL,
                        sentiment_notes TEXT NOT NULL,
                        harmfulness_notes TEXT NOT NULL,
                        intent_notes TEXT NOT NULL,
                        evolution_notes TEXT NOT NULL,
                        evidence_gaps TEXT NOT NULL,
                        updated_at REAL NOT NULL
                    )
                    """
                )

    def remember(
        self,
        *,
        topic: str,
        image_paths: list[str],
        image_urls: list[str],
        input_mode: str,
        analysis: str,
        visual_report: str = "",
        retrieval_plan: str = "",
        search_report: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not analysis.strip():
            return

        topic_key, image_key = build_memory_keys(topic, image_paths, image_urls)
        if not topic_key and not image_key:
            return

        with closing(self._connect()) as conn:
            with conn:
                conn.execute(
                    """
                    INSERT INTO meme_memory(
                        topic_key,
                        image_key,
                        topic,
                        input_mode,
                        analysis,
                        visual_report,
                        retrieval_plan,
                        search_report,
                        metadata_json,
                        created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        topic_key,
                        image_key,
                        topic,
                        input_mode,
                        analysis,
                        visual_report,
                        retrieval_plan,
                        search_report,
                        json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True),
                        time.time(),
                    ),
                )
                if topic_key:
                    self._upsert_memory_card(
                        conn=conn,
                        topic_key=topic_key,
                        topic=topic,
                        analysis=analysis,
                    )

    def recall(
        self,
        *,
        topic: str,
        image_paths: list[str],
        image_urls: list[str],
        limit: int = 3,
    ) -> list[MemoryRecord]:
        topic_key, image_key = build_memory_keys(topic, image_paths, image_urls)
        clauses: list[str] = []
        params: list[Any] = []

        if topic_key:
            clauses.append("topic_key = ?")
            params.append(topic_key)
        if image_key:
            clauses.append("image_key = ?")
            params.append(image_key)

        if not clauses:
            return []

        params.append(limit)
        query = f"""
            SELECT topic, input_mode, analysis, visual_report, retrieval_plan,
                   search_report, created_at
            FROM meme_memory
            WHERE {' OR '.join(clauses)}
            ORDER BY created_at DESC
            LIMIT ?
        """
        with closing(self._connect()) as conn:
            rows = conn.execute(query, params).fetchall()

        return [
            MemoryRecord(
                topic=row[0],
                input_mode=row[1],
                analysis=row[2],
                visual_report=row[3],
                retrieval_plan=row[4],
                search_report=row[5],
                created_at=float(row[6]),
            )
            for row in rows
        ]

    def recall_card(self, topic: str) -> MemoryCard | None:
        topic_key = _hash_text(_normalize_topic(topic)) if topic.strip() else ""
        if not topic_key:
            return None

        with closing(self._connect()) as conn:
            row = conn.execute(
                """
                SELECT topic, artifact_count, object_context, visual_traits,
                       sentiment_notes, harmfulness_notes, intent_notes,
                       evolution_notes, evidence_gaps, updated_at
                FROM meme_memory_cards
                WHERE topic_key = ?
                """,
                (topic_key,),
            ).fetchone()

        if not row:
            return None

        return MemoryCard(
            topic=row[0],
            artifact_count=int(row[1]),
            object_context=row[2],
            visual_traits=row[3],
            sentiment_notes=row[4],
            harmfulness_notes=row[5],
            intent_notes=row[6],
            evolution_notes=row[7],
            evidence_gaps=row[8],
            updated_at=float(row[9]),
        )

    def format_records(
        self,
        records: list[MemoryRecord],
        card: MemoryCard | None = None,
        max_chars: int = 1800,
    ) -> str:
        if not records and not card:
            return ""

        sections = [
            (
                "Local MemeAgent memory. Treat these as prior analysis notes, "
                "not authoritative external evidence."
            )
        ]
        if card:
            sections.append(self._format_card(card))

        for idx, record in enumerate(records, start=1):
            analysis = " ".join(record.analysis.split())
            if len(analysis) > max_chars:
                analysis = analysis[:max_chars].rstrip() + "..."
            sections.append(
                f"[Memory {idx}] Topic: {record.topic or 'N/A'} | "
                f"Input mode: {record.input_mode}\n"
                f"Prior analysis summary:\n{analysis}"
            )
        return "\n\n".join(sections)

    def _upsert_memory_card(
        self,
        *,
        conn: sqlite3.Connection,
        topic_key: str,
        topic: str,
        analysis: str,
    ) -> None:
        now = time.time()
        extracted = self._extract_card_fields(analysis)
        row = conn.execute(
            """
            SELECT artifact_count, object_context, visual_traits, sentiment_notes,
                   harmfulness_notes, intent_notes, evolution_notes, evidence_gaps
            FROM meme_memory_cards
            WHERE topic_key = ?
            """,
            (topic_key,),
        ).fetchone()

        if not row:
            conn.execute(
                """
                INSERT INTO meme_memory_cards(
                    topic_key,
                    topic,
                    artifact_count,
                    object_context,
                    visual_traits,
                    sentiment_notes,
                    harmfulness_notes,
                    intent_notes,
                    evolution_notes,
                    evidence_gaps,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    topic_key,
                    topic,
                    1,
                    extracted["object_context"],
                    extracted["visual_traits"],
                    extracted["sentiment_notes"],
                    extracted["harmfulness_notes"],
                    extracted["intent_notes"],
                    extracted["evolution_notes"],
                    extracted["evidence_gaps"],
                    now,
                ),
            )
            return

        conn.execute(
            """
            UPDATE meme_memory_cards
            SET topic = ?,
                artifact_count = ?,
                object_context = ?,
                visual_traits = ?,
                sentiment_notes = ?,
                harmfulness_notes = ?,
                intent_notes = ?,
                evolution_notes = ?,
                evidence_gaps = ?,
                updated_at = ?
            WHERE topic_key = ?
            """,
            (
                topic,
                int(row[0]) + 1,
                self._merge_note(row[1], extracted["object_context"]),
                self._merge_note(row[2], extracted["visual_traits"]),
                self._merge_note(row[3], extracted["sentiment_notes"]),
                self._merge_note(row[4], extracted["harmfulness_notes"]),
                self._merge_note(row[5], extracted["intent_notes"]),
                self._merge_note(row[6], extracted["evolution_notes"]),
                self._merge_note(row[7], extracted["evidence_gaps"]),
                now,
                topic_key,
            ),
        )

    def _extract_card_fields(self, analysis: str) -> dict[str, str]:
        return {
            "object_context": self._extract_numbered_section(analysis, 1),
            "visual_traits": self._extract_numbered_section(analysis, 2),
            "sentiment_notes": self._extract_numbered_section(analysis, 3),
            "harmfulness_notes": self._extract_numbered_section(analysis, 4),
            "intent_notes": self._extract_numbered_section(analysis, 6),
            "evolution_notes": self._extract_numbered_section(analysis, 7),
            "evidence_gaps": self._extract_numbered_section(analysis, 9),
        }

    def _extract_numbered_section(self, text: str, number: int, max_chars: int = 900) -> str:
        match = re.search(
            rf"(?ms)^\s*{number}\.\s+.*?(?=^\s*\d+\.\s+|\Z)",
            text,
        )
        if not match:
            return ""

        section = " ".join(match.group(0).split())
        return section[:max_chars].rstrip()

    def _merge_note(self, previous: str, new: str, max_chars: int = 1400) -> str:
        previous = previous.strip()
        new = new.strip()
        if not new:
            return previous[:max_chars]
        if not previous:
            return new[:max_chars]
        if new in previous:
            return previous[:max_chars]

        merged = f"{new}\n---\n{previous}"
        return merged[:max_chars].rstrip()

    def _format_card(self, card: MemoryCard) -> str:
        lines = [
            f"## Topic Memory Card: {card.topic}",
            f"Observed analyses: {card.artifact_count}",
        ]
        fields = [
            ("Object/context", card.object_context),
            ("Visual traits", card.visual_traits),
            ("Sentiment notes", card.sentiment_notes),
            ("Harmfulness notes", card.harmfulness_notes),
            ("Intent notes", card.intent_notes),
            ("Evolution notes", card.evolution_notes),
            ("Evidence gaps", card.evidence_gaps),
        ]
        for label, value in fields:
            if value.strip():
                lines.append(f"{label}: {value}")
        return "\n".join(lines)
