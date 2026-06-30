from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import threading
import time
from typing import Any, Iterable


def _now_iso(timestamp: float | None = None) -> str:
    value = time.time() if timestamp is None else timestamp
    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return str(value)


def _normalize_role(role: Any) -> str:
    normalized = str(role or "user")
    if normalized == "human":
        return "user"
    if normalized == "ai":
        return "assistant"
    return normalized


def _message_role(message: Any) -> str:
    return _normalize_role(getattr(message, "type", None) or getattr(message, "role", "user"))


def _message_content(message: Any) -> Any:
    return getattr(message, "content", message)


def _sanitize_image_ref(image_ref: Any) -> dict[str, Any]:
    value = str(image_ref or "")
    if value.startswith("data:"):
        match = re.match(r"data:([^;,]+)?;base64,(.*)", value, flags=re.S)
        if not match:
            return {
                "source": "data_url",
                "redacted": True,
                "sha256": hashlib.sha256(value.encode("utf-8")).hexdigest(),
                "encoded_length": len(value),
            }
        mime_type = match.group(1) or "application/octet-stream"
        encoded = match.group(2)
        digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        byte_length: int | None = None
        try:
            byte_length = len(base64.b64decode(encoded, validate=False))
        except Exception:
            byte_length = None
        return {
            "source": "data_url",
            "mime_type": mime_type,
            "redacted": True,
            "sha256": digest,
            "encoded_length": len(encoded),
            "byte_length": byte_length,
        }
    if value.startswith(("http://", "https://")):
        return {"source": "url", "url": value}
    if value:
        return {"source": "path_or_uri", "value": value}
    return {"source": "empty", "value": ""}


def _serialize_content(content: Any) -> Any:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return _json_safe(content)

    blocks: list[Any] = []
    for item in content:
        if not isinstance(item, dict):
            blocks.append(_json_safe(item))
            continue

        item_type = item.get("type")
        if item_type == "image_url" or "image_url" in item:
            image_url = item.get("image_url")
            url = image_url.get("url") if isinstance(image_url, dict) else image_url
            blocks.append(
                {
                    "type": "image_url",
                    "image_url": _sanitize_image_ref(url),
                }
            )
            continue
        if item_type == "image" or "image" in item:
            blocks.append(
                {
                    "type": "image",
                    "image": _sanitize_image_ref(item.get("image")),
                }
            )
            continue
        blocks.append(_json_safe(item))
    return blocks


def _serialize_messages(messages: Iterable[Any]) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for message in messages:
        content = _message_content(message)
        serialized.append(
            {
                "role": _message_role(message),
                "content": _serialize_content(content),
                "text_char_count": len(_text_from_content(content)),
            }
        )
    return serialized


def _text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and (item.get("type") == "text" or "text" in item):
                parts.append(str(item.get("text", "")))
        return "\n".join(part for part in parts if part)
    return str(content)


def _messages_text(messages: Iterable[Any]) -> str:
    return "\n\n".join(_text_from_content(_message_content(message)) for message in messages)


def _infer_call_kind(messages: Iterable[Any]) -> str:
    text = _messages_text(messages)
    lowered = text.lower()
    if "search-oriented visual description" in lowered:
        return "image_preanalysis"
    if "plan a small set of supplemental searches" in lowered:
        return "retrieval_planning"
    if "controller model for memeagent" in lowered:
        return "controller_iteration_planning"
    if "re-examine the attached meme image" in lowered:
        return "controller_image_followup"
    if "retrieval reflection planner" in lowered:
        return "retrieval_reflection"
    if "researcher-oriented meme analysis" in lowered:
        return "final_analysis"
    if "counterfactual reasoning reviewer" in lowered:
        return "harmfulness_counterfactual"
    if '"harmful_probability"' in text or "perspective vote summary" in lowered:
        return "harmfulness_perspective"
    if "task instruction:" in lowered:
        return "task_head"
    return "unknown"


def _extract_json_object(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _find_first_key(value: Any, keys: set[str]) -> Any:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key) in keys:
                return item
        for item in value.values():
            found = _find_first_key(item, keys)
            if found is not None:
                return found
    if isinstance(value, list):
        for item in value:
            found = _find_first_key(item, keys)
            if found is not None:
                return found
    return None


def _section_value(text: str, section: str) -> str:
    pattern = re.compile(
        rf"(?ims)^\s*{re.escape(section)}\s*:\s*\n(?P<body>.*?)(?=^\s*[A-Z][A-Z0-9_ ]{{2,}}\s*:\s*$|\Z)"
    )
    match = pattern.search(text)
    if not match:
        inline = re.search(rf"(?im)^\s*{re.escape(section)}\s*:\s*(.+)$", text)
        return inline.group(1).strip() if inline else ""
    return match.group("body").strip()


def _trim(value: Any, limit: int = 2000) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def _extract_visible_reasoning_summary(output: str) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    parsed = _extract_json_object(output)
    if parsed:
        for key in ("reasoning_summary", "summary", "rationale", "reason"):
            value = _find_first_key(parsed, {key})
            if value:
                summary[key] = _trim(value)

    for section in (
        "ITERATION_CONFIDENCE",
        "SHOULD_FINALIZE",
        "CONFIDENCE_REASON",
        "RETRIEVAL_SCORE",
        "SHOULD_CONTINUE",
        "STOP_REASON",
        "KEY_FINDINGS_SO_FAR",
        "EVIDENCE_GAPS",
        "FINAL_OUTPUT_NOTES",
    ):
        value = _section_value(output, section)
        if value:
            summary[section.lower()] = _trim(value)
    return summary


def _response_content(response: Any) -> str:
    return str(getattr(response, "content", response) or "")


class LLMTraceRecorder:
    """Writes visible LLM inputs and outputs to a structured JSON trace file."""

    def __init__(
        self,
        path: str | Path,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        now = time.time()
        self._lock = threading.Lock()
        self._data: dict[str, Any] = {
            "schema_version": 1,
            "created_at": now,
            "created_at_iso": _now_iso(now),
            "trace_path": str(self.path),
            "note": (
                "This file records visible prompts, visible model outputs, timings, "
                "errors, and concise reasoning summaries explicitly present in output. "
                "Hidden chain-of-thought is not available from most model APIs and is "
                "not requested by MemeAgent."
            ),
            "metadata": metadata or {},
            "calls": [],
        }
        self.flush()

    def record_start(
        self,
        *,
        llm_name: str,
        operation: str,
        messages: list[Any],
        model: str,
        llm_class: str,
    ) -> int:
        now = time.time()
        with self._lock:
            calls = self._data["calls"]
            call_index = len(calls) + 1
            calls.append(
                {
                    "call_index": call_index,
                    "llm_name": llm_name,
                    "operation": operation,
                    "call_kind": _infer_call_kind(messages),
                    "model": model,
                    "llm_class": llm_class,
                    "status": "running",
                    "started_at": now,
                    "started_at_iso": _now_iso(now),
                    "finished_at": None,
                    "finished_at_iso": None,
                    "duration_seconds": None,
                    "messages": _serialize_messages(messages),
                    "output": "",
                    "visible_reasoning_summary": {},
                    "stream_chunk_count": 0,
                    "error": "",
                }
            )
            self._flush_locked()
        return call_index

    def record_finish(
        self,
        call_index: int,
        *,
        output: str = "",
        status: str = "completed",
        error: str = "",
        stream_chunk_count: int = 0,
    ) -> None:
        now = time.time()
        with self._lock:
            call = self._data["calls"][call_index - 1]
            call["status"] = status
            call["finished_at"] = now
            call["finished_at_iso"] = _now_iso(now)
            call["duration_seconds"] = round(now - float(call["started_at"]), 6)
            call["output"] = output
            call["visible_reasoning_summary"] = (
                _extract_visible_reasoning_summary(output) if output else {}
            )
            call["stream_chunk_count"] = stream_chunk_count
            call["error"] = error
            self._flush_locked()

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def _flush_locked(self) -> None:
        tmp_path = self.path.with_name(f"{self.path.name}.tmp")
        tmp_path.write_text(_json_dumps(self._data) + "\n", encoding="utf-8")
        tmp_path.replace(self.path)


class TracedLLM:
    """Delegates to an LLM while recording each invoke/stream call."""

    def __init__(
        self,
        llm: Any,
        *,
        recorder: LLMTraceRecorder,
        name: str,
    ) -> None:
        self._llm = llm
        self._recorder = recorder
        self._name = name

    def __getattr__(self, name: str) -> Any:
        return getattr(self._llm, name)

    def invoke(self, messages: list[Any]) -> Any:
        call_index = self._recorder.record_start(
            llm_name=self._name,
            operation="invoke",
            messages=messages,
            model=self._model_name(),
            llm_class=type(self._llm).__name__,
        )
        try:
            response = self._llm.invoke(messages)
        except Exception as exc:
            self._recorder.record_finish(
                call_index,
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        self._recorder.record_finish(call_index, output=_response_content(response))
        return response

    def stream(self, messages: list[Any]):
        if not hasattr(self._llm, "stream"):
            response = self.invoke(messages)
            yield _response_content(response)
            return

        call_index = self._recorder.record_start(
            llm_name=self._name,
            operation="stream",
            messages=messages,
            model=self._model_name(),
            llm_class=type(self._llm).__name__,
        )
        chunks: list[str] = []
        try:
            for chunk in self._llm.stream(messages):
                text = str(chunk)
                chunks.append(text)
                yield chunk
        except Exception as exc:
            self._recorder.record_finish(
                call_index,
                output="".join(chunks),
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
                stream_chunk_count=len(chunks),
            )
            raise
        self._recorder.record_finish(
            call_index,
            output="".join(chunks),
            stream_chunk_count=len(chunks),
        )

    def _model_name(self) -> str:
        for attr in ("model", "model_path"):
            value = getattr(self._llm, attr, None)
            if value:
                return str(value)
        return type(self._llm).__name__
