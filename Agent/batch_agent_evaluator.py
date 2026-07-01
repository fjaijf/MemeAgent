from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

from memeagent.config import MemeAgentConfig, load_project_env


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
DEFAULT_ID_FIELDS = ("id", "image_id", "uid", "sample_id", "name", "filename", "file_name")
DEFAULT_IMAGE_FIELDS = (
    "image",
    "image_path",
    "img",
    "img_path",
    "path",
    "file",
    "filename",
)
DEFAULT_TEXT_FIELDS = ("text", "caption", "ocr", "context", "description", "prompt")
DEFAULT_LABEL_FIELDS = ("label", "harmful", "is_harmful", "target", "class")

POSITIVE_LABELS = {
    "1",
    "true",
    "yes",
    "y",
    "b",
    "harmful",
    "hateful",
    "offensive",
    "toxic",
    "abusive",
    "dangerous",
    "discrimination",
    "violence",
    "vulgar",
    "antagonism",
}
NEGATIVE_LABELS = {
    "0",
    "false",
    "no",
    "n",
    "a",
    "harmless",
    "safe",
    "benign",
    "normal",
    "neutral",
    "not_harmful",
    "not harmful",
    "non_harmful",
    "non harmful",
    "nonharmful",
}

MAIN_SYSTEM_PROMPT = """You are the multimodal analysis model in a meme harmfulness evaluation pipeline.
Analyze the image and any supplied context. Focus on visible text/OCR, visual entities, target group, image-text relation, harmful stereotypes, threats, dehumanization, harassment, obscenity, antagonism, and whether the meme endorses or criticizes harmful content.
Keep evidence separate from inference. Return concise JSON when possible."""

CONTROLLER_SYSTEM_PROMPT = """You are the controller model for an iterative meme harmfulness evaluation pipeline.
You review multimodal analyses, decide whether the sample can be finalized, estimate confidence, and ask targeted follow-up questions only when they are useful.
Return only one JSON object and no markdown."""


@dataclass
class SampleState:
    sample_id: str
    image_path: Path
    context: str = ""
    label: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)
    status: str = "needs_main"
    round_index: int = 0
    next_questions: list[str] = field(default_factory=list)
    main_outputs: list[dict[str, Any]] = field(default_factory=list)
    controller_outputs: list[dict[str, Any]] = field(default_factory=list)
    final_label: str = "unclear"
    final_confidence: float | None = None
    final_reason: str = ""
    error: str = ""


@dataclass(frozen=True)
class ChatEndpoint:
    base_url: str
    api_key: str
    timeout: float
    retries: int
    strip_thinking: bool

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int | None,
    ) -> str:
        url = self.base_url.rstrip("/") + "/chat/completions"
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None and max_tokens > 0:
            payload["max_tokens"] = max_tokens

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"].get("content") or ""
                return _postprocess_content(content, strip_thinking=self.strip_thinking)
            except Exception as exc:  # noqa: BLE001 - request failures are recorded per sample.
                last_error = exc
                if attempt >= self.retries:
                    break
                time.sleep(min(2.0 * (attempt + 1), 8.0))
        raise RuntimeError(str(last_error) if last_error else "chat completion failed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run wavefront batch inference for MemeAgent harmfulness evaluation "
            "against the OpenAI-compatible vLLM router."
        )
    )
    parser.add_argument("--dataset", default="", help="JSON, JSONL, or CSV dataset file.")
    parser.add_argument(
        "--image-dir",
        default="",
        help="Directory of images. Used when --dataset is not provided.",
    )
    parser.add_argument(
        "--image",
        action="append",
        default=[],
        help="Single image path. Can be passed multiple times.",
    )
    parser.add_argument(
        "--image-root",
        default="",
        help="Base directory for relative image paths in a dataset.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output directory. Defaults to runs/batch_agent_<timestamp>.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Maximum samples to run.")
    parser.add_argument("--offset", type=int, default=0, help="Skip this many samples.")
    parser.add_argument("--base-url", default="", help="OpenAI-compatible base URL.")
    parser.add_argument("--api-key", default="", help="OpenAI-compatible API key.")
    parser.add_argument("--main-model", default="", help="Main multimodal model name.")
    parser.add_argument("--controller-model", default="", help="Controller model name.")
    parser.add_argument(
        "--main-concurrency",
        type=int,
        default=int(os.getenv("MEMEAGENT_BATCH_MAIN_CONCURRENCY", "4")),
        help="Concurrent main-model requests per wave.",
    )
    parser.add_argument(
        "--controller-concurrency",
        type=int,
        default=int(os.getenv("MEMEAGENT_BATCH_CONTROLLER_CONCURRENCY", "8")),
        help="Concurrent controller-model requests per wave.",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=int(os.getenv("MEMEAGENT_BATCH_MAX_ROUNDS", "3")),
        help="Maximum main/controller analysis rounds per sample.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=float(os.getenv("MEMEAGENT_BATCH_CONFIDENCE_THRESHOLD", "0.85")),
        help="Controller confidence threshold for finalization.",
    )
    parser.add_argument(
        "--main-max-tokens",
        type=int,
        default=int(os.getenv("MEMEAGENT_BATCH_MAIN_MAX_TOKENS", "1400")),
        help="Max output tokens for main-model calls. Use 0 to omit.",
    )
    parser.add_argument(
        "--controller-max-tokens",
        type=int,
        default=int(os.getenv("MEMEAGENT_BATCH_CONTROLLER_MAX_TOKENS", "700")),
        help="Max output tokens for controller calls. Use 0 to omit.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(os.getenv("MEMEAGENT_BATCH_TEMPERATURE", "0")),
        help="Main-model temperature.",
    )
    parser.add_argument(
        "--controller-temperature",
        type=float,
        default=float(os.getenv("MEMEAGENT_BATCH_CONTROLLER_TEMPERATURE", "0")),
        help="Controller-model temperature.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.getenv("MEMEAGENT_BATCH_TIMEOUT", "240")),
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=int(os.getenv("MEMEAGENT_BATCH_RETRIES", "1")),
        help="Retries per failed request.",
    )
    parser.add_argument(
        "--save-prompts",
        action="store_true",
        help="Store full prompt text in each batch JSONL record.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load config and samples, print the plan, then exit.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    load_project_env(project_root)
    config = MemeAgentConfig.from_env()

    base_url = args.base_url.strip() or config.base_url or ""
    if not base_url:
        raise ValueError("OPENAI_BASE_URL is not set and --base-url was not provided.")

    endpoint = ChatEndpoint(
        base_url=base_url,
        api_key=args.api_key.strip() or _openai_compatible_api_key(),
        timeout=args.timeout,
        retries=max(args.retries, 0),
        strip_thinking=_env_bool("MEMEAGENT_LOCAL_STRIP_THINKING", True),
    )
    main_model = args.main_model.strip() or config.model
    controller_model = args.controller_model.strip() or config.controller_model or main_model
    samples = _load_samples(args)
    output_dir = _resolve_output_dir(args.output_dir, project_root)

    print(f"Loaded {len(samples)} samples", flush=True)
    print(f"Output directory: {output_dir}", flush=True)
    print(f"Base URL: {endpoint.base_url}", flush=True)
    print(f"Main model: {main_model} concurrency={args.main_concurrency}", flush=True)
    print(
        f"Controller model: {controller_model} "
        f"concurrency={args.controller_concurrency}",
        flush=True,
    )
    print(
        f"Max rounds: {args.max_rounds} confidence_threshold={args.confidence_threshold}",
        flush=True,
    )

    if args.dry_run:
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(
        output_dir / "samples.jsonl",
        [_sample_manifest_record(sample) for sample in samples],
    )

    batch_index = 0
    for round_index in range(max(args.max_rounds, 1)):
        main_candidates = [sample for sample in samples if sample.status == "needs_main"]
        if main_candidates:
            print(
                f"[round {round_index + 1}] main wave: {len(main_candidates)} samples",
                flush=True,
            )
            main_records = _run_main_wave(
                samples=main_candidates,
                endpoint=endpoint,
                model=main_model,
                round_index=round_index,
                concurrency=max(args.main_concurrency, 1),
                temperature=args.temperature,
                max_tokens=_optional_positive(args.main_max_tokens),
                save_prompts=args.save_prompts,
            )
            _write_jsonl(
                output_dir / f"batch_{batch_index:03d}_main_round_{round_index + 1}.jsonl",
                main_records,
            )
            batch_index += 1

        controller_candidates = [
            sample for sample in samples if sample.status == "needs_controller"
        ]
        if controller_candidates:
            print(
                f"[round {round_index + 1}] controller wave: "
                f"{len(controller_candidates)} samples",
                flush=True,
            )
            controller_records = _run_controller_wave(
                samples=controller_candidates,
                endpoint=endpoint,
                model=controller_model,
                round_index=round_index,
                concurrency=max(args.controller_concurrency, 1),
                temperature=args.controller_temperature,
                max_tokens=_optional_positive(args.controller_max_tokens),
                confidence_threshold=args.confidence_threshold,
                is_last_round=round_index >= max(args.max_rounds, 1) - 1,
                save_prompts=args.save_prompts,
            )
            _write_jsonl(
                output_dir
                / f"batch_{batch_index:03d}_controller_round_{round_index + 1}.jsonl",
                controller_records,
            )
            batch_index += 1

        active = [sample for sample in samples if sample.status not in {"final", "error"}]
        if not active:
            break

    for sample in samples:
        if sample.status not in {"final", "error"}:
            _force_finalize(sample)

    final_records = [_final_record(sample) for sample in samples]
    _write_jsonl(output_dir / "final_predictions.jsonl", final_records)
    summary = _summary(final_records, args=args, output_dir=output_dir)
    _write_json(output_dir / "summary.json", summary)
    print(
        "Done: "
        f"final={summary['final_count']} error={summary['error_count']} "
        f"output={output_dir / 'final_predictions.jsonl'}",
        flush=True,
    )
    return 0


def _run_main_wave(
    *,
    samples: list[SampleState],
    endpoint: ChatEndpoint,
    model: str,
    round_index: int,
    concurrency: int,
    temperature: float,
    max_tokens: int | None,
    save_prompts: bool,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(
                _call_main,
                input_index=index,
                sample=sample,
                endpoint=endpoint,
                model=model,
                round_index=round_index,
                temperature=temperature,
                max_tokens=max_tokens,
                save_prompts=save_prompts,
            ): index
            for index, sample in enumerate(samples)
        }
        for future in as_completed(futures):
            records.append(future.result())
    records.sort(key=lambda item: item["input_index"])
    return records


def _run_controller_wave(
    *,
    samples: list[SampleState],
    endpoint: ChatEndpoint,
    model: str,
    round_index: int,
    concurrency: int,
    temperature: float,
    max_tokens: int | None,
    confidence_threshold: float,
    is_last_round: bool,
    save_prompts: bool,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(
                _call_controller,
                input_index=index,
                sample=sample,
                endpoint=endpoint,
                model=model,
                round_index=round_index,
                temperature=temperature,
                max_tokens=max_tokens,
                confidence_threshold=confidence_threshold,
                is_last_round=is_last_round,
                save_prompts=save_prompts,
            ): index
            for index, sample in enumerate(samples)
        }
        for future in as_completed(futures):
            records.append(future.result())
    records.sort(key=lambda item: item["input_index"])
    return records


def _call_main(
    *,
    input_index: int,
    sample: SampleState,
    endpoint: ChatEndpoint,
    model: str,
    round_index: int,
    temperature: float,
    max_tokens: int | None,
    save_prompts: bool,
) -> dict[str, Any]:
    started = time.time()
    prompt = _main_prompt(sample, round_index)
    record = _base_record(
        sample,
        role="main",
        round_index=round_index,
        input_index=input_index,
    )
    if save_prompts:
        record["prompt"] = prompt
    try:
        messages = [
            {"role": "system", "content": MAIN_SYSTEM_PROMPT},
            {"role": "user", "content": _vision_content(prompt, sample.image_path)},
        ]
        output = endpoint.chat(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        parsed = _extract_json_object(output)
        record.update(
            {
                "output": output,
                "parsed": parsed,
                "latency_seconds": round(time.time() - started, 3),
                "error": "",
            }
        )
        sample.main_outputs.append(_compact_batch_record(record))
        sample.status = "needs_controller"
    except Exception as exc:  # noqa: BLE001 - continue other samples.
        sample.status = "error"
        sample.error = str(exc)
        record.update(
            {
                "output": "",
                "parsed": {},
                "latency_seconds": round(time.time() - started, 3),
                "error": str(exc),
            }
        )
    return record


def _call_controller(
    *,
    input_index: int,
    sample: SampleState,
    endpoint: ChatEndpoint,
    model: str,
    round_index: int,
    temperature: float,
    max_tokens: int | None,
    confidence_threshold: float,
    is_last_round: bool,
    save_prompts: bool,
) -> dict[str, Any]:
    started = time.time()
    prompt = _controller_prompt(sample, confidence_threshold=confidence_threshold)
    record = _base_record(
        sample,
        role="controller",
        round_index=round_index,
        input_index=input_index,
    )
    if save_prompts:
        record["prompt"] = prompt
    try:
        messages = [
            {"role": "system", "content": CONTROLLER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        output = endpoint.chat(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        parsed = _extract_json_object(output)
        decision = _normalize_controller_decision(
            output=output,
            parsed=parsed,
            confidence_threshold=confidence_threshold,
            is_last_round=is_last_round,
        )
        record.update(
            {
                "output": output,
                "parsed": parsed,
                "decision": decision,
                "latency_seconds": round(time.time() - started, 3),
                "error": "",
            }
        )
        sample.controller_outputs.append(_compact_batch_record(record))
        _apply_controller_decision(sample, decision)
    except Exception as exc:  # noqa: BLE001 - continue other samples.
        sample.status = "error"
        sample.error = str(exc)
        record.update(
            {
                "output": "",
                "parsed": {},
                "decision": {},
                "latency_seconds": round(time.time() - started, 3),
                "error": str(exc),
            }
        )
    return record


def _main_prompt(sample: SampleState, round_index: int) -> str:
    rubric = (
        "Harmfulness labels: Discrimination, Offensive, Violence, Vulgar, "
        "Antagonism, Not harmful, Unclear."
    )
    if round_index == 0 or not sample.main_outputs:
        return f"""Analyze this meme image for harmfulness.

{rubric}

Optional context:
{sample.context or "None"}

Return a compact JSON object with these keys:
- ocr_text: visible text, or empty string
- visual_description: concrete visible entities/actions
- target: target person/group/community if any
- image_text_relation: how text and image interact
- harmfulness_cues: list of evidence-based harmfulness cues
- benign_or_counter_speech_cues: list of cues suggesting parody, criticism, or non-harmful use
- provisional_label: harmful, not_harmful, or unclear
- confidence: number from 0 to 1
- missing_information: list of uncertainties
"""

    previous = _clip(_latest_output(sample.main_outputs), 4000)
    questions = "\n".join(f"- {item}" for item in sample.next_questions) or "- Re-check the image carefully."
    return f"""Re-examine the same meme image for harmfulness.

Previous multimodal analysis:
{previous}

Controller follow-up questions:
{questions}

Optional context:
{sample.context or "None"}

Return a compact JSON object with updated evidence and these keys:
- answers_to_followup: list of direct answers
- corrected_ocr_text: visible text, or empty string
- updated_visual_evidence: concrete image evidence
- updated_harmfulness_cues: list
- updated_benign_or_counter_speech_cues: list
- provisional_label: harmful, not_harmful, or unclear
- confidence: number from 0 to 1
- remaining_uncertainties: list
"""


def _controller_prompt(sample: SampleState, *, confidence_threshold: float) -> str:
    main_history = [
        {
            "round": item.get("round"),
            "output": _clip(str(item.get("output", "")), 3500),
            "parsed": item.get("parsed", {}),
        }
        for item in sample.main_outputs[-3:]
    ]
    controller_history = [
        {
            "round": item.get("round"),
            "decision": item.get("decision", {}),
        }
        for item in sample.controller_outputs[-3:]
    ]
    payload = {
        "sample_id": sample.sample_id,
        "image_path": str(sample.image_path),
        "context": sample.context,
        "gold_label_if_available": sample.label,
        "confidence_threshold": confidence_threshold,
        "main_analysis_history": main_history,
        "previous_controller_decisions": controller_history,
    }
    return f"""Review the following meme harmfulness analysis state.

{json.dumps(payload, ensure_ascii=False, indent=2)}

Decide whether the sample should be finalized. Finalize only when the label is clear enough or no useful visual follow-up remains. If not final, ask 1-5 specific questions for the multimodal model to answer in the next round.

Return exactly this JSON shape:
{{
  "should_finalize": true,
  "label": "harmful",
  "confidence": 0.0,
  "reason": "short evidence-based reason",
  "next_questions": []
}}

Valid labels: harmful, not_harmful, unclear.
"""


def _normalize_controller_decision(
    *,
    output: str,
    parsed: dict[str, Any],
    confidence_threshold: float,
    is_last_round: bool,
) -> dict[str, Any]:
    label = _normalize_prediction_label(
        parsed.get("label")
        or parsed.get("final_label")
        or parsed.get("provisional_label")
        or _label_from_text(output)
    )
    confidence = _parse_confidence(parsed.get("confidence"))
    should_finalize = _parse_bool(parsed.get("should_finalize"))
    if should_finalize is None:
        should_finalize = bool(
            label in {"harmful", "not_harmful"}
            and confidence is not None
            and confidence >= confidence_threshold
        )
    if is_last_round:
        should_finalize = True

    questions = _normalize_questions(
        parsed.get("next_questions")
        or parsed.get("followup_questions")
        or parsed.get("questions")
    )
    if not should_finalize and not questions:
        questions = [
            "Re-check OCR, target, image-text relation, harmful stereotype cues, and whether the meme endorses or criticizes the harmful content."
        ]

    return {
        "should_finalize": should_finalize,
        "label": label,
        "confidence": confidence,
        "reason": str(parsed.get("reason") or parsed.get("rationale") or "").strip(),
        "next_questions": questions,
    }


def _apply_controller_decision(sample: SampleState, decision: dict[str, Any]) -> None:
    sample.final_label = str(decision.get("label") or "unclear")
    sample.final_confidence = decision.get("confidence")
    sample.final_reason = str(decision.get("reason") or "")
    sample.next_questions = list(decision.get("next_questions") or [])
    sample.round_index += 1
    if decision.get("should_finalize"):
        sample.status = "final"
    else:
        sample.status = "needs_main"


def _force_finalize(sample: SampleState) -> None:
    last_decision = (
        sample.controller_outputs[-1].get("decision", {}) if sample.controller_outputs else {}
    )
    if last_decision:
        sample.final_label = str(last_decision.get("label") or "unclear")
        sample.final_confidence = last_decision.get("confidence")
        sample.final_reason = str(last_decision.get("reason") or "max rounds reached")
    elif sample.main_outputs:
        text = _latest_output(sample.main_outputs)
        sample.final_label = _normalize_prediction_label(_label_from_text(text))
        sample.final_confidence = _parse_confidence(_extract_json_object(text).get("confidence"))
        sample.final_reason = "max rounds reached before controller finalization"
    sample.status = "final"


def _vision_content(prompt: str, image_path: Path) -> list[dict[str, Any]]:
    return [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": _image_data_url(str(image_path))}},
    ]


@lru_cache(maxsize=256)
def _image_data_url(path_value: str) -> str:
    path = Path(path_value)
    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _load_samples(args: argparse.Namespace) -> list[SampleState]:
    rows: list[dict[str, Any]]
    dataset_path = Path(args.dataset).expanduser() if args.dataset else None
    if dataset_path:
        rows = _read_dataset(dataset_path)
        base_dir = Path(args.image_root).expanduser() if args.image_root else dataset_path.parent
    elif args.image:
        rows = [{"image_path": image} for image in args.image]
        base_dir = Path(args.image_root).expanduser() if args.image_root else Path.cwd()
    elif args.image_dir:
        image_dir = Path(args.image_dir).expanduser()
        rows = [{"image_path": str(path), "id": path.stem} for path in _iter_images(image_dir)]
        base_dir = Path(args.image_root).expanduser() if args.image_root else image_dir
    else:
        raise ValueError("Provide --dataset, --image-dir, or --image.")

    if args.offset:
        rows = rows[args.offset :]
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    samples: list[SampleState] = []
    seen_ids: dict[str, int] = {}
    for index, row in enumerate(rows):
        image_value = _image_value(row)
        if not image_value:
            continue
        image_path = _resolve_image_path(image_value, base_dir=base_dir)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found for row {index}: {image_path}")
        sample_id = _sample_id(row, image_path=image_path, index=index)
        sample_id = _unique_id(sample_id, seen_ids)
        samples.append(
            SampleState(
                sample_id=sample_id,
                image_path=image_path,
                context=_context_value(row),
                label=_normalize_label(_label_value(row)),
                raw=row,
            )
        )
    if not samples:
        raise ValueError("No valid samples found.")
    return samples


def _read_dataset(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise ValueError(f"Line {line_no} in {path} is not a JSON object.")
                rows.append(item)
        return rows
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))

    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        for key in ("data", "items", "samples", "annotations", "test"):
            value = data.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    raise ValueError(f"Could not find a list of samples in {path}.")


def _iter_images(image_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _image_value(row: dict[str, Any]) -> str:
    value = _first_present(row, DEFAULT_IMAGE_FIELDS)
    if isinstance(value, list) and value:
        value = value[0]
    if isinstance(value, dict):
        value = value.get("path") or value.get("image_path") or value.get("url") or ""
    if value:
        return str(value).strip()

    images = row.get("images")
    if isinstance(images, list) and images:
        return str(images[0]).strip()

    conversations = row.get("conversations")
    if isinstance(conversations, list):
        for message in conversations:
            if isinstance(message, dict):
                extracted = _extract_image_path(str(message.get("value") or message.get("content") or ""))
                if extracted:
                    return extracted
    return ""


def _resolve_image_path(value: str, *, base_dir: Path) -> Path:
    if _is_url(value):
        raise ValueError("Remote image URLs are not supported in this batch runner.")
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    candidates = [
        base_dir / path,
        Path.cwd() / path,
        Path(__file__).resolve().parent / path,
        Path(__file__).resolve().parents[1] / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (base_dir / path).resolve()


def _sample_id(row: dict[str, Any], *, image_path: Path, index: int) -> str:
    value = _first_present(row, DEFAULT_ID_FIELDS)
    if value is None or str(value).strip() == "":
        value = image_path.stem or f"sample_{index:06d}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())


def _unique_id(value: str, seen: dict[str, int]) -> str:
    count = seen.get(value, 0)
    seen[value] = count + 1
    if count == 0:
        return value
    return f"{value}_{count + 1}"


def _context_value(row: dict[str, Any]) -> str:
    value = _first_present(row, DEFAULT_TEXT_FIELDS)
    if value is not None:
        return str(value).strip()
    conversations = row.get("conversations")
    if isinstance(conversations, list):
        texts: list[str] = []
        for message in conversations:
            if not isinstance(message, dict):
                continue
            role = str(message.get("from") or message.get("role") or "").lower()
            if role in {"user", "human"}:
                text = str(message.get("value") or message.get("content") or "").strip()
                if text:
                    texts.append(text)
        return "\n".join(texts)
    return ""


def _label_value(row: dict[str, Any]) -> Any:
    value = _first_present(row, DEFAULT_LABEL_FIELDS)
    if value is not None:
        return value
    conversations = row.get("conversations")
    if isinstance(conversations, list):
        for message in conversations:
            if not isinstance(message, dict):
                continue
            role = str(message.get("from") or message.get("role") or "").lower()
            if role in {"assistant", "gpt"}:
                return message.get("value") or message.get("content")
    return None


def _first_present(row: dict[str, Any], fields: tuple[str, ...]) -> Any:
    lowered = {str(key).lower(): key for key in row}
    for field_name in fields:
        key = lowered.get(field_name.lower())
        if key is not None and row.get(key) not in {None, ""}:
            return row[key]
    return None


def _normalize_label(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        if int(value) == 1:
            return 1
        if int(value) == 0:
            return 0
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized in POSITIVE_LABELS:
        return 1
    if normalized in NEGATIVE_LABELS:
        return 0
    if re.search(r"\b(?:not|non)[-\s_]*harmful\b|\bnonharmful\b", str(value), flags=re.I):
        return 0
    labels = {
        match.group(1).lower()
        for match in re.finditer(r"\b(harmful|harmless)\b", str(value), flags=re.I)
    }
    if labels == {"harmful"}:
        return 1
    if labels == {"harmless"}:
        return 0
    return None


def _base_record(
    sample: SampleState,
    *,
    role: str,
    round_index: int,
    input_index: int,
) -> dict[str, Any]:
    return {
        "sample_id": sample.sample_id,
        "image_path": str(sample.image_path),
        "role": role,
        "round": round_index + 1,
        "input_index": input_index,
        "context": sample.context,
        "gold_label": sample.label,
    }


def _compact_batch_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in record.items()
        if key
        in {
            "sample_id",
            "image_path",
            "role",
            "round",
            "output",
            "parsed",
            "decision",
            "latency_seconds",
            "error",
        }
    }


def _sample_manifest_record(sample: SampleState) -> dict[str, Any]:
    return {
        "sample_id": sample.sample_id,
        "image_path": str(sample.image_path),
        "context": sample.context,
        "gold_label": sample.label,
        "raw": sample.raw,
    }


def _final_record(sample: SampleState) -> dict[str, Any]:
    return {
        "sample_id": sample.sample_id,
        "image_path": str(sample.image_path),
        "context": sample.context,
        "gold_label": sample.label,
        "prediction_label": sample.final_label,
        "prediction_binary": _prediction_binary(sample.final_label),
        "confidence": sample.final_confidence,
        "status": sample.status,
        "rounds": sample.round_index,
        "reason": sample.final_reason,
        "error": sample.error,
        "last_main_output": _latest_output(sample.main_outputs),
        "last_controller_decision": (
            sample.controller_outputs[-1].get("decision", {})
            if sample.controller_outputs
            else {}
        ),
    }


def _summary(
    records: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    counts = {
        "total": len(records),
        "final_count": sum(1 for item in records if item["status"] == "final"),
        "error_count": sum(1 for item in records if item["status"] == "error"),
        "harmful_predictions": sum(1 for item in records if item["prediction_binary"] == 1),
        "not_harmful_predictions": sum(
            1 for item in records if item["prediction_binary"] == 0
        ),
        "unclear_predictions": sum(
            1 for item in records if item["prediction_binary"] is None
        ),
    }
    metrics = _binary_metrics(records)
    return {
        **counts,
        "metrics": metrics,
        "output_dir": str(output_dir),
        "max_rounds": args.max_rounds,
        "confidence_threshold": args.confidence_threshold,
        "main_concurrency": args.main_concurrency,
        "controller_concurrency": args.controller_concurrency,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }


def _binary_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    labeled = [
        item
        for item in records
        if item.get("gold_label") in {0, 1} and item.get("prediction_binary") in {0, 1}
    ]
    if not labeled:
        return {}
    tp = sum(1 for item in labeled if item["gold_label"] == 1 and item["prediction_binary"] == 1)
    tn = sum(1 for item in labeled if item["gold_label"] == 0 and item["prediction_binary"] == 0)
    fp = sum(1 for item in labeled if item["gold_label"] == 0 and item["prediction_binary"] == 1)
    fn = sum(1 for item in labeled if item["gold_label"] == 1 and item["prediction_binary"] == 0)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(labeled)
    return {
        "evaluated": len(labeled),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def _extract_json_object(text: str) -> dict[str, Any]:
    if not text:
        return {}
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.I | re.S)
    candidates = [fenced.group(1)] if fenced else []
    candidates.extend(_balanced_json_candidates(text))
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return {}


def _balanced_json_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : index + 1])
                    break
        start = text.find("{", start + 1)
    return candidates


def _normalize_prediction_label(value: Any) -> str:
    if value is None:
        return "unclear"
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"harmful", "hateful", "offensive", "toxic", "dangerous"}:
        return "harmful"
    if normalized in {
        "not_harmful",
        "non_harmful",
        "nonharmful",
        "harmless",
        "safe",
        "benign",
    }:
        return "not_harmful"
    if normalized in {"unclear", "unknown", "ambiguous"}:
        return "unclear"
    if re.search(r"\b(?:not|non)[-\s_]*harmful\b|\bharmless\b", str(value), flags=re.I):
        return "not_harmful"
    if re.search(r"\bharmful\b", str(value), flags=re.I):
        return "harmful"
    return "unclear"


def _prediction_binary(label: Any) -> int | None:
    normalized = _normalize_prediction_label(label)
    if normalized == "harmful":
        return 1
    if normalized == "not_harmful":
        return 0
    return None


def _label_from_text(text: str) -> str:
    parsed = _extract_json_object(text)
    for key in ("label", "final_label", "provisional_label", "harmfulness_label"):
        if parsed.get(key):
            return str(parsed[key])
    match = re.search(
        r"\b(not[-_\s]*harmful|non[-_\s]*harmful|harmless|harmful|unclear)\b",
        text,
        flags=re.I,
    )
    return match.group(1) if match else "unclear"


def _parse_confidence(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return max(0.0, min(1.0, number if number <= 1 else number / 100.0))
    normalized = str(value).strip().lower()
    if normalized in {"high", "strong"}:
        return 0.9
    if normalized in {"medium", "moderate"}:
        return 0.6
    if normalized in {"low", "weak"}:
        return 0.3
    match = re.search(r"\d+(?:\.\d+)?", normalized)
    if not match:
        return None
    number = float(match.group(0))
    return max(0.0, min(1.0, number if number <= 1 else number / 100.0))


def _parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "final", "finalize"}:
        return True
    if normalized in {"0", "false", "no", "n", "continue", "more"}:
        return False
    return None


def _normalize_questions(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        lines = [line.strip(" -\t") for line in value.splitlines()]
        return [line for line in lines if line][:5]
    if isinstance(value, list):
        questions = [str(item).strip() for item in value if str(item).strip()]
        return questions[:5]
    return [str(value).strip()][:1]


def _latest_output(records: list[dict[str, Any]]) -> str:
    for record in reversed(records):
        output = str(record.get("output") or "").strip()
        if output:
            return output
    return ""


def _clip(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 20].rstrip() + "\n...[truncated]"


def _postprocess_content(content: str, *, strip_thinking: bool) -> str:
    if strip_thinking:
        content = re.sub(r"(?s)<think>.*?</think>\s*", "", content)
    return content.strip()


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _resolve_output_dir(value: str, project_root: Path) -> Path:
    if value:
        path = Path(value).expanduser()
        return path if path.is_absolute() else project_root / path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return project_root / "runs" / f"batch_agent_{timestamp}"


def _optional_positive(value: int) -> int | None:
    return value if value > 0 else None


def _openai_compatible_api_key() -> str:
    api_key = _env_secret(
        "OPENAI_API_KEY",
        "MEMEAGENT_API_KEY",
        "DASHSCOPE_API_KEY",
        "QWEN_API_KEY",
        "MEMEAGENT_QWEN_API_KEY",
    )
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set and --api-key was not provided.")
    return api_key


def _env_secret(*names: str) -> str | None:
    placeholders = {
        "your-api-key",
        "your_api_key_here",
        "your_dashscope_api_key",
        "your_qwen_api_key",
        "your_real_key",
    }
    for name in names:
        value = os.getenv(name, "").strip().strip('"').strip("'")
        if value and value not in placeholders:
            return value
    return None


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name, "").strip().lower()
    if not value:
        return default
    return value not in {"0", "false", "no", "off", "disabled"}


def _is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"}


def _extract_image_path(text: str) -> str:
    for pattern in (
        r"<\|vision_start\|>(.*?)<\|vision_(?:start|end)\|>",
        r"<image>\s*([^\s<>]+)",
        r"(/(?:data|home)/[^\s<>]+?\.(?:png|jpg|jpeg|webp|gif|bmp))",
        r"((?:Dataset|dataset)/[^\s<>]+?\.(?:png|jpg|jpeg|webp|gif|bmp))",
    ):
        match = re.search(pattern, text, flags=re.I | re.S)
        if match:
            return match.group(1).strip()
    return ""


if __name__ == "__main__":
    raise SystemExit(main())
