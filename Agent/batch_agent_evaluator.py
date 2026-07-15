from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import mimetypes
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

from memeagent.agent import build_analysis_iteration_prompt
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
DEFAULT_BATCH_MAIN_MODEL = "memeagent-main"
DEFAULT_BATCH_CONTROLLER_MODEL = "memeagent-controller"

@dataclass
class SampleState:
    sample_id: str
    image_path: Path
    context: str = ""
    gold_judgement: str = ""
    gold_binary: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)
    status: str = "needs_main"
    round_index: int = 0
    next_questions: list[str] = field(default_factory=list)
    retrieval_questions: list[str] = field(default_factory=list)
    no_progress_rounds: int = 0
    main_outputs: list[dict[str, Any]] = field(default_factory=list)
    controller_outputs: list[dict[str, Any]] = field(default_factory=list)
    final_outputs: list[dict[str, Any]] = field(default_factory=list)
    final_answer: str = ""
    prediction_label: str = "harmless"
    prediction_binary: int = 0
    prediction_source: str = "controller"
    harmfulness_analysis: str = ""
    harmfulness_labels: list[str] = field(default_factory=list)
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
        enable_thinking: bool = False,
    ) -> str:
        url = self.base_url.rstrip("/") + "/chat/completions"
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
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
            "Run wavefront batch inference for MemeAgent multimodal analysis "
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
    parser.add_argument(
        "--main-model",
        default="",
        help=(
            "Main multimodal service model name. "
            f"Defaults to {DEFAULT_BATCH_MAIN_MODEL}."
        ),
    )
    parser.add_argument(
        "--controller-model",
        default="",
        help=(
            "Controller service model name. "
            f"Defaults to {DEFAULT_BATCH_CONTROLLER_MODEL}."
        ),
    )
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
        "--final-concurrency",
        type=int,
        default=int(os.getenv("MEMEAGENT_BATCH_FINAL_CONCURRENCY", "4")),
        help="Concurrent final-analysis requests to the main model.",
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
        default=int(os.getenv("MEMEAGENT_BATCH_CONTROLLER_MAX_TOKENS", "2200")),
        help="Max output tokens for controller calls. Use 0 to omit.",
    )
    parser.add_argument(
        "--final-max-tokens",
        type=int,
        default=int(os.getenv("MEMEAGENT_BATCH_FINAL_MAX_TOKENS", "2048")),
        help="Max output tokens for the main model's final analysis. Use 0 to omit.",
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
    main_model = args.main_model.strip() or config.model or DEFAULT_BATCH_MAIN_MODEL
    controller_model = (
        args.controller_model.strip()
        or config.controller_model
        or DEFAULT_BATCH_CONTROLLER_MODEL
    )
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
    print(f"Final analysis: main model concurrency={args.final_concurrency}", flush=True)
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
                max_rounds=max(args.max_rounds, 1),
                is_last_round=round_index >= max(args.max_rounds, 1) - 1,
                save_prompts=args.save_prompts,
            )
            _write_jsonl(
                output_dir
                / f"batch_{batch_index:03d}_controller_round_{round_index + 1}.jsonl",
                controller_records,
            )
            batch_index += 1

        active = [
            sample
            for sample in samples
            if sample.status not in {"needs_final", "final", "error"}
        ]
        if not active:
            break

    for sample in samples:
        if sample.status not in {"needs_final", "final", "error"}:
            _force_finalize(sample)

    final_candidates = [sample for sample in samples if sample.status == "needs_final"]
    if final_candidates:
        print(f"[post] final analysis wave: {len(final_candidates)} samples", flush=True)
        final_analysis_records = _run_final_wave(
            samples=final_candidates,
            endpoint=endpoint,
            model=main_model,
            concurrency=max(args.final_concurrency, 1),
            temperature=args.temperature,
            max_tokens=_optional_positive(args.final_max_tokens),
            save_prompts=args.save_prompts,
        )
        _write_jsonl(
            output_dir / f"batch_{batch_index:03d}_final_analysis.jsonl",
            final_analysis_records,
        )
        batch_index += 1

    final_records = [_final_record(sample) for sample in samples]
    _write_jsonl(output_dir / "final_results.jsonl", final_records)
    summary = _summary(
        final_records,
        args=args,
        output_dir=output_dir,
        project_root=project_root,
        main_model=main_model,
        controller_model=controller_model,
        endpoint=endpoint,
    )
    _write_json(output_dir / "summary.json", summary)
    print(
        "Done: "
        f"final={summary['final_count']} error={summary['error_count']} "
        f"output={output_dir / 'final_results.jsonl'}",
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


def _run_final_wave(
    *,
    samples: list[SampleState],
    endpoint: ChatEndpoint,
    model: str,
    concurrency: int,
    temperature: float,
    max_tokens: int | None,
    save_prompts: bool,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(
                _call_final,
                input_index=index,
                sample=sample,
                endpoint=endpoint,
                model=model,
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
    max_rounds: int,
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
                max_rounds=max_rounds,
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
            {"role": "system", "content": _main_system_prompt()},
            {"role": "user", "content": _vision_content(prompt, sample.image_path)},
        ]
        output = endpoint.chat(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        parsed = _extract_json_object(output)
        normalized = _normalize_main_analysis(parsed)
        record.update(
            {
                "output": output,
                "parsed": normalized or parsed,
                "latency_seconds": round(time.time() - started, 3),
                "error": "",
            }
        )
        has_new_evidence = _has_genuinely_new_evidence(sample, normalized)
        sample.main_outputs.append(_compact_batch_record(record))
        if round_index > 0 and normalized:
            if has_new_evidence:
                sample.no_progress_rounds = 0
            else:
                sample.no_progress_rounds += 1
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


def _call_final(
    *,
    input_index: int,
    sample: SampleState,
    endpoint: ChatEndpoint,
    model: str,
    temperature: float,
    max_tokens: int | None,
    save_prompts: bool,
) -> dict[str, Any]:
    started = time.time()
    prompt = _final_prompt(sample)
    record = _base_record(
        sample,
        role="final",
        round_index=sample.round_index,
        input_index=input_index,
    )
    if save_prompts:
        record["prompt"] = prompt
    try:
        output = endpoint.chat(
            model=model,
            messages=[
                {"role": "system", "content": _main_system_prompt()},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        parsed = _extract_json_object(output)
        decision = _normalize_final_analysis(
            sample=sample,
            output=output,
            parsed=parsed,
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
        sample.final_outputs.append(_compact_batch_record(record))
        _apply_final_analysis(sample, decision)
        sample.status = "final"
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
    max_rounds: int,
    is_last_round: bool,
    save_prompts: bool,
) -> dict[str, Any]:
    started = time.time()
    prompt = _controller_prompt(
        sample,
        confidence_threshold=confidence_threshold,
        max_rounds=max_rounds,
    )
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
            {"role": "system", "content": _main_system_prompt()},
            {"role": "user", "content": prompt},
        ]
        output = endpoint.chat(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            enable_thinking=True,
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
    if round_index == 0 or not sample.main_outputs:
        task = """Perform the first image-grounded analysis. Inspect exact OCR,
visible entities, layout, symbols, gestures, editing, image-text relation,
target, stance, concrete harmfulness cues, and visible source clues. Separate
confirmed pixel evidence from inference."""
    else:
        previous = _batch_visual_report(sample)
        questions = "\n".join(f"- {item}" for item in sample.next_questions) or (
            "- Re-check the image carefully."
        )
        task = f"""Perform a targeted visual follow-up. Answer only the current
controller questions and report only genuinely new image-grounded evidence.

Current controller questions:
{questions}

Previous structured visual report:
{_clip(previous, 4000)}"""
    return f"""You are the multimodal evidence model in an image-only meme analysis pipeline.

Sample: {sample.sample_id}
User context: {sample.context or "None"}

{task}

Return only one JSON object with this exact shape:
{{
  "direct_answers": [
    {{
      "question": "controller question or initial analysis goal",
      "answer": "direct answer",
      "answerability": "answered|not_visible|uncertain|requires_retrieval",
      "evidence": "visible pixel/OCR evidence only",
      "confidence": 0.0
    }}
  ],
  "visible_evidence": [],
  "inferences": [],
  "new_evidence": [],
  "unresolved_visual_questions": [],
  "retrieval_required": false
}}

Do not output planning, a thinking process, task restatement, drafting notes,
self-correction, or markdown. On follow-up rounds, new_evidence must contain
only evidence not already present in the previous report. Use an empty list
when re-examination found nothing new. Mark source history, statistical
verification, creator intent, audience reception, and documented meme usage
as requires_retrieval rather than guessing.
"""


def _controller_prompt(
    sample: SampleState,
    *,
    confidence_threshold: float,
    max_rounds: int,
) -> str:
    visual_report = _batch_visual_report(sample)
    iteration_history = _batch_controller_history(sample)
    controller_prompt = build_analysis_iteration_prompt(
        topic=sample.sample_id,
        context=sample.context,
        visual_report=visual_report,
        search_report="None",
        iteration_history=iteration_history,
        input_mode="image_only",
        round_index=max(sample.round_index, 0) + 1,
        max_rounds=max(max_rounds, 1),
        confidence_threshold=confidence_threshold,
        retrieval_enabled=False,
    )
    return f"""{controller_prompt}

Ignore any earlier output-format instructions and return exactly this JSON
shape so the batch evaluator can parse it:
{{
  "should_finalize": true,
  "confidence": 0.0,
  "prediction_label": "harmless",
  "prediction_binary": 0,
  "harmfulness_labels": [],
  "harmfulness_analysis": "current evidence-based harmfulness judgement",
  "reason": "short evidence-based reason",
  "next_questions": [],
  "retrieval_questions": []
}}

prediction_label must be exactly one of: harmful, harmless.
prediction_binary must be 1 for harmful and 0 for harmless.
Image-text
combinations may provide an implicit target even when OCR does not name one;
do not default to harmless when that multimodal relation supports a concrete
harm mechanism.
This evaluator is running in image-only mode. next_questions may contain only
questions answerable by re-examining image pixels: OCR, visible watermarks,
entities, symbols, composition, target, stance, and image-text relation.
Put source history, fact/statistic verification, creator intent, audience
reception, and documented meme-template usage in retrieval_questions. If all
remaining questions require retrieval, set should_finalize=true. Do not repeat
a visual question that the latest main analysis already answered. If the latest
main analysis reports no new evidence, finalize with the remaining uncertainty.
Return only one JSON object and no markdown.
"""


def _final_prompt(sample: SampleState) -> str:
    context_parts = []
    if sample.context.strip():
        context_parts.append(f"[User Context]\n{sample.context.strip()}")
    context_parts.append(
        "Controller planning and confidence report:\n"
        + _batch_controller_history(sample)
    )
    context_parts.append(
        "Cumulative image analysis after controller-directed passes:\n"
        + _batch_visual_report(sample)
    )
    context_text = "\n\n".join(context_parts)
    return f"""Topic: {sample.sample_id}

The controller has already made the classification decision:
- prediction_label: {sample.prediction_label}
- prediction_binary: {sample.prediction_binary}
- confidence: {sample.final_confidence}

{context_text}

Write a concise evidence-based explanation of the controller's decision. Do
not reconsider or change the label. Return only one JSON object:
{{
  "meme_object_and_context": "concise description",
  "visual_ocr_evidence": "confirmed visible evidence",
  "harmfulness_analysis": "why the fixed label is supported",
  "limitations": "remaining uncertainty"
}}

Do not output planning, thinking, task restatement, drafting notes,
self-correction, or markdown. Keep the complete response under 500 words.
"""


def _main_system_prompt() -> str:
    return MemeAgentConfig.from_env().system_prompt


def _batch_visual_report(sample: SampleState) -> str:
    if not sample.main_outputs:
        return "None"
    blocks = []
    for item in sample.main_outputs[-3:]:
        round_label = item.get("round", "?")
        parsed = item.get("parsed", {})
        evidence = ""
        if parsed:
            evidence = _clip(json.dumps(parsed, ensure_ascii=False, indent=2), 3500)
        if not evidence:
            evidence = _clip(str(item.get("output", "")), 3500)
        blocks.append(
            f"## Batch Image Analysis Round {round_label}\n\n"
            f"{evidence or 'No usable visual evidence.'}"
        )
    return "\n\n".join(blocks).strip()


def _batch_controller_history(sample: SampleState) -> str:
    if not sample.controller_outputs:
        return "None"
    blocks = []
    for item in sample.controller_outputs[-3:]:
        round_label = item.get("round", "?")
        decision = item.get("decision", {})
        blocks.append(
            f"## Batch Controller Round {round_label}\n\n"
            f"{json.dumps(decision, ensure_ascii=False, indent=2)}"
        )
    return "\n\n".join(blocks).strip()


def _normalize_main_analysis(parsed: dict[str, Any]) -> dict[str, Any]:
    if not parsed:
        return {}
    expected_keys = {
        "direct_answers",
        "visible_evidence",
        "inferences",
        "new_evidence",
        "unresolved_visual_questions",
        "retrieval_required",
    }
    if not expected_keys.intersection(parsed):
        return {}

    direct_answers: list[dict[str, Any]] = []
    raw_answers = parsed.get("direct_answers")
    if isinstance(raw_answers, list):
        for item in raw_answers:
            if not isinstance(item, dict):
                continue
            answerability = str(item.get("answerability") or "uncertain").strip().lower()
            if answerability not in {
                "answered",
                "not_visible",
                "uncertain",
                "requires_retrieval",
            }:
                answerability = "uncertain"
            direct_answers.append(
                {
                    "question": str(item.get("question") or "").strip(),
                    "answer": str(item.get("answer") or "").strip(),
                    "answerability": answerability,
                    "evidence": str(item.get("evidence") or "").strip(),
                    "confidence": _parse_confidence(item.get("confidence")),
                }
            )

    return {
        "direct_answers": direct_answers,
        "visible_evidence": _normalize_string_list(parsed.get("visible_evidence")),
        "inferences": _normalize_string_list(parsed.get("inferences")),
        "new_evidence": _normalize_string_list(parsed.get("new_evidence")),
        "unresolved_visual_questions": _normalize_questions(
            parsed.get("unresolved_visual_questions")
        ),
        "retrieval_required": bool(_parse_bool(parsed.get("retrieval_required"))),
    }


def _has_genuinely_new_evidence(
    sample: SampleState,
    analysis: dict[str, Any],
) -> bool:
    def key(value: Any) -> str:
        return " ".join(str(value).casefold().split())

    previous_evidence: set[str] = set()
    for record in sample.main_outputs:
        parsed = record.get("parsed")
        if not isinstance(parsed, dict):
            continue
        for field_name in ("visible_evidence", "new_evidence"):
            previous_evidence.update(
                key(item) for item in _normalize_string_list(parsed.get(field_name))
            )

    candidates = {
        key(item) for item in _normalize_string_list(analysis.get("new_evidence"))
    }
    candidates.discard("")
    return bool(candidates - previous_evidence)


def _normalize_controller_decision(
    *,
    output: str,
    parsed: dict[str, Any],
    confidence_threshold: float,
    is_last_round: bool,
) -> dict[str, Any]:
    confidence = _parse_confidence(parsed.get("confidence"))
    label_value = (
        parsed.get("prediction_label")
        or parsed.get("label")
        or parsed.get("final_decision")
        or parsed.get("judgement")
    )
    prediction_binary = _prediction_binary_from_values(
        parsed.get("prediction_binary"),
        str(label_value or ""),
        "",
        str(parsed.get("harmfulness_analysis") or ""),
    )
    prediction_label = "harmful" if prediction_binary == 1 else "harmless"
    harmfulness_analysis = str(
        parsed.get("harmfulness_analysis")
        or parsed.get("harmfulness")
        or _extract_harmfulness_section(output)
    ).strip()
    should_finalize = _parse_bool(parsed.get("should_finalize"))
    if should_finalize is None:
        should_finalize = bool(confidence is not None and confidence >= confidence_threshold)
    if is_last_round:
        should_finalize = True

    questions = _normalize_questions(
        parsed.get("next_questions")
        or parsed.get("followup_questions")
        or parsed.get("questions")
    )
    retrieval_questions = _normalize_questions(parsed.get("retrieval_questions"))
    if retrieval_questions and not questions:
        should_finalize = True
    if not should_finalize and not questions:
        questions = [
            "Re-check OCR, concrete visual details, image-text relation, target, stance, and concrete harm mechanism."
        ]

    return {
        "should_finalize": should_finalize,
        "confidence": confidence,
        "prediction_label": prediction_label,
        "prediction_binary": prediction_binary,
        "harmfulness_analysis": harmfulness_analysis,
        "harmfulness_labels": _normalize_string_list(parsed.get("harmfulness_labels")),
        "reason": str(parsed.get("reason") or parsed.get("rationale") or "").strip(),
        "next_questions": questions,
        "retrieval_questions": retrieval_questions,
    }


def _normalize_final_analysis(
    *,
    sample: SampleState,
    output: str,
    parsed: dict[str, Any],
) -> dict[str, Any]:
    harmfulness_analysis = str(
        parsed.get("harmfulness_analysis")
        or parsed.get("harmfulness")
        or _extract_harmfulness_section(output)
        or sample.harmfulness_analysis
    ).strip()
    return {
        "prediction_label": sample.prediction_label,
        "prediction_binary": sample.prediction_binary,
        "prediction_source": sample.prediction_source,
        "harmfulness_analysis": harmfulness_analysis,
        "harmfulness_labels": _normalize_string_list(
            parsed.get("harmfulness_labels")
        ),
        "confidence": _parse_confidence(parsed.get("confidence"))
        or sample.final_confidence,
        "final_answer": output.strip(),
        "reason": str(
            parsed.get("reason") or sample.final_reason or "controller decision"
        ).strip(),
    }


def _apply_controller_decision(sample: SampleState, decision: dict[str, Any]) -> None:
    sample.prediction_label = str(decision.get("prediction_label") or "harmless")
    sample.prediction_binary = int(decision.get("prediction_binary") == 1)
    sample.prediction_source = "controller"
    sample.harmfulness_analysis = str(decision.get("harmfulness_analysis") or "")
    sample.harmfulness_labels = list(decision.get("harmfulness_labels") or [])
    sample.final_confidence = decision.get("confidence")
    sample.final_reason = str(decision.get("reason") or "")
    sample.next_questions = list(decision.get("next_questions") or [])
    sample.retrieval_questions = list(decision.get("retrieval_questions") or [])
    sample.round_index += 1
    no_progress = sample.no_progress_rounds > 0
    retrieval_only = bool(sample.retrieval_questions and not sample.next_questions)
    if no_progress:
        suffix = "No new image-grounded evidence was found in the latest follow-up."
        sample.final_reason = " ".join(part for part in (sample.final_reason, suffix) if part)
    if decision.get("should_finalize") or no_progress or retrieval_only:
        sample.status = "needs_final"
    else:
        sample.status = "needs_main"


def _apply_final_analysis(sample: SampleState, decision: dict[str, Any]) -> None:
    sample.final_answer = str(decision.get("final_answer") or "")
    sample.prediction_label = str(decision.get("prediction_label") or "harmless")
    sample.prediction_binary = int(decision.get("prediction_binary") == 1)
    sample.harmfulness_analysis = str(decision.get("harmfulness_analysis") or "")
    sample.final_confidence = decision.get("confidence")
    sample.final_reason = str(decision.get("reason") or "final main-model synthesis")


def _force_finalize(sample: SampleState) -> None:
    last_decision = (
        sample.controller_outputs[-1].get("decision", {}) if sample.controller_outputs else {}
    )
    if last_decision:
        sample.prediction_label = str(last_decision.get("prediction_label") or "harmless")
        sample.prediction_binary = int(last_decision.get("prediction_binary") == 1)
        sample.harmfulness_analysis = str(last_decision.get("harmfulness_analysis") or "")
        sample.harmfulness_labels = list(last_decision.get("harmfulness_labels") or [])
        sample.final_confidence = last_decision.get("confidence")
        sample.final_reason = str(last_decision.get("reason") or "max rounds reached")
    elif sample.main_outputs:
        text = _latest_output(sample.main_outputs)
        sample.final_answer = _fallback_final_answer(text, _extract_json_object(text))
        sample.harmfulness_analysis = _extract_harmfulness_section(sample.final_answer)
        sample.prediction_label = "harmless"
        sample.prediction_binary = _prediction_binary_from_values(
            None,
            "",
            sample.final_answer,
            sample.harmfulness_analysis,
        )
        sample.prediction_label = "harmful" if sample.prediction_binary == 1 else "harmless"
        sample.prediction_source = "main_fallback"
        sample.final_confidence = _parse_confidence(_extract_json_object(text).get("confidence"))
        sample.final_reason = "max rounds reached before controller finalization"
    sample.status = "needs_final"


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
                gold_judgement=_judgement_value(row),
                gold_binary=_judgement_binary(_judgement_value(row)),
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
    return ""


def _judgement_value(row: dict[str, Any]) -> str:
    judgement = row.get("gold_judgement")
    if judgement is not None and str(judgement).strip():
        return str(judgement).strip()

    binary = row.get("gold_binary", row.get("label"))
    if binary in {0, 1, "0", "1"}:
        return "JUDGEMENT: harmful" if int(binary) == 1 else "JUDGEMENT: harmless"

    conversations = row.get("conversations")
    if isinstance(conversations, list):
        for message in conversations:
            if not isinstance(message, dict):
                continue
            role = str(message.get("from") or message.get("role") or "").lower()
            if role in {"assistant", "gpt"}:
                text = str(message.get("value") or message.get("content") or "")
                judgement = _extract_tag(text, "JUDGEMENT")
                return judgement or text.strip()
    return ""


def _judgement_binary(value: str) -> int | None:
    if not value:
        return None
    normalized = value.strip().lower().replace("-", "_")
    if re.search(r"\b(?:harmless|not[_\s-]*harmful|non[_\s-]*harmful|nonharmful)\b", normalized):
        return 0
    if re.search(r"\bharmful\b", normalized):
        return 1
    return None


def _first_present(row: dict[str, Any], fields: tuple[str, ...]) -> Any:
    lowered = {str(key).lower(): key for key in row}
    for field_name in fields:
        key = lowered.get(field_name.lower())
        if key is not None and row.get(key) not in {None, ""}:
            return row[key]
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
        "gold_judgement": sample.gold_judgement,
        "gold_binary": sample.gold_binary,
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
        "gold_judgement": sample.gold_judgement,
        "gold_binary": sample.gold_binary,
        "raw": sample.raw,
    }


def _final_record(sample: SampleState) -> dict[str, Any]:
    return {
        "sample_id": sample.sample_id,
        "image_path": str(sample.image_path),
        "context": sample.context,
        "gold_judgement": sample.gold_judgement,
        "gold_binary": sample.gold_binary,
        "prediction_label": sample.prediction_label,
        "prediction_binary": sample.prediction_binary,
        "prediction_source": sample.prediction_source,
        "controller_prediction_label": sample.prediction_label,
        "controller_prediction_binary": sample.prediction_binary,
        "harmfulness_analysis": sample.harmfulness_analysis,
        "harmfulness_labels": sample.harmfulness_labels,
        "final_answer": sample.final_answer,
        "confidence": sample.final_confidence,
        "status": sample.status,
        "rounds": sample.round_index,
        "reason": sample.final_reason,
        "retrieval_questions": sample.retrieval_questions,
        "error": sample.error,
    }


def _summary(
    records: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    output_dir: Path,
    project_root: Path,
    main_model: str,
    controller_model: str,
    endpoint: ChatEndpoint,
) -> dict[str, Any]:
    counts = {
        "total": len(records),
        "final_count": sum(1 for item in records if item["status"] == "final"),
        "error_count": sum(1 for item in records if item["status"] == "error"),
        "gold_harmful_count": sum(1 for item in records if item["gold_binary"] == 1),
        "gold_harmless_count": sum(1 for item in records if item["gold_binary"] == 0),
        "gold_unknown_count": sum(1 for item in records if item["gold_binary"] is None),
        "prediction_harmful_count": sum(
            1 for item in records if item["prediction_binary"] == 1
        ),
        "prediction_harmless_count": sum(
            1 for item in records if item["prediction_binary"] == 0
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
        "final_concurrency": args.final_concurrency,
        "experiment": _experiment_metadata(
            args=args,
            project_root=project_root,
            main_model=main_model,
            controller_model=controller_model,
            endpoint=endpoint,
        ),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }


def _experiment_metadata(
    *,
    args: argparse.Namespace,
    project_root: Path,
    main_model: str,
    controller_model: str,
    endpoint: ChatEndpoint,
) -> dict[str, Any]:
    return {
        "entrypoint": Path(sys.argv[0]).name,
        "command": [sys.executable, *_redacted_argv(sys.argv)],
        "dataset": str(Path(args.dataset).expanduser()) if args.dataset else "",
        "image_dir": str(Path(args.image_dir).expanduser()) if args.image_dir else "",
        "image_root": str(Path(args.image_root).expanduser()) if args.image_root else "",
        "offset": args.offset,
        "limit": args.limit,
        "base_url": endpoint.base_url,
        "main_model": main_model,
        "controller_model": controller_model,
        "final_model": main_model,
        "main_temperature": args.temperature,
        "controller_temperature": args.controller_temperature,
        "main_max_tokens": _optional_positive(args.main_max_tokens),
        "controller_max_tokens": _optional_positive(args.controller_max_tokens),
        "final_max_tokens": _optional_positive(args.final_max_tokens),
        "timeout": args.timeout,
        "retries": args.retries,
        "strip_thinking": endpoint.strip_thinking,
        "save_prompts": args.save_prompts,
        "decision_policy": {
            "prediction_owner": "controller",
            "final_model_can_override_prediction": False,
            "hard_judgement_rules_enabled": False,
            "output_normalization_fallbacks_enabled": True,
        },
        "source": _source_metadata(project_root),
    }


def _redacted_argv(argv: list[str]) -> list[str]:
    redacted: list[str] = []
    hide_next = False
    for argument in argv:
        if hide_next:
            redacted.append("<redacted>")
            hide_next = False
            continue
        if argument == "--api-key":
            redacted.append(argument)
            hide_next = True
            continue
        if argument.startswith("--api-key="):
            redacted.append("--api-key=<redacted>")
            continue
        redacted.append(argument)
    return redacted


def _source_metadata(project_root: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "git_commit": None,
        "git_dirty": None,
        "file_sha256": {},
    }
    for relative_path in (
        "batch_agent_evaluator.py",
        "batch_agent_evaluator_main_as_controller.py",
        "memeagent/agent.py",
    ):
        path = project_root / relative_path
        if path.is_file():
            metadata["file_sha256"][relative_path] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        metadata["git_commit"] = commit.stdout.strip() or None
        metadata["git_dirty"] = bool(status.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return metadata


def _binary_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    evaluated = [
        item
        for item in records
        if item.get("gold_binary") in {0, 1}
        and item.get("prediction_binary") in {0, 1}
    ]
    if not evaluated:
        return {
            "evaluated": 0,
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
        }
    tp = sum(
        1
        for item in evaluated
        if item["gold_binary"] == 1 and item["prediction_binary"] == 1
    )
    tn = sum(
        1
        for item in evaluated
        if item["gold_binary"] == 0 and item["prediction_binary"] == 0
    )
    fp = sum(
        1
        for item in evaluated
        if item["gold_binary"] == 0 and item["prediction_binary"] == 1
    )
    fn = sum(
        1
        for item in evaluated
        if item["gold_binary"] == 1 and item["prediction_binary"] == 0
    )
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(evaluated)
    return {
        "evaluated": len(evaluated),
        "coverage": len(evaluated) / len(records) if records else 0.0,
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


def _extract_tag(text: str, tag: str) -> str:
    match = re.search(fr"<{tag}\b[^>]*>(.*?)</{tag}>", text, flags=re.I | re.S)
    return match.group(1).strip() if match else ""


def _extract_harmfulness_section(text: str) -> str:
    if not text:
        return ""
    patterns = (
        r"(?:^|\n)\s*(?:3|4)[.\)]\s*(?:Harmfulness analysis|有害性分析)[:：]?\s*(.*?)(?=\n\s*(?:4|5)[.\)]|Evidence map|证据地图|\Z)",
        r"(?:^|\n)\s*(?:Harmfulness|有害性)[:：]\s*(.*?)(?=\n\s*(?:Evidence|证据)\b|\Z)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I | re.S)
        if match:
            return match.group(1).strip()
    return ""


def _normalize_prediction_label(value: Any) -> str:
    if value is None or value == "":
        return "harmless"
    raw_value = str(value).strip()
    normalized = raw_value.lower().replace("-", "_").replace(" ", "_")
    if normalized in {"harmful", "unsafe", "toxic", "offensive"}:
        return "harmful"
    if normalized in {"harmless", "not_harmful", "non_harmful", "nonharmful", "safe"}:
        return "harmless"
    if raw_value in {"有害", "有害内容"}:
        return "harmful"
    if raw_value in {"无害", "无害内容"}:
        return "harmless"
    if raw_value in {"不明确", "不确定", "无法判断", "证据不足"}:
        return "harmless"
    if re.search(
        r"\b(?:insufficient evidence|not enough evidence|not clearly|not necessarily|not demonstrably)\b.{0,40}\bharmful\b",
        raw_value,
        flags=re.I | re.S,
    ):
        return "harmless"
    if re.search(r"(?:不明确|不确定|无法判断|证据不足).{0,20}有害", raw_value):
        return "harmless"
    if re.search(r"\b(?:harmless|not[_\s-]*harmful|non[_\s-]*harmful|nonharmful)\b", raw_value, flags=re.I):
        return "harmless"
    if re.search(r"\bharmful\b", raw_value, flags=re.I):
        return "harmful"
    return "harmless"


def _prediction_binary_from_values(
    explicit_binary: Any,
    label: str,
    final_answer: str,
    harmfulness_analysis: str,
) -> int:
    if isinstance(explicit_binary, bool):
        return int(explicit_binary)
    if isinstance(explicit_binary, (int, float)):
        if int(explicit_binary) == 1:
            return 1
        if int(explicit_binary) == 0:
            return 0
    if explicit_binary is not None and str(explicit_binary).strip() != "":
        normalized = str(explicit_binary).strip().lower()
        if normalized in {"1", "harmful", "true", "yes"}:
            return 1
        if normalized in {"0", "harmless", "not_harmful", "false", "no"}:
            return 0

    if str(label).strip():
        normalized_label = _normalize_prediction_label(label)
        if normalized_label == "harmful":
            return 1
        return 0

    text = f"{harmfulness_analysis}\n{final_answer}".lower().replace("-", "_")
    if re.search(r"\b(?:prediction_label|final_decision|judgement|label)\s*(?:is|:|：)?\s*(?:harmless|not[_\s]*harmful|non[_\s]*harmful|nonharmful)\b", text, flags=re.S):
        return 0
    if re.search(r"\b(?:prediction_label|final_decision|judgement|label)\s*(?:is|:|：)?\s*harmful\b", text, flags=re.S):
        return 1
    if re.search(
        r"\b(?:insufficient evidence|not enough evidence|not clearly|not necessarily|not demonstrably)\b.{0,40}\bharmful\b",
        text,
        flags=re.S,
    ):
        return 0
    if re.search(r"(?:不明确|不确定|无法判断|证据不足).{0,20}有害", text):
        return 0
    if re.search(r"\b(?:harmless|not[_\s]*harmful|non[_\s]*harmful|nonharmful)\b", text, flags=re.S):
        return 0
    if re.search(
        r"(?:\b(?:the|this)\s+meme\s+(?:is|appears|seems|is classified as|is judged)\s+harmful\b|(?:^|\n)\s*harmful\s*(?:because|:|：))",
        text,
        flags=re.S,
    ):
        return 1
    if re.search(r"(?:该|这个)?(?:迷因|梗图|表情包|内容).{0,8}(?:是|属于|判定为|可视为)有害", text):
        return 1
    if re.search(r"(?:该|这个)?(?:迷因|梗图|表情包|内容).{0,8}(?:是|属于|判定为|可视为)无害", text):
        return 0
    return 0


def _normalize_string_list(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [part.strip() for part in str(value).split(",") if part.strip()]


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


def _fallback_final_answer(output: str, parsed: dict[str, Any]) -> str:
    for key in (
        "final_answer",
        "answer",
        "analysis",
        "summary",
        "updated_interpretation",
        "visual_description",
    ):
        value = parsed.get(key)
        if value:
            if isinstance(value, (list, dict)):
                return json.dumps(value, ensure_ascii=False)
            return str(value).strip()
    return _clip(output.strip(), 2000)


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
