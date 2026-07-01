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

MAIN_SYSTEM_PROMPT = """You are the multimodal analysis model in a MemeAgent batch inference pipeline.
Analyze the image and any supplied context for a researcher-oriented meme analysis.
Focus on visible text/OCR, concrete visual entities, image-text relation, sentiment, harmfulness, audience, intent, cultural references, evolution cues, uncertainty, and evidence gaps.
Keep evidence separate from inference. Return concise JSON when possible."""

CONTROLLER_SYSTEM_PROMPT = """You are the controller model for an iterative MemeAgent batch inference pipeline.
You review multimodal analyses, decide whether the sample can be finalized, estimate confidence, and produce a researcher-oriented Chinese meme analysis matching the MemeAgent style.
Return only one JSON object and no markdown."""

PROJECT_RUBRIC = """
Use these project-specific meme analysis standards.

Harmfulness labels:
- Discrimination: discriminates based on race, gender, religion, nationality, LGBTQ+ identity, or comparable protected/social identity.
- Offensive: insults, belittles, or maliciously attacks individuals or groups. Any reference to sensitive events, including disasters, pandemics, historical tragedies, racial issues, or violent incidents, is offensive in any form: humorous, sarcastic, dismissive, factual, or otherwise.
- Violence: directly or indirectly encourages, endorses, threatens, celebrates, or depicts violent behavior.
- Vulgar: sexual implication, obscenity, nudity, vulgar language, or implicit sexual behavior information in any form.
- Antagonism: aggressive discontent, resentment, or negativity toward groups or society through hostile satire, malicious sarcasm, harmful exaggeration, pessimism, or nihilism without constructive intent.

Sentiment labels:
- Joy, Sadness, Anger, Disgust, Fear, Surprise.

Audience prediction:
- Gemeinschaft-oriented audience: micro-level, high-context, subcultural, domain-specific, ideological, or localized in-group.
- Gesellschaft-oriented audience: macro-level mass internet society.

Intent detection:
- Teleological, Normative, Dramaturgical, Communicative.

Evolution analysis:
- Multimodal phylogenetic tracking, Core kernel fidelity, Lifecycle and diffusion dynamics, Semiotic and intertextual splicing.

Evidence discipline:
- Separate direct image/OCR evidence, user context, retrieved evidence, and inference.
- Do not invent entities, source IDs, platforms, dates, intent, audience, or evolution history.
""".strip()


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
    main_outputs: list[dict[str, Any]] = field(default_factory=list)
    controller_outputs: list[dict[str, Any]] = field(default_factory=list)
    final_answer: str = ""
    prediction_label: str = "unclear"
    prediction_binary: int | None = None
    harmfulness_analysis: str = ""
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
        default=int(os.getenv("MEMEAGENT_BATCH_CONTROLLER_MAX_TOKENS", "2200")),
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
    _write_jsonl(output_dir / "final_results.jsonl", final_records)
    summary = _summary(final_records, args=args, output_dir=output_dir)
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
    if round_index == 0 or not sample.main_outputs:
        return f"""Analyze this meme image as evidence for a researcher-oriented MemeAgent report.

Optional context:
{sample.context or "None"}

Project rubric:
{PROJECT_RUBRIC}

Return a compact JSON object with these keys:
- ocr_text: visible text, or empty string
- visual_description: concrete visible entities/actions
- subjects: people, groups, objects, symbols, or communities referenced if any
- image_text_relation: how text and image interact
- sentiment_analysis: primary and secondary sentiment labels with evidence
- harmfulness_analysis: labels, target, harm type, evidence, uncertainty, and whether the image is harmful or harmless
- audience_prediction: Gemeinschaft-oriented or Gesellschaft-oriented with evidence
- intent_recognition: Teleological, Normative, Dramaturgical, Communicative, mixed, or unclear
- evolution_tracking: template lineage, visual drift, intertextual splicing, or unknown
- evidence: concrete evidence from the image
- confidence: number from 0 to 1
- missing_information: list of uncertainties
"""

    previous = _clip(_latest_output(sample.main_outputs), 4000)
    questions = "\n".join(f"- {item}" for item in sample.next_questions) or "- Re-check the image carefully."
    return f"""Re-examine the same meme image for a researcher-oriented MemeAgent report.

Previous multimodal analysis:
{previous}

Controller follow-up questions:
{questions}

Optional context:
{sample.context or "None"}

Project rubric:
{PROJECT_RUBRIC}

Return a compact JSON object with updated evidence and these keys:
- answers_to_followup: list of direct answers
- corrected_ocr_text: visible text, or empty string
- updated_visual_evidence: concrete image evidence
- updated_harmfulness_analysis: harmfulness labels, target, evidence, uncertainty, and harmful/harmless judgement
- updated_sentiment_audience_intent_evolution: concise notes
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
        "confidence_threshold": confidence_threshold,
        "main_analysis_history": main_history,
        "previous_controller_decisions": controller_history,
    }
    return f"""Review the following meme analysis state and produce the final batch output.

{json.dumps(payload, ensure_ascii=False, indent=2)}

Project rubric:
{PROJECT_RUBRIC}

Decide whether the sample should be finalized. Finalize only when the analysis is clear enough or no useful visual follow-up remains. If not final, ask 1-5 specific questions for the multimodal model to answer in the next round.

When finalizing, write final_answer as a researcher-oriented Chinese MemeAgent report with exactly these sections:
1. Meme object and context
2. Visual/OCR evidence
3. Sentiment analysis
4. Harmfulness analysis
5. Audience and reception prediction
6. Intent recognition
7. Evolution tracking
8. Evidence map
9. Evidence gaps and overall confidence

The Harmfulness analysis section must explicitly state whether the meme is harmful or harmless.

Return exactly this JSON shape:
{{
  "should_finalize": true,
  "confidence": 0.0,
  "prediction_label": "harmful",
  "prediction_binary": 1,
  "harmfulness_labels": ["Offensive"],
  "harmfulness_analysis": "the Harmfulness analysis section only",
  "final_answer": "complete Chinese report with the 9 sections above",
  "reason": "short evidence-based reason",
  "next_questions": []
}}

prediction_label must be exactly one of: harmful, harmless, unclear.
prediction_binary must be 1 for harmful, 0 for harmless, and null for unclear.
"""


def _normalize_controller_decision(
    *,
    output: str,
    parsed: dict[str, Any],
    confidence_threshold: float,
    is_last_round: bool,
) -> dict[str, Any]:
    confidence = _parse_confidence(parsed.get("confidence"))
    final_answer = str(
        parsed.get("final_answer")
        or parsed.get("answer")
        or parsed.get("analysis")
        or parsed.get("summary")
        or ""
    ).strip()
    prediction_label = _normalize_prediction_label(
        parsed.get("prediction_label")
        or parsed.get("label")
        or parsed.get("final_decision")
        or parsed.get("judgement")
    )
    prediction_binary = _prediction_binary_from_values(
        parsed.get("prediction_binary"),
        prediction_label,
        final_answer,
        str(parsed.get("harmfulness_analysis") or ""),
    )
    harmfulness_analysis = str(
        parsed.get("harmfulness_analysis")
        or parsed.get("harmfulness")
        or _extract_harmfulness_section(final_answer)
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
    if not should_finalize and not questions:
        questions = [
            "Re-check OCR, concrete visual details, image-text relation, likely intent, sentiment, audience, and remaining uncertainty."
        ]

    return {
        "should_finalize": should_finalize,
        "confidence": confidence,
        "prediction_label": prediction_label,
        "prediction_binary": prediction_binary,
        "harmfulness_analysis": harmfulness_analysis,
        "harmfulness_labels": _normalize_string_list(parsed.get("harmfulness_labels")),
        "final_answer": final_answer or _fallback_final_answer(output, parsed),
        "reason": str(parsed.get("reason") or parsed.get("rationale") or "").strip(),
        "next_questions": questions,
    }


def _apply_controller_decision(sample: SampleState, decision: dict[str, Any]) -> None:
    sample.final_answer = str(decision.get("final_answer") or "")
    sample.prediction_label = str(decision.get("prediction_label") or "unclear")
    sample.prediction_binary = decision.get("prediction_binary")
    sample.harmfulness_analysis = str(decision.get("harmfulness_analysis") or "")
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
        sample.final_answer = str(last_decision.get("final_answer") or "")
        sample.prediction_label = str(last_decision.get("prediction_label") or "unclear")
        sample.prediction_binary = last_decision.get("prediction_binary")
        sample.harmfulness_analysis = str(last_decision.get("harmfulness_analysis") or "")
        sample.final_confidence = last_decision.get("confidence")
        sample.final_reason = str(last_decision.get("reason") or "max rounds reached")
    elif sample.main_outputs:
        text = _latest_output(sample.main_outputs)
        sample.final_answer = _fallback_final_answer(text, _extract_json_object(text))
        sample.harmfulness_analysis = _extract_harmfulness_section(sample.final_answer)
        sample.prediction_label = _normalize_prediction_label(None)
        sample.prediction_binary = _prediction_binary_from_values(
            None,
            sample.prediction_label,
            sample.final_answer,
            sample.harmfulness_analysis,
        )
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
        "harmfulness_analysis": sample.harmfulness_analysis,
        "final_answer": sample.final_answer,
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
        "gold_harmful_count": sum(1 for item in records if item["gold_binary"] == 1),
        "gold_harmless_count": sum(1 for item in records if item["gold_binary"] == 0),
        "gold_unknown_count": sum(1 for item in records if item["gold_binary"] is None),
        "prediction_harmful_count": sum(
            1 for item in records if item["prediction_binary"] == 1
        ),
        "prediction_harmless_count": sum(
            1 for item in records if item["prediction_binary"] == 0
        ),
        "prediction_unknown_count": sum(
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
        r"(?:^|\n)\s*(?:4[.\)]\s*)?(?:Harmfulness analysis|有害性分析)[:：]?\s*(.*?)(?=\n\s*(?:5[.\)]|Audience and reception prediction|受众|Intent recognition|意图|Evolution tracking|演化|Evidence map|证据地图)\b|\Z)",
        r"(?:^|\n)\s*(?:Harmfulness|有害性)[:：]\s*(.*?)(?=\n\s*(?:Audience|受众|Intent|意图|Evolution|演化|Evidence|证据)\b|\Z)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I | re.S)
        if match:
            return match.group(1).strip()
    return ""


def _normalize_prediction_label(value: Any) -> str:
    if value is None or value == "":
        return "unclear"
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"harmful", "unsafe", "toxic", "offensive"}:
        return "harmful"
    if normalized in {"harmless", "not_harmful", "non_harmful", "nonharmful", "safe"}:
        return "harmless"
    if normalized in {"unclear", "unknown", "ambiguous"}:
        return "unclear"
    if re.search(r"\b(?:harmless|not[_\s-]*harmful|non[_\s-]*harmful|nonharmful)\b", str(value), flags=re.I):
        return "harmless"
    if re.search(r"\bharmful\b", str(value), flags=re.I):
        return "harmful"
    return "unclear"


def _prediction_binary_from_values(
    explicit_binary: Any,
    label: str,
    final_answer: str,
    harmfulness_analysis: str,
) -> int | None:
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

    normalized_label = _normalize_prediction_label(label)
    if normalized_label == "harmful":
        return 1
    if normalized_label == "harmless":
        return 0

    text = f"{harmfulness_analysis}\n{final_answer}".lower().replace("-", "_")
    if re.search(r"\b(?:prediction_label|final_decision|judgement|label)\s*(?:is|:|：)?\s*(?:harmless|not[_\s]*harmful|non[_\s]*harmful|nonharmful)\b", text, flags=re.S):
        return 0
    if re.search(r"\b(?:prediction_label|final_decision|judgement|label)\s*(?:is|:|：)?\s*harmful\b", text, flags=re.S):
        return 1
    if re.search(r"\b(?:harmless|not[_\s]*harmful|non[_\s]*harmful|nonharmful)\b", text, flags=re.S):
        return 0
    if re.search(r"\bharmful\b", text, flags=re.S):
        return 1
    return None


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
