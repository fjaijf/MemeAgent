from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from threading import local
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from memeagent.agent import MemeAgent, _normalize_content
from memeagent.config import MemeAgentConfig, load_project_env
from memeagent.llm import create_controller_llm, create_llm
from memeagent.memory import MemeMemoryStore
from memeagent.search_agent import SearchAgentConfig, WebSearchAgent
from memeagent.workflow import MemeResearchWorkflow


DEFAULT_ID_FIELDS = ("id", "image_id", "uid", "name", "filename", "file_name")
DEFAULT_IMAGE_FIELDS = (
    "image",
    "image_path",
    "img",
    "img_path",
    "path",
    "file",
    "filename",
)
DEFAULT_TEXT_FIELDS = ("text", "caption", "ocr", "context", "description")
DEFAULT_LABEL_FIELDS = ("label", "harmful", "is_harmful", "target", "class")

POSITIVE_LABELS = {
    "1",
    "true",
    "yes",
    "y",
    "harmful",
    "hateful",
    "offensive",
    "toxic",
    "abusive",
    "dangerous",
    "potentially_harmful",
    "potentially harmful",
}
NEGATIVE_LABELS = {
    "0",
    "false",
    "no",
    "n",
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

LABEL_TEST_SYSTEM_PROMPT = (
    "You are a strict binary meme safety classifier. "
    "Look only at the attached image and output exactly one label: A for harmless, "
    "B for harmful. No explanation."
)


@dataclass(frozen=True)
class EvalItem:
    sample_id: str
    image_path: str
    label: int | None
    context: str
    raw: dict[str, Any]


@dataclass(frozen=True)
class Prediction:
    label: str
    binary: int | None
    confidence: str
    rationale: str
    raw_output: str
    parsed: dict[str, Any]


_THREAD_STATE = local()


def _thread_direct_agent(config: MemeAgentConfig) -> MemeAgent:
    agent = getattr(_THREAD_STATE, "agent", None)
    if agent is None:
        agent = MemeAgent(llm=create_llm(config), system_prompt=config.system_prompt)
        _THREAD_STATE.agent = agent
    return agent


def _first_present(row: dict[str, Any], fields: tuple[str, ...]) -> Any:
    lowered = {str(key).lower(): key for key in row}
    for field in fields:
        key = lowered.get(field.lower())
        if key is not None and row.get(key) not in {None, ""}:
            return row[key]
    return None


def _get_field(row: dict[str, Any], field_name: str | None, fallback: tuple[str, ...]) -> Any:
    if field_name:
        lowered = {str(key).lower(): key for key in row}
        key = lowered.get(field_name.lower())
        if key is not None and row.get(key) not in {None, ""}:
            return row[key]
        return None
    return _first_present(row, fallback)


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
    return None


def _read_json_dataset(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        for key in ("data", "items", "samples", "annotations", "test"):
            value = data.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    raise ValueError(f"Could not find a list of samples in {path}")


def _read_jsonl_dataset(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"Line {line_no} in {path} is not a JSON object")
            rows.append(item)
    return rows


def _read_csv_dataset(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _detect_schema(rows: list[dict[str, Any]], requested_schema: str) -> str:
    if requested_schema != "auto":
        return requested_schema
    if rows:
        keys = {str(key).lower() for key in rows[0]}
        if {"messages", "images", "solution"}.issubset(keys):
            return "label_test"
    return "generic"


def _extract_label_test_prompt(row: dict[str, Any]) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return ""
    for message in messages:
        if not isinstance(message, dict):
            continue
        if str(message.get("role", "")).lower() != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
    return ""


def _resolve_image_path(raw_path: Any, dataset_path: Path, image_root: Path | None) -> Path:
    raw = str(raw_path).strip().replace("\\", "/")
    if not raw:
        return Path(raw)

    base_root = image_root or dataset_path.parent
    if "/Dataset/" in raw and image_root is None:
        suffix = raw.split("/Dataset/", 1)[1]
        candidate = dataset_path.parent / suffix
    else:
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = base_root / candidate

    if candidate.exists():
        return candidate

    normalized = str(candidate).replace("\\", "/")
    if "/MET/Img/" in normalized:
        match = re.fullmatch(r"(Image-|image_)(\d+)(\.[A-Za-z0-9]+)", candidate.name)
        if match:
            met_candidate = candidate.with_name(
                f"{match.group(1)} ({match.group(2)}){match.group(3)}"
            )
            if met_candidate.exists():
                return met_candidate
    return candidate


def _load_label_test_items(
    rows: list[dict[str, Any]],
    dataset_path: Path,
    image_root: Path | None,
) -> list[EvalItem]:
    items: list[EvalItem] = []
    for index, row in enumerate(rows):
        images = row.get("images")
        if not isinstance(images, list) or not images:
            raise ValueError(f"Sample {index} has no images list.")

        image_path = _resolve_image_path(images[0], dataset_path, image_root)
        solution = row.get("solution")
        if solution in {None, ""}:
            messages = row.get("messages")
            if isinstance(messages, list) and messages:
                for message in reversed(messages):
                    if isinstance(message, dict) and message.get("role") == "assistant":
                        solution = message.get("content")
                        break

        items.append(
            EvalItem(
                sample_id=str(row.get("id") or image_path),
                image_path=str(image_path),
                label=_normalize_label(solution),
                context=_extract_label_test_prompt(row),
                raw=row,
            )
        )
    return items


def load_dataset(
    dataset_path: Path,
    image_root: Path | None,
    id_field: str | None,
    image_field: str | None,
    label_field: str | None,
    text_field: str | None,
    schema: str,
) -> list[EvalItem]:
    suffix = dataset_path.suffix.lower()
    if suffix == ".json":
        rows = _read_json_dataset(dataset_path)
    elif suffix in {".jsonl", ".ndjson"}:
        rows = _read_jsonl_dataset(dataset_path)
    elif suffix == ".csv":
        rows = _read_csv_dataset(dataset_path)
    else:
        raise ValueError("Dataset must be .json, .jsonl, or .csv")

    detected_schema = _detect_schema(rows, schema)
    if detected_schema == "label_test":
        return _load_label_test_items(rows, dataset_path, image_root)

    root = image_root or dataset_path.parent
    items: list[EvalItem] = []
    for index, row in enumerate(rows):
        sample_id_value = _get_field(row, id_field, DEFAULT_ID_FIELDS)
        sample_id = str(sample_id_value or index)
        image_value = _get_field(row, image_field, DEFAULT_IMAGE_FIELDS)
        if image_value in {None, ""}:
            raise ValueError(
                f"Sample {sample_id} has no image path. Pass --image-field explicitly."
            )
        image_path = Path(str(image_value)).expanduser()
        if not image_path.is_absolute():
            image_path = root / image_path

        label_value = _get_field(row, label_field, DEFAULT_LABEL_FIELDS)
        context_value = _get_field(row, text_field, DEFAULT_TEXT_FIELDS)
        items.append(
            EvalItem(
                sample_id=sample_id,
                image_path=str(image_path),
                label=_normalize_label(label_value),
                context=str(context_value or ""),
                raw=row,
            )
        )
    return items


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.S)
    if fenced:
        stripped = fenced.group(1)
    else:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start != -1 and end != -1 and end > start:
            stripped = stripped[start : end + 1]
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _prediction_binary(label: Any) -> int | None:
    if label is None:
        return None
    normalized = str(label).strip().lower().replace("-", "_")
    if normalized in {"a", "harmless"}:
        return 0
    if normalized in {"b", "harmful"}:
        return 1
    if normalized in {"potentially_harmful"}:
        return 1
    if normalized in {"not_harmful", "unclear"}:
        return 0
    return _normalize_label(normalized)


def _prediction_from_text(output: str) -> Prediction:
    parsed = _extract_json_object(output)
    label = str(parsed.get("harmfulness_label") or parsed.get("label") or "").strip()
    confidence = str(parsed.get("confidence") or "unknown").strip()
    rationale = str(parsed.get("rationale") or parsed.get("reason") or "").strip()

    boxed = re.search(r"\\boxed\{\s*([AB])\s*\}", output, flags=re.I)
    if boxed:
        label = boxed.group(1).upper()

    if not label:
        match = re.search(
            r"harmfulness_label\s*[:：]\s*([A-Za-z_ -]+)",
            output,
            flags=re.I,
        )
        if match:
            label = match.group(1).strip().split()[0]

    if not label:
        stripped = output.strip()
        if re.fullmatch(r"[AB]", stripped, flags=re.I):
            label = stripped.upper()

    if not label:
        compact = output.strip().lower().replace("-", "_")
        if "nonharmful" in compact or "non_harmful" in compact or "harmless" in compact:
            label = "nonharmful"
        elif "harmful" in compact:
            label = "harmful"

    return Prediction(
        label=label or "unknown",
        binary=_prediction_binary(label),
        confidence=confidence,
        rationale=rationale,
        raw_output=output,
        parsed=parsed,
    )


def predict_direct(agent: MemeAgent, item: EvalItem) -> Prediction:
    prompt = item.context.strip() or (
        "Please judge if the image is harmful: A. harmless B. harmful. "
        "Please output strictly in the form of \\boxed{A} or \\boxed{B}:"
    )
    messages = [
        SystemMessage(content=LABEL_TEST_SYSTEM_PROMPT),
        HumanMessage(
            content=agent._build_user_content_with_images(
                prompt,
                [item.image_path],
                [],
            )
        ),
    ]
    response = agent.llm.invoke(messages)
    output = _normalize_content(getattr(response, "content", response))
    return _prediction_from_text(output)


def make_workflow(config: MemeAgentConfig, project_root: Path, disable_memory: bool) -> MemeResearchWorkflow:
    llm = create_llm(config)
    agent = MemeAgent(llm=llm, system_prompt=config.system_prompt)
    controller_llm = create_controller_llm(config)
    controller_agent = (
        MemeAgent(llm=controller_llm, system_prompt=config.system_prompt)
        if controller_llm is not None
        else None
    )

    cache_dir = Path(config.cache_dir).expanduser()
    if not cache_dir.is_absolute():
        cache_dir = project_root / cache_dir
    memory_dir = Path(config.memory_dir).expanduser()
    if not memory_dir.is_absolute():
        memory_dir = project_root / memory_dir

    memory_store = None
    if config.memory_enabled and not disable_memory:
        memory_store = MemeMemoryStore(memory_dir / "memory.sqlite3")

    return MemeResearchWorkflow(
        meme_agent=agent,
        controller_agent=controller_agent,
        search_agent=WebSearchAgent(
            SearchAgentConfig(
                search_provider=config.search_provider,
                search_api_key=config.search_api_key,
                tavily_api_key=config.tavily_api_key,
                zhihu_api_key=config.zhihu_api_key,
                glm_search_api_key=config.glm_search_api_key,
                glm_search_engine=config.glm_search_engine,
                glm_search_recency_filter=config.glm_search_recency_filter,
                glm_search_content_size=config.glm_search_content_size,
                glm_search_domain_filter=config.glm_search_domain_filter,
                search_proxy=config.search_proxy,
                search_max_results=config.search_max_results,
                news_max_results=config.news_max_results,
                search_timeout=config.search_timeout,
                search_country=config.search_country,
                search_lang=config.search_lang,
                search_context_sites=config.search_context_sites,
                tavily_search_depth=config.tavily_search_depth,
                cache_enabled=config.cache_enabled,
                search_cache_path=str(cache_dir / "search.sqlite3"),
                search_cache_ttl_seconds=config.search_cache_ttl_seconds,
                news_cache_ttl_seconds=config.news_cache_ttl_seconds,
            )
        ),
        memory_store=memory_store,
        memory_recall_limit=config.memory_recall_limit,
    )


def predict_workflow(
    workflow: MemeResearchWorkflow,
    item: EvalItem,
    use_search: bool,
    force_search: bool,
) -> Prediction:
    result = workflow.run_heads(
        topic=item.sample_id,
        context=item.context,
        image_paths=[item.image_path],
        task_heads=["harmfulness"],
        use_search=use_search,
        force_search=force_search,
        progress=None,
    )
    return _prediction_from_text(result.formatted_output)


def _read_completed_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    completed: set[str] = set()
    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            sample_id = item.get("id")
            if sample_id is not None:
                completed.add(str(sample_id))
    return completed


def _update_counts(counts: dict[str, int], truth: int | None, pred: int | None) -> None:
    if truth is None or pred is None:
        counts["unknown"] += 1
        return
    if truth == 1 and pred == 1:
        counts["tp"] += 1
    elif truth == 0 and pred == 0:
        counts["tn"] += 1
    elif truth == 0 and pred == 1:
        counts["fp"] += 1
    elif truth == 1 and pred == 0:
        counts["fn"] += 1


def _metrics(counts: dict[str, int]) -> dict[str, Any]:
    tp = counts["tp"]
    tn = counts["tn"]
    fp = counts["fp"]
    fn = counts["fn"]
    evaluated = tp + tn + fp + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / evaluated if evaluated else 0.0
    return {
        "evaluated": evaluated,
        "unknown": counts["unknown"],
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": round(accuracy, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
    }


def evaluate_direct_sample(
    config: MemeAgentConfig,
    item: EvalItem,
) -> tuple[Prediction, str | None, float]:
    started_sample = time.time()
    try:
        prediction = predict_direct(_thread_direct_agent(config), item)
        error = None
    except Exception as exc:
        error = str(exc)
        prediction = Prediction(
            label="error",
            binary=None,
            confidence="unknown",
            rationale="",
            raw_output="",
            parsed={},
        )
    return prediction, error, round(time.time() - started_sample, 3)


def build_record(
    item: EvalItem,
    prediction: Prediction,
    error: str | None,
    latency_seconds: float,
) -> dict[str, Any]:
    return {
        "id": item.sample_id,
        "image": item.image_path,
        "truth": item.label,
        "prediction": prediction.binary,
        "prediction_label": prediction.label,
        "confidence": prediction.confidence,
        "rationale": prediction.rationale,
        "latency_seconds": latency_seconds,
        "error": error,
        "raw_output": prediction.raw_output,
        "parsed": prediction.parsed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate MemeAgent on a harmful meme detection dataset."
    )
    parser.add_argument("--dataset", required=True, help="Path to .json, .jsonl, or .csv dataset.")
    parser.add_argument("--image-root", default=None, help="Root directory for relative image paths.")
    parser.add_argument("--output", default="eval_results.jsonl", help="Per-sample JSONL output.")
    parser.add_argument("--summary", default="eval_summary.json", help="Summary metrics JSON output.")
    parser.add_argument("--mode", choices=["direct", "workflow"], default="direct")
    parser.add_argument(
        "--schema",
        choices=["auto", "label_test", "generic"],
        default="auto",
        help="Dataset schema. auto detects messages/images/solution label_test format.",
    )
    parser.add_argument("--id-field", default=None)
    parser.add_argument("--image-field", default=None)
    parser.add_argument("--label-field", default=None)
    parser.add_argument("--text-field", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--resume", action="store_true", help="Skip ids already present in --output.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output and summary files.")
    parser.add_argument("--search", action="store_true", help="Enable retrieval in workflow mode.")
    parser.add_argument("--force-search", action="store_true", help="Force retrieval in workflow mode.")
    parser.add_argument("--disable-memory", action="store_true", help="Do not read/write local memory.")
    parser.add_argument(
        "--include-unlabeled",
        action="store_true",
        help="Also run samples whose ground-truth label cannot be parsed.",
    )
    parser.add_argument("--sleep", type=float, default=0.0, help="Delay between samples in seconds.")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel workers for direct mode. Use 1 for safest rate-limit behavior.",
    )
    return parser.parse_args()


def main() -> int:
    project_root = Path(__file__).resolve().parent
    load_project_env(project_root)
    args = parse_args()

    dataset_path = Path(args.dataset).expanduser()
    output_path = Path(args.output).expanduser()
    summary_path = Path(args.summary).expanduser()
    image_root = Path(args.image_root).expanduser() if args.image_root else None

    if args.overwrite:
        for path in (output_path, summary_path):
            if path.exists():
                path.unlink()

    items = load_dataset(
        dataset_path=dataset_path,
        image_root=image_root,
        id_field=args.id_field,
        image_field=args.image_field,
        label_field=args.label_field,
        text_field=args.text_field,
        schema=args.schema,
    )
    if args.offset:
        items = items[args.offset :]
    if args.limit is not None:
        items = items[: args.limit]

    completed = _read_completed_ids(output_path) if args.resume else set()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    config = MemeAgentConfig.from_env()
    workflow = None
    if args.mode == "workflow":
        workflow = make_workflow(config, project_root, disable_memory=args.disable_memory)
        if args.workers > 1:
            print("--workers is only used in direct mode; workflow mode will run sequentially.")

    counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "unknown": 0}
    processed = 0
    started_at = time.time()

    with output_path.open("a", encoding="utf-8") as output:
        runnable_direct_items: list[tuple[int, EvalItem]] = []
        for index, item in enumerate(items, start=1):
            if item.sample_id in completed:
                print(f"[{index}/{len(items)}] skip {item.sample_id} (resume)")
                continue
            if item.label is None and not args.include_unlabeled:
                print(f"[{index}/{len(items)}] skip {item.sample_id} (unlabeled)")
                continue
            if not Path(item.image_path).exists():
                print(f"[{index}/{len(items)}] missing image: {item.image_path}")
                record = {
                    "id": item.sample_id,
                    "image": item.image_path,
                    "truth": item.label,
                    "error": "image_not_found",
                }
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                output.flush()
                counts["unknown"] += 1
                continue

            if args.mode == "direct" and args.workers > 1:
                runnable_direct_items.append((index, item))
                continue

            print(f"[{index}/{len(items)}] evaluating {item.sample_id}")
            started_sample = time.time()
            error = None
            try:
                if args.mode == "workflow":
                    assert workflow is not None
                    prediction = predict_workflow(
                        workflow,
                        item,
                        use_search=args.search or args.force_search,
                        force_search=args.force_search,
                    )
                else:
                    prediction = predict_direct(_thread_direct_agent(config), item)
            except Exception as exc:
                error = str(exc)
                prediction = Prediction(
                    label="error",
                    binary=None,
                    confidence="unknown",
                    rationale="",
                    raw_output="",
                    parsed={},
                )

            _update_counts(counts, item.label, prediction.binary)
            processed += 1
            record = build_record(
                item=item,
                prediction=prediction,
                error=error,
                latency_seconds=round(time.time() - started_sample, 3),
            )
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()

            current_metrics = _metrics(counts)
            print(
                "    pred={pred} truth={truth} f1={f1:.4f} acc={acc:.4f}".format(
                    pred=prediction.binary,
                    truth=item.label,
                    f1=current_metrics["f1"],
                    acc=current_metrics["accuracy"],
                )
            )
            if args.sleep > 0:
                time.sleep(args.sleep)

        if runnable_direct_items:
            total_runnable = len(runnable_direct_items)
            max_workers = max(1, args.workers)
            print(f"Running direct evaluation with {max_workers} workers.")
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_map = {
                    executor.submit(evaluate_direct_sample, config, item): (index, item)
                    for index, item in runnable_direct_items
                }
                for done_count, future in enumerate(as_completed(future_map), start=1):
                    index, item = future_map[future]
                    prediction, error, latency_seconds = future.result()
                    _update_counts(counts, item.label, prediction.binary)
                    processed += 1
                    record = build_record(
                        item=item,
                        prediction=prediction,
                        error=error,
                        latency_seconds=latency_seconds,
                    )
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    output.flush()

                    current_metrics = _metrics(counts)
                    print(
                        "[{done}/{total}] {sample_id} pred={pred} truth={truth} "
                        "f1={f1:.4f} acc={acc:.4f}".format(
                            done=done_count,
                            total=total_runnable,
                            sample_id=item.sample_id,
                            pred=prediction.binary,
                            truth=item.label,
                            f1=current_metrics["f1"],
                            acc=current_metrics["accuracy"],
                        )
                    )
                    if args.sleep > 0:
                        time.sleep(args.sleep)

    summary = {
        "dataset": str(dataset_path),
        "mode": args.mode,
        "schema": args.schema,
        "total_loaded": len(items),
        "processed_this_run": processed,
        "elapsed_seconds": round(time.time() - started_at, 3),
        **_metrics(counts),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
