from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_SOURCE = Path("/data/ggbond/cache_10.160.8.205/FHM_data/train.jsonl")
DEFAULT_OUTPUT = Path(
    "/data/ggbond/cache_10.160.8.205/FHM_data/train_trajectory.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize the FHM training split for trajectory sampling."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def normalize(row: dict[str, Any], *, line_no: int, image_root: Path) -> dict[str, Any]:
    sample_id = row.get("id")
    image = str(row.get("img") or "").strip()
    text = str(row.get("text") or "").strip()
    label = row.get("label")

    if sample_id is None:
        raise ValueError(f"Line {line_no}: missing id")
    if not image:
        raise ValueError(f"Line {line_no}: missing img")
    if label not in {0, 1}:
        raise ValueError(f"Line {line_no}: label must be 0 or 1, got {label!r}")
    if not (image_root / image).is_file():
        raise FileNotFoundError(f"Line {line_no}: image not found: {image_root / image}")

    judgement = "harmful" if label == 1 else "harmless"
    return {
        "id": str(sample_id),
        "image": image,
        "text": text,
        "gold_binary": label,
        "gold_judgement": f"JUDGEMENT: {judgement}",
        "source": "FHM/train",
    }


def main() -> int:
    args = parse_args()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    image_root = source.parent
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    label_counts = {0: 0, 1: 0}

    with source.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError(f"Line {line_no}: expected a JSON object")
            record = normalize(raw, line_no=line_no, image_root=image_root)
            if record["id"] in seen_ids:
                raise ValueError(f"Line {line_no}: duplicate id {record['id']!r}")
            seen_ids.add(record["id"])
            label_counts[record["gold_binary"]] += 1
            records.append(record)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} samples to {output}")
    print(f"harmless={label_counts[0]} harmful={label_counts[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
