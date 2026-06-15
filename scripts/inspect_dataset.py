"""Inspect a PathGen-1.6M (jamessyx/PathGen) metadata JSON.

PathGen-1.6M ships a single gated JSON file at
``https://huggingface.co/datasets/jamessyx/PathGen``. The file is a flat
list of records of the form::

    {
      "wsi_id":   "TCGA-AA-3844-01Z-00-DX1.<uuid>",
      "position": ["35136", "33344"],
      "caption":  "...",
      "file_id":  "<gdc-file-uuid>"
    }

This script loads the JSON, prints its size, the set of top-level keys
seen in the first N rows, and a small preview. It does not call the
``datasets`` library (the file is gated and trivially read with stdlib
``json``).

Usage
-----
Bash::

    python scripts/inspect_dataset.py --pathgen_json /path/to/PathGen-1.6M.json

Windows PowerShell::

    python scripts/inspect_dataset.py --pathgen_json .\\PathGen-1.6M.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, List

DEFAULT_DATASET = "jamessyx/PathGen"
PREVIEW_MAX_LEN = 300
HEAD_SIZE = 50


def _summarize_value(value: Any) -> str:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return f"<bytes len={len(value)}>"
    if isinstance(value, list):
        if not value:
            return "[]"
        first = _summarize_value(value[0])
        return f"<list len={len(value)} first={first}>"
    s = repr(value)
    if len(s) > PREVIEW_MAX_LEN:
        s = s[:PREVIEW_MAX_LEN] + "..."
    return s


def _load_pathgen_json(path: Path) -> List[dict]:
    # utf-8-sig transparently strips a BOM if present (some redistributions
    # of PathGen-1.6M.json saved on Windows include one); falls back to
    # plain utf-8 behavior otherwise.
    with path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for k in ("data", "samples", "items"):
            if isinstance(data.get(k), list):
                return data[k]
    raise ValueError(
        f"Unexpected JSON shape in {path}: top-level {type(data).__name__}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect a PathGen-1.6M JSON.")
    parser.add_argument(
        "--pathgen_json",
        required=True,
        help="Path to PathGen-1.6M.json (download manually from the gated HF dataset).",
    )
    parser.add_argument(
        "--dataset_name",
        default=DEFAULT_DATASET,
        help="HF dataset id, printed for reference only.",
    )
    args = parser.parse_args()

    path = Path(args.pathgen_json)
    if not path.exists():
        print(
            f"[inspect_dataset] ERROR: file not found: {path}\n"
            f"Download {args.dataset_name} first (gated):\n"
            f"  1. Open https://huggingface.co/datasets/{args.dataset_name} and accept the terms.\n"
            f"  2. Run: hf download {args.dataset_name} PathGen-1.6M.json "
            f"--repo-type=dataset --local-dir .",
            file=sys.stderr,
        )
        return 1

    print(f"[inspect_dataset] Dataset id   : {args.dataset_name}")
    print(f"[inspect_dataset] JSON file    : {path}")
    print(f"[inspect_dataset] File size    : {path.stat().st_size:,} bytes")

    try:
        rows = _load_pathgen_json(path)
    except Exception as exc:  # noqa: BLE001
        print(f"[inspect_dataset] ERROR loading JSON: {exc}", file=sys.stderr)
        return 1

    print(f"[inspect_dataset] Top-level    : list[{len(rows)}]")
    if not rows:
        return 0

    head = [r for r in rows[:HEAD_SIZE] if isinstance(r, dict)]
    key_counts: Counter[str] = Counter()
    for r in head:
        key_counts.update(r.keys())
    print(f"[inspect_dataset] Keys in first {len(head)} rows (key: count):")
    for k, c in key_counts.most_common():
        print(f"  - {k}: {c}")

    sample = head[0] if head else rows[0]
    if not isinstance(sample, dict):
        print(f"[inspect_dataset] First sample is not a dict: {_summarize_value(sample)}")
        return 0
    print(f"[inspect_dataset] First sample keys: {list(sample.keys())}")
    print("[inspect_dataset] First sample preview:")
    for k, v in sample.items():
        print(f"  - {k}: {_summarize_value(v)}")

    distinct_file_ids = {r.get("file_id") for r in head if isinstance(r, dict)}
    distinct_file_ids.discard(None)
    print(
        f"[inspect_dataset] Distinct file_id in first {len(head)} rows: "
        f"{len(distinct_file_ids)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
