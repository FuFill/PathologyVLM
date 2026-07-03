"""Show model output and CSV fields for one Quilt-1M image.

This script is meant for quick inspection, not report generation.
It prints:

* the model row from ``outputs/vlm_outputs.jsonl`` or ``outputs/vlm_outputs.csv``
* the full row from ``quilt_1M_lookup.csv``
* a compact side-by-side summary for one specific image

Select the image by exact basename, image_id, or full path fragment.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_JSONL = PROJECT_ROOT / "outputs" / "vlm_outputs.jsonl"
DEFAULT_MODEL_CSV = PROJECT_ROOT / "outputs" / "vlm_outputs.csv"
DEFAULT_LOOKUP_CSV = PROJECT_ROOT / "quilt_1M_lookup.csv"

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def _norm(text: object) -> str:
    return str(text).replace("\\", "/").strip().lower()


def _name(text: object) -> str:
    return Path(_norm(text)).name


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fin:
        return list(csv.DictReader(fin))


def _find_row(rows: list[dict[str, Any]], query: str, keys: tuple[str, ...]) -> dict[str, Any] | None:
    qn = _name(query)
    q = _norm(query)
    for row in rows:
        for key in keys:
            value = row.get(key, "")
            if not value:
                continue
            nv = _norm(value)
            if nv == q or _name(value) == qn or q in nv:
                return row
    return None


def _print_kv(title: str, row: dict[str, Any]) -> None:
    print(f"\n== {title} ==")
    print(json.dumps(row, ensure_ascii=False, indent=2))


def _print_pretty_json(title: str, row: dict[str, Any]) -> None:
    print(f"\n== {title} (JSON) ==")
    print(json.dumps(row, ensure_ascii=False, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser(description="Inspect one Quilt-1M image's outputs.")
    ap.add_argument(
        "--image",
        required=True,
        help="Image basename, image_id, or path fragment to inspect.",
    )
    ap.add_argument(
        "--model-jsonl",
        default=str(DEFAULT_MODEL_JSONL),
        help="Path to outputs/vlm_outputs.jsonl.",
    )
    ap.add_argument(
        "--model-csv",
        default=str(DEFAULT_MODEL_CSV),
        help="Path to outputs/vlm_outputs.csv.",
    )
    ap.add_argument(
        "--lookup-csv",
        default=str(DEFAULT_LOOKUP_CSV),
        help="Path to quilt_1M_lookup.csv.",
    )
    args = ap.parse_args()

    model_jsonl = Path(args.model_jsonl)
    model_csv = Path(args.model_csv)
    lookup_csv = Path(args.lookup_csv)

    for path, label in ((model_jsonl, "model JSONL"), (model_csv, "model CSV"), (lookup_csv, "lookup CSV")):
        if not path.exists():
            print(f"[inspect_quilt_1m_image] ERROR: {label} not found: {path}", file=sys.stderr)
            return 1

    jsonl_rows = _load_jsonl(model_jsonl)
    csv_rows = _load_csv(model_csv)
    lookup_rows = _load_csv(lookup_csv)

    model_row = _find_row(
        jsonl_rows,
        args.image,
        keys=("image_id", "image_path", "image_name"),
    )
    if model_row is None:
        print(f"[inspect_quilt_1m_image] ERROR: image not found in model JSONL: {args.image}", file=sys.stderr)
        return 1

    image_path = model_row.get("image_path", "")
    image_id = model_row.get("image_id", "")
    lookup_row = _find_row(
        lookup_rows,
        image_path or image_id or args.image,
        keys=("image_path", "path", "image_id"),
    )
    csv_row = _find_row(
        csv_rows,
        image_path or image_id or args.image,
        keys=("image_id", "image_path"),
    )

    print(f"[inspect_quilt_1m_image] image: {args.image}")
    print(f"[inspect_quilt_1m_image] model image_id: {image_id}")
    print(f"[inspect_quilt_1m_image] model image_path: {image_path}")

    _print_pretty_json("Model output row", model_row)
    if csv_row is not None:
        _print_kv("Model CSV row (all fields)", csv_row)
    else:
        print("\n== Model CSV row ==")
        print("(not found)")

    if lookup_row is not None:
        _print_kv("Quilt-1M lookup row (all fields)", lookup_row)
    else:
        print("\n== Quilt-1M lookup row ==")
        print("(not found)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
