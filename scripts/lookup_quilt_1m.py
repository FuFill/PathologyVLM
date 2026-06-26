"""Look up Quilt-1M metadata for images in ``data/quilt-1m``.

The repository includes a very large CSV index named
``quilt_1M_lookup.csv``. This script streams that file line by line so it
can resolve metadata for one or more image filenames without loading the
entire 2.2 GB table into memory.

The lookup is performed by image basename because the CSV stores entries
like ``example_123.jpg`` rather than full directory paths.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Iterable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = PROJECT_ROOT / "quilt_1M_lookup.csv"
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}
DISPLAY_FIELDS = [
    "caption",
    "subset",
    "split",
    "pathology",
    "roi_text",
    "corrected_text",
    "magnification",
    "height",
    "width",
    "not_histology",
    "single_wsi",
]


def _normalize_name(value: object) -> str:
    return Path(str(value).replace("\\", "/")).name.lower()


def _truncate(value: object, limit: int = 220) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\r", " ").replace("\n", " ").strip()
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def _collect_targets(image_dir: Path, limit: int | None) -> list[Path]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    if not image_dir.is_dir():
        raise NotADirectoryError(f"Image directory is not a folder: {image_dir}")
    results = [
        p for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    ]
    results.sort(key=lambda p: str(p).lower())
    if limit is not None and limit > 0:
        results = results[:limit]
    return results


def _iter_query_names(args: argparse.Namespace) -> list[str]:
    names: list[str] = []
    if args.image:
        names.extend(args.image)
    if args.image_dir:
        targets = _collect_targets(Path(args.image_dir), args.max_images)
        names.extend(str(p) for p in targets)
    return names


def _print_text(matches_by_name: dict[str, list[dict]], query_names: list[str]) -> None:
    for query in query_names:
        key = _normalize_name(query)
        matches = matches_by_name.get(key, [])
        print(f"\n{query}")
        if not matches:
            print("  (no match)")
            continue
        for idx, row in enumerate(matches, start=1):
            prefix = f"  [{idx}]"
            print(f"{prefix} image_path: {row.get('image_path', '')}")
            for field in DISPLAY_FIELDS:
                value = row.get(field, "")
                if value in ("", None):
                    continue
                print(f"{prefix} {field}: {_truncate(value)}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Look up Quilt-1M metadata for images in data/quilt-1m."
    )
    parser.add_argument(
        "--csv",
        default=str(DEFAULT_CSV),
        help="Path to quilt_1M_lookup.csv.",
    )
    parser.add_argument(
        "--image_dir",
        default="",
        help="Local folder containing images to look up, e.g. data/quilt-1m.",
    )
    parser.add_argument(
        "--image",
        action="append",
        default=[],
        help="Image filename or path to look up. Can be repeated.",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=0,
        help="Optional limit when scanning an image directory.",
    )
    parser.add_argument(
        "--jsonl",
        action="store_true",
        help="Print each matched row as JSONL instead of human-readable text.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"[lookup_quilt_1m] ERROR: CSV not found: {csv_path}", file=sys.stderr)
        return 1

    query_names = _iter_query_names(args)
    if not query_names:
        print(
            "[lookup_quilt_1m] ERROR: provide --image_dir or at least one --image.",
            file=sys.stderr,
        )
        return 1

    targets = {_normalize_name(name) for name in query_names}
    matches: DefaultDict[str, list[dict]] = defaultdict(list)
    pending = set(targets)

    with csv_path.open("r", encoding="utf-8-sig", newline="") as fin:
        reader = csv.DictReader(fin)
        if reader.fieldnames is None:
            print(f"[lookup_quilt_1m] ERROR: CSV has no header: {csv_path}", file=sys.stderr)
            return 1

        for row in reader:
            name = _normalize_name(row.get("image_path", ""))
            if name in pending:
                matches[name].append(row)
                pending.discard(name)
                if not pending:
                    break

    missing = [name for name in query_names if _normalize_name(name) not in matches]

    if args.jsonl:
        for query in query_names:
            key = _normalize_name(query)
            for row in matches.get(key, []):
                payload = {"query": query, **row}
                print(json.dumps(payload, ensure_ascii=False))
    else:
        _print_text(matches, query_names)

    if missing:
        print("\nMissing:", file=sys.stderr)
        for name in missing:
            print(f"  - {name}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
