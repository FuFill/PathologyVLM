"""Build a stratified subset of the CAMELYON16 ABMIL patch set.

The full ABMIL export (``data/vlm_patches_wsi_abmil/``) has 4656 patches
described by ``vlm_patch_metadata.csv``. Running the VLM on all of them at
once is wasteful for a baseline; instead we draw a *stratified* subset so
every sampling regime and every clinically-relevant slice is represented.

Stratification key: ``(source, slide_label, tile_in_mask)``. The realized
strata in the data are, e.g.::

    top_k / label=1 / tile_in_mask=1   (attention hits inside tumor mask)
    top_k / label=1 / tile_in_mask=0   (attention misses the mask)
    oracle_tumor / label=1 / mask=1    (ground-truth tumor tiles)
    oracle_non_tumor / label=0 / mask=0
    hard_negative / label=1 / mask=0   (tumor slide, benign tile)
    random / ...

Selection is deterministic (rows sorted by ``patch_id`` within each
stratum, take the first K) so the same subset is reproducible with no RNG.
Allocation is proportional to stratum size with a per-stratum floor so
small-but-important groups (oracle_tumor, tile_in_mask=1) are never starved.

Output: images copied into a ``<slide_id>/<source>/<patch_id>.png`` layout
plus ``subset_metadata.csv`` at the root, so ``run_remote_vlm.py`` can join
metadata back by path tail / basename.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SRC_DIR = PROJECT_ROOT / "data" / "vlm_patches_wsi_abmil"
DEFAULT_METADATA = DEFAULT_SRC_DIR / "vlm_patch_metadata.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "c16_stratified_subset"

STRATUM_KEYS = ("source", "slide_label", "tile_in_mask")


def _stratum(row: dict[str, Any]) -> tuple[str, str, str]:
    return tuple(str(row.get(k, "")).strip() for k in STRATUM_KEYS)  # type: ignore[return-value]


def _local_image_path(src_dir: Path, row: dict[str, Any]) -> Path:
    """Reconstruct the local patch path: <slide_id>/<source>/<patch_id>.png."""
    return src_dir / row["slide_id"] / row["source"] / f"{row['patch_id']}.png"


def _allocate(strata: dict[tuple, list], target_n: int, floor: int) -> dict[tuple, int]:
    """Proportional allocation with a per-stratum floor, capped at stratum size."""
    total = sum(len(v) for v in strata.values())
    alloc: dict[tuple, int] = {}
    for key, rows in strata.items():
        want = round(target_n * len(rows) / total) if total else 0
        want = max(floor, want)
        alloc[key] = min(want, len(rows))
    return alloc


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a stratified C16 patch subset.")
    ap.add_argument("--metadata_csv", default=str(DEFAULT_METADATA))
    ap.add_argument("--src_dir", default=str(DEFAULT_SRC_DIR))
    ap.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--target_n", type=int, default=400, help="Approx total patches.")
    ap.add_argument(
        "--floor", type=int, default=20, help="Minimum patches per non-empty stratum."
    )
    ap.add_argument(
        "--include_diverse",
        action="store_true",
        help="Include is_diverse_topk=1 rows (default: only non-diverse).",
    )
    args = ap.parse_args()

    metadata_path = Path(args.metadata_csv)
    src_dir = Path(args.src_dir)
    out_dir = Path(args.out_dir)

    if not metadata_path.is_file():
        print(f"[build_c16_subset] ERROR: metadata not found: {metadata_path}", file=sys.stderr)
        return 1

    with metadata_path.open("r", encoding="utf-8-sig", newline="") as fin:
        reader = csv.DictReader(fin)
        fieldnames = reader.fieldnames or []
        rows = [r for r in reader]

    if not args.include_diverse:
        rows = [r for r in rows if str(r.get("is_diverse_topk", "0")).strip() != "1"]

    strata: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        strata[_stratum(r)].append(r)

    # Deterministic ordering within each stratum.
    for key in strata:
        strata[key].sort(key=lambda r: str(r.get("patch_id", "")))

    alloc = _allocate(strata, args.target_n, args.floor)

    out_dir.mkdir(parents=True, exist_ok=True)
    selected: list[dict[str, Any]] = []
    missing = 0
    realized: dict[tuple, int] = defaultdict(int)

    for key in sorted(strata.keys()):
        take = alloc[key]
        for row in strata[key][:take]:
            src_img = _local_image_path(src_dir, row)
            if not src_img.is_file():
                missing += 1
                continue
            rel = Path(row["slide_id"]) / row["source"] / f"{row['patch_id']}.png"
            dst_img = out_dir / rel
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_img, dst_img)
            new_row = dict(row)
            # Point patch_path at the subset-relative location so run_remote_vlm's
            # tail-path metadata join lines up with the copied image.
            new_row["patch_path"] = rel.as_posix()
            selected.append(new_row)
            realized[key] += 1

    # Write subset metadata (preserve original columns + any we touched).
    out_meta = out_dir / "subset_metadata.csv"
    with out_meta.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in selected:
            writer.writerow(row)

    print(f"[build_c16_subset] Source rows considered: {len(rows)} "
          f"({'incl' if args.include_diverse else 'excl'} diverse)")
    print(f"[build_c16_subset] Realized subset: {len(selected)} patches "
          f"(missing images skipped: {missing})")
    print(f"[build_c16_subset] Layout + metadata written under: {out_dir}")
    print("[build_c16_subset] Per-stratum (source, label, tile_in_mask) -> n:")
    by_source: dict[str, int] = defaultdict(int)
    for key in sorted(realized.keys()):
        print(f"    {key} -> {realized[key]}")
        by_source[key[0]] += realized[key]
    print("[build_c16_subset] Per-source totals:")
    for src, n in sorted(by_source.items()):
        print(f"    {src} -> {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
