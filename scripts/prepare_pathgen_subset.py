"""Extract a small image subset from PathGen-1.6M (jamessyx/PathGen).

PathGen-1.6M is distributed in two pieces:

  1. A gated metadata file ``PathGen-1.6M.json`` hosted at
     ``https://huggingface.co/datasets/jamessyx/PathGen``. Each entry has
     the shape::

         {
           "wsi_id":   "TCGA-AA-3844-01Z-00-DX1.<uuid>",
           "position": ["35136", "33344"],
           "caption":  "...",
           "file_id":  "<gdc-file-uuid>"
         }

     The user must accept the terms on the Hugging Face dataset page,
     download the JSON manually, and pass its path via ``--pathgen_json``
     (the file is too large to ship with this repo).

  2. The actual whole-slide images (``.svs``) hosted by the GDC (TCGA).
     They are *not* on Hugging Face. Each entry references one slide by
     its GDC ``file_id``; the slide is downloaded via the ``gdc-client``
     command-line tool.

This script:

  * loads the PathGen JSON;
  * picks the first ``--max_images`` entries with *distinct* ``file_id``
    values, so the user only has to download N slides instead of many
    patches from the same slide;
  * for each selected entry, makes sure the ``.svs`` file is present
    locally (under one of several supported layouts) and, if not and
    ``--auto_download`` is set, calls ``gdc-client download <file_id>``
    inside the chosen WSI directory;
  * opens the slide with ``openslide``, reads a level-0 patch of size
    ``--patch_size`` at the given ``(x, y)`` position, saves it as JPEG;
  * writes ``manifest.jsonl`` with one row per saved patch, carrying
    the PathGen caption as ``reference_answer``.

Requires:
  * Python package ``openslide-python`` AND the native OpenSlide library
    (https://openslide.org/download/). On Windows, ``openslide-python``
    only loads if the OpenSlide DLLs are on PATH.
  * ``gdc-client`` on PATH if ``--auto_download`` is used
    (https://gdc.cancer.gov/access-data/gdc-data-transfer-tool).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from PIL import Image

DEFAULT_DATASET = "jamessyx/PathGen"
DEFAULT_PATCH_SIZE = 672
DEFAULT_GDC_CLIENT = "gdc-client"


# ----------------------------------------------------------------------------
# JSON loading
# ----------------------------------------------------------------------------
def _load_pathgen_json(path: Path) -> List[dict]:
    """Load PathGen-1.6M.json. Accepts both list and dict-of-list shapes."""
    # utf-8-sig transparently strips a BOM if present (some Windows
    # redistributions of PathGen-1.6M.json have one); behaves like utf-8
    # otherwise.
    with path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)

    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        # Some redistributions wrap the list, e.g. {"data": [...]}.
        for candidate_key in ("data", "samples", "items"):
            if isinstance(data.get(candidate_key), list):
                rows = data[candidate_key]
                break
        else:
            raise ValueError(
                f"Unsupported JSON top-level dict with keys {list(data.keys())}. "
                f"Expected a list, or a dict containing one of "
                f"'data' / 'samples' / 'items'."
            )
    else:
        raise ValueError(
            f"Expected a JSON list, got {type(data).__name__} in {path}."
        )

    return rows


def _normalize_entry(entry: dict) -> Optional[dict]:
    """Map keys to a canonical shape; tolerate the two casings seen in the wild."""
    wsi_id = entry.get("wsi_id") or entry.get("WSI_id")
    position = entry.get("position")
    caption = entry.get("caption")
    file_id = entry.get("file_id")
    if not wsi_id or not position or not isinstance(position, (list, tuple)) or len(position) < 2:
        return None
    try:
        x = int(position[0])
        y = int(position[1])
    except (TypeError, ValueError):
        return None
    return {
        "wsi_id": str(wsi_id),
        "x": x,
        "y": y,
        "caption": caption if isinstance(caption, str) else "",
        "file_id": str(file_id) if file_id else None,
    }


# ----------------------------------------------------------------------------
# Subset selection: distinct file_id, preserving order
# ----------------------------------------------------------------------------
def _select_distinct_by_file_id(
    entries: Iterable[dict],
    max_items: int,
    require_file_id: bool,
) -> List[dict]:
    seen: set = set()
    out: List[dict] = []
    for raw in entries:
        norm = _normalize_entry(raw)
        if norm is None:
            continue
        key = norm["file_id"] if require_file_id else (norm["file_id"] or norm["wsi_id"])
        if require_file_id and not norm["file_id"]:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(norm)
        if len(out) >= max_items:
            break
    return out


# ----------------------------------------------------------------------------
# WSI resolution and download
# ----------------------------------------------------------------------------
def _find_wsi_file(wsi_dir: Path, wsi_id: str, file_id: Optional[str]) -> Optional[Path]:
    """Try the layouts produced by gdc-client and by ad-hoc unpacking."""
    candidates: List[Path] = []
    if file_id:
        candidates.append(wsi_dir / file_id / f"{wsi_id}.svs")
        # gdc-client sometimes places the slide directly under the file_id dir
        # even when the inner name differs; try any .svs in that dir.
        d = wsi_dir / file_id
        if d.is_dir():
            for p in d.iterdir():
                if p.suffix.lower() == ".svs":
                    candidates.append(p)
    candidates.append(wsi_dir / f"{wsi_id}.svs")

    for c in candidates:
        if c.exists() and c.is_file():
            return c
    return None


def _download_with_gdc_client(
    file_id: str,
    wsi_dir: Path,
    gdc_client: str,
    token_file: Optional[Path],
) -> bool:
    """Run ``gdc-client download <file_id>`` inside ``wsi_dir``.

    Returns True on a clean exit, False otherwise. The slide ends up under
    ``<wsi_dir>/<file_id>/...``.
    """
    if shutil.which(gdc_client) is None:
        print(
            f"[prepare_pathgen_subset] ERROR: '{gdc_client}' not found on PATH. "
            f"Install gdc-client from "
            f"https://gdc.cancer.gov/access-data/gdc-data-transfer-tool "
            f"or pass --no_auto_download and download slides manually.",
            file=sys.stderr,
        )
        return False
    wsi_dir.mkdir(parents=True, exist_ok=True)
    cmd: List[str] = [gdc_client, "download", file_id]
    if token_file:
        cmd.extend(["-t", str(token_file)])
    print(f"[prepare_pathgen_subset] Running: {' '.join(cmd)}  (cwd={wsi_dir})")
    try:
        result = subprocess.run(cmd, cwd=str(wsi_dir), check=False)
    except OSError as exc:
        print(f"[prepare_pathgen_subset] gdc-client failed to launch: {exc}", file=sys.stderr)
        return False
    if result.returncode != 0:
        print(
            f"[prepare_pathgen_subset] gdc-client exited with code "
            f"{result.returncode} for file_id={file_id}",
            file=sys.stderr,
        )
        return False
    return True


# ----------------------------------------------------------------------------
# Patch extraction
# ----------------------------------------------------------------------------
def _extract_patch(wsi_path: Path, x: int, y: int, patch_size: int) -> Optional[Image.Image]:
    """Read a level-0 RGB patch from ``wsi_path`` at ``(x, y)``."""
    try:
        import openslide  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001 (covers OSError on missing DLLs)
        print(
            "[prepare_pathgen_subset] ERROR: failed to import openslide. "
            "Install the openslide-python Python package AND the native "
            "OpenSlide library, then ensure its DLLs/.so files are on PATH/"
            "LD_LIBRARY_PATH. https://openslide.org/download/  "
            f"Underlying error: {exc!r}",
            file=sys.stderr,
        )
        return None
    try:
        slide = openslide.OpenSlide(str(wsi_path))
    except Exception as exc:  # noqa: BLE001
        print(
            f"[prepare_pathgen_subset] ERROR opening WSI {wsi_path}: {exc}",
            file=sys.stderr,
        )
        return None
    try:
        region = slide.read_region((x, y), 0, (patch_size, patch_size))
    except Exception as exc:  # noqa: BLE001
        print(
            f"[prepare_pathgen_subset] ERROR reading region "
            f"({x},{y}) {patch_size}x{patch_size} from {wsi_path.name}: {exc}",
            file=sys.stderr,
        )
        slide.close()
        return None
    slide.close()
    # OpenSlide returns RGBA; flatten to RGB on a white background to
    # avoid black holes where the slide is transparent.
    if region.mode != "RGB":
        bg = Image.new("RGB", region.size, (255, 255, 255))
        bg.paste(region, mask=region.split()[-1] if region.mode == "RGBA" else None)
        region = bg
    return region


# ----------------------------------------------------------------------------
# Safe preview helper for the first sample log line
# ----------------------------------------------------------------------------
def _summarize_entry(entry: dict) -> str:
    caption = entry.get("caption") or ""
    short_cap = caption if len(caption) <= 120 else caption[:117] + "..."
    return (
        f"wsi_id={entry.get('wsi_id')!r}, file_id={entry.get('file_id')!r}, "
        f"x={entry.get('x')}, y={entry.get('y')}, caption={short_cap!r}"
    )


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a small PathGen-1.6M image subset by downloading TCGA WSIs "
            "via gdc-client and extracting fixed-size patches at the positions "
            "listed in PathGen-1.6M.json."
        )
    )
    parser.add_argument("--out_dir", required=True, help="Output directory for patches + manifest.")
    parser.add_argument("--max_images", type=int, default=10, help="Max patches to extract (distinct WSIs).")
    parser.add_argument(
        "--pathgen_json",
        required=True,
        help=(
            "Path to PathGen-1.6M.json. Download it from "
            "https://huggingface.co/datasets/jamessyx/PathGen (gated; accept terms first)."
        ),
    )
    parser.add_argument(
        "--wsi_dir",
        required=True,
        help=(
            "Directory holding TCGA .svs files. Both flat "
            "('<wsi_dir>/<wsi_id>.svs') and gdc-client "
            "('<wsi_dir>/<file_id>/<wsi_id>.svs') layouts are supported."
        ),
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=DEFAULT_PATCH_SIZE,
        help=f"Patch edge length in level-0 pixels (default: {DEFAULT_PATCH_SIZE}).",
    )
    parser.add_argument(
        "--dataset_name",
        default=DEFAULT_DATASET,
        help="HF dataset id, recorded in the manifest only.",
    )
    parser.add_argument(
        "--auto_download",
        dest="auto_download",
        action="store_true",
        default=True,
        help="Run gdc-client to fetch missing slides (default).",
    )
    parser.add_argument(
        "--no_auto_download",
        dest="auto_download",
        action="store_false",
        help="Do not call gdc-client; only use slides already on disk.",
    )
    parser.add_argument(
        "--gdc_client",
        default=DEFAULT_GDC_CLIENT,
        help="Path or name of the gdc-client executable.",
    )
    parser.add_argument(
        "--gdc_token_file",
        default=None,
        help="Optional GDC user token file for controlled-access slides.",
    )
    parser.add_argument(
        "--jpeg_quality",
        type=int,
        default=95,
        help="JPEG quality for saved patches (default 95).",
    )
    parser.add_argument(
        "--allow_repeat_slides",
        action="store_true",
        help=(
            "Allow multiple patches from the same WSI. By default the script "
            "selects entries with distinct file_id to minimize downloads."
        ),
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Validate inputs.
    # ------------------------------------------------------------------
    pathgen_json = Path(args.pathgen_json)
    if not pathgen_json.exists():
        print(
            f"[prepare_pathgen_subset] ERROR: PathGen JSON not found: {pathgen_json}\n"
            "Download it from https://huggingface.co/datasets/jamessyx/PathGen "
            "(accept the dataset terms first), e.g.:\n"
            "  hf download jamessyx/PathGen PathGen-1.6M.json --repo-type=dataset \\\n"
            "    --local-dir .",
            file=sys.stderr,
        )
        return 1

    wsi_dir = Path(args.wsi_dir)
    wsi_dir.mkdir(parents=True, exist_ok=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"

    token_file = Path(args.gdc_token_file) if args.gdc_token_file else None
    if token_file and not token_file.exists():
        print(
            f"[prepare_pathgen_subset] ERROR: GDC token file not found: {token_file}",
            file=sys.stderr,
        )
        return 1

    # ------------------------------------------------------------------
    # Load and select.
    # ------------------------------------------------------------------
    print(f"[prepare_pathgen_subset] Reading PathGen JSON: {pathgen_json}")
    try:
        raw_entries = _load_pathgen_json(pathgen_json)
    except Exception as exc:  # noqa: BLE001
        print(f"[prepare_pathgen_subset] ERROR loading JSON: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1
    print(f"[prepare_pathgen_subset] Total entries in JSON: {len(raw_entries)}")

    if raw_entries:
        first_norm = _normalize_entry(raw_entries[0])
        if first_norm:
            print(f"[prepare_pathgen_subset] First sample: {_summarize_entry(first_norm)}")

    if args.allow_repeat_slides:
        selected: List[dict] = []
        for raw in raw_entries:
            norm = _normalize_entry(raw)
            if norm is None:
                continue
            selected.append(norm)
            if len(selected) >= args.max_images:
                break
    else:
        selected = _select_distinct_by_file_id(
            raw_entries, max_items=args.max_images, require_file_id=args.auto_download
        )
    if not selected:
        print(
            "[prepare_pathgen_subset] ERROR: no usable entries selected from JSON. "
            "If you cannot auto-download (no file_id), re-run with --no_auto_download "
            "and pre-populate --wsi_dir.",
            file=sys.stderr,
        )
        return 1
    print(
        f"[prepare_pathgen_subset] Selected {len(selected)} entries "
        f"({'distinct file_id' if not args.allow_repeat_slides else 'any'})."
    )

    # ------------------------------------------------------------------
    # For each selected entry: ensure slide is available, extract patch.
    # ------------------------------------------------------------------
    saved = 0
    failures = 0
    with manifest_path.open("w", encoding="utf-8") as fout:
        for source_index, entry in enumerate(selected):
            wsi_id = entry["wsi_id"]
            file_id = entry["file_id"]
            x = entry["x"]
            y = entry["y"]
            caption = entry["caption"]

            wsi_path = _find_wsi_file(wsi_dir, wsi_id, file_id)
            if wsi_path is None and args.auto_download and file_id:
                ok = _download_with_gdc_client(
                    file_id=file_id,
                    wsi_dir=wsi_dir,
                    gdc_client=args.gdc_client,
                    token_file=token_file,
                )
                if ok:
                    wsi_path = _find_wsi_file(wsi_dir, wsi_id, file_id)

            if wsi_path is None:
                failures += 1
                print(
                    f"[prepare_pathgen_subset] Skip wsi_id={wsi_id} file_id={file_id}: "
                    f"no .svs found in {wsi_dir}",
                    file=sys.stderr,
                )
                continue

            patch = _extract_patch(wsi_path, x, y, args.patch_size)
            if patch is None:
                failures += 1
                continue

            image_id = f"pathgen_{saved:04d}"
            image_path = out_dir / f"{image_id}.jpg"
            try:
                patch.save(image_path, format="JPEG", quality=args.jpeg_quality)
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(
                    f"[prepare_pathgen_subset] Skip {image_id}: failed to save JPEG: {exc}",
                    file=sys.stderr,
                )
                continue

            row = {
                "image_id": image_id,
                "image_path": image_path.name,  # relative within out_dir
                "dataset": args.dataset_name,
                "source_index": source_index,
                "wsi_id": wsi_id,
                "file_id": file_id,
                "x": x,
                "y": y,
                "patch_size": args.patch_size,
                "magnification": "level0",
                "source": "PathGen-1.6M",
                "question_or_instruction": None,
                "reference_answer": caption,
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            saved += 1
            print(
                f"[prepare_pathgen_subset] Saved {image_path} "
                f"(wsi_id={wsi_id}, x={x}, y={y})"
            )

    print(
        f"[prepare_pathgen_subset] Done. Saved {saved} patches, "
        f"{failures} failures, into {out_dir}."
    )
    if saved == 0:
        print(
            "[prepare_pathgen_subset] ERROR: no patches were saved.\n"
            "Check that:\n"
            "  - openslide-python and the native OpenSlide library are installed,\n"
            "  - gdc-client is on PATH (or pass --no_auto_download with slides on disk),\n"
            "  - the file_id values you tried are public TCGA slides "
            "(controlled-access ones need --gdc_token_file).",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
