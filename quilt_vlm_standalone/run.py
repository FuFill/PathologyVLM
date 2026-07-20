"""Quilt-LLaVA lymph-node describer — standalone entry point.

Runs the pathology vision-language model ``wisdomik/Quilt-Llava-v1.5-7b`` on H&E
histology tiles and writes one structured JSON result per image, streamed to a
JSONL file (so a long run's progress survives interruption).

Two input modes (mutually exclusive):

* ``--image_dir FOLDER`` — describe every tile under a folder (recursive). No
  patch provenance is carried (metadata fields are left empty).
* ``--manifest CSV`` — describe the tiles named in a manifest CSV, carrying each
  row's provenance (source, label, tile_in_mask, coordinates, slide_id, score,
  ...) straight through into the output. Image paths are resolved from a path
  column, optionally against ``--image_root``.

Usage
-----
    # folder mode, neutral prompt, deterministic control:
    python run.py --image_dir tiles --output out.jsonl --temperature 0 --seed 0

    # manifest mode (real integration), 4-bit smoke test:
    python run.py --manifest patches.csv --image_root /data --output out.jsonl \
        --max_images 20 --load_4bit

Output
------
A JSONL file (one JSON object per line): schema fields + manifest provenance +
run params + raw_response + the three JSON provenance flags + error. A sidecar
``<output>.run_manifest.json`` records the full reproducibility context.

SAFETY: the prompt instructs the model not to name a clinical diagnosis, and
``should_abstain`` signals insufficient evidence — but output is NOT guaranteed;
always inspect ``raw_response``. This is a research prototype, not a diagnostic
tool. Known bias: it over-calls ``tumor_suspicious=yes`` on benign tissue.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from json_utils import SCHEMA_VERSION, normalize_json, parse_with_provenance
from model import (
    MODEL_REVISION,
    QUILT_LLAVA_GIT,
    bootstrap_llava,
    find_images,
    generate_answer,
    load_model,
)
from prompt import (
    DEFAULT_PROMPT_VARIANT,
    PROMPT_VARIANTS,
    get_prompt,
    get_prompt_version,
)

# --- Defaults --------------------------------------------------------------- #
DEFAULT_MODEL = "wisdomik/Quilt-Llava-v1.5-7b"
DEFAULT_TEMPERATURE = 0.4
DEFAULT_REPETITION_PENALTY = 1.08
DEFAULT_TOP_P = 0.95
DEFAULT_MAX_NEW_TOKENS = 768
DEFAULT_SEED = 0

# Manifest columns that may hold the image path, in priority order.
_PATH_COLUMNS = ("patch_path", "image_path", "path", "minio_path", "filepath")
# Provenance columns carried straight through from the manifest into output.
_CARRY_COLUMNS = (
    "source", "label", "prediction", "confidence", "x", "y",
    "slide_id", "patch_id", "attention_score", "attention_rank",
    "tile_size", "tile_in_mask", "dataset", "split", "fold",
    "dataset_id", "dataset_version",
)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Quilt-LLaVA lymph-node describer (standalone). Emits JSONL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--image_dir",
                     help="Folder of H&E tiles (searched recursively).")
    src.add_argument("--manifest",
                     help="CSV naming tiles + carrying patch provenance.")
    ap.add_argument("--image_root", default="",
                    help="Root prefix to resolve relative manifest image paths.")
    ap.add_argument("--output", default="quilt_vlm_results.jsonl",
                    help="Output JSONL file (one result object per line).")
    ap.add_argument("--model_name", default=DEFAULT_MODEL,
                    help="Hugging Face model id.")
    ap.add_argument("--model_revision", default=MODEL_REVISION or "",
                    help="Pin the HF weights commit/tag. Empty = repo default.")
    ap.add_argument("--prompt_variant", default=DEFAULT_PROMPT_VARIANT,
                    choices=PROMPT_VARIANTS,
                    help="Prompt to use.")
    ap.add_argument("--max_images", type=int, default=0,
                    help="Cap number of images. 0 = all.")
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE,
                    help="Sampling temperature. 0 = greedy/deterministic.")
    ap.add_argument("--repetition_penalty", type=float,
                    default=DEFAULT_REPETITION_PENALTY,
                    help="Repetition penalty (>1.0 applies).")
    ap.add_argument("--top_p", type=float, default=DEFAULT_TOP_P,
                    help="Nucleus sampling top_p (used when sampling).")
    ap.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS,
                    help="Generation length limit.")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED,
                    help="RNG seed applied before each image (repeatable sampling).")
    ap.add_argument("--load_4bit", action="store_true",
                    help="Use bitsandbytes 4-bit quantization (needs a GPU, "
                         "~half the VRAM). Omit for full fp16.")
    return ap.parse_args()


def _git_commit() -> str:
    """Best-effort git commit of this standalone folder's repo."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(Path(__file__).resolve().parent),
            capture_output=True, text=True,
        )
        if res.returncode == 0:
            return res.stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    return ""


def _resolve_path(raw: str, image_root: str) -> Path:
    p = Path(raw)
    if image_root and not p.is_absolute():
        return Path(image_root) / p
    return p


def _load_manifest(manifest: str, image_root: str, max_images: int
                   ) -> list[tuple[Path, dict[str, Any]]]:
    """Return [(image_path, carried_metadata), ...] from a manifest CSV."""
    out: list[tuple[Path, dict[str, Any]]] = []
    with open(manifest, "r", encoding="utf-8", newline="") as fin:
        reader = csv.DictReader(fin)
        cols = reader.fieldnames or []
        path_col = next((c for c in _PATH_COLUMNS if c in cols), None)
        if path_col is None:
            raise ValueError(
                f"Manifest {manifest!r} has no recognized path column "
                f"(looked for {_PATH_COLUMNS}). Columns: {cols}"
            )
        for row in reader:
            raw_path = str(row.get(path_col, "")).strip()
            if not raw_path:
                continue
            img_path = _resolve_path(raw_path, image_root)
            meta = {c: row[c] for c in _CARRY_COLUMNS if c in row}
            out.append((img_path, meta))
            if max_images and len(out) >= max_images:
                break
    return out


def main() -> int:
    args = _parse_args()

    # Build the work list: (image_path, metadata) pairs.
    if args.manifest:
        work = _load_manifest(args.manifest, args.image_root, args.max_images)
        if not work:
            print(f"[run] Manifest {args.manifest!r} produced no images.",
                  file=sys.stderr)
            return 1
        print(f"[run] Manifest mode: {len(work)} image(s) from {args.manifest}")
    else:
        images = find_images(args.image_dir, max_images=args.max_images)
        if not images:
            print(f"[run] No images found under {args.image_dir!r}", file=sys.stderr)
            return 1
        work = [(p, {}) for p in images]
        print(f"[run] Folder mode: {len(work)} image(s) under {args.image_dir}")

    prompt_variant = args.prompt_variant
    prompt_text = get_prompt(prompt_variant)
    prompt_version = get_prompt_version(prompt_variant)
    revision = args.model_revision or None

    print(f"[run] prompt={prompt_version}, temp={args.temperature}, "
          f"seed={args.seed}, rep={args.repetition_penalty}, top_p={args.top_p}, "
          f"4bit={args.load_4bit}")

    bootstrap_llava()
    tokenizer, model, image_processor, _ = load_model(
        args.model_name, args.load_4bit, revision=revision
    )

    standalone_commit = _git_commit()
    run_params = {
        "model_name": args.model_name,
        "model_revision": revision or "",
        "llava_git": QUILT_LLAVA_GIT,
        "standalone_commit": standalone_commit,
        "prompt_variant": prompt_variant,
        "prompt_version": prompt_version,
        "schema_version": SCHEMA_VERSION,
        "temperature": args.temperature,
        "repetition_penalty": args.repetition_penalty,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "load_4bit": args.load_4bit,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Reproducibility sidecar written up front.
    manifest_path = out_path.with_suffix(out_path.suffix + ".run_manifest.json")
    manifest_path.write_text(
        json.dumps({**run_params,
                    "input_mode": "manifest" if args.manifest else "image_dir",
                    "input": args.manifest or args.image_dir,
                    "image_root": args.image_root,
                    "n_images": len(work)},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    n_valid = n_strict = n_schema = 0
    t0 = time.time()
    with out_path.open("w", encoding="utf-8") as fout:
        for i, (img_path, meta) in enumerate(work, 1):
            row: dict[str, Any] = {
                "image_id": img_path.stem,
                "image_path": str(img_path),
            }
            row.update(run_params)
            row.update(meta)  # manifest provenance (empty in folder mode)
            try:
                raw = generate_answer(
                    image_path=img_path,
                    tokenizer=tokenizer,
                    model=model,
                    image_processor=image_processor,
                    prompt=prompt_text,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    repetition_penalty=args.repetition_penalty,
                    top_p=args.top_p,
                    seed=args.seed,
                )
                parsed, prov = parse_with_provenance(raw)
                row["raw_response"] = raw
                row["strict_json_valid"] = prov["strict_json_valid"]
                row["parse_valid"] = prov["parse_valid"]
                row["schema_valid"] = prov["schema_valid"]
                row["repair_stage"] = prov["repair_stage"]
                row["json_valid"] = prov["parse_valid"]  # deprecated alias
                row.update(normalize_json(parsed))
                n_valid += int(prov["parse_valid"])
                n_strict += int(prov["strict_json_valid"])
                n_schema += int(prov["schema_valid"])
                row["error"] = ""
            except Exception as exc:  # noqa: BLE001 — one image must not abort the run
                row["raw_response"] = ""
                row["strict_json_valid"] = False
                row["parse_valid"] = False
                row["schema_valid"] = False
                row["repair_stage"] = "none"
                row["json_valid"] = False
                row.update(normalize_json(None))
                row["error"] = f"{type(exc).__name__}: {exc}"
                print(f"[run] ERROR on {img_path.name}: {row['error']}",
                      file=sys.stderr)

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()  # stream so progress survives interruption

            if i % 10 == 0 or i == len(work):
                print(f"[run] {i}/{len(work)} done "
                      f"(parse_valid {n_valid}/{i}, strict {n_strict}/{i})")

    dt = time.time() - t0
    n = len(work)
    print(f"[run] Wrote {n} result(s) to {out_path}")
    print(f"[run] run_manifest: {manifest_path}")
    print(f"[run] parse_valid={n_valid}/{n} strict_json={n_strict}/{n} "
          f"schema_valid={n_schema}/{n} in {dt:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
