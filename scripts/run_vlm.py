"""Run VLM inference on a set of patches from the registry.

Supports single-patch, multi-patch context, and separate-patch modes.
Outputs JSONL with raw responses and provenance.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.s3_utils import get_minio_path_components, get_s3_client, upload_to_s3

REGISTRY_CSV_DEFAULT = "s3://pershin-medailab/Pathomorphology/CAMELYON/mil/vlm_patches_registry/patch_registry.csv"


PROMPT_TEMPLATE_SINGLE = """You are a pathology AI analyzing an H&E stained lymph node tissue patch.

Below is a tissue patch (P1) from a lymph node biopsy.

Decide:
- A: Tumor features are clearly visible in this patch
- B: Tumor features are NOT visible in this patch
- C: The presented data is insufficient to decide

First, analyze the patch carefully. Then provide your FINAL ANSWER as a single letter (A, B, or C).

FINAL ANSWER:"""

PROMPT_TEMPLATE_CONTEXT = """You are a pathology AI analyzing H&E stained lymph node tissue patches.

Below are three tissue patches (P1, P2, P3) from a lymph node biopsy.

For each patch, decide:
- A: Tumor features are clearly visible in this patch
- B: Tumor features are NOT visible in this patch
- C: The presented data is insufficient to decide

First, analyze each patch independently. Then provide your FINAL ANSWER as a single letter (A, B, or C) based on the overall assessment:
- A if tumor is evident in at least one patch
- B if no tumor features are seen in any patch and tissue is adequate
- C if tissue is inadequate, ambiguous, or you cannot make a determination

FINAL ANSWER:"""

PROMPT_TEMPLATE_SEPARATE = """You are a pathology AI analyzing an H&E stained lymph node tissue patch.

Below is a tissue patch (P1) from a lymph node biopsy.

Decide:
- A: Tumor features are clearly visible in this patch
- B: Tumor features are NOT visible in this patch
- C: The presented data is insufficient to decide

First, analyze the patch carefully. Then provide your FINAL ANSWER as a single letter (A, B, or C).

FINAL ANSWER:"""


def _git_commit() -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=10,
        )
        if res.returncode == 0:
            return res.stdout.strip()
    except Exception:
        pass
    return ""


def _parse_answer(raw: str) -> tuple[str, bool]:
    """Extract A, B, or C from raw response."""
    text = raw.strip().upper()

    m = re.search(r'\bFINAL\s*ANSWER\s*:\s*([ABC])', text)
    if m:
        return m.group(1), True

    m = re.search(r'\bANSWER\s*:\s*([ABC])', text)
    if m:
        return m.group(1), True

    m = re.search(r'\b([ABC])\b', text)
    if m:
        return m.group(1), True

    return text[:50], False


def _download_patch(minio_path: str, cache_dir: Path) -> Optional[Path]:
    """Download a single patch from S3 tar.gz to cache."""
    try:
        tar_key, internal_path = get_minio_path_components(minio_path)
    except Exception:
        return None

    if not internal_path:
        cached = cache_dir / minio_path.replace("/", "_").replace(":", "_")
        if cached.exists():
            return cached
        return None

    cached = cache_dir / internal_path.replace("/", "_")
    if cached.exists():
        return cached

    client = get_s3_client()
    s3_path = tar_key

    import io
    import tarfile

    try:
        from botocore.config import Config
        obj = client.get_object(
            Bucket="pershin-medailab",
            Key=s3_path,
        )
        body = obj["Body"].read()
        with tarfile.open(fileobj=io.BytesIO(body), mode="r:gz") as tf:
            try:
                member = tf.getmember(internal_path)
            except KeyError:
                alt_path = internal_path.replace("vlm_patches/", "vlm_patches_standard/")
                member = tf.getmember(alt_path)
            f = tf.extractfile(member)
            if f is None:
                return None
            img_data = f.read()
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.write_bytes(img_data)
            return cached
    except Exception as exc:
        print(f"  [run_vlm] WARNING: download failed for {internal_path}: {exc}")
        return None


def _load_image(path: Path) -> Optional[Image.Image]:
    try:
        img = Image.open(path)
        img.load()
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img
    except Exception:
        return None


def _build_patch_set_id(patches: list[dict]) -> str:
    raw = "|".join(p.get("region_uid", "") for p in patches)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _resolve_aggregate_answer(separate_results: list[str]) -> str:
    """Aggregate separate-patch answers: majority vote; on a tie -> C."""
    votes = [r for r in separate_results if r in ("A", "B", "C")]
    counts = Counter(votes)
    if not counts:
        return "C"
    top = max(counts.values())
    winners = [k for k, v in counts.items() if v == top]
    if len(winners) == 1:
        return winners[0]
    return "C"


def _resolve_registry_csv(path: str) -> str:
    if path.startswith("s3://"):
        parts = path.replace("s3://", "", 1).split("/", 1)
        if len(parts) == 2:
            bucket, key = parts
            client = get_s3_client()
            tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
            client.download_file(bucket, key, tmp.name)
            print(f"[run_vlm] Downloaded registry from {path} to {tmp.name}")
            return tmp.name
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run VLM inference on patches")
    parser.add_argument("--registry_csv", default=REGISTRY_CSV_DEFAULT, help="Path to patch_registry.csv (local or s3://)")
    parser.add_argument("--model", default="quilt_llava", choices=[
        "quilt_llava", "med_gemma", "med_siglip",
    ])
    parser.add_argument("--mode", default="context", choices=["single", "context", "separate"])
    parser.add_argument("--n_patches", type=int, default=3, help="Patches per set (for context/separate)")
    parser.add_argument("--source", default="top_k", help="Patch source filter (or 'all')")
    parser.add_argument("--dataset", default="c17_native", help="Dataset filter")
    parser.add_argument("--max_slides", type=int, default=100, help="Max slides to process")
    parser.add_argument("--max_patch_sets", type=int, default=0, help="Max patch sets (0=all)")
    parser.add_argument("--random_seed", type=int, default=0, help="Random seed filter (0=all)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--load_4bit", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="", help="Output JSONL path (default: auto)")
    parser.add_argument("--output_s3", default="mil/vlm_results", help="S3 prefix for results")
    parser.add_argument("--cache_dir", default="/tmp/vlm_patch_cache", help="Patch cache dir")
    args = parser.parse_args()

    model_key = args.model
    if model_key == "quilt_llava":
        from vlm_backends.quilt_llava import QuiltLLaVABackend
        backend = QuiltLLaVABackend()
    elif model_key == "med_gemma":
        from vlm_backends.med_gemma import MedGemmaBackend
        backend = MedGemmaBackend()
    elif model_key == "med_siglip":
        from vlm_backends.med_siglip import MedSigLIPBackend
        backend = MedSigLIPBackend()
    else:
        print(f"[run_vlm] Unknown model: {model_key}")
        return 1

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    resolved_csv = _resolve_registry_csv(args.registry_csv)
    print(f"[run_vlm] Loading registry from {resolved_csv}")
    registry = pd.read_csv(resolved_csv)
    total_registry = len(registry)
    print(f"  Total registry entries: {total_registry}")

    if args.dataset != "all":
        registry = registry[registry["dataset"] == args.dataset]
    if args.random_seed > 0:
        registry = registry[registry["random_seed"] == args.random_seed]
    if args.source != "all":
        registry = registry[registry["selection_source"] == args.source]

    print(f"  After filtering: {len(registry)} entries")

    non_random = registry[registry["selection_source"] != "random"]
    random_part = registry[registry["selection_source"] == "random"]

    patch_sets = []
    slides = sorted(non_random["slide_id"].unique())
    if args.max_slides:
        slides = slides[:args.max_slides]

    for slide in slides:
        slide_patches = non_random[non_random["slide_id"] == slide]
        slide_patches = slide_patches.sort_values("rank")
        patches = slide_patches.head(args.n_patches).to_dict("records")
        if len(patches) >= 1:
            patch_sets.append(patches)

    n_sets = len(patch_sets)
    if args.max_patch_sets > 0:
        patch_sets = patch_sets[:args.max_patch_sets]

    print(f"[run_vlm] Patch sets to process: {len(patch_sets)} (from {len(slides)} slides)")

    if not patch_sets:
        print("[run_vlm] No patch sets found. Nothing to do.")
        return 0

    print(f"[run_vlm] Loading model: {backend.model_id()}")
    try:
        backend.load(load_4bit=args.load_4bit)
    except Exception as exc:
        print(f"[run_vlm] ERROR loading model: {exc}")
        traceback.print_exc()
        return 1
    print(f"[run_vlm] Model loaded successfully")

    git_commit = _git_commit()

    if args.output:
        jsonl_path = Path(args.output)
    else:
        jsonl_path = Path(tempfile.gettempdir()) / f"vlm_output_{model_key}_{args.mode}_{args.dataset}_{args.source}.jsonl"

    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    total = len(patch_sets)
    n_valid = 0

    with jsonl_path.open("w", encoding="utf-8") as fout:
        for idx, patches in enumerate(patch_sets):
            if idx % 20 == 0:
                print(f"[run_vlm] [{idx}/{total}] processing slide {patches[0].get('slide_id', '?')}")

            set_id = _build_patch_set_id(patches)
            slide_id = patches[0].get("slide_id", "unknown")

            pil_images: list[Image.Image] = []
            patch_records: dict[str, dict] = {}

            for pi, patch in enumerate(patches[:args.n_patches]):
                patch_key = f"P{pi + 1}"
                minio_path = patch.get("minio_path", "")
                local_path = _download_patch(str(minio_path), cache_dir) if minio_path else None

                if local_path is not None and local_path.exists():
                    img = _load_image(local_path)
                else:
                    img = None

                if img is None:
                    print(f"  WARNING: could not load patch {patch.get('patch_uid', '?')}")
                    continue

                pil_images.append(img)
                patch_records[patch_key] = {
                    "region_uid": str(patch.get("region_uid", "")),
                    "relative_path": str(patch.get("relative_path", "")),
                    "task_id": str(patch.get("task_id", "")),
                    "model_hash": str(patch.get("model_hash", "")),
                    "rank": int(patch.get("rank", 0)) if pd.notna(patch.get("rank")) else 0,
                    "slide_id": slide_id,
                    "selection_source": str(patch.get("selection_source", "")),
                }

            if not pil_images:
                continue

            if args.mode == "single":
                prompt = PROMPT_TEMPLATE_SINGLE
            elif args.mode == "context":
                prompt = PROMPT_TEMPLATE_CONTEXT
            else:
                prompt = PROMPT_TEMPLATE_SEPARATE

            raw_responses: list[str] = []
            answers: list[str] = []

            try:
                if args.mode == "separate":
                    for img in pil_images:
                        raw = backend.generate(
                            images=[img],
                            prompt=prompt,
                            max_new_tokens=args.max_new_tokens,
                            temperature=args.temperature,
                            repetition_penalty=args.repetition_penalty,
                            seed=args.seed,
                        )
                        raw_responses.append(raw)
                        ans, valid = _parse_answer(raw)
                        answers.append(ans)
                    aggregate = _resolve_aggregate_answer(answers)
                else:
                    raw = backend.generate(
                        images=pil_images,
                        prompt=prompt,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        repetition_penalty=args.repetition_penalty,
                        seed=args.seed,
                    )
                    raw_responses = [raw]
                    ans, valid = _parse_answer(raw)
                    aggregate = ans
                    answers = [ans]

                if aggregate in ("A", "B", "C"):
                    n_valid += 1

                row = {
                    "patch_set_uid": set_id,
                    "slide_id": slide_id,
                    "model_name": backend.model_id(),
                    "model_type": model_key,
                    "prompt_version": f"v1_{args.mode}",
                    "mode": args.mode,
                    "temperature": args.temperature,
                    "repetition_penalty": args.repetition_penalty,
                    "max_new_tokens": args.max_new_tokens,
                    "load_4bit": args.load_4bit,
                    "seed": args.seed,
                    "git_commit": git_commit,
                    "patches": patch_records,
                    "raw_responses": raw_responses,
                    "per_patch_answers": answers,
                    "answer": aggregate,
                    "parse_valid": aggregate in ("A", "B", "C"),
                    "error": "",
                }

            except Exception as exc:
                err = f"{type(exc).__name__}: {exc}"
                print(f"  ERROR on set {set_id}: {err}")
                traceback.print_exc()
                row = {
                    "patch_set_uid": set_id,
                    "slide_id": slide_id,
                    "model_name": backend.model_id(),
                    "model_type": model_key,
                    "prompt_version": f"v1_{args.mode}",
                    "mode": args.mode,
                    "temperature": args.temperature,
                    "repetition_penalty": args.repetition_penalty,
                    "max_new_tokens": args.max_new_tokens,
                    "load_4bit": args.load_4bit,
                    "seed": args.seed,
                    "git_commit": git_commit,
                    "patches": patch_records,
                    "raw_responses": [],
                    "per_patch_answers": [],
                    "answer": "",
                    "parse_valid": False,
                    "error": err,
                }

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()

    print(f"\n[run_vlm] Results:")
    print(f"  File: {jsonl_path}")
    print(f"  Total sets: {total}")
    print(f"  Valid answers: {n_valid} ({n_valid/max(total,1)*100:.1f}%)")

    s3_key = f"{args.output_s3}/{jsonl_path.name}"
    s3_url = upload_to_s3(str(jsonl_path), s3_key)
    print(f"  S3: {s3_url}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
