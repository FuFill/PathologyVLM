"""Benchmark VLM models on C16 mask-positive vs mask-negative patches.

Runs all configured models sequentially on the same clean control patches
and produces a comparison table to select the best model.

Usage:
  python scripts/benchmark_vlm_c16.py --registry_csv <path> [--output_s3_prefix <s3>]
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_vlm import (
    PROMPT_TEMPLATE_SINGLE,
    _build_patch_set_id,
    _download_patch,
    _git_commit,
    _load_image,
    _parse_answer,
)
from src.s3_utils import get_s3_client, upload_to_s3

REGISTRY_CSV_DEFAULT = "s3://pershin-medailab/Pathomorphology/CAMELYON/mil/vlm_patches_registry/patch_registry.csv"


def _compute_metrics(results: list[dict]) -> dict:
    n_total = len(results)
    n_parsable = sum(1 for r in results if r.get("parse_valid"))
    n_a = sum(1 for r in results if r.get("answer") == "A")
    n_b = sum(1 for r in results if r.get("answer") == "B")
    n_c = sum(1 for r in results if r.get("answer") == "C")

    mask_pos = [r for r in results if r.get("patch_tile_in_mask") == 1]
    mask_neg = [r for r in results if r.get("patch_tile_in_mask") == 0]

    tp = sum(1 for r in mask_pos if r.get("answer") == "A")
    fn = sum(1 for r in mask_pos if r.get("answer") in ("B", "C"))
    tn = sum(1 for r in mask_neg if r.get("answer") == "B")
    fp = sum(1 for r in mask_neg if r.get("answer") == "A")

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced_acc = (sensitivity + specificity) / 2

    unique_raw = len(set(r.get("raw_responses", [""])[0] for r in results if r.get("raw_responses")))

    return {
        "n_total": n_total,
        "n_parsable": n_parsable,
        "parse_rate": n_parsable / n_total if n_total else 0.0,
        "n_A": n_a,
        "n_B": n_b,
        "n_C": n_c,
        "mask_pos_n": len(mask_pos),
        "mask_neg_n": len(mask_neg),
        "TP": tp,
        "FN": fn,
        "TN": tn,
        "FP": fp,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": balanced_acc,
        "unique_raw_responses": unique_raw,
        "mode_collapse_ratio": unique_raw / n_total if n_total else 0.0,
    }


def _run_model(
    backend,
    model_key: str,
    patches_df: pd.DataFrame,
    cache_dir: Path,
    seed: int,
    temperature: float,
) -> list[dict]:
    print(f"\n{'='*60}")
    print(f"Running model: {model_key} ({backend.model_id()})")
    print(f"{'='*60}")

    backend.load(load_4bit=True)

    results = []
    patch_records = patches_df.to_dict("records")
    total = len(patch_records)

    for idx, patch in enumerate(patch_records):
        if idx % 50 == 0:
            print(f"  [{idx}/{total}] {patch.get('slide_id', '?')} / {patch.get('selection_source', '?')}")

        minio_path = str(patch.get("minio_path", ""))
        local_path = _download_patch(minio_path, cache_dir) if minio_path else None
        img = _load_image(local_path) if local_path and local_path.exists() else None

        if img is None:
            continue

        try:
            raw = backend.generate(
                images=[img],
                prompt=PROMPT_TEMPLATE_SINGLE,
                max_new_tokens=128,
                temperature=temperature,
                repetition_penalty=1.0,
                seed=seed,
            )
        except Exception as exc:
            raw = f"ERROR: {exc}"

        ans, valid = _parse_answer(raw)
        results.append({
            "model": model_key,
            "patch_uid": str(patch.get("patch_uid", "")),
            "slide_id": str(patch.get("slide_id", "")),
            "selection_source": str(patch.get("selection_source", "")),
            "tile_in_mask": int(patch.get("tumor_mask_overlap", 0)) if pd.notna(patch.get("tumor_mask_overlap")) else 0,
            "raw_response": raw,
            "answer": ans,
            "parse_valid": valid,
        })

    return results


def _resolve_registry(path: str) -> str:
    if path.startswith("s3://"):
        parts = path.replace("s3://", "", 1).split("/", 1)
        if len(parts) == 2:
            bucket, key = parts
            client = get_s3_client()
            tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
            client.download_file(bucket, key, tmp.name)
            print(f"[benchmark] Downloaded registry from {path} to {tmp.name}")
            return tmp.name
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark VLM models on C16 control patches")
    parser.add_argument("--registry_csv", default=REGISTRY_CSV_DEFAULT)
    parser.add_argument("--model", default="all", choices=["all", "quilt_llava", "med_gemma", "med_siglip"])
    parser.add_argument("--output", default="")
    parser.add_argument("--output_s3", default="mil/vlm_results/c16_benchmark")
    parser.add_argument("--cache_dir", default="/tmp/vlm_patch_cache")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_patches", type=int, default=200)
    args = parser.parse_args()

    registry_path = _resolve_registry(args.registry_csv)
    print(f"[benchmark] Loading registry: {registry_path}")
    registry = pd.read_csv(registry_path)

    c16_mask_pos = registry[
        (registry["dataset"].isin(["c16_native", "c17_to_c16"]))
        & (registry["selection_source"].isin(["top_k", "oracle_tumor"]))
        & (registry["tumor_mask_overlap"] == 1)
        & (registry["is_diverse"] == 0)
    ].head(args.max_patches // 2)

    c16_mask_neg = registry[
        (registry["dataset"].isin(["c16_native", "c17_to_c16"]))
        & (registry["selection_source"].isin(["oracle_non_tumor", "hard_negative"]))
        & (registry["tumor_mask_overlap"] == 0)
        & (registry["is_diverse"] == 0)
    ].head(args.max_patches // 2)

    patches_df = pd.concat([c16_mask_pos, c16_mask_neg], ignore_index=True)
    print(f"[benchmark] Selected {len(patches_df)} patches:")
    print(f"  Mask-positive: {len(c16_mask_pos)}")
    print(f"  Mask-negative: {len(c16_mask_neg)}")

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    all_configs = [
        ("quilt_llava", "vlm_backends.quilt_llava", "QuiltLLaVABackend"),
        ("med_gemma", "vlm_backends.med_gemma", "MedGemmaBackend"),
        ("med_siglip", "vlm_backends.med_siglip", "MedSigLIPBackend"),
    ]
    model_configs = [c for c in all_configs if args.model in ("all", c[0])]

    print(f"[benchmark] Models to run: {[c[0] for c in model_configs]}")

    all_results = {}
    for model_key, module_path, class_name in model_configs:
        try:
            mod = importlib.import_module(module_path)
            backend_cls = getattr(mod, class_name)
            backend = backend_cls()
            results = _run_model(
                backend=backend,
                model_key=model_key,
                patches_df=patches_df,
                cache_dir=cache_dir,
                seed=args.seed,
                temperature=args.temperature,
            )
            all_results[model_key] = results
        except Exception as exc:
            print(f"[benchmark] SKIPPING {model_key}: {exc}")
            all_results[model_key] = []

    print(f"\n{'='*70}")
    print("C16 VLM BENCHMARK RESULTS")
    print(f"{'='*70}")

    rows = []
    for model_key, results in all_results.items():
        metrics = _compute_metrics(results)
        rows.append(metrics)
        print(f"\n  --- {model_key} ---")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.4f}")
            else:
                print(f"    {k}: {v}")

    summary_df = pd.DataFrame(rows)
    summary_df.index = [r for r in all_results.keys()]

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(tempfile.gettempdir()) / "c16_vlm_benchmark.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "config": {
            "seed": args.seed,
            "temperature": args.temperature,
            "max_patches": args.max_patches,
        },
        "models": all_results,
        "metrics": {k: v for k, v in zip(all_results.keys(), rows)},
    }
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    csv_path = output_path.with_suffix(".csv")
    summary_df.to_csv(csv_path)

    for f in [output_path, csv_path]:
        s3_key = f"{args.output_s3}/{f.name}"
        url = upload_to_s3(str(f), s3_key)
        print(f"  Uploaded: {url}")

    print(f"\n[benchmark] Done. Results at {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
