"""VLM inference pipeline for C16 control and C17 final evaluation.

Modes:
  single   — 1 best patch (top-1 by rank) per slide, 1 VLM call
  separate — 3 patches, 3 separate VLM calls, aggregate (any A→A, all B→B, else C)
  context  — 3 patches in one multi-image prompt, 1 VLM call

Sources:
  top_k              — MIL top attention patches
  random             — random patches (3 fixed seeds: 42, 123, 456)
  oracle_tumor       — mask-positive controls
  oracle_non_tumor   — mask-negative controls
  hard_negative      — high-attention non-tumor patches
  diverse            — spatially-diversified top_k (context_set=diverse)
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import io
import json
import re
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.s3_utils import (
    get_minio_path_components,
    get_s3_client,
    presign_url,
    read_csv_from_s3,
    upload_to_s3,
)

REGISTRY_CSV_DEFAULT = (
    "s3://pershin-medailab/Pathomorphology/CAMELYON/"
    "mil/vlm_patches_registry/patch_registry.csv"
)

C16_DATASETS = ("c16_native", "c17_to_c16")
C17_DATASETS = ("c17_native", "c16_to_c17")

RANDOM_SEEDS = (42, 123, 456)

PROMPT_SINGLE = """You are a pathology AI analyzing an H&E stained lymph node tissue patch.

Below is a tissue patch (P1) from a lymph node biopsy.

Decide:
- A: Tumor features are clearly visible in this patch
- B: Tumor features are NOT visible in this patch
- C: The presented data is insufficient to decide

First, analyze the patch carefully. Then provide your FINAL ANSWER as a single letter (A, B, or C).

FINAL ANSWER:"""

PROMPT_CONTEXT = """You are a pathology AI analyzing H&E stained lymph node tissue patches.

Below are three tissue patches (P1, P2, P3) from a lymph node biopsy.

Decide:
- A: Tumor features are clearly visible in at least one patch
- B: Tumor features are NOT visible in any patch
- C: The presented data is insufficient to decide

First, analyze each patch independently. Then provide your FINAL ANSWER as a single letter (A, B, or C).

FINAL ANSWER:"""


def _git_commit() -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=10,
        )
        if res.returncode == 0:
            return res.stdout.strip()
    except Exception:
        pass
    return ""


def _parse_answer(raw: str) -> tuple[str, bool]:
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
    if text.startswith("A") or text.startswith("B") or text.startswith("C"):
        return text[0], True
    return text[:80], False


def _aggregate_separate(answers: list[str]) -> str:
    if any(a == "A" for a in answers):
        return "A"
    if all(a == "B" for a in answers):
        return "B"
    return "C"


def _download_patch(minio_path: str, cache_dir: Path) -> Optional[Path]:
    try:
        tar_key, internal_path = get_minio_path_components(minio_path)
    except Exception:
        return None
    if not internal_path:
        cached = cache_dir / minio_path.replace("/", "_").replace(":", "_")
        return cached if cached.exists() else None
    cached = cache_dir / internal_path.replace("/", "_")
    if cached.exists():
        return cached
    client = get_s3_client()
    s3_path = tar_key
    try:
        obj = client.get_object(Bucket="pershin-medailab", Key=s3_path)
        body = obj["Body"].read()
        with tarfile.open(fileobj=io.BytesIO(body), mode="r:gz") as tf:
            try:
                member = tf.getmember(internal_path)
            except KeyError:
                alt = internal_path.replace("vlm_patches/", "vlm_patches_standard/")
                member = tf.getmember(alt)
            f = tf.extractfile(member)
            if f is None:
                return None
            img_data = f.read()
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.write_bytes(img_data)
            return cached
    except Exception as exc:
        print(f"  [pipeline] WARNING: download failed {internal_path}: {exc}")
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


def _patch_set_uid(patches: list[dict]) -> str:
    raw = "|".join(
        str(p.get("region_uid", p.get("patch_uid", ""))) for p in patches
    )
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _test_s3_access():
    from src.s3_utils import get_s3_client
    client = get_s3_client()
    key = "Pathomorphology/CAMELYON/mil/vlm_patches_registry/patch_registry.csv"
    try:
        client.head_object(Bucket="pershin-medailab", Key=key)
        print("[pipeline] S3 access OK")
    except Exception as exc:
        print(f"[pipeline] S3 access FAILED: {exc}")
        raise


def load_registry(
    csv_path: str,
    datasets: tuple[str, ...],
    sources: tuple[str, ...] | None,
    max_slides: int = 0,
) -> pd.DataFrame:
    s3_registry_key = "mil/vlm_patches_registry/patch_registry.csv"
    if csv_path.startswith("s3://"):
        print(f"[pipeline] Loading registry from S3 via read_csv_from_s3")
        df = read_csv_from_s3(s3_registry_key)
    else:
        print(f"[pipeline] Loading registry: {csv_path}")
        df = pd.read_csv(csv_path)
    print(f"  Total: {len(df)} entries")

    df = df[df["dataset"].isin(datasets)]
    print(f"  After dataset filter {datasets}: {len(df)}")

    if sources:
        df = df[df["selection_source"].isin(sources)]
        print(f"  After source filter {sources}: {len(df)}")

    slides = sorted(df["slide_id"].dropna().unique())
    if max_slides > 0:
        keep = set(slides[:max_slides])
        df = df[df["slide_id"].isin(keep)]
        print(f"  After slide cap {max_slides}: {len(df)}")

    return df


def build_patch_sets(
    registry: pd.DataFrame,
    source: str,
    n_patches: int,
    context_set: str = "standard",
) -> list[list[dict]]:
    """Build list of patch sets (each set is list of patch dicts).

    For random source: returns 3 separate lists (one per seed).
    For all others: 1 list.
    """
    if source == "diverse":
        sub = registry[registry["context_set"] == "diverse"].copy()
        source_key = "top_k"
    else:
        sub = registry[registry["context_set"] == context_set].copy()
        source_key = source

    if source == "random":
        all_sets = []
        for seed_val in RANDOM_SEEDS:
            seed_sub = sub[
                (sub["selection_source"] == "random") &
                (sub["random_seed"] == seed_val)
            ]
            sets = _build_per_slide_sets(seed_sub, source_key, n_patches)
            all_sets.extend(sets)
        return all_sets

    sub = sub[sub["selection_source"] == source_key]
    return _build_per_slide_sets(sub, source_key, n_patches)


def _build_per_slide_sets(
    df: pd.DataFrame,
    source_key: str,
    n_patches: int,
) -> list[list[dict]]:
    sets = []
    for slide_id in sorted(df["slide_id"].unique()):
        slide_patches = df[df["slide_id"] == slide_id].sort_values("rank")
        patches = slide_patches.head(n_patches).to_dict("records")
        if len(patches) >= 1:
            sets.append(patches)
    return sets


def run_inference(
    backend,
    patch_sets: list[list[dict]],
    mode: str,
    cache_dir: Path,
    seed: int,
    temperature: float,
    repetition_penalty: float,
    max_new_tokens: int,
    backend_config: dict,
    git_commit: str,
) -> list[dict]:
    all_records = []
    total = len(patch_sets)
    t0 = time.time()

    print(f"[pipeline] Running {mode} mode on {total} patch sets")

    for idx, patches in enumerate(patch_sets):
        slide_id = patches[0].get("slide_id", "unknown")
        source = patches[0].get("selection_source", "unknown")
        group = f"{slide_id}/{source}"
        task_id = str(patches[0].get("task_id", patches[0].get("model_hash", ""))) if patches else ""

        set_id = _patch_set_uid(patches)

        pil_images: list[Image.Image] = []
        patch_metas: dict[str, dict] = {}

        for pi, patch in enumerate(patches):
            key = f"P{pi + 1}"
            minio_path = str(patch.get("minio_path", ""))
            local_path = _download_patch(minio_path, cache_dir) if minio_path else None
            img = _load_image(local_path) if local_path and local_path.exists() else None
            if img is None:
                print(f"    WARNING: could not load {key} ({patch.get('patch_uid', '?')})")
                continue
            pil_images.append(img)
            patch_metas[key] = {
                "region_uid": str(patch.get("region_uid", "")),
                "patch_uid": str(patch.get("patch_uid", "")),
                "rank": int(patch.get("rank", 0)) if pd.notna(patch.get("rank")) else 0,
                "selection_source": source,
                "tumor_mask_overlap": int(patch.get("tumor_mask_overlap", 0)),
                "relative_path": str(patch.get("relative_path", "")),
                "tissue_fraction": float(patch.get("tissue_fraction", 1.0)),
                "context_set": str(patch.get("context_set", "")),
            }

        if not pil_images:
            continue

        raw_responses: list[str] = []
        per_patch_answers: list[str] = []
        error: str = ""

        try:
            if mode == "separate":
                prompt = PROMPT_SINGLE
                for img in pil_images:
                    raw = backend.generate(
                        images=[img],
                        prompt=prompt,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        repetition_penalty=repetition_penalty,
                        seed=seed,
                    )
                    raw_responses.append(raw)
                    ans, _ = _parse_answer(raw)
                    per_patch_answers.append(ans)
                aggregate = _aggregate_separate(per_patch_answers)
            elif mode == "context":
                prompt = PROMPT_CONTEXT
                raw = backend.generate(
                    images=pil_images,
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                    seed=seed,
                )
                raw_responses = [raw]
                ans, _ = _parse_answer(raw)
                aggregate = ans
                per_patch_answers = [ans]
            else:
                prompt = PROMPT_SINGLE
                raw = backend.generate(
                    images=[pil_images[0]],
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                    seed=seed,
                )
                raw_responses = [raw]
                ans, _ = _parse_answer(raw)
                aggregate = ans
                per_patch_answers = [ans]

            parse_valid = aggregate in ("A", "B", "C")

        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
            aggregate = ""
            parse_valid = False

        elapsed = time.time() - t0
        ans_str = aggregate if aggregate else "∅"
        if ans_str not in ("A", "B", "C"):
            for ri, r in enumerate(raw_responses):
                trunc = r[:300].replace("\n", "\\n")
                print(f"    RAW[{ri}]: {trunc}")
        print(f"  [{idx}/{total}] {group}/{mode} → {ans_str} ({elapsed:.0f}s)")

        n_shown = 1 if mode == "single" else len(pil_images)

        record = {
            "patch_set_uid": set_id,
            "slide_id": slide_id,
            "patient_id": str(patches[0].get("patient_id", "")),
            "dataset": str(patches[0].get("dataset", "")),
            "task_id": task_id,
            "selection_source": source,
            "group_label": f"{slide_id}/{source}/{mode}/{task_id}",
            "model_name": backend_config["model_id"],
            "model_revision": backend_config["revision"],
            "quantization": backend_config["quantization"],
            "model_config": backend_config,
            "mode": mode,
            "temperature": temperature,
            "repetition_penalty": repetition_penalty,
            "max_new_tokens": max_new_tokens,
            "seed": seed,
            "git_commit": git_commit,
            "n_patches_shown": n_shown,
            "patches": patch_metas,
            "prompt": prompt,
            "raw_responses": raw_responses,
            "per_patch_answers": per_patch_answers,
            "answer": aggregate,
            "parse_valid": parse_valid,
            "error": error,
        }
        all_records.append(record)

    elapsed = time.time() - t0
    print(f"[pipeline] Done: {len(all_records)}/{total} sets, {elapsed:.0f}s")
    return all_records


def _per_set_ground_truth(r: dict) -> int:
    return 1 if any(
        p.get("tumor_mask_overlap") == 1 for p in r["patches"].values()
    ) else 0


def compute_metrics(
    results: list[dict],
    slide_labels: dict[str, int] | None = None,
) -> dict:
    metrics = {"total_sets": len(results)}

    valid = [r for r in results if r["parse_valid"]]
    metrics["valid_sets"] = len(valid)
    metrics["coverage"] = len(valid) / max(len(results), 1)

    if not valid:
        return metrics

    source = valid[0]["selection_source"]
    mode = valid[0]["mode"]
    dataset = valid[0]["dataset"]
    metrics["source"] = source
    metrics["mode"] = mode
    metrics["dataset"] = dataset

    a = sum(1 for r in valid if r["answer"] == "A")
    b = sum(1 for r in valid if r["answer"] == "B")
    c = sum(1 for r in valid if r["answer"] == "C")
    total = a + b + c
    metrics["A"] = a
    metrics["B"] = b
    metrics["C"] = c

    if source == "oracle_tumor":
        metrics["sensitivity"] = a / total if total else float("nan")
        metrics["specificity"] = float("nan")

    elif source in ("oracle_non_tumor", "hard_negative"):
        metrics["sensitivity"] = float("nan")
        metrics["specificity"] = b / total if total else float("nan")

    elif source == "top_k":
        tp = fn = tn = fp = 0
        for r in valid:
            gt = _per_set_ground_truth(r)
            ans = r["answer"]
            if gt == 1 and ans == "A":
                tp += 1
            elif gt == 1 and ans in ("B", "C"):
                fn += 1
            elif gt == 0 and ans == "B":
                tn += 1
            elif gt == 0 and ans == "A":
                fp += 1
        sens = tp / (tp + fn) if (tp + fn) else float("nan")
        spec = tn / (tn + fp) if (tn + fp) else float("nan")
        metrics["TP"] = tp
        metrics["FN"] = fn
        metrics["TN"] = tn
        metrics["FP"] = fp
        metrics["sensitivity"] = sens
        metrics["specificity"] = spec
        metrics["accuracy"] = (tp + tn) / max(tp + fn + tn + fp, 1)

    else:
        metrics["sensitivity"] = float("nan")
        metrics["specificity"] = float("nan")

    if slide_labels:
        slide_preds: dict[str, list[str]] = defaultdict(list)
        for r in valid:
            slide_preds[r["slide_id"]].append(r["answer"])
        tp = fn = tn = fp = 0
        for slide_id, answers in slide_preds.items():
            gt = slide_labels.get(slide_id, 0)
            any_a = any(a == "A" for a in answers)
            if gt == 1 and any_a:
                tp += 1
            elif gt == 1 and not any_a:
                fn += 1
            elif gt == 0 and not any_a:
                tn += 1
            elif gt == 0 and any_a:
                fp += 1
        sens = tp / (tp + fn) if (tp + fn) else float("nan")
        spec = tn / (tn + fp) if (tn + fp) else float("nan")
        metrics["slide_level"] = {
            "TP": tp, "FN": fn, "TN": tn, "FP": fp,
            "sensitivity": sens,
            "specificity": spec,
            "balanced_accuracy": (sens + spec) / 2,
        }

    return metrics


def print_metrics(metrics: dict) -> None:
    print(f"\n{'='*70}")
    print("METRICS")
    print(f"{'='*70}")
    print(f"  Coverage: {metrics['coverage']:.4f} ({metrics['valid_sets']}/{metrics['total_sets']})")

    a = metrics.get("A", 0)
    b = metrics.get("B", 0)
    c = metrics.get("C", 0)
    total = a + b + c
    sens = metrics.get("sensitivity", float("nan"))
    spec = metrics.get("specificity", float("nan"))
    acc = metrics.get("accuracy", float("nan"))
    src = metrics.get("source", "?")
    md = metrics.get("mode", "?")
    ds = metrics.get("dataset", "?")

    s = f"{sens:.3f}" if not np.isnan(sens) else " n/a"
    sp = f"{spec:.3f}" if not np.isnan(spec) else " n/a"
    ac = f"{acc:.3f}" if not np.isnan(acc) else " n/a"

    print(f"  {ds:<20} {src:<16} {md:<10} A={a:>4} B={b:>4} C={c:>4} Total={total:>4}  "
          f"Sens={s}  Spec={sp}  Acc={ac}")

    sl = metrics.get("slide_level")
    if sl:
        print(f"  Slide-level (any A = tumor):")
        print(f"    TP={sl['TP']} FN={sl['FN']} TN={sl['TN']} FP={sl['FP']}")
        ss = f"{sl['sensitivity']:.3f}" if not np.isnan(sl['sensitivity']) else "n/a"
        sp = f"{sl['specificity']:.3f}" if not np.isnan(sl['specificity']) else "n/a"
        ba = f"{sl['balanced_accuracy']:.3f}" if not np.isnan(sl['balanced_accuracy']) else "n/a"
        print(f"    Sensitivity: {ss}  Specificity: {sp}  Balanced acc: {ba}")


def save_results(results: list[dict], output_path: Path, output_s3_prefix: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    task_ids = sorted(set(r.get("task_id", "") for r in results if r.get("task_id")))
    tid_slug = f"_{task_ids[0]}" if len(task_ids) == 1 else ""
    stem = output_path.stem

    jsonl_path = output_path.with_name(f"{stem}{tid_slug}.jsonl")
    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

    summary = []
    for r in results:
        summary.append({
            "patch_set_uid": r["patch_set_uid"],
            "slide_id": r["slide_id"],
            "patient_id": r["patient_id"],
            "dataset": r["dataset"],
            "selection_source": r["selection_source"],
            "mode": r["mode"],
            "answer": r["answer"],
            "parse_valid": r["parse_valid"],
            "n_patches": r["n_patches_shown"],
            "error": r["error"],
        })
    summary_df = pd.DataFrame(summary)
    csv_path = output_path.with_name(f"{stem}{tid_slug}.csv")
    summary_df.to_csv(csv_path, index=False)

    urls = {}
    for f in [jsonl_path, csv_path]:
        s3_key = f"{output_s3_prefix}/{f.name}"
        url = upload_to_s3(str(f), s3_key)
        urls[f.name] = (url, s3_key)
        print(f"  Uploaded: {url}")

    try:
        from clearml import Task
        clearml_task = Task.current_task()
        if clearml_task:
            for name, local_path in (("vlm_outputs_jsonl", jsonl_path), ("vlm_outputs_csv", csv_path)):
                s3_uri, _key = urls[local_path.name]
                clearml_task.upload_artifact(name=name, artifact_object=presign_url(s3_uri))
                clearml_task.set_parameter(f"outputs/{name}_uri", s3_uri)
            print(f"  Uploaded to ClearML artifacts")
    except Exception as exc:
        print(f"  ClearML artifact upload skipped: {exc}")


SOURCES_ALL = ["top_k", "oracle_tumor", "oracle_non_tumor", "hard_negative", "random", "diverse"]
MODES_ALL = ["single", "separate", "context"]


def main() -> int:
    parser = argparse.ArgumentParser(description="VLM inference pipeline")
    parser.add_argument("--registry_csv", default=REGISTRY_CSV_DEFAULT)
    parser.add_argument("--dataset", default="c16", choices=["c16", "c17", "c16_c17"])
    parser.add_argument("--model", default="med_gemma")
    parser.add_argument("--mode", default="all",
                        help="single / separate / context / all (default: all)")
    parser.add_argument("--source", default="all",
                        help="source type or 'all' (default: all)")
    parser.add_argument("--n_patches", type=int, default=3)
    parser.add_argument("--max_slides", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--4bit", action="store_true", dest="four_bit",
                        help="Enable 4-bit quantization (default: float16)")
    parser.add_argument("--cache_dir", default="/tmp/vlm_patch_cache")
    parser.add_argument("--output", default="")
    parser.add_argument("--output_s3", default="mil/vlm_results")
    args = parser.parse_args()
    load_4bit = args.four_bit

    if args.dataset == "c16":
        datasets = C16_DATASETS
    elif args.dataset == "c17":
        datasets = C17_DATASETS
    else:
        datasets = C16_DATASETS + C17_DATASETS

    sources = SOURCES_ALL if args.source == "all" else [args.source]
    modes = MODES_ALL if args.mode == "all" else [args.mode]

    _test_s3_access()

    registry = load_registry(
        args.registry_csv, datasets,
        sources=None, max_slides=args.max_slides,
    )

    slide_labels: dict[str, int] = {}
    if args.registry_csv.startswith("s3://") and len(registry) > 0:
        first = registry.iloc[0]
        minio_path = str(first.get("minio_path", ""))
        m = re.search(r'(\w+)_vlm_patches_([a-f0-9]+)\.tar\.gz', minio_path)
        if m:
            prefix = m.group(1)
            task_id = m.group(2)
            meta_key = f"mil/vlm_patches/{prefix}_vlm_metadata_{task_id}.csv"
            try:
                meta_df = read_csv_from_s3(meta_key)
                slide_labels = meta_df.groupby("slide_id")["label"].max().to_dict()
                n_tumor = sum(1 for v in slide_labels.values() if v == 1)
                print(f"  Loaded metadata: {len(slide_labels)} slides ({n_tumor} tumor, {len(slide_labels) - n_tumor} normal)")
            except Exception as exc:
                print(f"  WARNING: metadata not loaded ({exc})")

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    mod = importlib.import_module(f"vlm_backends.{args.model}")
    backend_cls_name = {
        "med_gemma": "MedGemmaBackend",
        "med_siglip": "MedSigLIPBackend",
        "quilt_llava": "QuiltLLaVABackend",
    }.get(args.model)
    if backend_cls_name is None:
        print(f"[pipeline] Unknown model: {args.model}")
        return 1
    BackendClass = getattr(mod, backend_cls_name)

    print(f"[pipeline] Model: {BackendClass.model_id()}")
    print(f"[pipeline] Run config: dataset={args.dataset} sources={sources} modes={modes} "
          f"temp={args.temperature} 4bit={load_4bit}")

    backend = BackendClass()
    print(f"[pipeline] Loading model...")
    try:
        backend.load(load_4bit=load_4bit)
    except Exception as exc:
        print(f"[pipeline] ERROR loading model: {exc}")
        traceback.print_exc()
        return 1

    backend_config = backend.config_snapshot()
    git_commit = _git_commit()
    all_results = []

    for ds in datasets:
        ds_registry = registry[registry["dataset"] == ds]
        if ds_registry.empty:
            continue

        print(f"\n{'='*60}")
        print(f"DATASET: {ds} ({len(ds_registry)} entries)")
        print(f"{'='*60}")

        for src in sources:
            print(f"\n  SOURCE: {src}")
            print(f"  {'-'*40}")

            patch_sets = build_patch_sets(
                ds_registry, source=src, n_patches=args.n_patches,
            )
            print(f"  Patch sets built: {len(patch_sets)}")

            for mode in modes:
                print(f"\n    --- Mode: {mode} ---")
                label = f"{ds}/{src}/{mode}"

                results = run_inference(
                    backend=backend,
                    patch_sets=patch_sets,
                    mode=mode,
                    cache_dir=cache_dir,
                    seed=args.seed,
                    temperature=args.temperature,
                    repetition_penalty=args.repetition_penalty,
                    max_new_tokens=args.max_new_tokens,
                    backend_config=backend_config,
                    git_commit=git_commit,
                )

                metrics = compute_metrics(results, slide_labels=slide_labels)
                print(f"    Metrics for {label}:")
                print_metrics(metrics)

                for r in results:
                    r["run_label"] = label
                    all_results.append(r)

    if not all_results:
        print("[pipeline] No results produced.")
        return 0

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(tempfile.gettempdir()) / f"vlm_{args.dataset}_{args.model}"

    save_results(all_results, output_path, args.output_s3)

    print(f"\n[pipefine] Done. {len(all_results)} total results at {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
