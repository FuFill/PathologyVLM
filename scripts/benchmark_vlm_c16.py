"""Benchmark VLM models on C16 mask-positive vs mask-negative patches.

Per-slide per-group logic: for each (slide_id, selection_source) group,
run VLM on every patch individually, compute sensitivity/specificity
against tumor_mask_overlap ground truth.

Models to compare:
  - MedGemma (generative)
  - Quilt-LLaVA (generative)
  - Gemma-3-27B (generative)
  - MedSigLIP (contrastive control)

Model selection via --model (or BENCHMARK_MODELS env var):
  all / quilt_llava / med_gemma / gemma3_27b / gemma_family (med_gemma +
  gemma3_27b) / med_siglip. MedSigLIP is always appended as control.

Precision via BENCHMARK_DTYPE env var: bf16 (default) or 4bit.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.s3_utils import get_minio_path_components, get_s3_client, presign_url, upload_to_s3

PROMPT_TEMPLATE_SINGLE = """You are a pathology AI analyzing an H&E stained lymph node tissue patch.

Below is a tissue patch (P1) from a lymph node biopsy.

Decide:
- A: Tumor features are clearly visible in this patch
- B: Tumor features are NOT visible in this patch
- C: The presented data is insufficient to decide

First, analyze the patch carefully. Then provide your FINAL ANSWER as a single letter (A, B, or C).

Reply with exactly one letter: A, B, or C. Do not add any other text, explanations, or JSON.

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
    text = raw.strip().upper()
    m = re.search(r'\bFINAL\s*ANSWER\s*:\s*([ABC])', text)
    if m:
        return m.group(1), True
    m = re.search(r'\bANSWER\s*:\s*([ABC])', text)
    if m:
        return m.group(1), True

    # Quilt-LLaVA emits Quilt-VQA-style JSON; try to read a decision key from it.
    try:
        json_blob = raw[raw.find("{"):]
        if json_blob:
            depth = 0
            end = -1
            for i, ch in enumerate(json_blob):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            if end > 0:
                obj = json.loads(json_blob[:end])
                for key in ("final_answer", "answer", "decision", "final", "label"):
                    val = str(obj.get(key, "")).strip().upper()
                    if val in ("A", "B", "C"):
                        return val, True
    except Exception:
        pass

    # Fallback: take the LAST standalone A/B/C (the decision comes at the end).
    letters = re.findall(r'\b([ABC])\b', text)
    if letters:
        return letters[-1], True
    return text[:50], False


def _download_patch(minio_path: str, cache_dir: Path) -> Optional[Path]:
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
    try:
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
        print(f"  [benchmark] WARNING: download failed for {internal_path}: {exc}")
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


def _resolve_aggregate_answer(separate_results: list[str]) -> str:
    if any(r == "A" for r in separate_results):
        return "A"
    if all(r == "B" for r in separate_results):
        return "B"
    return "C"

REGISTRY_CSV_DEFAULT = (
    "s3://pershin-medailab/Pathomorphology/CAMELYON/"
    "mil/vlm_patches_registry/patch_registry.csv"
)

C16_DATASETS = ("c16_native", "c17_to_c16")
SOURCES_SINGLE = ("top_k", "oracle_tumor", "oracle_non_tumor", "hard_negative")
RANDOM_SEEDS = (0, 1, 2)


def _per_group_metrics(records: list[dict]) -> dict:
    """Compute sensitivity / specificity for one source group.

    tumor_mask_overlap=1 (mask-pos): A = TP (tumor seen), B/C = FN
    tumor_mask_overlap=0 (mask-neg): B = TN (no tumor), A/C = FP
    """
    n = len(records)
    n_parsable = sum(1 for r in records if r.get("parse_valid"))

    pos = [r for r in records if r.get("tile_in_mask") == 1]
    neg = [r for r in records if r.get("tile_in_mask") == 0]

    tp = sum(1 for r in pos if r.get("answer") == "A")
    fn = sum(1 for r in pos if r.get("answer") in ("B", "C"))
    tn = sum(1 for r in neg if r.get("answer") == "B")
    fp = sum(1 for r in neg if r.get("answer") == "A")

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    balanced_acc = (sensitivity + specificity) / 2

    unique_raw = len(set(r.get("raw_response", "") for r in records if r.get("raw_response")))

    return {
        "n": n,
        "n_parsable": n_parsable,
        "n_mask_pos": len(pos),
        "n_mask_neg": len(neg),
        "TP": tp,
        "FN": fn,
        "TN": tn,
        "FP": fp,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": balanced_acc,
        "unique_raw": unique_raw,
        "mode_collapse": unique_raw / n if n else 0.0,
    }


def _run_model(
    backend,
    model_key: str,
    patches_df: pd.DataFrame,
    cache_dir: Path,
    seed: int,
    temperature: float,
    load_4bit: bool,
    max_patches: int = 0,
) -> list[dict]:
    print(f"\n{'='*60}")
    print(f"Running model: {model_key} ({backend.model_id()})")
    print(f"{'='*60}")

    print(f"  Loading model weights... (load_4bit={load_4bit})")
    try:
        backend.load(load_4bit=load_4bit)
    except Exception as exc:
        print(f"  [ERROR] Failed to load {model_key}: {exc}")
        import traceback
        traceback.print_exc()
        raise
    print(f"  Model loaded.")

    groups: dict[str, list[dict]] = defaultdict(list)
    for _, row in patches_df.iterrows():
        src = str(row.get("selection_source", "unknown"))
        seed_val = row.get("random_seed")
        if src == "random" and pd.notna(seed_val):
            group_key = f"random_seed_{int(seed_val)}"
        else:
            group_key = src
        groups[group_key].append(row.to_dict())

    all_records: list[dict] = []
    t0 = time.time()
    total = sum(len(v) for v in groups.values())

    for group_key in sorted(groups):
        group_patches = groups[group_key]
        print(f"\n  --- {group_key} ({len(group_patches)} patches) ---")

        for pi, patch in enumerate(group_patches):
            minio_path = str(patch.get("minio_path", ""))
            local_path = _download_patch(minio_path, cache_dir) if minio_path else None
            img = _load_image(local_path) if local_path and local_path.exists() else None

            if img is None:
                print(f"    [{pi+1}/{len(group_patches)}] SKIP (no image)")
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

            if pi < 3 or not valid:
                trunc = raw[:2000].replace("\n", "\\n")
                print(f"    RAW[{pi+1}]: {trunc}")

            all_records.append({
                "model": model_key,
                "patch_uid": str(patch.get("patch_uid", "")),
                "slide_id": str(patch.get("slide_id", "")),
                "group": group_key,
                "selection_source": str(patch.get("selection_source", "")),
                "tile_in_mask": (
                    int(patch.get("tumor_mask_overlap", 0))
                    if pd.notna(patch.get("tumor_mask_overlap"))
                    else 0
                ),
                "raw_response": raw,
                "answer": ans,
                "parse_valid": valid,
            })
            print(f"    [{pi+1}/{len(group_patches)}] answer: {ans}", end="\r")

        group_as = [r["answer"] for r in all_records if r["group"] == group_key]
        print(f"    A={group_as.count('A')} B={group_as.count('B')} C={group_as.count('C')}")

    elapsed = time.time() - t0
    print(f"\n  Done: {total} patches, {elapsed:.0f}s")
    return all_records


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


MIG_PROFILES = ("7g.79gb", "4g.40gb", "3g.40gb", "2g.20gb", "1g.10gb")


def _print_nvidia_smi() -> None:
    try:
        res = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=30
        )
        print(res.stdout or res.stderr or "(no output)")
    except Exception as exc:
        print(f"nvidia-smi failed: {exc}")


def _mig_enabled_no_instances() -> bool:
    try:
        res = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=30
        )
        smi = res.stdout or ""
    except Exception as exc:
        print(f"[benchmark] nvidia-smi failed: {exc}")
        return False
    return "Enabled" in smi and "No MIG devices found" in smi


def _provision_mig() -> bool:
    """Try to create a MIG instance from inside the container (requires
    NVIDIA_MIG_CONFIG_DEVICES=all in docker args). Sets CUDA_VISIBLE_DEVICES
    to the new MIG UUID. Must run BEFORE the first torch.cuda call, because
    cuInit reads CUDA_VISIBLE_DEVICES once.
    """
    for profile in MIG_PROFILES:
        print(f"[benchmark] MIG enabled with no instances - trying profile {profile}...")
        try:
            r = subprocess.run(
                ["nvidia-smi", "mig", "-cgi", profile, "-C"],
                capture_output=True,
                text=True,
                timeout=120,
            )
            print((r.stdout or r.stderr or "").strip()[:400])
            if r.returncode == 0:
                break
        except Exception as exc:
            print(f"  nvidia-smi mig failed: {exc}")
    else:
        print("[benchmark] MIG provisioning failed for all profiles")
        return False

    try:
        res = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30
        )
        for line in (res.stdout or "").splitlines():
            if "MIG" in line and "UUID:" in line:
                uuid = line.split("UUID:")[-1].strip().split(")")[0].strip()
                os.environ["CUDA_VISIBLE_DEVICES"] = uuid
                print(f"[benchmark] CUDA_VISIBLE_DEVICES={uuid}")
                return True
    except Exception as exc:
        print(f"  failed to read MIG UUID: {exc}")
    return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark VLM models on C16 control patches"
    )
    parser.add_argument("--registry_csv", default=REGISTRY_CSV_DEFAULT)
    parser.add_argument(
        "--model",
        default=os.environ.get("BENCHMARK_MODELS", "quilt_llava"),
        choices=["all", "quilt_llava", "med_gemma", "med_siglip", "gemma3_27b", "gemma_family"],
    )
    parser.add_argument("--output", default="")
    parser.add_argument("--output_s3", default="mil/vlm_results/c16_benchmark")
    parser.add_argument("--cache_dir", default="/tmp/vlm_patch_cache")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--max_patches", type=int, default=0,
        help="Debug: run only the first N patches (0 = all)",
    )
    args = parser.parse_args()

    registry_path = _resolve_registry(args.registry_csv)
    print(f"[benchmark] Loading registry: {registry_path}")
    registry = pd.read_csv(registry_path)
    print(f"  Total registry entries: {len(registry)}")

    registry = registry[registry["dataset"].isin(C16_DATASETS)]
    print(f"  After C16 dataset filter: {len(registry)}")

    non_diverse = registry[registry.get("is_diverse", 0) == 0].copy()
    non_random = non_diverse[non_diverse["selection_source"] != "random"]
    random_part = non_diverse[non_diverse["selection_source"] == "random"].copy()

    if "random_seed" in random_part.columns:
        for seed_val in RANDOM_SEEDS:
            subset = random_part[random_part["random_seed"] == seed_val]
            if not subset.empty:
                n = len(subset)
                print(f"    random_seed_{seed_val}: {n} patches")
    else:
        print("    WARNING: no random_seed column in registry")

    patches_df = pd.concat([non_random, random_part], ignore_index=True)
    print(f"  Total patches for benchmark: {len(patches_df)}")

    slides_in_data = patches_df["slide_id"].nunique()
    print(f"  Slides: {slides_in_data}")
    for src in sorted(patches_df["selection_source"].unique()):
        n = len(patches_df[patches_df["selection_source"] == src])
        print(f"    {src}: {n}")

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    all_configs = [
        ("quilt_llava", "vlm_backends.quilt_llava", "QuiltLLaVABackend"),
        ("med_gemma", "vlm_backends.med_gemma", "MedGemmaBackend"),
        ("med_siglip", "vlm_backends.med_siglip", "MedSigLIPBackend"),
        ("gemma3_27b", "vlm_backends.gemma3", "Gemma3Backend"),
    ]
    family_models = {"med_gemma", "gemma3_27b"} if args.model == "gemma_family" else set()
    model_configs = [
        c for c in all_configs
        if args.model in ("all", c[0]) or c[0] in family_models or c[0] == "med_siglip"
    ]
    print(f"[benchmark] Models to run: {[c[0] for c in model_configs]}")

    dtype = os.environ.get("BENCHMARK_DTYPE", "bf16").strip().lower()
    if dtype in ("bf16", "fp16", "16bit", ""):
        load_4bit = False
        print("[benchmark] Precision: bf16 (no quantization)")
    elif dtype in ("4bit", "int4"):
        load_4bit = True
        print("[benchmark] Precision: 4bit nf4 (bitsandbytes)")
    else:
        print(f"[benchmark] WARNING: unknown BENCHMARK_DTYPE={dtype!r}, using bf16")
        load_4bit = False

    cuda_models = [
        c for c in model_configs
        if getattr(getattr(importlib.import_module(c[1]), c[2]), "requires_cuda", False)
    ]
    if cuda_models and _mig_enabled_no_instances():
        print("[benchmark] MIG enabled but no MIG instances - attempting provisioning...")
        print("--- nvidia-smi output (before) ---")
        _print_nvidia_smi()
        provisioned = _provision_mig()
        print("--- nvidia-smi output (after) ---")
        _print_nvidia_smi()
        print(f"[benchmark] MIG provisioning success: {provisioned}")

    cuda_ok = torch.cuda.is_available()
    print(f"[benchmark] CUDA available: {cuda_ok}")
    if cuda_ok:
        try:
            print(f"[benchmark] CUDA device: {torch.cuda.get_device_name(0)}")
            props = torch.cuda.get_device_properties(0)
            print(f"[benchmark] CUDA VRAM: {props.total_memory / 1e9:.1f} GB")
        except Exception:
            pass

    if cuda_models and not cuda_ok:
        print("[benchmark] ERROR: selected models require CUDA but no GPU is available "
              f"inside the container: {[c[0] for c in cuda_models]}")
        print("[benchmark] Failing fast (expected: every agent has a working CUDA GPU).")
        return 2

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
                load_4bit=load_4bit,
                max_patches=args.max_patches,
            )
            all_results[model_key] = results
        except Exception as exc:
            print(f"[benchmark] SKIPPING {model_key}: {exc}")
            import traceback
            traceback.print_exc()
            all_results[model_key] = []

    print(f"\n{'='*70}")
    print("C16 VLM BENCHMARK RESULTS")
    print(f"{'='*70}")

    rows = []
    for model_key, records in all_results.items():
        print(f"\n  === {model_key} ===")

        groups_in_data = sorted(set(r["group"] for r in records))
        for gk in groups_in_data:
            grp = [r for r in records if r["group"] == gk]
            m = _per_group_metrics(grp)
            rows.append({"model": model_key, "group": gk, **m})
            print(f"\n    --- {gk} (n={m['n']}) ---")
            for k in ("n_parsable", "sensitivity", "specificity", "balanced_accuracy",
                      "n_mask_pos", "n_mask_neg", "TP", "FN", "TN", "FP",
                      "unique_raw", "mode_collapse"):
                v = m[k]
                if isinstance(v, float):
                    print(f"      {k}: {v:.4f}")
                else:
                    print(f"      {k}: {v}")
            print()

        total_n = len(records)
        pos_recs = [r for r in records if r.get("tile_in_mask") == 1]
        neg_recs = [r for r in records if r.get("tile_in_mask") == 0]
        tp = sum(1 for r in pos_recs if r.get("answer") == "A")
        fn = sum(1 for r in pos_recs if r.get("answer") in ("B", "C"))
        tn = sum(1 for r in neg_recs if r.get("answer") == "B")
        fp = sum(1 for r in neg_recs if r.get("answer") == "A")
        sens_total = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        spec_total = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
        bacc_total = (sens_total + spec_total) / 2
        print(f"    --- ALL GROUPS (n={total_n}) ---")
        print(f"      mask-pos: {len(pos_recs)}, mask-neg: {len(neg_recs)}")
        print(f"      TP={tp} FN={fn} TN={tn} FP={fp}")
        print(f"      sensitivity: {sens_total:.4f}")
        print(f"      specificity: {spec_total:.4f}")
        print(f"      balanced_accuracy: {bacc_total:.4f}")

        # --- Model selection decision ---
        print(f"\n    === MODEL SELECTION: {model_key} ===")
        oracle_tumor_recs = [r for r in records if r.get("group") == "oracle_tumor"]
        oracle_non_tumor_recs = [r for r in records if r.get("group") == "oracle_non_tumor"]

        if oracle_tumor_recs:
            ot_m = _per_group_metrics(oracle_tumor_recs)
            print(f"      oracle_tumor: sens={ot_m['sensitivity']:.4f}  "
                  f"parsable={ot_m['n_parsable']}/{ot_m['n']}  "
                  f"unique_raw={ot_m['unique_raw']}")
        if oracle_non_tumor_recs:
            ont_m = _per_group_metrics(oracle_non_tumor_recs)
            print(f"      oracle_non_tumor: spec={ont_m['specificity']:.4f}  "
                  f"parsable={ont_m['n_parsable']}/{ont_m['n']}  "
                  f"unique_raw={ont_m['unique_raw']}")

        if oracle_tumor_recs and oracle_non_tumor_recs:
            ot_sens = ot_m.get("sensitivity", 0)
            ont_spec = ont_m.get("specificity", 0)
            ot_ba = ot_m.get("balanced_accuracy", 0)
            ont_ba = ont_m.get("balanced_accuracy", 0)
            mean_ba = (ot_ba + ont_ba) / 2
            parse_rate = (ot_m.get("n_parsable", 0) + ont_m.get("n_parsable", 0)) / max(ot_m.get("n", 1) + ont_m.get("n", 1), 1)
            mode_collapse_ot = ot_m.get("mode_collapse", 1) > 0.8
            mode_collapse_ont = ont_m.get("mode_collapse", 1) > 0.8

            print(f"      Combined: mean_balanced_acc={mean_ba:.4f}  parse_rate={parse_rate:.4f}")
            discriminates = ot_sens > 0.4 and ont_spec > 0.4 and mean_ba > 0.45
            if mode_collapse_ot or mode_collapse_ont:
                print(f"      WARNING: mode collapse detected (unique_raw close to 1)")
            if parse_rate < 0.8:
                print(f"      WARNING: low parse rate ({parse_rate:.4f})")

            if discriminates:
                print(f"      >>> PASS: model distinguishes mask-positive from mask-negative")
            else:
                print(f"      >>> FAIL: model does NOT reliably distinguish groups")
                print(f"      >>> C17 full run will NOT be meaningful for {model_key}")

    summary_df = pd.DataFrame(rows)
    tid = os.environ.get("CLEARML_TASK_ID", "")
    suffix = f"_{tid}" if tid else ""
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(tempfile.gettempdir()) / f"c16_vlm_benchmark{suffix}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "config": {
            "seed": args.seed,
            "temperature": args.temperature,
        },
        "models": {
            k: v for k, v in all_results.items()
        },
    }
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    csv_path = output_path.with_suffix(".csv")
    summary_df.to_csv(csv_path, index=False)

    urls = {}
    for f in [output_path, csv_path]:
        s3_key = f"{args.output_s3}/{f.name}"
        url = upload_to_s3(str(f), s3_key)
        urls[f.name] = url
        print(f"  Uploaded: {url}")

    try:
        from clearml import Task
        clearml_task = Task.current_task()
        if clearml_task:
            clearml_task.upload_artifact(name="benchmark_json", artifact_object=presign_url(urls[output_path.name]))
            clearml_task.upload_artifact(name="benchmark_csv", artifact_object=presign_url(urls[csv_path.name]))
            clearml_task.set_parameter("outputs/benchmark_json_uri", urls[output_path.name])
            clearml_task.set_parameter("outputs/benchmark_csv_uri", urls[csv_path.name])
            print(f"  Uploaded to ClearML artifacts")
    except Exception as exc:
        print(f"  ClearML artifact upload skipped: {exc}")

    print(f"\n[benchmark] Done. Results at {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
