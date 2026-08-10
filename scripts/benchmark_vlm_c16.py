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

First, analyze the patch carefully and briefly explain your reasoning.

Then, after your reasoning, append the phrase "FINAL ANSWER:" followed by your choice (A, B, or C) at the end of your response.

FINAL ANSWER:"""

PROMPT_TEMPLATE_SINGLE_QUILT = """You are a pathology AI analyzing an H&E stained lymph node tissue patch.

Below is a tissue patch (P1) from a lymph node biopsy.

Which of the following best describes this patch?
A: Tumor features are clearly visible in this patch
B: Tumor features are NOT visible in this patch
C: The presented data is insufficient to decide

Begin your answer with exactly one letter (A, B, or C), then briefly explain your reasoning.

Answer:"""

PROMPT_TEMPLATE_CONTEXT = """You are a pathology AI analyzing H&E stained lymph node tissue patches.

Below are three tissue patches (P1, P2, P3) from a lymph node biopsy.

For each patch, decide:
- A: Tumor features are clearly visible in this patch
- B: Tumor features are NOT visible in this patch
- C: The presented data is insufficient to decide

First, analyze each patch independently and briefly explain your reasoning. Then give your overall assessment:
- A if tumor is evident in at least one patch
- B if no tumor features are seen in any patch and tissue is adequate
- C if tissue is inadequate, ambiguous, or you cannot make a determination

Then, after your reasoning, append the phrase "FINAL ANSWER:" followed by your choice (A, B, or C) at the end of your response.

FINAL ANSWER:"""

PROMPT_TEMPLATE_SEPARATE = """You are a pathology AI analyzing an H&E stained lymph node tissue patch.

Below is a tissue patch (P1) from a lymph node biopsy.

Decide:
- A: Tumor features are clearly visible in this patch
- B: Tumor features are NOT visible in this patch
- C: The presented data is insufficient to decide

First, analyze the patch carefully and briefly explain your reasoning.

Then, after your reasoning, append the phrase "FINAL ANSWER:" followed by your choice (A, B, or C) at the end of your response.

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

    # Fallback: decision-letter-first responses (e.g. "B\nRationale: ...").
    # Case-sensitive on the raw text: the model copies the option letters
    # ("A"/"B"/"C") from the prompt in uppercase, while prose articles are
    # lowercase ("a few ..."). The leading letter must be followed by a
    # separator or end-of-string; otherwise take the LAST standalone
    # uppercase letter in the trailing window (the answer letter, if
    # written, is at the end).
    raw_stripped = raw.strip()
    m = re.match(r'^([ABC])(?=[\n:.,;)]|$)', raw_stripped)
    if m:
        return m.group(1), True
    matches = re.findall(r'\b([ABC])\b', raw_stripped[-80:])
    if matches:
        return matches[-1], True
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
RANDOM_SEEDS = (42, 123, 456)


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
    revision: Optional[str] = None,
) -> list[dict]:
    print(f"\n{'='*60}")
    print(f"Running model: {model_key} ({backend.model_id()})")
    print(f"{'='*60}")

    if getattr(backend, "_model", None) is None:
        print(f"  Loading model weights... (load_4bit={load_4bit}, revision={revision})")
        try:
            backend.load(load_4bit=load_4bit, revision=revision)
        except Exception as exc:
            print(f"  [ERROR] Failed to load {model_key}: {exc}")
            import traceback
            traceback.print_exc()
            raise
    else:
        print("  Model already loaded, reusing.")
    print(f"  Model loaded.")
    print(f"  Resolved revision: {getattr(backend, '_revision', None)}")
    print(f"  Prompt (mode=single): {PROMPT_TEMPLATE_SINGLE[:140].replace(chr(10), ' ')}...")

    groups: dict[str, list[dict]] = defaultdict(list)
    for _, row in patches_df.iterrows():
        src = str(row.get("selection_source", "unknown"))
        ctx = str(row.get("context_set", "")).strip().lower() or "unknown"
        seed_val = row.get("random_seed")
        if src == "random" and pd.notna(seed_val):
            group_key = f"random_seed_{int(seed_val)}|{ctx}"
        else:
            group_key = f"{src}|{ctx}"
        groups[group_key].append(row.to_dict())

    all_records: list[dict] = []
    t0 = time.time()
    total = sum(len(v) for v in groups.values())

    group_keys = sorted(groups)
    group_keys = [k for k in group_keys if k.startswith("top_k")] + [
        k for k in group_keys if not k.startswith("top_k")
    ]
    for group_key in group_keys:
        group_patches = groups[group_key]
        print(f"\n  --- {group_key} ({len(group_patches)} patches) ---")

        for pi, patch in enumerate(group_patches):
            minio_path = str(patch.get("minio_path", ""))
            local_path = _download_patch(minio_path, cache_dir) if minio_path else None
            img = _load_image(local_path) if local_path and local_path.exists() else None

            if img is None:
                print(f"    [{pi+1}/{len(group_patches)}] SKIP (no image)")
                continue

            if pi == 0:
                print(f"    image size: {img.size}")

            prompt = (
                PROMPT_TEMPLATE_SINGLE_QUILT
                if model_key == "quilt_llava"
                else PROMPT_TEMPLATE_SINGLE
            )
            try:
                raw = backend.generate(
                    images=[img],
                    prompt=prompt,
                    max_new_tokens=128,
                    temperature=temperature,
                    repetition_penalty=1.0,
                    seed=seed,
                )
            except Exception as exc:
                raw = f"ERROR: {exc}"

            ans, valid = _parse_answer(raw)

            trunc = raw.replace("\n", "\\n")
            print(f"    [{pi+1}/{len(group_patches)}] RAW: {trunc}  -> {ans}")

            metrics = {}
            if hasattr(backend, "diagnostics"):
                metrics = backend.diagnostics()

            all_records.append({
                "model": model_key,
                "mode": "single",
                "patch_uid": str(patch.get("patch_uid", "")),
                "slide_id": str(patch.get("slide_id", "")),
                "dataset": str(patch.get("dataset", "")),
                "context_set": str(patch.get("context_set", "")),
                "group": group_key,
                "selection_source": str(patch.get("selection_source", "")),
                "tile_in_mask": (
                    int(patch.get("tumor_mask_overlap", 0))
                    if pd.notna(patch.get("tumor_mask_overlap"))
                    else 0
                ),
                "prompt": prompt,
                "raw_response": raw,
                "answer": ans,
                "parse_valid": valid,
                **({"metrics": metrics} if metrics else {}),
            })

        group_records = [r for r in all_records if r["group"] == group_key]
        group_as = [r["answer"] for r in group_records]
        group_raws = [r["raw_response"] for r in group_records if r["raw_response"]]
        avg_raw_len = sum(len(r) for r in group_raws) / len(group_raws) if group_raws else 0
        unique_raw = len(set(group_raws))
        print(f"    A={group_as.count('A')} B={group_as.count('B')} C={group_as.count('C')} "
              f"unique_raw={unique_raw} avg_raw_len={avg_raw_len:.0f}")

    elapsed = time.time() - t0
    print(f"\n  Done: {total} patches, {elapsed:.0f}s")
    return all_records


def _build_patch_sets(patches_df: pd.DataFrame, n_patches: int) -> list[list[dict]]:
    """Group registry rows into per-slide patch sets for separate/context modes.

    Key = (dataset, slide_id, selection_source, context_set, random_seed), so
    the three random draws stay independent sets, standard/diverse stay
    separate, and c16_native / c17_to_c16 slides do not mix. Within a group,
    patches are taken by ascending rank (top-n for top_k, first n otherwise).
    """
    df = patches_df.copy()
    if "random_seed" not in df.columns:
        df["random_seed"] = 0
    df["random_seed"] = df["random_seed"].fillna(0)
    df["rank"] = pd.to_numeric(df.get("rank", 0), errors="coerce").fillna(2 ** 31)
    keys = ["dataset", "slide_id", "selection_source", "context_set", "random_seed"]
    sets: list[list[dict]] = []
    for _, grp in df.groupby(keys, sort=False):
        grp = grp.sort_values("rank")
        sets.append([r.to_dict() for _, r in grp.head(n_patches).iterrows()])
    return sets


def _run_model_sets(
    backend,
    model_key: str,
    patches_df: pd.DataFrame,
    cache_dir: Path,
    seed: int,
    temperature: float,
    load_4bit: bool,
    mode: str,
    n_patches: int,
    max_patches: int = 0,
    revision: Optional[str] = None,
) -> list[dict]:
    print(f"\n{'='*60}")
    print(f"Running model: {model_key} ({backend.model_id()}) mode={mode}")
    print(f"{'='*60}")

    if getattr(backend, "_model", None) is None:
        print(f"  Loading model weights... (load_4bit={load_4bit}, revision={revision})")
        try:
            backend.load(load_4bit=load_4bit, revision=revision)
        except Exception as exc:
            print(f"  [ERROR] Failed to load {model_key}: {exc}")
            import traceback
            traceback.print_exc()
            raise
    else:
        print("  Model already loaded, reusing.")
    print("  Model loaded.")
    print(f"  Resolved revision: {getattr(backend, '_revision', None)}")

    patch_sets = _build_patch_sets(patches_df, n_patches)
    patch_sets = [s for s in patch_sets if s[0].get("selection_source") == "top_k"] + [
        s for s in patch_sets if s[0].get("selection_source") != "top_k"
    ]
    if max_patches > 0:
        patch_sets = patch_sets[:max_patches]
    print(f"  Patch sets: {len(patch_sets)}")

    prompt = PROMPT_TEMPLATE_SEPARATE if mode == "separate" else PROMPT_TEMPLATE_CONTEXT
    print(f"  Prompt (mode={mode}): {prompt[:140].replace(chr(10), ' ')}...")

    all_records: list[dict] = []
    t0 = time.time()
    for si, patches in enumerate(patch_sets):
        if si % 20 == 0:
            print(f"  [{si}/{len(patch_sets)}] slide {patches[0].get('slide_id', '?')}")

        pil_images: list[Image.Image] = []
        patch_info: dict[str, dict] = {}
        for patch in patches:
            minio_path = str(patch.get("minio_path", ""))
            local_path = _download_patch(minio_path, cache_dir) if minio_path else None
            img = _load_image(local_path) if local_path and local_path.exists() else None
            if img is None:
                continue
            key = f"P{len(pil_images) + 1}"
            pil_images.append(img)
            patch_info[key] = {
                "patch_uid": str(patch.get("patch_uid", "")),
                "region_uid": str(patch.get("region_uid", "")),
                "relative_path": str(patch.get("relative_path", "")),
                "rank": int(patch.get("rank", 0)) if pd.notna(patch.get("rank")) else 0,
                "tile_in_mask": (
                    int(patch.get("tumor_mask_overlap", 0))
                    if pd.notna(patch.get("tumor_mask_overlap"))
                    else 0
                ),
            }
            if len(pil_images) == n_patches:
                break

        if not pil_images:
            continue

        p0 = patches[0]
        set_id = hashlib.sha256(
            "|".join([
                str(p0.get("dataset", "")),
                str(p0.get("slide_id", "")),
                str(p0.get("selection_source", "")),
                str(p0.get("context_set", "")),
                str(p0.get("random_seed", 0)),
                mode,
            ]).encode()
        ).hexdigest()[:16]

        raw_responses: list[str] = []
        per_patch_answers: list[str] = []
        metrics = {}

        try:
            if mode == "separate":
                for img in pil_images:
                    raw = backend.generate(
                        images=[img],
                        prompt=prompt,
                        max_new_tokens=128,
                        temperature=temperature,
                        repetition_penalty=1.0,
                        seed=seed,
                    )
                    raw_responses.append(raw)
                    ans, _ = _parse_answer(raw)
                    per_patch_answers.append(ans)
                answer = _resolve_aggregate_answer(per_patch_answers)
                parse_valid = answer in ("A", "B", "C")
            else:
                raw = backend.generate(
                    images=pil_images,
                    prompt=prompt,
                    max_new_tokens=128,
                    temperature=temperature,
                    repetition_penalty=1.0,
                    seed=seed,
                )
                raw_responses = [raw]
                answer, parse_valid = _parse_answer(raw)
                per_patch_answers = [answer]
                if hasattr(backend, "diagnostics"):
                    metrics = backend.diagnostics() or {}
        except Exception as exc:
            raw_responses.append(f"ERROR: {exc}")
            answer = ""
            parse_valid = False

        if si < 3 or not parse_valid:
            for ri, r in enumerate(raw_responses):
                print(f"    RAW[{ri}]: {r.replace(chr(10), chr(92) + 'n')}")

        seed_val = p0.get("random_seed")
        src = str(p0.get("selection_source", "unknown"))
        ctx = str(p0.get("context_set", "")).strip().lower() or "unknown"
        if src == "random" and pd.notna(seed_val):
            group_key = f"random_seed_{int(seed_val)}|{ctx}"
        else:
            group_key = f"{src}|{ctx}"

        all_records.append({
            "model": model_key,
            "mode": mode,
            "patch_set_uid": set_id,
            "slide_id": str(p0.get("slide_id", "")),
            "dataset": str(p0.get("dataset", "")),
            "context_set": str(p0.get("context_set", "")),
            "group": group_key,
            "selection_source": src,
            "random_seed": int(seed_val) if pd.notna(seed_val) else 0,
            "tile_in_mask": int(any(p["tile_in_mask"] == 1 for p in patch_info.values())),
            "n_patches": len(pil_images),
            "patches": patch_info,
            "prompt": prompt,
            "raw_responses": raw_responses,
            "per_patch_answers": per_patch_answers,
            "answer": answer,
            "parse_valid": parse_valid,
            **({"metrics": metrics} if metrics else {}),
        })

    elapsed = time.time() - t0
    print(f"\n  Done: {len(all_records)} sets, {elapsed:.0f}s")
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
        "--mode", default="single",
        choices=["single", "separate", "context"],
        help="Patch feeding mode: single = per-patch calls (default); "
             "separate = n_patches per slide, one call each, aggregated answer "
             "(any A->A, all B->B, else C); context = n_patches per slide in "
             "one multi-image call. May be overridden by BENCHMARK_MODE env.",
    )
    parser.add_argument(
        "--n_patches", type=int, default=3,
        help="Patches per slide for separate/context modes",
    )
    parser.add_argument(
        "--max_patches", type=int, default=0,
        help="Debug: run only the first N patches (0 = all)",
    )
    args = parser.parse_args()

    mode = os.environ.get("BENCHMARK_MODE", args.mode)
    if mode == "both":
        run_modes = ["separate", "context"]
    elif mode in ("single", "separate", "context"):
        run_modes = [mode]
    else:
        print(f"[benchmark] WARNING: unknown BENCHMARK_MODE={mode!r}, falling back to {args.mode}")
        run_modes = [args.mode]

    registry_path = _resolve_registry(args.registry_csv)
    print(f"[benchmark] Loading registry: {registry_path}")
    registry = pd.read_csv(registry_path)
    print(f"  Total registry entries: {len(registry)}")

    registry = registry[registry["dataset"].isin(C16_DATASETS)]
    print(f"  After C16 dataset filter: {len(registry)}")

    non_random = registry[registry["selection_source"] != "random"].copy()
    random_part = registry[registry["selection_source"] == "random"].copy()

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
        if args.model in ("all", c[0]) or c[0] in family_models
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

    revision = os.environ.get("BENCHMARK_REVISION", "").strip() or None
    if revision:
        print(f"[benchmark] Pinned model revision: {revision}")
    else:
        print("[benchmark] Model revision: latest (not pinned)")

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
    models_meta: dict[str, dict] = {}
    for model_key, module_path, class_name in model_configs:
        try:
            mod = importlib.import_module(module_path)
            backend_cls = getattr(mod, class_name)
            backend = backend_cls()
            if mode == "single":
                results = _run_model(
                    backend=backend,
                    model_key=model_key,
                    patches_df=patches_df,
                    cache_dir=cache_dir,
                    seed=args.seed,
                    temperature=args.temperature,
                    load_4bit=load_4bit,
                    max_patches=args.max_patches,
                    revision=revision,
                )
            else:
                mode_results = []
                for _mode in run_modes:
                    print(f"\n===== MODE: {_mode} =====")
                    mode_results.append(_run_model_sets(
                        backend=backend,
                        model_key=model_key,
                        patches_df=patches_df,
                        cache_dir=cache_dir,
                        seed=args.seed,
                        temperature=args.temperature,
                        load_4bit=load_4bit,
                        mode=_mode,
                        n_patches=args.n_patches,
                        max_patches=args.max_patches,
                        revision=revision,
                    ))
                results = [r for res in mode_results for r in res]
            models_meta[model_key] = {
                "model_id": backend.model_id(),
                "revision": getattr(backend, "_revision", None),
                "quantization": "bf16" if not load_4bit else "nf4",
            }
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
        modes_in_data = sorted({r.get("mode", "unknown") for r in records})
        for rec_mode in modes_in_data:
            recs = [r for r in records if r.get("mode") == rec_mode]
            print(f"\n  === {model_key} [mode={rec_mode}] ===")

            groups_in_data = sorted(set(r["group"] for r in recs))
            for gk in groups_in_data:
                grp = [r for r in recs if r["group"] == gk]
                m = _per_group_metrics(grp)
                rows.append({"model": model_key, "mode": rec_mode, "group": gk, **m})
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

            total_n = len(recs)
            pos_recs = [r for r in recs if r.get("tile_in_mask") == 1]
            neg_recs = [r for r in recs if r.get("tile_in_mask") == 0]
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
            print(f"\n    === MODEL SELECTION: {model_key} [mode={rec_mode}] ===")
            oracle_tumor_recs = [r for r in recs if r.get("group", "").startswith("oracle_tumor")]
            oracle_non_tumor_recs = [r for r in recs if r.get("group", "").startswith("oracle_non_tumor")]

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
            "mode": mode,
            "n_patches": args.n_patches,
            "git_commit": _git_commit(),
            "revision": revision,
            "modes": run_modes,
            "prompts": {
                "single": PROMPT_TEMPLATE_SINGLE,
                "separate": PROMPT_TEMPLATE_SEPARATE,
                "context": PROMPT_TEMPLATE_CONTEXT,
            },
            "models": models_meta,
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
