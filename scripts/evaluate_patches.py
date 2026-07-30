"""Patch-level VLM answer consistency evaluation.

Computes false-positive rate, false-negative rate, correct abstain,
coverage, and calibration.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.s3_utils import get_s3_client, upload_to_s3

REGISTRY_CSV_DEFAULT = "s3://pershin-medailab/Pathomorphology/CAMELYON/mil/vlm_patches_registry/patch_registry.csv"


def _load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _resolve_registry_csv(path: str) -> str:
    if path.startswith("s3://"):
        parts = path.replace("s3://", "", 1).split("/", 1)
        if len(parts) == 2:
            bucket, key = parts
            client = get_s3_client()
            tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
            client.download_file(bucket, key, tmp.name)
            print(f"[evaluate_patches] Downloaded registry from {path} to {tmp.name}")
            return tmp.name
    return path


def _compute_patch_metrics(results_path: str, registry_csv: str) -> dict:
    rows = _load_jsonl(results_path)
    registry = pd.read_csv(_resolve_registry_csv(registry_csv))

    uid_to_mask = {}
    for _, r in registry.iterrows():
        uid = str(r.get("region_uid", ""))
        if uid:
            uid_to_mask[uid] = int(r.get("tumor_mask_overlap", 0))

    per_patch_results = []
    for r in rows:
        patches = r.get("patches", {})
        patch_tile_in_mask = []
        for pk, pv in patches.items():
            uid = pv.get("region_uid", "")
            mask_val = uid_to_mask.get(uid, 0)
            patch_tile_in_mask.append(mask_val)

        answer = r.get("answer", "")
        per_patch_results.append({
            "slide_id": r.get("slide_id", ""),
            "answer": answer,
            "parse_valid": r.get("parse_valid", False),
            "n_patches": len(patches),
            "tile_in_mask_values": patch_tile_in_mask,
            "any_mask_positive": any(v == 1 for v in patch_tile_in_mask),
            "all_mask_negative": all(v == 0 for v in patch_tile_in_mask),
        })

    n = len(per_patch_results)
    n_parsable = sum(1 for r in per_patch_results if r["parse_valid"] and r["answer"])

    false_positive = [
        r for r in per_patch_results
        if r["answer"] == "A" and r["all_mask_negative"] and r["parse_valid"]
    ]
    false_negative = [
        r for r in per_patch_results
        if r["answer"] in ("B", "C") and r["any_mask_positive"] and r["parse_valid"]
    ]
    correct_abstain = [
        r for r in per_patch_results
        if r["answer"] == "C" and r["all_mask_negative"] and r["parse_valid"]
    ]

    n_positive_sets = sum(1 for r in per_patch_results if r["any_mask_positive"])
    n_negative_sets = sum(1 for r in per_patch_results if r["all_mask_negative"])

    fp_rate = len(false_positive) / n_negative_sets if n_negative_sets > 0 else 0.0
    fn_rate = len(false_negative) / n_positive_sets if n_positive_sets > 0 else 0.0
    abstain_rate = sum(1 for r in per_patch_results if r["answer"] == "C") / n if n > 0 else 0.0

    answer_dist = Counter(r["answer"] for r in per_patch_results if r["parse_valid"])

    pos_sets = [r for r in per_patch_results if r["any_mask_positive"] and r["parse_valid"]]
    neg_sets = [r for r in per_patch_results if r["all_mask_negative"] and r["parse_valid"]]
    mixed_sets = [r for r in per_patch_results if not r["all_mask_negative"] and r["any_mask_positive"] and r["parse_valid"]]

    def call_rate(sets: list[dict], answer: str) -> float:
        return sum(1 for r in sets if r["answer"] == answer) / len(sets) if sets else 0.0

    tumor_call_rate_pos = call_rate(pos_sets, "A")
    tumor_call_rate_neg = call_rate(neg_sets, "A")
    abstain_on_negative = call_rate(neg_sets, "C")
    mixed_call_rate = call_rate(mixed_sets, "A")
    call_rate_gap = tumor_call_rate_pos - tumor_call_rate_neg

    return {
        "n_total_sets": n,
        "n_parsable": n_parsable,
        "coverage": n_parsable / n if n > 0 else 0.0,
        "n_positive_sets": n_positive_sets,
        "n_negative_sets": n_negative_sets,
        "false_positives": len(false_positive),
        "false_negatives": len(false_negative),
        "correct_abstains": len(correct_abstain),
        "false_positive_rate": fp_rate,
        "false_negative_rate": fn_rate,
        "abstain_rate": abstain_rate,
        "answer_distribution": dict(answer_dist),
        "tumor_call_rate_pos": tumor_call_rate_pos,
        "tumor_call_rate_neg": tumor_call_rate_neg,
        "abstain_on_negative": abstain_on_negative,
        "mixed_call_rate": mixed_call_rate,
        "call_rate_gap": call_rate_gap,
        "error_slides_fp": list(set(r["slide_id"] for r in false_positive)),
        "error_slides_fn": list(set(r["slide_id"] for r in false_negative)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Patch-level VLM evaluation")
    parser.add_argument("--jsonl", required=True, help="VLM results JSONL")
    parser.add_argument("--registry_csv", default=REGISTRY_CSV_DEFAULT, help="Patch registry CSV")
    parser.add_argument("--output", default="")
    parser.add_argument("--output_s3", default="mil/vlm_results")
    args = parser.parse_args()

    print(f"[evaluate_patches] Loading results from {args.jsonl}")
    print(f"[evaluate_patches] Loading registry from {args.registry_csv}")

    metrics = _compute_patch_metrics(args.jsonl, args.registry_csv)

    print(f"\n{'='*60}")
    print("PATCH-LEVEL EVALUATION")
    print(f"{'='*60}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        elif isinstance(v, list) and len(v) > 10:
            print(f"  {k}: {v[:5]}... ({len(v)} total)")
        else:
            print(f"  {k}: {v}")

    if args.output:
        output_path = Path(args.output)
    else:
        rows = _load_jsonl(args.jsonl)
        task_ids = sorted(set(r.get("task_id", "") for r in rows if r.get("task_id")))
        tid_slug = f"_{task_ids[0]}" if len(task_ids) == 1 else ""
        output_path = Path(tempfile.gettempdir()) / f"patch_evaluation{tid_slug}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

    s3_key = f"{args.output_s3}/{output_path.name}"
    upload_to_s3(str(output_path), s3_key)
    print(f"\n[evaluate_patches] Results uploaded: s3://pershin-medailab/{s3_key}")

    try:
        from clearml import Task
        clearml_task = Task.current_task()
        if clearml_task:
            clearml_task.upload_artifact(name="patch_evaluation", artifact_object=str(output_path))
            print(f"  Uploaded to ClearML artifacts")
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
