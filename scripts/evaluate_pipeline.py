"""Patient-level pipeline evaluation.

Computes sensitivity, specificity, balanced accuracy, and top-3 vs random
advantage with bootstrap confidence intervals by patient.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.s3_utils import upload_to_s3


def _load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _patient_id(slide_id: str) -> str:
    sid = str(slide_id).strip()
    if sid.startswith("patient_"):
        parts = sid.split("_node_")
        return parts[0]
    if sid.startswith("test_"):
        parts = sid.split("_tile_embeddings")
        return parts[0]
    return sid


def _bootstrap_ci(values: list[float], n_bootstrap: int = 1000, ci: float = 0.95) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    means = []
    rng = np.random.RandomState(42)
    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        means.append(np.mean(sample))
    lower = np.percentile(means, (1 - ci) / 2 * 100)
    upper = np.percentile(means, (1 + ci) / 2 * 100)
    return float(lower), float(upper)


def _compute_patient_metrics(rows: list[dict], ground_truth: dict[str, int]) -> dict:
    patient_results: dict[str, list[str]] = defaultdict(list)

    for r in rows:
        pid = _patient_id(r.get("slide_id", ""))
        ans = r.get("answer", "")
        if ans in ("A", "B", "C"):
            patient_results[pid].append(ans)

    patient_scores = {}
    for pid, answers in patient_results.items():
        true_label = ground_truth.get(pid)
        if true_label is None:
            continue
        agg = "A" if any(a == "A" for a in answers) else "C" if any(a == "C" for a in answers) else "B"
        patient_scores[pid] = {
            "true_label": true_label,
            "prediction": agg,
            "correct": (agg == "A" and true_label == 1) or (agg == "B" and true_label == 0),
        }

    n = len(patient_scores)
    tp = sum(1 for v in patient_scores.values() if v["prediction"] == "A" and v["true_label"] == 1)
    fn = sum(1 for v in patient_scores.values() if v["prediction"] in ("B", "C") and v["true_label"] == 1)
    tn = sum(1 for v in patient_scores.values() if v["prediction"] == "B" and v["true_label"] == 0)
    fp = sum(1 for v in patient_scores.values() if v["prediction"] == "A" and v["true_label"] == 0)

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    bacc = (sens + spec) / 2

    patient_accuracies = [1.0 if v["correct"] else 0.0 for v in patient_scores.values()]
    ci_low, ci_high = _bootstrap_ci(patient_accuracies)
    sens_ci = _bootstrap_ci(
        [1.0 if v["prediction"] == "A" and v["true_label"] == 1 else 0.0
         for v in patient_scores.values() if v["true_label"] == 1]
    ) if sum(1 for v in patient_scores.values() if v["true_label"] == 1) > 0 else (0.0, 0.0)
    spec_ci = _bootstrap_ci(
        [1.0 if v["prediction"] == "B" and v["true_label"] == 0 else 0.0
         for v in patient_scores.values() if v["true_label"] == 0]
    ) if sum(1 for v in patient_scores.values() if v["true_label"] == 0) > 0 else (0.0, 0.0)

    return {
        "n_patients": n,
        "n_slide_sets": len(rows),
        "tp": tp,
        "fn": fn,
        "tn": tn,
        "fp": fp,
        "sensitivity": sens,
        "sensitivity_ci": sens_ci,
        "specificity": spec,
        "specificity_ci": spec_ci,
        "balanced_accuracy": bacc,
        "accuracy": (tp + tn) / n if n > 0 else 0.0,
        "accuracy_ci": (ci_low, ci_high),
        "a_count": sum(1 for v in patient_scores.values() if v["prediction"] == "A"),
        "b_count": sum(1 for v in patient_scores.values() if v["prediction"] == "B"),
        "c_count": sum(1 for v in patient_scores.values() if v["prediction"] == "C"),
    }


def _build_ground_truth(registry_csv: str) -> dict[str, int]:
    df = pd.read_csv(registry_csv)
    ground_truth: dict[str, int] = {}

    for slide_id in df["slide_id"].unique():
        slide_df = df[df["slide_id"] == slide_id]
        pid = _patient_id(slide_id)

        if pid in ground_truth:
            continue

        has_tumor = (slide_df["tumor_mask_overlap"] == 1).any()
        ground_truth[pid] = 1 if has_tumor else 0

    return ground_truth


def _compare_runs(top3_path: str, random_paths: list[str], ground_truth: dict[str, int]) -> dict:
    top3_rows = _load_jsonl(top3_path)
    top3_metrics = _compute_patient_metrics(top3_rows, ground_truth)

    random_metrics_list = []
    for rp in random_paths:
        rows = _load_jsonl(rp)
        random_metrics_list.append(_compute_patient_metrics(rows, ground_truth))

    result = {
        "top3": top3_metrics,
        "random": random_metrics_list,
    }

    if random_metrics_list:
        random_baccs = [m["balanced_accuracy"] for m in random_metrics_list]
        result["advantage"] = {
            "top3_balanced_accuracy": top3_metrics["balanced_accuracy"],
            "mean_random_balanced_accuracy": np.mean(random_baccs),
            "advantage": top3_metrics["balanced_accuracy"] - np.mean(random_baccs),
        }

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Pipeline-level evaluation")
    parser.add_argument("--top3_jsonl", required=True, help="Top-3 MIL VLM results JSONL")
    parser.add_argument("--random_jsonl", nargs="+", default=[], help="Random-3 VLM results JSONL(s)")
    parser.add_argument("--registry_csv", required=True, help="Patch registry for ground truth")
    parser.add_argument("--output", default="")
    parser.add_argument("--output_s3", default="mil/vlm_results")
    args = parser.parse_args()

    ground_truth = _build_ground_truth(args.registry_csv)
    print(f"[evaluate] Ground truth: {len(ground_truth)} patients")
    print(f"  Tumor: {sum(1 for v in ground_truth.values() if v == 1)}")
    print(f"  Normal: {sum(1 for v in ground_truth.values() if v == 0)}")

    print(f"\n{evaluate} Top-3 MIL analysis:")
    top3_rows = _load_jsonl(args.top3_jsonl)
    top3_metrics = _compute_patient_metrics(top3_rows, ground_truth)
    for k, v in top3_metrics.items():
        if isinstance(v, tuple):
            print(f"  {k}: [{v[0]:.4f}, {v[1]:.4f}]")
        elif isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    if args.random_jsonl:
        for i, rp in enumerate(args.random_jsonl):
            print(f"\n[evaluate] Random-3 (seed {i}):")
            rows = _load_jsonl(rp)
            metrics = _compute_patient_metrics(rows, ground_truth)
            for k, v in metrics.items():
                if isinstance(v, tuple):
                    print(f"  {k}: [{v[0]:.4f}, {v[1]:.4f}]")
                elif isinstance(v, float):
                    print(f"  {k}: {v:.4f}")
                else:
                    print(f"  {k}: {v}")

        comparison = _compare_runs(args.top3_jsonl, args.random_jsonl, ground_truth)
        print(f"\n[evaluate] Top-3 vs Random advantage:")
        for k, v in comparison["advantage"].items():
            print(f"  {k}: {v:.4f}")

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path("/tmp/pipeline_evaluation.json")

    all_results = {
        "ground_truth": {
            "n_patients": len(ground_truth),
            "n_tumor": sum(1 for v in ground_truth.values() if v == 1),
            "n_normal": sum(1 for v in ground_truth.values() if v == 0),
        },
        "top3": top3_metrics,
    }
    if args.random_jsonl:
        all_results["random"] = [
            _compute_patient_metrics(_load_jsonl(rp), ground_truth)
            for rp in args.random_jsonl
        ]
        all_results["advantage"] = _compare_runs(
            args.top3_jsonl, args.random_jsonl, ground_truth
        )["advantage"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False))

    s3_key = f"{args.output_s3}/{output_path.name}"
    upload_to_s3(str(output_path), s3_key)
    print(f"\n[evaluate] Results uploaded: s3://pershin-medailab/{s3_key}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
