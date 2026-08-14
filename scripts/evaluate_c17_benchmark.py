"""Evaluate a frozen benchmark output (c17_vlm_benchmark_*.json) from
scripts/benchmark_vlm_c16.py.

Level 1 (pipeline usefulness, per slide AND per patient):
  sensitivity / specificity / balanced accuracy on top_k vs each random seed,
  top_k-vs-random advantage, 95% bootstrap CI by patient.

Level 2 (answer consistency with the shown patches, per slide-set):
  FP  : answered A but no mask-pos patch was shown
  FN  : answered B/C but a mask-pos patch was shown
  C   : answered C (insufficient data) rate
  coverage: fraction of slide-sets with a parsable answer (A/B/C).
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

from src.s3_utils import get_s3_client, presign_url, upload_to_s3

REGISTRY_CSV_DEFAULT = "s3://pershin-medailab/Pathomorphology/CAMELYON/mil/vlm_patches_registry/patch_registry.csv"


def _patient_id(slide_id: str) -> str:
    sid = str(slide_id).strip()
    if sid.startswith("patient_"):
        return sid.split("_node_")[0]
    if sid.startswith("test_"):
        return sid.split("_tile_embeddings")[0]
    return sid


def _bootstrap_ci(values: list[float], n_bootstrap: int = 1000, ci: float = 0.95) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    means = []
    rng = np.random.RandomState(42)
    arr = np.asarray(values, dtype=float)
    for _ in range(n_bootstrap):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means.append(float(np.mean(sample)))
    lower = np.percentile(means, (1 - ci) / 2 * 100)
    upper = np.percentile(means, (1 + ci) / 2 * 100)
    return float(lower), float(upper)


def _confusion(records: list[dict], ground_truth: dict[str, int]) -> dict:
    tp = sum(1 for r in records if ground_truth.get(r["slide_id"]) == 1 and r["answer"] == "A")
    fn = sum(1 for r in records if ground_truth.get(r["slide_id"]) == 1 and r["answer"] in ("B", "C"))
    tn = sum(1 for r in records if ground_truth.get(r["slide_id"]) == 0 and r["answer"] == "B")
    fp = sum(1 for r in records if ground_truth.get(r["slide_id"]) == 0 and r["answer"] == "A")
    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    return {
        "n": len(records),
        "tp": tp, "fn": fn, "tn": tn, "fp": fp,
        "sensitivity": sens,
        "specificity": spec,
        "balanced_accuracy": (sens + spec) / 2,
        "a_count": sum(1 for r in records if r["answer"] == "A"),
        "b_count": sum(1 for r in records if r["answer"] == "B"),
        "c_count": sum(1 for r in records if r["answer"] == "C"),
        "parse_rate": sum(1 for r in records if r["answer"] in ("A", "B", "C")) / len(records) if records else 0.0,
    }


def _per_patient_metrics(records: list[dict], ground_truth: dict[str, int]) -> dict:
    by_patient: dict[str, list[str]] = defaultdict(list)
    for r in records:
        if r["answer"] in ("A", "B", "C"):
            by_patient[_patient_id(r["slide_id"])].append(r["answer"])

    accs = []
    sens_vals, spec_vals = [], []
    counts = Counter()
    for pid, answers in by_patient.items():
        true_label = ground_truth.get(pid)
        if true_label is None:
            continue
        agg = "A" if any(a == "A" for a in answers) else "C" if any(a == "C" for a in answers) else "B"
        counts[agg] += 1
        accs.append(1.0 if ((agg == "A" and true_label == 1) or (agg == "B" and true_label == 0)) else 0.0)
        if true_label == 1:
            sens_vals.append(1.0 if agg == "A" else 0.0)
        else:
            spec_vals.append(1.0 if agg == "B" else 0.0)

    sens = np.mean(sens_vals) if sens_vals else float("nan")
    spec = np.mean(spec_vals) if spec_vals else float("nan")
    return {
        "n_patients": len(accs),
        "sensitivity": float(sens),
        "sensitivity_ci": _bootstrap_ci(sens_vals),
        "specificity": float(spec),
        "specificity_ci": _bootstrap_ci(spec_vals),
        "balanced_accuracy": (float(sens) + float(spec)) / 2,
        "accuracy": float(np.mean(accs)) if accs else float("nan"),
        "accuracy_ci": _bootstrap_ci(accs),
        "a_count": counts["A"], "b_count": counts["B"], "c_count": counts["C"],
    }


def _level2(records: list[dict]) -> dict:
    fp = fn = correct_c = c_count = coverage = n_parsed = 0
    for r in records:
        ans = r["answer"]
        shown = [int(p.get("tile_in_mask", 0) or 0) for p in r.get("patches", {}).values()]
        has_pos = any(t == 1 for t in shown)
        if ans in ("A", "B", "C"):
            n_parsed += 1
        if ans == "A" and shown and not has_pos:
            fp += 1
        if ans in ("B", "C") and shown and has_pos:
            fn += 1
        if ans == "C":
            c_count += 1
            # C is "correct" (adequate abstention) when the shown patches are
            # not consistently tumor-positive; approximated as not all tumor.
            if shown and not all(t == 1 for t in shown):
                correct_c += 1
    n = len(records)
    return {
        "n_slide_sets": n,
        "n_parsable": n_parsed,
        "coverage": n_parsed / n if n else 0.0,
        "fp_no_tumor_patch": fp,
        "fn_despite_tumor_patch": fn,
        "c_total": c_count,
        "c_adequate": correct_c,
    }


def _load_benchmark_json(path: str) -> list[dict]:
    with open(path, "r") as f:
        data = json.load(f)
    recs = []
    for model, records in data.get("models", {}).items():
        for r in records:
            r = dict(r)
            r["model"] = model
            recs.append(r)
    return recs


def _resolve_registry_csv(path: str) -> str:
    if path.startswith("s3://"):
        parts = path.replace("s3://", "", 1).split("/", 1)
        if len(parts) == 2:
            bucket, key = parts
            client = get_s3_client()
            tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
            client.download_file(bucket, key, tmp.name)
            print(f"[evaluate_c17] Downloaded registry from {path} to {tmp.name}")
            return tmp.name
    return path


def _build_ground_truth(registry_csv: str) -> dict[str, int]:
    df = pd.read_csv(_resolve_registry_csv(registry_csv))
    ground_truth: dict[str, int] = {}
    for slide_id, g in df.groupby("slide_id"):
        pid = _patient_id(slide_id)
        if pid not in ground_truth:
            ground_truth[pid] = 1 if (g["tumor_mask_overlap"] == 1).any() else 0
    slide_truth = {
        slide: (1 if g["tumor_mask_overlap"].eq(1).any() else 0)
        for slide, g in df.groupby("slide_id")
    }
    return ground_truth, slide_truth


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate frozen benchmark output")
    parser.add_argument("--benchmark_json", required=True, help="c17_vlm_benchmark_*.json")
    parser.add_argument("--registry_csv", default=REGISTRY_CSV_DEFAULT)
    parser.add_argument("--mode", default="context", choices=["single", "separate", "context"])
    parser.add_argument("--output", default="")
    parser.add_argument("--output_s3", default="mil/vlm_results")
    args = parser.parse_args()

    recs = [r for r in _load_benchmark_json(args.benchmark_json) if r.get("mode") == args.mode]
    if not recs:
        print(f"[evaluate_c17] No records for mode={args.mode!r}", file=sys.stderr)
        return 2
    print(f"[evaluate_c17] Records (mode={args.mode}): {len(recs)}")

    ground_truth, slide_truth = _build_ground_truth(args.registry_csv)
    print(f"[evaluate_c17] Patients: {len(ground_truth)} "
          f"(tumor={sum(1 for v in ground_truth.values() if v == 1)}, "
          f"normal={sum(1 for v in ground_truth.values() if v == 0)})")

    # Attach slide-level ground truth to each record.
    for r in recs:
        r["slide_gt"] = slide_truth.get(r["slide_id"])

    # --- Level 1: top_k vs random, per slide and per patient ---
    def _by_source(source):
        return [r for r in recs if r["selection_source"] == source]

    top3 = _by_source("top_k")
    random_seeds = sorted({r["random_seed"] for r in recs if r["selection_source"] == "random"})

    print(f"\n=== LEVEL 1 (per slide) ===")
    top3_slide = _confusion(top3, slide_truth)
    print(f"  top_k: n={top3_slide['n']} sens={top3_slide['sensitivity']:.4f} "
          f"spec={top3_slide['specificity']:.4f} bacc={top3_slide['balanced_accuracy']:.4f} "
          f"parse={top3_slide['parse_rate']:.4f}")

    random_baccs = []
    for seed in random_seeds:
        m = _confusion([r for r in recs if r["selection_source"] == "random" and r["random_seed"] == seed], slide_truth)
        random_baccs.append(m["balanced_accuracy"])
        print(f"  random_{seed}: n={m['n']} sens={m['sensitivity']:.4f} "
              f"spec={m['specificity']:.4f} bacc={m['balanced_accuracy']:.4f}")

    adv = top3_slide["balanced_accuracy"] - (np.mean(random_baccs) if random_baccs else float("nan"))
    print(f"  advantage (top_k - mean_random): {adv:.4f}")

    print(f"\n=== LEVEL 1 (per patient, 95% CI) ===")
    pt = _per_patient_metrics(top3, ground_truth)
    print(f"  top_k patients: n={pt['n_patients']} sens={pt['sensitivity']:.4f} "
          f"[{pt['sensitivity_ci'][0]:.4f},{pt['sensitivity_ci'][1]:.4f}] "
          f"spec={pt['specificity']:.4f} [{pt['specificity_ci'][0]:.4f},{pt['specificity_ci'][1]:.4f}] "
          f"bacc={(pt['sensitivity'] + pt['specificity']) / 2:.4f} "
          f"acc={pt['accuracy']:.4f} [{pt['accuracy_ci'][0]:.4f},{pt['accuracy_ci'][1]:.4f}]")

    # --- Level 2: answer consistency with shown patches ---
    l2 = _level2(recs)
    print(f"\n=== LEVEL 2 (per slide-set, mode={args.mode}) ===")
    for k, v in l2.items():
        print(f"  {k}: {v}")

    result = {
        "mode": args.mode,
        "level1_slide": {"top_k": top3_slide, "random": {s: _confusion([r for r in recs if r["selection_source"] == "random" and r["random_seed"] == s], slide_truth) for s in random_seeds}, "advantage": adv},
        "level1_patient": {"top_k": pt},
        "level2": l2,
        "n_records": len(recs),
    }

    if args.output:
        output_path = Path(args.output)
    else:
        tid = ""
        for r in recs:
            tid = r.get("patch_set_uid", "")[:0] or ""
            break
        import re
        m = re.search(r"benchmark_([0-9a-f]+)", args.benchmark_json)
        slug = f"_{m.group(1)}" if m else ""
        output_path = Path(tempfile.gettempdir()) / f"c17_evaluation{slug}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=float))
    s3_key = f"{args.output_s3}/{output_path.name}"
    s3_uri = upload_to_s3(str(output_path), s3_key)
    print(f"\n[evaluate_c17] Results uploaded: {s3_uri}")

    try:
        from clearml import Task
        task = Task.current_task()
        if task:
            task.upload_artifact(name="c17_evaluation", artifact_object=presign_url(s3_uri))
            task.set_parameter("outputs/c17_evaluation_uri", s3_uri)
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())