"""Re-evaluate a vlm_pipeline output JSONL against a (fixed) patch registry.

Joins every patch in the JSONL to the registry by patch_uid (fallback
region_uid) to obtain the corrected tumor_mask_overlap, then recomputes
per-set ground truth and the C16 model/mode selection metrics:

  - Step 4 (model selection): oracle_tumor vs oracle_non_tumor
    sensitivity / specificity / balanced accuracy, parse rate, mode
    collapse (unique answers), oracle group purity after the join.
  - Step 5 (mode selection): single / separate / context side by side
    for top_k and the oracle control groups.

Inputs:
  --jsonl         output of scripts/vlm_pipeline.py (local path)
  --registry_csv  s3://... or local patch_registry.csv
  --labels_csv    optional s3:// or local MIL metadata CSV with slide_id/label

Outputs:
  markdown report + csv + json written into --out_dir.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.s3_utils import read_csv_from_s3, upload_to_s3

REGISTRY_CSV_DEFAULT = (
    "s3://pershin-medailab/Pathomorphology/CAMELYON/"
    "mil/vlm_patches_registry/patch_registry.csv"
)
JSONL_DEFAULT = (
    "s3://pershin-medailab/Pathomorphology/CAMELYON/"
    "mil/vlm_results/c16_med_gemma/vlm_c16_med_gemma.jsonl"
)
LABELS_CSV_DEFAULT = (
    "s3://pershin-medailab/Pathomorphology/CAMELYON/"
    "mil/vlm_patches/c16_abmil_vlm_metadata_a50fbde29aa04e9d829a4580fd5c68b8.csv"
)

MODES = ("single", "separate", "context")
SOURCES_STEP4 = ("oracle_tumor", "oracle_non_tumor")


def _load_registry(path: str) -> pd.DataFrame:
    if path.startswith("s3://"):
        key = path.replace("s3://", "", 1).split("/", 1)[1]
        return read_csv_from_s3(key)
    return pd.read_csv(path)


def _load_labels(path: str) -> dict[str, int]:
    if path.startswith("s3://"):
        bucket, key = path.replace("s3://", "", 1).split("/", 1)
        client = get_s3_client()
        obj = client.get_object(Bucket=bucket, Key=key)
        df = pd.read_csv(io.BytesIO(obj["Body"].read()))
    else:
        df = pd.read_csv(path)
    if "label" not in df.columns or "slide_id" not in df.columns:
        return {}
    return df.groupby("slide_id")["label"].max().to_dict()


def _metrics_from_confusion(tp: int, fn: int, tn: int, fp: int) -> dict:
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    return {
        "TP": tp, "FN": fn, "TN": tn, "FP": fp,
        "sensitivity": sens,
        "specificity": spec,
        "balanced_accuracy": (sens + spec) / 2,
    }


def _fmt(v: float) -> str:
    return f"{v:.3f}" if v == v else "n/a"


def main() -> int:
    ap = argparse.ArgumentParser(description="Re-evaluate vlm_pipeline JSONL vs registry")
    ap.add_argument("--jsonl", default=JSONL_DEFAULT,
                    help="vlm_pipeline output jsonl (local path or s3:// URL)")
    ap.add_argument("--registry_csv", default=REGISTRY_CSV_DEFAULT)
    ap.add_argument("--labels_csv", default=LABELS_CSV_DEFAULT,
                    help="MIL metadata CSV with slide_id/label (local or s3:// URL)")
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--upload_s3", default="mil/vlm_results/c16_med_gemma_reeval",
                    help="S3 prefix to upload results to ('' = no upload)")
    args = ap.parse_args()

    jsonl_path = Path(args.jsonl)
    if args.jsonl.startswith("s3://"):
        bucket, key = args.jsonl.replace("s3://", "", 1).split("/", 1)
        client = get_s3_client()
        obj = client.get_object(Bucket=bucket, Key=key)
        jsonl_path = Path(tempfile.gettempdir()) / Path(key).name
        jsonl_path.write_bytes(obj["Body"].read())
        print(f"[evaluate] downloaded {args.jsonl} to {jsonl_path}")
    if not jsonl_path.exists():
        print(f"[evaluate] ERROR: jsonl not found: {jsonl_path}")
        return 1

    rows = [json.loads(line) for line in jsonl_path.open("r", encoding="utf-8")]
    print(f"[evaluate] loaded {len(rows)} result rows from {jsonl_path}")

    registry = _load_registry(args.registry_csv)
    print(f"[evaluate] registry: {len(registry)} rows")

    by_uid: dict[str, dict] = {}
    for _, r in registry.iterrows():
        by_uid[str(r.get("patch_uid", ""))] = r.to_dict()
        by_uid.setdefault(str(r.get("region_uid", "")), r.to_dict())
    by_uid_set = set(by_uid.keys())
    n_reg = len(by_uid_set)
    print(f"[evaluate] registry index: {n_reg} keys (patch_uid + region_uid)")

    slide_labels: dict[str, int] = {}
    if args.labels_csv:
        slide_labels = _load_labels(args.labels_csv)
        n_t = sum(1 for v in slide_labels.values() if v == 1)
        print(f"[evaluate] labels: {len(slide_labels)} slides "
              f"({n_t} tumor, {len(slide_labels) - n_t} normal)")

    # --- join patches to corrected mask labels ---
    missing: Counter = Counter()
    for row in rows:
        for key, p in row.get("patches", {}).items():
            puid = str(p.get("patch_uid", ""))
            ruid = str(p.get("region_uid", ""))
            rec = by_uid.get(puid) or by_uid.get(ruid)
            if rec is None:
                missing[key] += 1
                p["_mask"] = None
            else:
                p["_mask"] = int(rec.get("tumor_mask_overlap", 0))
                p["_overlap_frac"] = rec.get("tumor_overlap_fraction", float("nan"))
                p["_registry_source"] = str(rec.get("selection_source", ""))

    n_missing_patches = int(sum(missing.values()))
    print(f"[evaluate] patches not found in registry: {n_missing_patches}")
    for key, cnt in missing.most_common(5):
        print(f"    {key}: {cnt}")

    usable = []
    dropped_missing = 0
    for row in rows:
        if any(p.get("_mask") is None for p in row.get("patches", {}).values()):
            dropped_missing += 1
            continue
        row["_set_mask"] = int(any(p["_mask"] == 1 for p in row["patches"].values()))
        usable.append(row)
    print(f"[evaluate] rows dropped (unmatched patches): {dropped_missing}; "
          f"usable: {len(usable)}")

    # --- per group x mode metrics ---
    records = []
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in usable:
        groups[(row["dataset"], row["selection_source"], row["mode"])].append(row)

    print(f"\n{'=' * 100}")
    print("STEP 4 / 5 — per (dataset, source, mode)")
    print(f"{'=' * 100}")

    for (ds, src, mode), grp in sorted(groups.items()):
        valid = [r for r in grp if r["parse_valid"]]
        answers = Counter(r["answer"] for r in valid)
        n = len(grp)
        n_valid = len(valid)

        pos = [r for r in valid if r["_set_mask"] == 1]
        neg = [r for r in valid if r["_set_mask"] == 0]
        tp = sum(1 for r in pos if r["answer"] == "A")
        fn = sum(1 for r in pos if r["answer"] in ("B", "C"))
        tn = sum(1 for r in neg if r["answer"] == "B")
        fp = sum(1 for r in neg if r["answer"] == "A")
        m = _metrics_from_confusion(tp, fn, tn, fp)

        uniq = len(set(r["answer"] for r in valid))
        records.append({
            "dataset": ds, "source": src, "mode": mode,
            "n_sets": n, "n_valid": n_valid,
            "A": answers.get("A", 0), "B": answers.get("B", 0),
            "C": answers.get("C", 0),
            "n_set_pos": len(pos), "n_set_neg": len(neg),
            "TP": tp, "FN": fn, "TN": tn, "FP": fp,
            "sensitivity": m["sensitivity"],
            "specificity": m["specificity"],
            "balanced_accuracy": m["balanced_accuracy"],
            "unique_answers": uniq,
            "mode_collapse": uniq / max(n_valid, 1),
        })
        print(
            f"  {ds:<12} {src:<16} {mode:<9} n={n:>3} valid={n_valid:>3} "
            f"A={answers.get('A', 0):>3} B={answers.get('B', 0):>3} C={answers.get('C', 0):>3} "
            f"| posGT={len(pos):>3} negGT={len(neg):>3} "
            f"| sens={_fmt(m['sensitivity'])} spec={_fmt(m['specificity'])} "
            f"bacc={_fmt(m['balanced_accuracy'])} uniq={uniq}"
        )

    # --- oracle purity check ---
    print(f"\nOracle group purity after join (per-patch corrected mask):")
    for ds in sorted(set(r["dataset"] for r in usable)):
        for src in SOURCES_STEP4:
            sub = [r for r in usable if r["dataset"] == ds and r["selection_source"] == src]
            if not sub:
                continue
            all_masks = [p["_mask"] for r in sub for p in r["patches"].values()]
            pure = all(m == 1 for m in all_masks) if src == "oracle_tumor" else all(m == 0 for m in all_masks)
            cnt = Counter(all_masks)
            print(f"  {ds:<12} {src:<16} patches={len(all_masks):>4} "
                  f"mask_dist={dict(cnt)} pure={'YES' if pure else 'NO'}")

    # --- slide-level (any A == tumor) ---
    slide_level = []
    if slide_labels:
        print(f"\nSlide-level (any A == tumor), top_k:")
        for ds in sorted(set(r["dataset"] for r in usable)):
            for mode in MODES:
                sub = [r for r in usable
                       if r["dataset"] == ds and r["selection_source"] == "top_k"
                       and r["mode"] == mode]
                if not sub:
                    continue
                by_slide: dict[str, list[str]] = defaultdict(list)
                for r in sub:
                    by_slide[r["slide_id"]].append(r["answer"])
                tp = fn = tn = fp = 0
                for sid, answers in by_slide.items():
                    gt = slide_labels.get(sid)
                    if gt is None:
                        continue
                    any_a = any(a == "A" for a in answers)
                    if gt == 1 and any_a:
                        tp += 1
                    elif gt == 1:
                        fn += 1
                    elif gt == 0 and not any_a:
                        tn += 1
                    elif gt == 0:
                        fp += 1
                m = _metrics_from_confusion(tp, fn, tn, fp)
                slide_level.append({"dataset": ds, "mode": mode, **m,
                                    "n_slides_scored": tp + fn + tn + fp})
                print(f"  {ds:<12} {mode:<9} TP={tp} FN={fn} TN={tn} FP={fp} "
                      f"sens={_fmt(m['sensitivity'])} spec={_fmt(m['specificity'])} "
                      f"bacc={_fmt(m['balanced_accuracy'])}")

    out_dir = Path(args.out_dir) if args.out_dir else jsonl_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(records)
    csv_path = out_dir / "evaluation_summary.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"\n[evaluate] wrote {csv_path}")

    md_path = out_dir / "evaluation_report.md"
    lines = [
        "# VLM run re-evaluation vs fixed registry",
        "",
        f"- jsonl: `{jsonl_path}` ({len(rows)} rows, {n_missing_patches} unmatched patches)",
        f"- registry: `{args.registry_csv}` ({len(registry)} rows)",
        "",
        "## Per (dataset, source, mode)",
        "",
        "| dataset | source | mode | n | valid | A | B | C | posGT | negGT | sens | spec | bacc | uniq |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for rec in records:
        lines.append(
            f"| {rec['dataset']} | {rec['source']} | {rec['mode']} | {rec['n_sets']} "
            f"| {rec['n_valid']} | {rec['A']} | {rec['B']} | {rec['C']} "
            f"| {rec['n_set_pos']} | {rec['n_set_neg']} "
            f"| {_fmt(rec['sensitivity'])} | {_fmt(rec['specificity'])} "
            f"| {_fmt(rec['balanced_accuracy'])} | {rec['unique_answers']} |"
        )
    if slide_level:
        lines += ["", "## Slide-level (any A == tumor), top_k", "",
                  "| dataset | mode | TP | FN | TN | FP | sens | spec | bacc |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for rec in slide_level:
            lines.append(
                f"| {rec['dataset']} | {rec['mode']} | {rec['TP']} | {rec['FN']} "
                f"| {rec['TN']} | {rec['FP']} | {_fmt(rec['sensitivity'])} "
                f"| {_fmt(rec['specificity'])} | {_fmt(rec['balanced_accuracy'])} |"
            )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[evaluate] wrote {md_path}")

    if args.upload_s3:
        for f in (csv_path, md_path):
            url = upload_to_s3(str(f), f"{args.upload_s3}/{f.name}")
            print(f"  Uploaded: {url}")

    print("[evaluate] Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
