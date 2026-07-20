"""Grounding metrics for a CAMELYON16 Quilt-LLaVA run.

This report answers a question the grouped/JSON reports do NOT: does the
model's morphological call agree with the CAMELYON16 tile mask? It is a
*grounding* check, not a validated diagnostic metric — see the disclaimer in
the generated header.

Definitions
-----------
Ground truth per tile is ``tile_in_mask``:
    "1" -> tile overlaps the annotated tumor region (positive)
    "0" -> tile is outside it (negative)
Model call is ``tumor_suspicious``:
    "yes" -> model-positive
    "no"  -> model-negative
    "uncertain" / abstain -> NOT scored (reported separately, never silently
    folded into a yes/no).

Over the *scored* subset (decisive yes/no AND tile_in_mask in {0,1}):
    TP  yes & in_mask==1        FN  no  & in_mask==1
    FP  yes & in_mask==0        TN  no  & in_mask==0
    sensitivity = TP/(TP+FN)    specificity = TN/(TN+FP)
    balanced_accuracy = mean(sensitivity, specificity)

We deliberately print sensitivity and specificity side by side and never a
single "accuracy" number: on this heavily yes-biased run a lone accuracy figure
would hide that specificity is near zero. We also print coverage
(n_scored / n) so the reader sees how much was excluded by uncertain/abstain.

The false-``yes`` rate is split by mask (a ``yes`` on ``in_mask==0`` is a false
positive; on ``in_mask==1`` it is a true positive) and further by ``source``.

Input: the ``vlm_outputs.jsonl`` from ``run_remote_vlm.py``.
Output: a Markdown report + a flat CSV of the per-source breakdown.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "outputs" / "c16_best_safe_full" / "vlm_outputs.jsonl"
DEFAULT_OUT_MD = PROJECT_ROOT / "outputs" / "c16_grounding_metrics.md"
DEFAULT_OUT_CSV = PROJECT_ROOT / "outputs" / "c16_grounding_metrics.csv"


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return False


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _mask(row: dict[str, Any]) -> str:
    return str(row.get("tile_in_mask", "")).strip()


def _call(row: dict[str, Any]) -> str:
    return str(row.get("tumor_suspicious", "")).strip().lower()


def _fmt_rate(num: int, den: int) -> str:
    return f"{num / den:.3f} ({num}/{den})" if den else "n/a (0)"


def _pct(num: int, den: int) -> str:
    return f"{num / den:.0%}" if den else "-"


def _confusion(rows: list[dict[str, Any]]) -> dict[str, int]:
    """TP/FP/TN/FN over the scored subset + counts of what was excluded."""
    tp = fp = tn = fn = 0
    n_uncertain = n_abstain = n_no_mask = 0
    for r in rows:
        m = _mask(r)
        c = _call(r)
        if _coerce_bool(r.get("should_abstain")):
            n_abstain += 1
        if c not in {"yes", "no"}:
            n_uncertain += 1
            continue
        if m not in {"0", "1"}:
            n_no_mask += 1
            continue
        positive = m == "1"
        if c == "yes":
            if positive:
                tp += 1
            else:
                fp += 1
        else:  # "no"
            if positive:
                fn += 1
            else:
                tn += 1
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "n_uncertain": n_uncertain,
        "n_abstain": n_abstain,
        "n_no_mask": n_no_mask,
    }


def _metrics(c: dict[str, int]) -> dict[str, Any]:
    tp, fp, tn, fn = c["tp"], c["fp"], c["tn"], c["fn"]
    n_scored = tp + fp + tn + fn
    sens = tp / (tp + fn) if (tp + fn) else None
    spec = tn / (tn + fp) if (tn + fp) else None
    if sens is not None and spec is not None:
        bal = (sens + spec) / 2
    else:
        bal = None
    return {"n_scored": n_scored, "sensitivity": sens, "specificity": spec,
            "balanced_accuracy": bal}


def _fmt_metric(v: Any) -> str:
    return f"{v:.3f}" if isinstance(v, float) else "n/a"


def main() -> int:
    ap = argparse.ArgumentParser(description="C16 grounding metrics report.")
    ap.add_argument("--input", default=str(DEFAULT_INPUT), help="vlm_outputs.jsonl.")
    ap.add_argument("--out-md", default=str(DEFAULT_OUT_MD))
    ap.add_argument("--out-csv", default=str(DEFAULT_OUT_CSV))
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.is_file():
        print(f"[report_c16_metrics] ERROR: input not found: {in_path}", file=sys.stderr)
        return 1

    rows = load_jsonl(in_path)
    if not rows:
        print("[report_c16_metrics] ERROR: no rows in input.", file=sys.stderr)
        return 1

    n = len(rows)
    conf = _confusion(rows)
    met = _metrics(conf)
    n_scored = met["n_scored"]

    # False-yes / true-yes split by mask.
    in0 = [r for r in rows if _mask(r) == "0"]
    in1 = [r for r in rows if _mask(r) == "1"]
    yes0 = sum(1 for r in in0 if _call(r) == "yes")
    yes1 = sum(1 for r in in1 if _call(r) == "yes")

    # Per-source false-yes on out-of-mask tiles.
    src_out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    src_in: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in in0:
        src_out[str(r.get("source", "")).strip()].append(r)
    for r in in1:
        src_in[str(r.get("source", "")).strip()].append(r)

    md: list[str] = ["# CAMELYON16 Quilt-LLaVA grounding metrics", ""]
    md.append("> **Not a diagnostic metric.** This measures agreement between "
              "the model's `tumor_suspicious` morphology call and the CAMELYON16 "
              "tile mask (`tile_in_mask`). It is a research grounding check on "
              "one benchmark subset, NOT validation of clinical performance.")
    md.append("")
    md.append(f"Input: `{in_path.as_posix()}` — {n} patches.")
    md.append("")
    md.append("## Coverage")
    md.append("")
    md.append(f"- Scored (decisive yes/no with a valid mask): **{n_scored}/{n}** "
              f"({_pct(n_scored, n)}).")
    md.append(f"- Excluded — `uncertain`: {conf['n_uncertain']}; "
              f"missing/invalid `tile_in_mask`: {conf['n_no_mask']}.")
    md.append(f"- `should_abstain=true`: {conf['n_abstain']} "
              f"({_pct(conf['n_abstain'], n)}) — reported, not folded into yes/no.")
    md.append("")
    md.append("## Confusion (scored subset)")
    md.append("")
    md.extend([
        "| | in_mask=1 (tumor) | in_mask=0 (benign) |",
        "|---|---|---|",
        f"| call=yes | TP {conf['tp']} | FP {conf['fp']} |",
        f"| call=no  | FN {conf['fn']} | TN {conf['tn']} |",
    ])
    md.append("")
    md.append("## Headline metrics")
    md.append("")
    md.extend([
        "| metric | value |",
        "|---|---|",
        f"| sensitivity (recall on tumor tiles) | {_fmt_metric(met['sensitivity'])} |",
        f"| specificity (recall on benign tiles) | {_fmt_metric(met['specificity'])} |",
        f"| balanced accuracy | {_fmt_metric(met['balanced_accuracy'])} |",
    ])
    md.append("")
    md.append("Sensitivity and specificity are shown side by side on purpose: a "
              "model that always answers `yes` scores perfect sensitivity and "
              "zero specificity, so no single accuracy number is reported.")
    md.append("")
    md.append("## `yes` rate split by mask")
    md.append("")
    md.extend([
        "| tiles | n | yes | yes-rate | interpretation |",
        "|---|---|---|---|---|",
        f"| in_mask=0 (benign) | {len(in0)} | {yes0} | {_pct(yes0, len(in0))} | "
        f"false-positive rate |",
        f"| in_mask=1 (tumor) | {len(in1)} | {yes1} | {_pct(yes1, len(in1))} | "
        f"true-positive rate |",
    ])
    md.append("")
    md.append("## False-`yes` on benign tiles, by source")
    md.append("")
    md.extend([
        "| source | benign tiles (in_mask=0) | yes | false-yes rate |",
        "|---|---|---|---|",
    ])
    for src in sorted(src_out.keys()):
        grp = src_out[src]
        y = sum(1 for r in grp if _call(r) == "yes")
        md.append(f"| {src or '(none)'} | {len(grp)} | {y} | {_pct(y, len(grp))} |")
    md.append("")

    out_md = Path(args.out_md)
    out_csv = Path(args.out_csv)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")

    # CSV: per-source false-yes (benign) + true-yes (tumor).
    all_sources = sorted(set(src_out) | set(src_in))
    with out_csv.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=[
            "source", "benign_n", "benign_yes", "benign_false_yes_rate",
            "tumor_n", "tumor_yes", "tumor_true_yes_rate"])
        writer.writeheader()
        for src in all_sources:
            b = src_out.get(src, [])
            t = src_in.get(src, [])
            by = sum(1 for r in b if _call(r) == "yes")
            ty = sum(1 for r in t if _call(r) == "yes")
            writer.writerow({
                "source": src,
                "benign_n": len(b),
                "benign_yes": by,
                "benign_false_yes_rate": (by / len(b)) if b else "",
                "tumor_n": len(t),
                "tumor_yes": ty,
                "tumor_true_yes_rate": (ty / len(t)) if t else "",
            })

    print(f"[report_c16_metrics] wrote {out_md}")
    print(f"[report_c16_metrics] wrote {out_csv}")
    print(f"[report_c16_metrics] scored={n_scored}/{n} "
          f"sens={_fmt_metric(met['sensitivity'])} "
          f"spec={_fmt_metric(met['specificity'])} "
          f"bal_acc={_fmt_metric(met['balanced_accuracy'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
