"""Cross-run agreement report for repeated Quilt-LLaVA runs.

Point 4 of the rigor overhaul: temperature 0.4 is stochastic, so "no repetition"
is not the same as "reproducible". This script takes N JSONL outputs of the SAME
fixed subset run with the SAME seed (or just repeated) and measures how much the
per-patch calls agree across runs.

For each patch (joined by `patch_id`, falling back to `image_id`), and for each
field in {tumor_suspicious, should_abstain, visual_description_confidence,
conclusion_confidence}, we compute:

    unanimous    all N runs gave the same value
    majority     the modal value and how many runs backed it

Overall, per field: the fraction of patches that were unanimous, and the mean
majority fraction. A field that is 100% unanimous is fully reproducible across
the runs; a low unanimity fraction quantifies the stochastic drift the user
asked to see.

Usage:
    python scripts/report_agreement.py run1.jsonl run2.jsonl run3.jsonl \
        --out-md outputs/agreement.md
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

FIELDS = (
    "tumor_suspicious",
    "should_abstain",
    "visual_description_confidence",
    "conclusion_confidence",
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _key(row: dict[str, Any]) -> str:
    for k in ("patch_id", "image_id"):
        v = str(row.get(k, "")).strip()
        if v:
            return v
    return str(row.get("image_path", "")).strip()


def _val(row: dict[str, Any], field: str) -> str:
    return str(row.get(field, "")).strip().lower()


def _pct(num: int, den: int) -> str:
    return f"{num / den:.0%}" if den else "-"


def main() -> int:
    ap = argparse.ArgumentParser(description="Cross-run agreement of repeated runs.")
    ap.add_argument("inputs", nargs="+", help="Two or more vlm_outputs.jsonl files.")
    ap.add_argument("--out-md", default="outputs/agreement_report.md")
    args = ap.parse_args()

    if len(args.inputs) < 2:
        print("[report_agreement] ERROR: need at least 2 input files.", file=sys.stderr)
        return 1

    paths = [Path(p) for p in args.inputs]
    for p in paths:
        if not p.is_file():
            print(f"[report_agreement] ERROR: not found: {p}", file=sys.stderr)
            return 1

    runs = [load_jsonl(p) for p in paths]
    n_runs = len(runs)

    # index each run by join key
    indexed: list[dict[str, dict[str, Any]]] = []
    for r in runs:
        idx: dict[str, dict[str, Any]] = {}
        for row in r:
            idx[_key(row)] = row
        indexed.append(idx)

    # only patches present in ALL runs are comparable
    common = set(indexed[0].keys())
    for idx in indexed[1:]:
        common &= set(idx.keys())
    common_keys = sorted(common)

    if not common_keys:
        print("[report_agreement] ERROR: no patches common to all runs.", file=sys.stderr)
        return 1

    # per-field aggregates
    field_unanimous: dict[str, int] = defaultdict(int)
    field_majority_sum: dict[str, float] = defaultdict(float)
    # collect per-patch disagreements for optional detail
    disagreements: dict[str, list[str]] = defaultdict(list)

    for key in common_keys:
        for field in FIELDS:
            vals = [_val(indexed[i][key], field) for i in range(n_runs)]
            counts = Counter(vals)
            modal_val, modal_n = counts.most_common(1)[0]
            if modal_n == n_runs:
                field_unanimous[field] += 1
            else:
                disagreements[field].append(
                    f"{key}: {'/'.join(vals)}"
                )
            field_majority_sum[field] += modal_n / n_runs

    n = len(common_keys)

    md: list[str] = ["# Cross-run agreement report", ""]
    md.append(f"Runs ({n_runs}): " + ", ".join(f"`{p.as_posix()}`" for p in paths))
    md.append("")
    md.append(f"Patches common to all runs: **{n}** "
              f"(run sizes: {', '.join(str(len(r)) for r in runs)}).")
    md.append("")
    md.append("`unanimous` = fraction of patches where all runs gave the same "
              "value. `mean majority` = average fraction of runs backing the "
              "modal value per patch (1.0 = always unanimous).")
    md.append("")
    md.extend([
        "| field | unanimous | mean majority |",
        "|---|---|---|",
    ])
    for field in FIELDS:
        md.append(
            f"| {field} | {field_unanimous[field]}/{n} "
            f"({_pct(field_unanimous[field], n)}) | "
            f"{field_majority_sum[field] / n:.3f} |"
        )
    md.append("")

    # show up to 20 disagreements per field for auditing
    for field in FIELDS:
        dis = disagreements[field]
        if not dis:
            continue
        md.append(f"## Disagreements: {field} ({len(dis)})")
        md.append("")
        for line in dis[:20]:
            md.append(f"- {line}")
        if len(dis) > 20:
            md.append(f"- … and {len(dis) - 20} more")
        md.append("")

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"[report_agreement] wrote {out_md}")
    for field in FIELDS:
        print(f"[report_agreement] {field}: unanimous "
              f"{_pct(field_unanimous[field], n)}, "
              f"mean majority {field_majority_sum[field] / n:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
