"""Side-by-side model-output report: Quilt-LLaVA vs ScaleReasoner-R1.

Places the two models' outputs for the SAME C16 tile next to each other so a
reviewer can qualitatively eyeball how each describes the tissue. Joined by
``patch_id`` (file stem), which both runners emit.

IMPORTANT — this is a QUALITATIVE view only. ScaleReasoner-R1 is a multiple-
choice model run here OFF its trained contract (single-tile free-text), so this
report draws **no** "which model is better" conclusion. It exists to inspect
description style/grounding, not to benchmark.

Ground truth per tile is ``tile_in_mask`` (1 = tile inside the tumor annotation
mask). The Quilt-LLaVA JSONL carries the C16 metadata (source/label/
tile_in_mask); the ScaleReasoner describe JSONL carries only free text, so
metadata is taken from the Quilt-LLaVA side of the join.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _s(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, list):
        return "; ".join(str(x) for x in v)
    return str(v)


def _key(row: dict[str, Any]) -> str:
    """Join key: patch_id, else image_id, else image_path stem."""
    for k in ("patch_id", "image_id"):
        v = _s(row.get(k)).strip()
        if v:
            return v
    ip = _s(row.get("image_path")).strip()
    return Path(ip).stem if ip else ""


def _truth_label(row: dict[str, Any]) -> str:
    tim = _s(row.get("tile_in_mask")).strip()
    if tim == "1":
        return "TUMOR tile (in mask)"
    if tim == "0":
        return "non-tumor tile (outside mask)"
    return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description="Quilt-LLaVA vs ScaleReasoner side-by-side.")
    ap.add_argument("--quilt", required=True, help="Quilt-LLaVA vlm_outputs.jsonl")
    ap.add_argument("--scalereasoner", required=True, help="ScaleReasoner describe JSONL")
    ap.add_argument("--output", default=None, help="Output .md (default: outputs/model_compare_report.md)")
    ap.add_argument("--per_group", type=int, default=4,
                    help="Max examples per (source, tile_in_mask) group. 0 = all.")
    args = ap.parse_args()

    quilt_path = Path(args.quilt)
    sr_path = Path(args.scalereasoner)
    for p in (quilt_path, sr_path):
        if not p.is_file():
            print(f"[report_model_compare] ERROR: not found: {p}", file=sys.stderr)
            return 1
    out_path = Path(args.output) if args.output else Path("outputs/model_compare_report.md")

    quilt = _load(quilt_path)
    sr = _load(sr_path)
    sr_by_key = {_key(r): r for r in sr}

    # Group by (source, tile_in_mask) using the Quilt-LLaVA metadata side.
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    matched = 0
    for q in quilt:
        k = _key(q)
        if k in sr_by_key:
            matched += 1
        groups[(_s(q.get("source")) or "?", _s(q.get("tile_in_mask")) or "?")].append(q)

    lines: list[str] = []
    lines.append("# Quilt-LLaVA vs ScaleReasoner-R1 — side-by-side microdescriptions")
    lines.append("")
    lines.append(
        "> **Qualitative only.** ScaleReasoner-R1 is a multiple-choice model run "
        "here OFF its trained contract (single-tile free-text description). This "
        "report makes **no performance comparison** and declares no winner — the "
        "two models were trained for different tasks. It is for eyeballing "
        "description style and grounding side by side."
    )
    lines.append("")
    lines.append(f"- Quilt-LLaVA: `{quilt_path.as_posix()}` ({len(quilt)} patches)")
    lines.append(f"- ScaleReasoner: `{sr_path.as_posix()}` ({len(sr)} tiles)")
    lines.append(f"- Joined on patch_id: {matched}/{len(quilt)} patches have both outputs.")
    lines.append("")
    lines.append(
        "Ground truth per tile is `tile_in_mask` (1 = inside tumor annotation)."
    )
    lines.append("")

    for key in sorted(groups.keys()):
        src, tim = key
        grp = groups[key]
        take = grp if args.per_group == 0 else grp[: args.per_group]
        lines.append(f"## source=`{src}`  tile_in_mask=`{tim}`  ({_truth_label(take[0])})")
        lines.append("")
        lines.append(f"Showing {len(take)} of {len(grp)} in this group.")
        lines.append("")
        for q in take:
            k = _key(q)
            s = sr_by_key.get(k)
            lines.append(f"### `{k}`")
            lines.append("")
            lines.append(f"- **ground truth:** {_truth_label(q)} (slide label={_s(q.get('label'))})")
            lines.append(
                f"- **Quilt-LLaVA** (tumor_suspicious=**{_s(q.get('tumor_suspicious'))}**, "
                f"abstain={_s(q.get('should_abstain'))}, "
                f"concl_conf={_s(q.get('conclusion_confidence'))}): "
                f"{_s(q.get('tissue_description'))}"
            )
            q_evid = _s(q.get("evidence"))
            if q_evid:
                lines.append(f"    - evidence: {q_evid}")
            if s is not None:
                lines.append(f"- **ScaleReasoner** (off-contract describe): {_s(s.get('description'))}")
            else:
                lines.append("- **ScaleReasoner**: _(no matching output for this patch)_")
            lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report_model_compare] wrote {out_path}")
    print(f"[report_model_compare] {matched}/{len(quilt)} patches matched across models, "
          f"{len(groups)} groups")
    return 0


if __name__ == "__main__":
    sys.exit(main())
