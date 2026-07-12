"""Emit a human-checkable model-output vs ground-truth markdown report.

For each patch we place the model's structured description next to the
CAMELYON16 ground truth so a reviewer can eyeball agreement:

- Ground truth per tile: ``tile_in_mask`` (1 = tile lies inside the tumor
  annotation mask -> genuinely tumor tissue; 0 = not) plus the slide-level
  ``label`` (1 = tumor slide) and the sampling ``source`` regime. The ABMIL
  ``prediction`` is shown too (that is the upstream model's call, not truth).
- Model output: ``tumor_suspicious`` / ``should_abstain`` and the free-text
  ``tissue_description`` / ``evidence`` / ``visible_abnormalities``.

A ``match`` flag compares the model's tumor call against ``tile_in_mask``
(the only real per-tile label) for quick scanning. It is a convenience, not
a metric — the model must never emit a diagnosis, so treat it as directional.

Default output is a *sample* per (source, tile_in_mask) group so the file
stays short; use ``--per_group 0`` to dump everything.
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


def _truth_label(row: dict[str, Any]) -> str:
    """Human phrasing of per-tile ground truth."""
    tim = _s(row.get("tile_in_mask")).strip()
    if tim == "1":
        return "TUMOR tile (in mask)"
    if tim == "0":
        return "non-tumor tile (outside mask)"
    return "unknown"


def _model_call(row: dict[str, Any]) -> str:
    if row.get("should_abstain"):
        return "abstain"
    return _s(row.get("tumor_suspicious")).strip().lower() or "?"


def _match(row: dict[str, Any]) -> str:
    """Directional agreement of model tumor call vs tile_in_mask."""
    tim = _s(row.get("tile_in_mask")).strip()
    call = _model_call(row)
    if call == "abstain" or tim not in ("0", "1") or call not in ("yes", "no"):
        return "—"
    truth_pos = tim == "1"
    call_pos = call == "yes"
    return "✓" if truth_pos == call_pos else "✗"


def main() -> int:
    ap = argparse.ArgumentParser(description="Model-vs-ground-truth example report.")
    ap.add_argument("--input", required=True, help="vlm_outputs.jsonl")
    ap.add_argument("--output", default=None, help="Output .md (default: next to input).")
    ap.add_argument(
        "--per_group",
        type=int,
        default=4,
        help="Max examples per (source, tile_in_mask) group. 0 = all.",
    )
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.is_file():
        print(f"[report_c16_examples] ERROR: not found: {in_path}", file=sys.stderr)
        return 1
    out_path = Path(args.output) if args.output else in_path.parent / "c16_examples_report.md"

    rows = _load(in_path)

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        key = (_s(r.get("source")) or "?", _s(r.get("tile_in_mask")) or "?")
        groups[key].append(r)

    lines: list[str] = []
    lines.append("# CAMELYON16 Quilt-LLaVA — model output vs ground truth")
    lines.append("")
    lines.append(f"Input: `{in_path.as_posix()}` — {len(rows)} patches.")
    lines.append("")
    lines.append(
        "Ground truth per tile is `tile_in_mask` (1 = tile inside tumor "
        "annotation = real tumor tissue). `prediction` is the upstream ABMIL "
        "call, **not** truth. `match` compares the model's tumor call against "
        "`tile_in_mask` for scanning only — the model is forbidden from "
        "emitting a diagnosis, so this is directional, not a score."
    )
    lines.append("")
    shown = 0
    for key in sorted(groups.keys()):
        src, tim = key
        grp = groups[key]
        take = grp if args.per_group == 0 else grp[: args.per_group]
        lines.append(f"## source=`{src}`  tile_in_mask=`{tim}`  ({_truth_label(take[0])})")
        lines.append("")
        lines.append(f"Showing {len(take)} of {len(grp)} in this group.")
        lines.append("")
        for r in take:
            shown += 1
            pid = _s(r.get("patch_id")) or _s(r.get("image_id"))
            lines.append(f"### `{pid}`")
            lines.append("")
            lines.append("| field | ground truth | model |")
            lines.append("|---|---|---|")
            lines.append(
                f"| tumor | **{_truth_label(r)}** | tumor_suspicious=**{_model_call(r)}** "
                f"(match: {_match(r)}) |"
            )
            lines.append(
                f"| slide label | {_s(r.get('label'))} | tissue_organ={_s(r.get('tissue_organ'))} |"
            )
            lines.append(
                f"| ABMIL prediction | {_s(r.get('prediction'))} "
                f"(attn={_s(r.get('attention_score'))[:8]}) | "
                f"conf desc/concl={_s(r.get('visual_description_confidence'))}/"
                f"{_s(r.get('conclusion_confidence'))} |"
            )
            lines.append(f"| json_valid | — | {_s(r.get('json_valid'))} |")
            lines.append("")
            lines.append(f"- **description:** {_s(r.get('tissue_description'))}")
            lines.append(f"- **visible_abnormalities:** {_s(r.get('visible_abnormalities'))}")
            lines.append(f"- **evidence:** {_s(r.get('evidence'))}")
            lines.append(f"- **cellularity:** {_s(r.get('cellularity'))}")
            lines.append(f"- **limitations:** {_s(r.get('limitations'))}")
            if not int(_s(r.get("json_valid")) == "True" or r.get("json_valid") is True):
                lines.append(f"- **raw_response:** `{_s(r.get('raw_response'))[:300]}`")
            lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report_c16_examples] wrote {out_path}")
    print(f"[report_c16_examples] {shown} examples across {len(groups)} groups")
    return 0


if __name__ == "__main__":
    sys.exit(main())
