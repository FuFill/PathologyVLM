"""Generate a WSI Explainability & Microdescription Report from VLM outputs.

Takes the structured JSON outputs from `run_remote_vlm.py` (specifically
the top-k attention patches retrieved by the MIL WSI baseline) and builds
slide-level explainability summaries for pathologists.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_vlm_outputs(path: Path) -> list[dict[str, Any]]:
    rows = []
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        pass
    elif path.suffix == ".csv":
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                # Parse list fields stored as JSON strings in CSV
                for field in (
                    "visible_abnormalities",
                    "evidence",
                    "artifacts",
                    "limitations",
                ):
                    if r.get(field):
                        try:
                            r[field] = json.loads(r[field])
                        except Exception:
                            r[field] = [r[field]]
                rows.append(r)
    return rows


def generate_slide_report(rows: list[dict[str, Any]], out_md: Path) -> None:
    # Group by slide_id
    slides: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        slide_id = str(r.get("slide_id", "unknown")).strip() or "unknown"
        slides[slide_id].append(r)

    out_md.parent.mkdir(parents=True, exist_ok=True)
    with out_md.open("w", encoding="utf-8") as f:
        f.write("# MIL + VLM WSI Microdescription & Explainability Report\n\n")
        f.write(
            "This report provides structured morphological microdescriptions and visual evidence for the top attention patches retrieved by the MIL WSI baseline.\n\n"
        )

        for slide_id in sorted(slides.keys()):
            patches = slides[slide_id]

            slide_label = patches[0].get("label", "N/A")
            slide_pred = patches[0].get("prediction", "N/A")

            f.write(f"## Whole-Slide Image: `{slide_id}`\n\n")
            f.write(
                f"* **MIL Baseline Prediction:** `{slide_pred}` (Ground Truth Label: `{slide_label}`)\n"
            )
            f.write(f"* **Total Patches Evaluated:** {len(patches)}\n\n")

            # Group patches by sampling source (top_k, oracle_tumor, etc.)
            by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for p in patches:
                src = str(p.get("source", "unknown")).strip() or "unknown"
                by_source[src].append(p)

            # Slide summary across all evaluated patches
            high_cell = sum(
                1 for p in patches if str(p.get("cellularity")).lower() == "high"
            )
            distorted = sum(
                1 for p in patches if "distorted" in str(p.get("architecture")).lower()
            )
            suspicious = sum(
                1 for p in patches if str(p.get("tumor_suspicious")).lower() == "yes"
            )

            f.write("### Slide-Level Morphological Summary (All Patches)\n")
            f.write(f"* **High Cellularity Patches:** {high_cell} / {len(patches)}\n")
            f.write(
                f"* **Architectural Distortion Patches:** {distorted} / {len(patches)}\n"
            )
            f.write(
                f"* **VLM Tumor Suspicion Flags:** {suspicious} / {len(patches)}\n\n"
            )

            # Focus the clinical report on the top_k MIL attention patches
            report_sources = (
                ["top_k"] if "top_k" in by_source else sorted(by_source.keys())
            )

            for src in report_sources:
                src_patches = by_source[src]
                # Sort by attention rank ascending, or attention score descending
                src_patches.sort(
                    key=lambda x: (
                        int(x.get("attention_rank", 999))
                        if str(x.get("attention_rank", "")).isdigit()
                        else 999,
                        -float(x.get("attention_score", 0))
                        if str(x.get("attention_score", ""))
                        .replace(".", "-")
                        .replace("e", "-")
                        .isdigit()
                        else 0,
                    )
                )

                f.write(
                    f"### Section: `{src.upper()}` Patches (MIL Attention Selection)\n\n"
                )

                for idx, p in enumerate(src_patches[:10], start=1):
                    rank = p.get("attention_rank", idx)
                    score = p.get("attention_score", "N/A")
                    if isinstance(score, float) or (
                        isinstance(score, str)
                        and score.replace(".", "").replace("e", "-").isdigit()
                    ):
                        try:
                            score = f"{float(score):.6f}"
                        except Exception:
                            pass

                    f.write(f"#### Rank #{rank} (Attention Score: `{score}`)\n")
                    f.write(
                        f"* **Tile ID / Path:** `{Path(str(p.get('image_path', ''))).name}`\n"
                    )
                    f.write(
                        f"* **Microdescription:** {p.get('tissue_description', 'N/A')}\n"
                    )
                    f.write(
                        f"* **Cellularity:** `{p.get('cellularity', 'N/A')}` | **Architecture:** `{p.get('architecture', 'N/A')}`\n"
                    )

                    abnorms = p.get("visible_abnormalities", [])
                    if isinstance(abnorms, list) and abnorms:
                        f.write(
                            f"* **Visible Abnormalities:** {', '.join(str(a) for a in abnorms)}\n"
                        )

                    ev = p.get("evidence", [])
                    if isinstance(ev, list) and ev:
                        f.write(
                            f"* **Visual Evidence:** {', '.join(str(e) for e in ev)}\n"
                        )

                    f.write(
                        f"* **Tumor Suspicious:** `{p.get('tumor_suspicious', 'uncertain')}` (Confidence: `{p.get('conclusion_confidence', 'low')}`)\n\n"
                    )

            f.write("---\n\n")

    print(f"[report generator] Wrote WSI Explainability Report to {out_md}")


def main():
    ap = argparse.ArgumentParser(
        description="Generate WSI Microdescription Explainability Report."
    )
    ap.add_argument(
        "--input", required=True, help="Path to vlm_outputs.jsonl or vlm_outputs.csv"
    )
    ap.add_argument(
        "--out_md",
        default="outputs/wsi_explainability_report.md",
        help="Output markdown report path",
    )
    args = ap.parse_args()

    rows = load_vlm_outputs(Path(args.input))
    print(f"[report generator] Loaded {len(rows)} patch records from {args.input}")
    generate_slide_report(rows, Path(args.out_md))


if __name__ == "__main__":
    main()
