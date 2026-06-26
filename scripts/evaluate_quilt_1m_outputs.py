"""Evaluate Quilt-1M inference outputs and build a small report.

This script joins a model output JSONL file with ``quilt_1M_lookup.csv``
and computes proxy metrics for:

* JSON validity
* hallucination risk
* unsupported diagnosis risk
* abstain behavior
* visible-feature alignment
* usefulness

It also writes a Markdown report with good and bad examples that embeds
the corresponding images.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "outputs" / "vlm_outputs.jsonl"
DEFAULT_LOOKUP = PROJECT_ROOT / "quilt_1M_lookup.csv"
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "data" / "quilt-1m"
DEFAULT_OUT_MD = PROJECT_ROOT / "outputs" / "quilt1m_report.md"
DEFAULT_OUT_CSV = PROJECT_ROOT / "outputs" / "quilt1m_metrics.csv"

STOPWORDS = {
    "and", "or", "the", "a", "an", "of", "to", "in", "on", "for", "with",
    "by", "from", "at", "is", "are", "was", "were", "this", "that", "these",
    "those", "there", "it", "as", "be", "been", "being", "can", "may",
    "likely", "shows", "show", "image", "tissue", "slide", "field", "view",
}


def normalize_tokens(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return {tok for tok in tokens if len(tok) >= 3 and tok not in STOPWORDS}


def flatten_model_text(row: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in (
        "tissue_organ",
        "tissue_description",
        "cellularity",
        "architecture",
        "tumor_suspicious",
        "visual_description_confidence",
        "conclusion_confidence",
        "confidence",
    ):
        value = row.get(key)
        if value:
            parts.append(str(value))
    for key in ("visible_abnormalities", "evidence", "artifacts", "limitations"):
        value = row.get(key, [])
        if isinstance(value, list):
            parts.extend(str(item) for item in value if item)
    return " ".join(parts)


def flatten_reference_text(row: dict[str, Any]) -> str:
    parts = [
        row.get("caption", ""),
        row.get("corrected_text", ""),
        row.get("roi_text", ""),
        row.get("pathology", ""),
    ]
    return " ".join(str(part) for part in parts if part)


def text_overlap(model_text: str, reference_text: str) -> tuple[float, float]:
    model_tokens = normalize_tokens(model_text)
    reference_tokens = normalize_tokens(reference_text)
    if not model_tokens or not reference_tokens:
        return 0.0, 0.0
    shared = model_tokens & reference_tokens
    precision = len(shared) / len(model_tokens)
    recall = len(shared) / len(reference_tokens)
    return precision, recall


def coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return False


def load_outputs(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_lookup_for_images(csv_path: Path, wanted_names: set[str]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as fin:
        reader = csv.DictReader(fin)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")
        for row in reader:
            name = Path(str(row.get("image_path", "")).replace("\\", "/")).name.lower()
            if name in wanted_names and name not in lookup:
                lookup[name] = row
                if len(lookup) == len(wanted_names):
                    break
    return lookup


def score_row(model_row: dict[str, Any], lookup_row: dict[str, Any], image_dir: Path) -> dict[str, Any]:
    reference_text = flatten_reference_text(lookup_row)
    model_text = flatten_model_text(model_row)
    precision, recall = text_overlap(model_text, reference_text)
    overlap = (precision + recall) / 2.0
    json_valid = bool(model_row.get("json_valid", False))
    abstain = coerce_bool(model_row.get("should_abstain", False))
    tumor_suspicious = str(model_row.get("tumor_suspicious", "")).strip().lower()
    evidence = model_row.get("evidence", [])
    if not isinstance(evidence, list):
        evidence = [evidence] if evidence else []
    visible_abnormalities = model_row.get("visible_abnormalities", [])
    if not isinstance(visible_abnormalities, list):
        visible_abnormalities = [visible_abnormalities] if visible_abnormalities else []

    evidence_tokens = normalize_tokens(" ".join(str(item) for item in evidence))
    text_tokens = normalize_tokens(model_text)
    reference_tokens = normalize_tokens(reference_text)
    shared_tokens = text_tokens & reference_tokens
    unsupported_diagnosis = (
        tumor_suspicious in {"yes", "no"}
        and json_valid
        and (not evidence_tokens or overlap < 0.15)
        and not abstain
    )
    hallucination_risk = bool(
        json_valid
        and not abstain
        and text_tokens
        and overlap < 0.15
        and len(text_tokens) >= 8
    )

    evidence_density = min(1.0, len(evidence_tokens) / 6.0)
    abstain_bonus = 1.0 if abstain and overlap < 0.15 else 0.0
    usefulness = (
        0.35 * float(json_valid)
        + 0.25 * overlap
        + 0.20 * evidence_density
        + 0.20 * abstain_bonus
    )
    if unsupported_diagnosis:
        usefulness -= 0.20
    if hallucination_risk:
        usefulness -= 0.25
    usefulness = max(0.0, min(1.0, usefulness))

    image_path = str(lookup_row.get("image_path", ""))
    image_rel = None
    if image_path:
        try:
            image_rel = (Path("..") / image_dir.relative_to(PROJECT_ROOT) / image_path).as_posix()
        except ValueError:
            image_rel = (image_dir / image_path).as_posix()

    return {
        "image_id": model_row.get("image_id", ""),
        "image_path": image_path,
        "json_valid": json_valid,
        "abstain": abstain,
        "hallucination_risk": hallucination_risk,
        "unsupported_diagnosis": unsupported_diagnosis,
        "visible_alignment": round(overlap, 4),
        "usefulness": round(usefulness, 4),
        "reference_text": reference_text,
        "model_text": model_text,
        "image_rel": image_rel or "",
        "shared_token_count": len(shared_tokens),
        "evidence_count": len(evidence_tokens),
        "visible_abnormalities_count": len(visible_abnormalities),
        "model": model_row,
        "lookup": lookup_row,
    }


def write_metrics_csv(rows: list[dict[str, Any]], out_csv: Path) -> None:
    fieldnames = [
        "image_id",
        "image_path",
        "json_valid",
        "abstain",
        "hallucination_risk",
        "unsupported_diagnosis",
        "visible_alignment",
        "usefulness",
        "shared_token_count",
        "evidence_count",
        "visible_abnormalities_count",
        "image_rel",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def render_examples(title: str, rows: list[dict[str, Any]], limit: int) -> list[str]:
    lines: list[str] = [f"## {title}\n"]
    if not rows:
        lines.append("_No examples found._\n")
        return lines

    for row in rows[:limit]:
        model = row["model"]
        lookup = row["lookup"]
        img = row["image_rel"].replace("\\", "/")
        lines.append(f"### {row['image_id']}\n")
        if img:
            lines.append(f"![]({img})\n")
        lines.append("| metric | value |")
        lines.append("|---|---|")
        lines.append(f"| json_valid | `{row['json_valid']}` |")
        lines.append(f"| abstain | `{row['abstain']}` |")
        lines.append(f"| hallucination_risk | `{row['hallucination_risk']}` |")
        lines.append(f"| unsupported_diagnosis | `{row['unsupported_diagnosis']}` |")
        lines.append(f"| visible_alignment | `{row['visible_alignment']}` |")
        lines.append(f"| usefulness | `{row['usefulness']}` |")
        lines.append("")
        lines.append("**Model output**")
        lines.append(f"- tissue_organ: `{model.get('tissue_organ', '')}`")
        lines.append(f"- tissue_description: {model.get('tissue_description', '')}")
        lines.append(f"- tumor_suspicious: `{model.get('tumor_suspicious', '')}`")
        lines.append(
            f"- confidences: `{model.get('visual_description_confidence', model.get('confidence', ''))} / "
            f"{model.get('conclusion_confidence', model.get('confidence', ''))}`"
        )
        lines.append(f"- evidence: {model.get('evidence', [])}")
        lines.append("")
        lines.append("**Reference text**")
        lines.append(lookup.get("caption", "") or lookup.get("corrected_text", "") or "(empty)")
        lines.append("")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate Quilt-1M outputs and build a report.")
    ap.add_argument("--input", default=str(DEFAULT_INPUT), help="Model output JSONL.")
    ap.add_argument("--lookup", default=str(DEFAULT_LOOKUP), help="quilt_1M_lookup.csv path.")
    ap.add_argument("--image_dir", default=str(DEFAULT_IMAGE_DIR), help="Directory with Quilt-1M images.")
    ap.add_argument("--out-md", default=str(DEFAULT_OUT_MD), help="Markdown report path.")
    ap.add_argument("--out-csv", default=str(DEFAULT_OUT_CSV), help="Per-image metrics CSV path.")
    ap.add_argument("--good-count", type=int, default=6, help="How many good examples to show.")
    ap.add_argument("--bad-count", type=int, default=6, help="How many bad examples to show.")
    args = ap.parse_args()

    in_path = Path(args.input)
    lookup_path = Path(args.lookup)
    image_dir = Path(args.image_dir)
    out_md = Path(args.out_md)
    out_csv = Path(args.out_csv)

    if not in_path.exists():
        print(f"[evaluate_quilt_1m] ERROR: output file not found: {in_path}", file=sys.stderr)
        return 1
    if not lookup_path.exists():
        print(f"[evaluate_quilt_1m] ERROR: lookup CSV not found: {lookup_path}", file=sys.stderr)
        return 1

    model_rows = load_outputs(in_path)
    wanted = {Path(str(r.get("image_path", "")).replace("\\", "/")).name.lower() for r in model_rows}
    wanted.discard("")
    lookup_rows = load_lookup_for_images(lookup_path, wanted)

    joined: list[dict[str, Any]] = []
    missing = 0
    for row in model_rows:
        image_name = Path(str(row.get("image_path", "")).replace("\\", "/")).name.lower()
        lookup_row = lookup_rows.get(image_name)
        if lookup_row is None:
            missing += 1
            continue
        joined.append(score_row(row, lookup_row, image_dir))

    joined.sort(key=lambda r: (r["usefulness"], r["visible_alignment"]), reverse=True)
    if not joined:
        print("[evaluate_quilt_1m] ERROR: no joined rows found.", file=sys.stderr)
        return 1

    counts = Counter()
    counts["n"] = len(joined)
    counts["json_valid"] = sum(1 for r in joined if r["json_valid"])
    counts["abstain"] = sum(1 for r in joined if r["abstain"])
    counts["hallucination_risk"] = sum(1 for r in joined if r["hallucination_risk"])
    counts["unsupported_diagnosis"] = sum(1 for r in joined if r["unsupported_diagnosis"])
    avg_alignment = sum(r["visible_alignment"] for r in joined) / len(joined)
    avg_usefulness = sum(r["usefulness"] for r in joined) / len(joined)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_metrics_csv(joined, out_csv)

    good_rows = [r for r in joined if not r["hallucination_risk"] and not r["unsupported_diagnosis"]]
    bad_rows = sorted(
        joined,
        key=lambda r: (r["usefulness"], int(not r["hallucination_risk"]), int(not r["unsupported_diagnosis"])),
    )

    out_md.parent.mkdir(parents=True, exist_ok=True)
    md_lines: list[str] = []
    md_lines.append("# Quilt-1M evaluation report\n")
    md_lines.append("| metric | value |")
    md_lines.append("|---|---|")
    md_lines.append(f"| images evaluated | {counts['n']} |")
    md_lines.append(f"| JSON valid | {counts['json_valid']} ({counts['json_valid'] / counts['n']:.1%}) |")
    md_lines.append(f"| abstain | {counts['abstain']} ({counts['abstain'] / counts['n']:.1%}) |")
    md_lines.append(f"| hallucination risk | {counts['hallucination_risk']} ({counts['hallucination_risk'] / counts['n']:.1%}) |")
    md_lines.append(f"| unsupported diagnosis | {counts['unsupported_diagnosis']} ({counts['unsupported_diagnosis'] / counts['n']:.1%}) |")
    md_lines.append(f"| mean visible alignment | {avg_alignment:.3f} |")
    md_lines.append(f"| mean usefulness | {avg_usefulness:.3f} |")
    md_lines.append(f"| lookup misses | {missing} |")
    md_lines.append("")
    md_lines.append("> Metrics are proxy scores based on the lookup CSV and the model's structured text; they still need manual review.\n")
    md_lines.extend(render_examples("Good examples", good_rows, args.good_count))
    md_lines.extend(render_examples("Bad examples", bad_rows, args.bad_count))
    out_md.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"[evaluate_quilt_1m] wrote {out_csv}")
    print(f"[evaluate_quilt_1m] wrote {out_md}")
    print(
        f"[evaluate_quilt_1m] n={counts['n']} json_valid={counts['json_valid']} "
        f"abstain={counts['abstain']} hallucination_risk={counts['hallucination_risk']} "
        f"unsupported_diagnosis={counts['unsupported_diagnosis']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
