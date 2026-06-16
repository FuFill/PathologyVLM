"""Build a human-readable comparison report joining:

  * data/he_test_10/manifest.jsonl        (PathGen reference_answer + TCGA wsi_id)
  * outputs/vlm_outputs_reparsed.jsonl    (model's normalized JSON output)

Produces:
  * outputs/report.md                     -> Markdown, one section per image
  * outputs/report.txt                    -> Plain text, aligned blocks
  * outputs/pretty/<image_id>.json        -> Indented per-image JSON

Important caveat surfaced in the report header:
  PathGen `reference_answer` is an auto-generated description, NOT a verified
  clinical diagnosis. The TCGA cohort (derived from wsi_id) is the closest
  thing to a verified label and is shown alongside.
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# Make src/ importable so we can reuse extract_json's repair logic for the
# pretty per-image JSON files.
sys.path.insert(0, str(REPO_ROOT / "src"))
try:
    from json_utils import extract_json  # type: ignore[import-not-found]
except Exception:  # noqa: BLE001
    extract_json = None  # type: ignore[assignment]
MANIFEST = REPO_ROOT / "data" / "he_test_10" / "manifest.jsonl"
REPARSED = REPO_ROOT / "outputs" / "vlm_outputs_reparsed.jsonl"
OUT_MD = REPO_ROOT / "outputs" / "report.md"
OUT_TXT = REPO_ROOT / "outputs" / "report.txt"
PRETTY_DIR = REPO_ROOT / "outputs" / "pretty"

# TCGA TSS code -> (source site, study name) for the 10 codes we use here.
# Source: https://gdc.cancer.gov/resources-tcga-users/tcga-code-tables/tissue-source-site-codes
TSS_MAP: dict[str, tuple[str, str]] = {
    "22": ("Mayo Clinic - Rochester", "Lung squamous cell carcinoma"),
    "55": ("International Genomics Consortium", "Lung adenocarcinoma"),
    "AA": ("Indivumed", "Colon adenocarcinoma"),
    "AH": ("International Genomics Consortium", "Rectum adenocarcinoma"),
    "BP": ("MSKCC", "Kidney renal clear cell carcinoma"),
    "CJ": ("MD Anderson Cancer Center", "Kidney renal clear cell carcinoma"),
    "CM": ("MSKCC", "Colon adenocarcinoma"),
    "D8": ("Greater Poland Cancer Center", "Breast invasive carcinoma"),
    "H9": ("ABS - IUPUI", "Prostate adenocarcinoma"),
    "HT": ("Case Western - St Joes", "Brain Lower Grade Glioma"),
}

# Map TCGA study string -> the canonical tissue_organ token expected from the
# model. Used to score model tissue identification against cohort-level truth.
STUDY_TO_ORGAN: dict[str, str] = {
    "Lung squamous cell carcinoma": "lung",
    "Lung adenocarcinoma": "lung",
    "Colon adenocarcinoma": "colon",
    "Rectum adenocarcinoma": "rectum",
    "Kidney renal clear cell carcinoma": "kidney",
    "Breast invasive carcinoma": "breast",
    "Prostate adenocarcinoma": "prostate",
    "Brain Lower Grade Glioma": "brain",
}


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def tss_of(wsi_id: str) -> str:
    parts = wsi_id.split("-")
    return parts[1] if len(parts) > 1 else ""


def wrap(text: str, width: int = 90, indent: str = "    ") -> str:
    if not text:
        return f"{indent}(empty)"
    lines: list[str] = []
    for paragraph in str(text).splitlines() or [str(text)]:
        wrapped = textwrap.fill(
            paragraph,
            width=width,
            initial_indent=indent,
            subsequent_indent=indent,
        )
        lines.append(wrapped if wrapped else indent)
    return "\n".join(lines)


def render_list(items: list, indent: str = "    ") -> str:
    if not items:
        return f"{indent}(none)"
    return "\n".join(f"{indent}- {item}" for item in items)


HEADER_NOTE = (
    "NOTE on 'ground truth':\n"
    "  PathGen 'reference_answer' is an auto-generated description, NOT a verified\n"
    "  clinical diagnosis. Use it as a reference caption only. The TCGA cohort\n"
    "  (derived from the WSI barcode's TSS code) is the closest verified label\n"
    "  and is shown for each image.\n"
)


def render_txt(rows: list[dict]) -> str:
    out: list[str] = []
    out.append("=" * 96)
    out.append("VLM baseline comparison report")
    out.append(f"Model:     wisdomik/Quilt-Llava-v1.5-7b (4-bit)")
    out.append(f"Dataset:   PathGen subset (10 H&E patches, 672x672 @ level0)")
    out.append(f"N images:  {len(rows)}")
    n_valid = sum(1 for r in rows if r["model"]["json_valid"])
    out.append(f"Valid JSON: {n_valid}/{len(rows)}  ({n_valid / len(rows):.0%})")
    n_with_organ_truth = sum(
        1 for r in rows
        if STUDY_TO_ORGAN.get(TSS_MAP.get(tss_of(r["manifest"]["wsi_id"]), ("", ""))[1])
    )
    n_organ_correct = sum(
        1 for r in rows
        if r["model"].get("tissue_organ")
        and STUDY_TO_ORGAN.get(TSS_MAP.get(tss_of(r["manifest"]["wsi_id"]), ("", ""))[1])
        == r["model"]["tissue_organ"]
    )
    if n_with_organ_truth:
        out.append(
            f"Tissue organ vs TCGA cohort: {n_organ_correct}/{n_with_organ_truth} "
            f"({n_organ_correct / n_with_organ_truth:.0%})"
        )
    out.append("=" * 96)
    out.append("")
    out.append(HEADER_NOTE)
    out.append("=" * 96)
    out.append("")

    for r in rows:
        m = r["manifest"]
        v = r["model"]
        tss = tss_of(m["wsi_id"])
        site, study = TSS_MAP.get(tss, ("(unknown TSS)", "(unknown study)"))
        expected_organ = STUDY_TO_ORGAN.get(study, "")
        model_organ = v.get("tissue_organ", "")
        organ_match = (
            "MATCH" if expected_organ and model_organ == expected_organ
            else ("MISS" if expected_organ else "n/a")
        )

        out.append(f"--- {r['image_id']} " + "-" * (96 - 5 - len(r["image_id"])))
        out.append(f"  file:        {m['image_path']}")
        out.append(f"  wsi_id:      {m['wsi_id']}")
        out.append(f"  tss_code:    {tss}    site: {site}")
        out.append(f"  TCGA study:  {study}   <- closest verified label (cohort-level)")
        out.append(f"  expected organ (from cohort): {expected_organ or '(unknown)'}")
        out.append(f"  patch:       ({m['x']}, {m['y']})  size={m['patch_size']}  mag={m['magnification']}")
        out.append("")
        out.append("  [PathGen reference_answer]  (auto-generated, NOT verified diagnosis)")
        out.append(wrap(m["reference_answer"]))
        out.append("")
        out.append("  [Model output]")
        out.append(f"    json_valid:         {v['json_valid']}")
        out.append(f"    tissue_organ:       {model_organ}   [{organ_match}]")
        out.append(f"    tissue_description: {v['tissue_description']}")
        out.append(f"    cellularity:        {v['cellularity']}")
        out.append(f"    architecture:       {v['architecture']}")
        out.append(f"    tumor_suspicious:   {v['tumor_suspicious']}")
        out.append(f"    confidence:         {v['confidence']}")
        out.append(f"    should_abstain:     {v['should_abstain']}")
        out.append("    visible_abnormalities:")
        out.append(render_list(v["visible_abnormalities"], indent="      "))
        out.append("    evidence:")
        out.append(render_list(v["evidence"], indent="      "))
        out.append("    artifacts:")
        out.append(render_list(v["artifacts"], indent="      "))
        out.append("    limitations:")
        out.append(render_list(v["limitations"], indent="      "))
        out.append("")

    return "\n".join(out) + "\n"


def render_md(rows: list[dict]) -> str:
    n_valid = sum(1 for r in rows if r["model"]["json_valid"])
    n_with_organ_truth = sum(
        1 for r in rows
        if STUDY_TO_ORGAN.get(TSS_MAP.get(tss_of(r["manifest"]["wsi_id"]), ("", ""))[1])
    )
    n_organ_correct = sum(
        1 for r in rows
        if r["model"].get("tissue_organ")
        and STUDY_TO_ORGAN.get(TSS_MAP.get(tss_of(r["manifest"]["wsi_id"]), ("", ""))[1])
        == r["model"]["tissue_organ"]
    )

    lines: list[str] = []
    lines.append("# VLM baseline comparison report\n")
    lines.append("| field | value |")
    lines.append("|---|---|")
    lines.append("| Model | `wisdomik/Quilt-Llava-v1.5-7b` (4-bit) |")
    lines.append("| Dataset | PathGen subset (10 H&E patches, 672x672 @ level0) |")
    lines.append(f"| N images | {len(rows)} |")
    lines.append(f"| Valid JSON | {n_valid}/{len(rows)} ({n_valid / len(rows):.0%}) |")
    if n_with_organ_truth:
        lines.append(
            f"| Tissue organ vs TCGA cohort | {n_organ_correct}/{n_with_organ_truth} "
            f"({n_organ_correct / n_with_organ_truth:.0%}) |"
        )
    lines.append("")

    lines.append("> **Note on \"ground truth\":** PathGen `reference_answer` is an "
                 "auto-generated description, *not* a verified clinical diagnosis. Use it "
                 "as a reference caption only. The TCGA cohort (derived from the WSI "
                 "barcode's TSS code) is the closest verified label and is shown for "
                 "each image.\n")

    # Summary table
    lines.append("## Summary\n")
    lines.append("| image | TCGA study (cohort) | expected organ | model organ | match | tumor | conf | abstain |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        m = r["manifest"]
        v = r["model"]
        study = TSS_MAP.get(tss_of(m["wsi_id"]), ("", ""))[1]
        expected_organ = STUDY_TO_ORGAN.get(study, "")
        model_organ = v.get("tissue_organ", "")
        match = (
            "OK" if expected_organ and model_organ == expected_organ
            else ("MISS" if expected_organ else "-")
        )
        lines.append(
            f"| `{r['image_id']}` | {study} | {expected_organ or '-'} | "
            f"{model_organ or '-'} | **{match}** | "
            f"{v['tumor_suspicious']} | {v['confidence']} | {v['should_abstain']} |"
        )
    lines.append("")

    # Per-image detail
    lines.append("## Per-image detail\n")
    for r in rows:
        m = r["manifest"]
        v = r["model"]
        tss = tss_of(m["wsi_id"])
        site, study = TSS_MAP.get(tss, ("(unknown TSS)", "(unknown study)"))
        expected_organ = STUDY_TO_ORGAN.get(study, "")
        model_organ = v.get("tissue_organ", "")
        if not expected_organ:
            organ_badge = "n/a"
        elif model_organ == expected_organ:
            organ_badge = "MATCH"
        else:
            organ_badge = "MISS"

        lines.append("---\n")
        lines.append(f"### `{r['image_id']}`\n")

        # --- Ground truth block ---
        lines.append("**Ground truth (verified at SLIDE level, not patch level):**\n")
        lines.append("| field | value |")
        lines.append("|---|---|")
        lines.append(f"| file | `{m['image_path']}` |")
        lines.append(f"| WSI id | `{m['wsi_id']}` |")
        lines.append(f"| TCGA TSS | `{tss}` ({site}) |")
        lines.append(f"| TCGA study (cohort) | **{study}** |")
        lines.append(f"| Expected organ | **`{expected_organ or '(no canonical mapping)'}`** |")
        lines.append(f"| Patch | x=`{m['x']}`, y=`{m['y']}`, "
                     f"size=`{m['patch_size']}`, mag=`{m['magnification']}` |")
        lines.append("")

        # --- PathGen reference (caveated) ---
        lines.append("**PathGen `reference_answer`** "
                     "_(auto-generated by GPT-4V, NOT a verified diagnosis -- "
                     "treat as a reference caption only)_:\n")
        lines.append(f"> {m['reference_answer']}\n")

        # --- Model output (structured) ---
        lines.append("**Model output (structured):**\n")
        lines.append("| field | value |")
        lines.append("|---|---|")
        lines.append(f"| json_valid | `{v['json_valid']}` |")
        lines.append(f"| tissue_organ | **`{model_organ or '-'}`** |")
        lines.append(f"| tissue_description | {v['tissue_description'] or '_(empty)_'} |")
        lines.append(f"| cellularity | {v['cellularity'] or '_(empty)_'} |")
        lines.append(f"| architecture | {v['architecture'] or '_(empty)_'} |")
        lines.append(f"| tumor_suspicious | **{v['tumor_suspicious']}** |")
        lines.append(f"| confidence | **{v['confidence']}** |")
        lines.append(f"| should_abstain | **{v['should_abstain']}** |")
        lines.append("")

        # --- List-valued fields, one block per category ---
        for key, title in [
            ("visible_abnormalities", "Visible abnormalities"),
            ("evidence",              "Evidence cited"),
            ("artifacts",             "Artifacts noted"),
            ("limitations",           "Limitations / caveats"),
        ]:
            items = v.get(key, [])
            lines.append(f"**{title}** (`{key}`):")
            if not items:
                lines.append("- _(none)_")
            else:
                for it in items:
                    lines.append(f"- {it}")
            lines.append("")

        # --- Verdict ---
        lines.append("**Verdict vs ground truth:**\n")
        lines.append(f"- expected organ: `{expected_organ or '(none)'}`")
        lines.append(f"- model organ:    `{model_organ or '-'}`")
        lines.append(f"- organ match:    **{organ_badge}**")
        lines.append("")

    return "\n".join(lines) + "\n"


def write_pretty_jsons(rows: list[dict]) -> None:
    PRETTY_DIR.mkdir(parents=True, exist_ok=True)
    for r in rows:
        m = r["manifest"]
        v = r["model"]
        tss = tss_of(m["wsi_id"])
        site, study = TSS_MAP.get(tss, ("(unknown TSS)", "(unknown study)"))
        expected_organ = STUDY_TO_ORGAN.get(study, "")
        model_organ = v.get("tissue_organ", "")
        if not expected_organ:
            organ_match = "n/a (no canonical mapping for this study)"
        elif model_organ == expected_organ:
            organ_match = "MATCH"
        else:
            organ_match = f"MISS (expected {expected_organ!r}, got {model_organ!r})"

        # Try to expose the model's JSON response as a real nested object
        # instead of an escaped string. Uses the same repair pipeline as the
        # main parser (handles "\\_" / "\\*" / markdown-escaped keys).
        raw_response = v.get("raw_response", "")
        response_object = None
        if raw_response:
            if extract_json is not None:
                response_object = extract_json(raw_response)
            if response_object is None:
                try:
                    response_object = json.loads(raw_response)
                except json.JSONDecodeError:
                    response_object = None

        # Strip the verbose / redundant keys from the model row so the file
        # stays scannable. raw_response is kept as a separate block.
        model_clean = {
            k: v[k] for k in (
                "tissue_organ",
                "tissue_description",
                "cellularity",
                "architecture",
                "visible_abnormalities",
                "tumor_suspicious",
                "evidence",
                "artifacts",
                "limitations",
                "confidence",
                "should_abstain",
            )
            if k in v
        }

        doc = {
            "image_id": r["image_id"],
            "image_file": m["image_path"],
            "ground_truth": {
                "wsi_id": m["wsi_id"],
                "tcga": {
                    "tss_code": tss,
                    "source_site": site,
                    "study": study,
                },
                "expected_organ": expected_organ or None,
                "patch": {
                    "x": m["x"],
                    "y": m["y"],
                    "size": m["patch_size"],
                    "magnification": m["magnification"],
                },
                "notes": [
                    "TCGA study is verified at the WHOLE-SLIDE level (cohort label).",
                    "It tells you the organ of origin and that the slide came from a "
                    "cancer patient. It does NOT tell you whether this specific patch "
                    "is on tumor, benign tissue, stroma, or a normal margin.",
                    "No per-patch pathologist annotation exists for this dataset.",
                ],
            },
            "pathgen_reference_answer": {
                "text": m["reference_answer"],
                "warning": (
                    "AUTO-GENERATED by PathGen using GPT-4V on source-paper context. "
                    "This is a reference caption, NOT a verified clinical diagnosis. "
                    "Do not score the model against this as if it were truth."
                ),
            },
            "model": {
                "name": v.get("model_name", ""),
                "json_valid": v.get("json_valid", False),
                "error": v.get("error", ""),
                "structured_output": model_clean,
            },
            "verdict_vs_ground_truth": {
                "expected_organ": expected_organ or None,
                "model_organ": model_organ or None,
                "organ_match": organ_match,
            },
            "raw_model_response": {
                "as_object": response_object,
                "as_text": raw_response,
            },
        }
        out_path = PRETTY_DIR / f"{r['image_id']}.json"
        out_path.write_text(
            json.dumps(doc, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


def main() -> None:
    manifest_rows = load_jsonl(MANIFEST)
    model_rows = load_jsonl(REPARSED)
    by_id = {r["image_id"]: r for r in model_rows}

    joined: list[dict] = []
    for m in manifest_rows:
        v = by_id.get(m["image_id"])
        if v is None:
            print(f"[WARN] No model output for {m['image_id']}, skipping")
            continue
        joined.append({"image_id": m["image_id"], "manifest": m, "model": v})

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(render_md(joined), encoding="utf-8")
    OUT_TXT.write_text(render_txt(joined), encoding="utf-8")
    write_pretty_jsons(joined)

    print(f"[report] joined {len(joined)} rows")
    print(f"[report] wrote {OUT_MD}")
    print(f"[report] wrote {OUT_TXT}")
    print(f"[report] wrote {len(joined)} files to {PRETTY_DIR}")


if __name__ == "__main__":
    main()
