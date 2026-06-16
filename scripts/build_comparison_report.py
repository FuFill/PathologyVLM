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
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
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

        out.append(f"--- {r['image_id']} " + "-" * (96 - 5 - len(r["image_id"])))
        out.append(f"  file:        {m['image_path']}")
        out.append(f"  wsi_id:      {m['wsi_id']}")
        out.append(f"  tss_code:    {tss}    site: {site}")
        out.append(f"  TCGA study:  {study}   <- closest verified label (cohort-level)")
        out.append(f"  patch:       ({m['x']}, {m['y']})  size={m['patch_size']}  mag={m['magnification']}")
        out.append("")
        out.append("  [PathGen reference_answer]  (auto-generated, NOT verified diagnosis)")
        out.append(wrap(m["reference_answer"]))
        out.append("")
        out.append("  [Model output]")
        out.append(f"    json_valid:         {v['json_valid']}")
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
    lines: list[str] = []
    lines.append("# VLM baseline comparison report\n")
    lines.append("| field | value |")
    lines.append("|---|---|")
    lines.append("| Model | `wisdomik/Quilt-Llava-v1.5-7b` (4-bit) |")
    lines.append("| Dataset | PathGen subset (10 H&E patches, 672x672 @ level0) |")
    lines.append(f"| N images | {len(rows)} |")
    lines.append(f"| Valid JSON | {n_valid}/{len(rows)} ({n_valid / len(rows):.0%}) |\n")

    lines.append("> **Note on \"ground truth\":** PathGen `reference_answer` is an "
                 "auto-generated description, *not* a verified clinical diagnosis. Use it "
                 "as a reference caption only. The TCGA cohort (derived from the WSI "
                 "barcode's TSS code) is the closest verified label and is shown for "
                 "each image.\n")

    # Summary table
    lines.append("## Summary\n")
    lines.append("| image | TCGA study (cohort) | model: tissue | tumor | conf | abstain |")
    lines.append("|---|---|---|---|---|---|")
    for r in rows:
        m = r["manifest"]
        v = r["model"]
        study = TSS_MAP.get(tss_of(m["wsi_id"]), ("", ""))[1]
        tissue = (v["tissue_description"] or "").replace("|", "\\|")
        if len(tissue) > 70:
            tissue = tissue[:67] + "..."
        lines.append(
            f"| `{r['image_id']}` | {study} | {tissue} | "
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

        lines.append(f"### `{r['image_id']}`\n")
        lines.append(f"- **file:** `{m['image_path']}`")
        lines.append(f"- **wsi_id:** `{m['wsi_id']}`")
        lines.append(f"- **TSS:** `{tss}` -- {site}")
        lines.append(f"- **TCGA study (cohort label):** **{study}**")
        lines.append(f"- **patch:** `({m['x']}, {m['y']})` size={m['patch_size']} "
                     f"mag={m['magnification']}\n")

        lines.append("**PathGen reference_answer** _(auto-generated, not verified)_:\n")
        lines.append(f"> {m['reference_answer']}\n")

        lines.append("**Model output:**\n")
        lines.append(f"- `json_valid`: `{v['json_valid']}`")
        lines.append(f"- `tissue_description`: {v['tissue_description']}")
        lines.append(f"- `cellularity`: {v['cellularity']}")
        lines.append(f"- `architecture`: {v['architecture']}")
        lines.append(f"- `tumor_suspicious`: **{v['tumor_suspicious']}**")
        lines.append(f"- `confidence`: **{v['confidence']}**")
        lines.append(f"- `should_abstain`: **{v['should_abstain']}**")

        for key in ("visible_abnormalities", "evidence", "artifacts", "limitations"):
            items = v.get(key, [])
            lines.append(f"- `{key}`:")
            if not items:
                lines.append("    - _(none)_")
            else:
                for it in items:
                    lines.append(f"    - {it}")
        lines.append("")

    return "\n".join(lines) + "\n"


def write_pretty_jsons(rows: list[dict]) -> None:
    PRETTY_DIR.mkdir(parents=True, exist_ok=True)
    for r in rows:
        m = r["manifest"]
        v = r["model"]
        tss = tss_of(m["wsi_id"])
        site, study = TSS_MAP.get(tss, ("(unknown TSS)", "(unknown study)"))
        doc = {
            "image_id": r["image_id"],
            "manifest": {
                "image_path": m["image_path"],
                "wsi_id": m["wsi_id"],
                "patch": {"x": m["x"], "y": m["y"], "size": m["patch_size"]},
                "magnification": m["magnification"],
                "tcga": {"tss_code": tss, "source_site": site, "study": study},
                "reference_answer": m["reference_answer"],
                "reference_answer_note": (
                    "Auto-generated PathGen description, NOT a verified diagnosis."
                ),
            },
            "model_output": v,
        }
        out_path = PRETTY_DIR / f"{r['image_id']}.json"
        out_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")


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
