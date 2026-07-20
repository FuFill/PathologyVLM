"""Grouped evaluation report for a CAMELYON16 Quilt-LLaVA baseline run.

The point of this report is NOT to produce one aggregate number. A single
"JSON-valid: 94%" hides mode collapse and per-slice failure. Instead we break
the run down by group ``(source, label, tile_in_mask)`` and show, per group:

    n                       patches in the group
    json_valid %            fraction with json_valid=True
    parse_valid %           fraction that normalized to a usable dict
    unique_raw              distinct raw_response strings (flags mode collapse:
                            n=40 but unique_raw=1 means the model said the same
                            thing 40 times)
    tumor_suspicious        yes / no / uncertain counts
    abstain %               fraction with should_abstain=True

Input is the ``vlm_outputs.jsonl`` produced by ``run_remote_vlm.py`` (rows
already carry the merged patch metadata: source, label, tile_in_mask, ...).
Output is a Markdown table + a flat CSV, both grouped, plus per-source
rollups. No headline global metric.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.json_utils import parse_with_provenance  # noqa: E402

DEFAULT_INPUT = PROJECT_ROOT / "outputs" / "abmil_standard_100" / "vlm_outputs.jsonl"
DEFAULT_OUT_MD = PROJECT_ROOT / "outputs" / "c16_grouped_report.md"
DEFAULT_OUT_CSV = PROJECT_ROOT / "outputs" / "c16_grouped_metrics.csv"

GROUP_KEYS = ("source", "label", "tile_in_mask")


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


def _group_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return tuple(str(row.get(k, "")).strip() for k in GROUP_KEYS)  # type: ignore[return-value]


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    json_valid = sum(1 for r in rows if _coerce_bool(r.get("json_valid")))
    # parse_valid is the newer provenance field; fall back to json_valid for
    # older runs that predate it.
    parse_valid = sum(
        1 for r in rows
        if _coerce_bool(r.get("parse_valid", r.get("json_valid")))
    )
    # strict_json_valid / schema_valid: the raw response was valid JSON with NO
    # repair / matched the schema. For runs that predate these flags we
    # recompute them from raw_response so the report still shows the TRUE strict
    # rate rather than a misleading 0.
    def _flag(r: dict[str, Any], key: str) -> bool:
        if key in r and r[key] is not None:
            return _coerce_bool(r[key])
        _, prov = parse_with_provenance(str(r.get("raw_response", "")))
        return bool(prov[key])

    strict_json_valid = sum(1 for r in rows if _flag(r, "strict_json_valid"))
    schema_valid = sum(1 for r in rows if _flag(r, "schema_valid"))
    unique_raw = len({str(r.get("raw_response", "")) for r in rows})
    ts = Counter(str(r.get("tumor_suspicious", "")).strip().lower() or "uncertain" for r in rows)
    abstain = sum(1 for r in rows if _coerce_bool(r.get("should_abstain")))
    return {
        "n": n,
        "strict_json_valid": strict_json_valid,
        "json_valid": json_valid,
        "parse_valid": parse_valid,
        "schema_valid": schema_valid,
        "unique_raw": unique_raw,
        "ts_yes": ts.get("yes", 0),
        "ts_no": ts.get("no", 0),
        "ts_uncertain": ts.get("uncertain", 0),
        "abstain": abstain,
    }


def _pct(num: int, den: int) -> str:
    return f"{num / den:.0%}" if den else "-"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    return lines


def _summary_cells(label: str, s: dict[str, Any]) -> list[str]:
    return [
        label,
        str(s["n"]),
        f"{s['strict_json_valid']} ({_pct(s['strict_json_valid'], s['n'])})",
        f"{s['parse_valid']} ({_pct(s['parse_valid'], s['n'])})",
        f"{s['schema_valid']} ({_pct(s['schema_valid'], s['n'])})",
        str(s["unique_raw"]),
        f"{s['ts_yes']}/{s['ts_no']}/{s['ts_uncertain']}",
        _pct(s["abstain"], s["n"]),
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description="Grouped C16 baseline report.")
    ap.add_argument("--input", default=str(DEFAULT_INPUT), help="vlm_outputs.jsonl.")
    ap.add_argument("--out-md", default=str(DEFAULT_OUT_MD))
    ap.add_argument("--out-csv", default=str(DEFAULT_OUT_CSV))
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.is_file():
        print(f"[report_c16_groups] ERROR: input not found: {in_path}", file=sys.stderr)
        return 1

    rows = load_jsonl(in_path)
    if not rows:
        print("[report_c16_groups] ERROR: no rows in input.", file=sys.stderr)
        return 1

    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[_group_key(r)].append(r)
        by_source[str(r.get("source", "")).strip()].append(r)

    header = ["group (source/label/tile_in_mask)", "n", "strict_json",
              "parse_valid", "schema_valid", "unique_raw", "tumor yes/no/unc",
              "abstain"]

    group_body: list[list[str]] = []
    for key in sorted(groups.keys()):
        s = _summarize(groups[key])
        group_body.append(_summary_cells("/".join(key), s))

    src_header = ["source", "n", "strict_json", "parse_valid", "schema_valid",
                  "unique_raw", "tumor yes/no/unc", "abstain"]
    src_body: list[list[str]] = []
    for src in sorted(by_source.keys()):
        s = _summarize(by_source[src])
        src_body.append(_summary_cells(src, s))

    overall = _summarize(rows)

    md: list[str] = ["# CAMELYON16 Quilt-LLaVA grouped baseline report", ""]
    md.append(f"Input: `{in_path.as_posix()}` — {len(rows)} patches, "
              f"{len(groups)} groups, {len(by_source)} sources.")
    md.append("")
    md.append("`unique_raw` = distinct raw responses in the group. "
              "If it is far below `n`, the model is repeating itself (mode "
              "collapse) rather than describing each patch.")
    md.append("")
    md.append("`strict_json` = raw response parsed as JSON with NO repair; "
              "`parse_valid` = a dict was recovered by any means (fence-strip, "
              "`\\_` escape repair, or brace-salvage); `schema_valid` = the "
              "parsed dict had every required field with in-range enums. A large "
              "gap between `strict_json` and `parse_valid` means the reported "
              "JSON rate depended on repair. (For runs predating these flags, "
              "`strict_json`/`schema_valid` are recomputed from `raw_response`.)")
    md.append("")
    md.append("## By group (source / label / tile_in_mask)")
    md.extend(_table(header, group_body))
    md.append("")
    md.append("## By source")
    md.extend(_table(src_header, src_body))
    md.append("")
    md.append("## Overall (context only — not the headline)")
    md.extend(_table(header[1:], [_summary_cells("", overall)[1:]]))
    md.append("")

    out_md = Path(args.out_md)
    out_csv = Path(args.out_csv)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")

    csv_fields = ["source", "label", "tile_in_mask", "n", "strict_json_valid",
                  "json_valid", "parse_valid", "schema_valid", "unique_raw",
                  "ts_yes", "ts_no", "ts_uncertain", "abstain"]
    with out_csv.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=csv_fields)
        writer.writeheader()
        for key in sorted(groups.keys()):
            s = _summarize(groups[key])
            writer.writerow({"source": key[0], "label": key[1],
                             "tile_in_mask": key[2], **{k: s[k] for k in
                             ("n", "strict_json_valid", "json_valid",
                              "parse_valid", "schema_valid", "unique_raw",
                              "ts_yes", "ts_no", "ts_uncertain", "abstain")}})

    print(f"[report_c16_groups] wrote {out_md}")
    print(f"[report_c16_groups] wrote {out_csv}")
    print(f"[report_c16_groups] {len(rows)} patches across {len(groups)} groups")
    return 0


if __name__ == "__main__":
    sys.exit(main())
