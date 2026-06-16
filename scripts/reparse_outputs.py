"""Re-parse a vlm_outputs JSONL file using the current json_utils logic.

Useful when ``src/json_utils.py`` has been improved (e.g. to handle new
LLM-emitted quirks like markdown-style ``\\_`` escapes) and we want to
recover structured fields from already-collected ``raw_response`` strings
without spending more GPU time.

Reads:   outputs/vlm_outputs_jsonl.jsonl  (one JSON per line)
Writes:  outputs/vlm_outputs_reparsed.jsonl
         outputs/vlm_outputs_reparsed.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.json_utils import DEFAULT_SCHEMA, extract_json, normalize_json  # noqa: E402

# Preserve column order: meta first, then schema fields in DEFAULT_SCHEMA order.
META_FIELDS = ("image_id", "image_path", "model_name", "json_valid", "error")
SCHEMA_FIELDS = tuple(DEFAULT_SCHEMA.keys())
CSV_FIELDS = META_FIELDS + SCHEMA_FIELDS + ("raw_response",)


def reparse_row(row: dict) -> dict:
    raw = row.get("raw_response", "")
    parsed = extract_json(raw)
    normalized = normalize_json(parsed)
    out = {
        "image_id": row.get("image_id", ""),
        "image_path": row.get("image_path", ""),
        "model_name": row.get("model_name", ""),
        "raw_response": raw,
        "json_valid": parsed is not None,
        "error": "" if parsed is not None else "extract_json returned None",
    }
    out.update(normalized)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default=str(REPO_ROOT / "outputs" / "vlm_outputs_jsonl.jsonl"),
        help="Source JSONL (each line must contain a 'raw_response' field).",
    )
    ap.add_argument(
        "--out-jsonl",
        default=str(REPO_ROOT / "outputs" / "vlm_outputs_reparsed.jsonl"),
    )
    ap.add_argument(
        "--out-csv",
        default=str(REPO_ROOT / "outputs" / "vlm_outputs_reparsed.csv"),
    )
    args = ap.parse_args()

    in_path = Path(args.input)
    out_jsonl = Path(args.out_jsonl)
    out_csv = Path(args.out_csv)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    n_valid = 0
    n_recovered = 0  # was invalid before, now valid
    rows: list[dict] = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[WARN] Skipping unparseable line {n + 1}: {exc}")
                continue
            was_valid_before = bool(row.get("json_valid", False))
            new_row = reparse_row(row)
            n += 1
            if new_row["json_valid"]:
                n_valid += 1
                if not was_valid_before:
                    n_recovered += 1
            rows.append(new_row)

    with out_jsonl.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            row_for_csv = dict(r)
            for k in ("visible_abnormalities", "evidence", "artifacts", "limitations"):
                row_for_csv[k] = json.dumps(r.get(k, []), ensure_ascii=False)
            writer.writerow(row_for_csv)

    print(
        f"[reparse] n={n} valid_now={n_valid} ({n_valid / n:.1%}) "
        f"newly_recovered={n_recovered}"
    )
    print(f"[reparse] wrote {out_jsonl}")
    print(f"[reparse] wrote {out_csv}")


if __name__ == "__main__":
    main()
