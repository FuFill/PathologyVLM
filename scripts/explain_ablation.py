"""Hybrid explanation pipeline (variant B->A) for the C17 VLM benchmark.

Subcommands:
  select   - build a stratified explanation subset from the context-run JSON
             (+ slide labels from MIL metadata) and emit a filtered patch
             registry CSV for the siglip ablation run.
  flips    - detect answer flips (trigger ii: A<->non-A) in the ablation
             run JSON, derive the essential patch, add deterministic
             non-flipped control sets, emit the registry CSV for the
             med_gemma explain run.
  analyze  - join the gemma explain run JSON with the flip table, parse
             the patches cited by gemma and report agreement with the
             essential patches.

Inputs are local file paths; filtered registries are uploaded to MinIO.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.s3_utils import upload_to_s3

GROUPS = [
    "TP",
    "FP",
    "FN",
    "hard_negative",
    "oracle_tumor",
    "oracle_non_tumor_a",
    "oracle_non_tumor_c",
    "divergence",
]

PAIR_MISSING = {"P1P2": "P3", "P1P3": "P2", "P2P3": "P1"}
PAIR_NAMES = tuple(PAIR_MISSING)
SINGLE_NAMES = ("P1", "P2", "P3")

FLIP_SRC_GROUP = "top_k|standard"
RANDOM_GROUP = "random_seed_42|standard"


def _patient(slide_id: str) -> str:
    return slide_id.split("_node_")[0]


def _pool_records(recs: list[dict], label_map: dict[str, int]) -> dict[str, list[dict]]:
    topk = [r for r in recs if r["group"] == FLIP_SRC_GROUP]
    rnd42 = {r["slide_id"]: r for r in recs if r["group"] == RANDOM_GROUP}
    return {
        "TP": [r for r in topk if label_map.get(r["slide_id"], -1) == 1
               and r["tile_in_mask"] == 1 and r["answer"] == "A"],
        "FP": [r for r in topk if label_map.get(r["slide_id"], -1) == 0
               and r["answer"] == "A"],
        "FN": [r for r in topk if label_map.get(r["slide_id"], -1) == 1
               and r["answer"] in ("B", "C")],
        "hard_negative": [r for r in recs if r["group"] == "hard_negative|standard"
                          and r["answer"] in ("A", "C")],
        "oracle_tumor": [r for r in recs if r["group"] == "oracle_tumor|standard"
                         and r["answer"] == "A"],
        "oracle_non_tumor_a": [r for r in recs if r["group"] == "oracle_non_tumor|standard"
                               and r["answer"] == "A"],
        "oracle_non_tumor_c": [r for r in recs if r["group"] == "oracle_non_tumor|standard"
                               and r["answer"] == "C"],
        "divergence": [r for r in topk if r["answer"] == "A"
                       and r["slide_id"] in rnd42
                       and rnd42[r["slide_id"]]["answer"] != "A"],
    }


def _sample_pool(pool: list[dict], per_group: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_patient: dict[str, list[dict]] = defaultdict(list)
    for r in sorted(pool, key=lambda x: x["patch_set_uid"]):
        by_patient[_patient(r["slide_id"])].append(r)
    picked: list[dict] = []
    while len(picked) < per_group:
        patients = [p for p in by_patient if by_patient[p]]
        if not patients:
            break
        p = rng.choice(patients)
        picked.append(by_patient[p].pop(0))
    return picked


def _registry_rows_for_sets(
    registry: pd.DataFrame,
    sets: list[dict],
    n_patches: int = 3,
) -> pd.DataFrame:
    keys = ["dataset", "slide_id", "selection_source", "context_set", "random_seed"]
    sel: list[pd.DataFrame] = []
    for r in sets:
        src = r["selection_source"]
        seed = r.get("random_seed") or 0
        m = (
            (registry["dataset"] == r["dataset"])
            & (registry["slide_id"] == r["slide_id"])
            & (registry["selection_source"] == src)
            & (registry["context_set"] == r["context_set"])
        )
        if "random_seed" in registry.columns:
            rs = registry["random_seed"].fillna(0)
            m = m & (rs == seed)
        rows = registry[m].sort_values("rank").head(n_patches)
        if len(rows) == n_patches:
            sel.append(rows)
    if not sel:
        raise SystemExit("[select] ERROR: no registry rows matched the selected sets")
    return pd.concat(sel, ignore_index=True)


def cmd_select(args: argparse.Namespace) -> None:
    with open(args.benchmark_json, encoding="utf-8") as f:
        data = json.load(f)
    recs = [r for r in data["models"]["med_siglip"]
            if r["dataset"] == "c17_native" and r["mode"] == "context"]
    metadata = pd.read_csv(args.metadata)
    label_map = metadata.groupby("slide_id")["label"].first().to_dict()
    registry = pd.read_csv(args.registry)
    registry = registry[registry["dataset"] == "c17_native"]

    pools = _pool_records(recs, label_map)
    selected: list[dict] = []
    for g in GROUPS:
        sampled = _sample_pool(pools[g], args.per_group, args.seed)
        selected.extend(sampled)
        patients = len({_patient(r["slide_id"]) for r in sampled})
        print(f"  {g}: pool={len(pools[g])} selected={len(sampled)} patients={patients}")

    out_df = _registry_rows_for_sets(registry, selected)
    out_df.to_csv(args.out, index=False)
    print(f"[select] registry rows: {len(out_df)} ({len(selected)} sets) -> {args.out}")
    if args.upload_s3:
        url = upload_to_s3(args.out, args.upload_s3)
        print(f"[select] uploaded: {url}")


def _flip_table(recs: list[dict]) -> dict[str, dict]:
    table: dict[str, dict] = {}
    for r in recs:
        ablation = r.get("ablation") or {}
        full = r.get("answer", "")
        pairs = {k: v.get("answer", "") for k, v in ablation.items() if k in PAIR_NAMES}
        flip_pairs = [
            k for k, a in pairs.items()
            if (full == "A" and a in ("B", "C")) or (full in ("B", "C") and a == "A")
        ]
        if flip_pairs:
            essential = set(PAIR_MISSING[k] for k in flip_pairs)
            if len(essential) == 1:
                essential_p = next(iter(essential))
            elif len(flip_pairs) > 1 and len(essential) < 3:
                essential_p = "|".join(sorted(essential))
            else:
                essential_p = "none"
        else:
            essential_p = ""
        table[r["patch_set_uid"]] = {
            "rec": r,
            "flipped": bool(flip_pairs),
            "flip_pairs": flip_pairs,
            "essential": essential_p,
        }
    return table


def cmd_flips(args: argparse.Namespace) -> None:
    with open(args.ablate_json, encoding="utf-8") as f:
        data = json.load(f)
    recs = [r for r in data["models"]["med_siglip"] if r["mode"] == "ablate"]
    print(f"[flips] ablation records: {len(recs)}")
    table = _flip_table(recs)
    flipped = [t for t in table.values() if t["flipped"]]
    non_flipped = [t for t in table.values() if not t["flipped"]]
    print(f"[flips] flipped: {len(flipped)}  non-flipped: {len(non_flipped)}")
    essential_counts = Counter(t["essential"] for t in flipped)
    print(f"[flips] essential patches: {dict(essential_counts)}")

    rng = random.Random(args.seed)
    control = rng.sample(
        sorted(non_flipped, key=lambda t: t["rec"]["patch_set_uid"]),
        min(args.n_control, len(non_flipped)),
    )
    print(f"[flips] control sets: {len(control)}")

    registry = pd.read_csv(args.registry)
    registry = registry[registry["dataset"] == "c17_native"]
    sel = [t["rec"] for t in flipped] + [t["rec"] for t in control]
    out_df = _registry_rows_for_sets(registry, sel)
    out_df.to_csv(args.out, index=False)
    print(f"[flips] registry rows: {len(out_df)} ({len(sel)} sets) -> {args.out}")
    if args.upload_s3:
        url = upload_to_s3(args.out, args.upload_s3)
        print(f"[flips] uploaded: {url}")


FINAL_ANSWER_RE = re.compile(r"FINAL\s*ANSWER\s*:\s*([ABC])", re.IGNORECASE)
CITED_RE = re.compile(r"\bP([123])\b")


def _parse_gemma(raw: str) -> tuple[str, list[str]]:
    m = FINAL_ANSWER_RE.search(raw)
    answer = m.group(1) if m else ""
    cited = sorted({f"P{n}" for n in CITED_RE.findall(raw)})
    return answer, cited


def cmd_analyze(args: argparse.Namespace) -> None:
    with open(args.ablate_json, encoding="utf-8") as f:
        ablate = json.load(f)
    ablate_recs = [r for r in ablate["models"]["med_siglip"] if r["mode"] == "ablate"]
    table = _flip_table(ablate_recs)

    with open(args.explain_json, encoding="utf-8") as f:
        explain = json.load(f)
    explain_recs = [r for r in explain["models"]["med_gemma"] if r["mode"] == "explain"]
    print(f"[analyze] explain records: {len(explain_recs)}")
    gemma = {r["patch_set_uid"]: r for r in explain_recs}

    rows = []
    for uid, t in table.items():
        g = gemma.get(uid)
        if g is None:
            continue
        raw = g["raw_responses"][0] if g.get("raw_responses") else ""
        g_answer, g_cited = _parse_gemma(raw)
        sig = t["rec"]["answer"]
        if t["flipped"]:
            cites_essential = t["essential"] and g_cited == set(t["essential"].split("|"))
            cites_overlap = t["essential"] and bool(set(g_cited) & set(t["essential"].split("|")))
        else:
            cites_essential = cites_overlap = False
        rows.append({
            "patch_set_uid": uid,
            "group": t["rec"]["group"],
            "slide_id": t["rec"]["slide_id"],
            "siglip_answer": sig,
            "flipped": t["flipped"],
            "flip_pairs": ";".join(t["flip_pairs"]),
            "essential": t["essential"],
            "gemma_answer": g_answer,
            "gemma_agree": g_answer == sig,
            "gemma_cited": ";".join(g_cited),
            "gemma_cites_essential": cites_essential,
            "gemma_cites_overlap": cites_overlap,
        })
    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(f"[analyze] joined rows: {len(out)} -> {args.out}")
    if args.upload_s3:
        url = upload_to_s3(args.out, args.upload_s3)
        print(f"[analyze] uploaded: {url}")

    fl = out[out["flipped"]]
    ctrl = out[~out["flipped"]]
    print("\n=== Flipped sets ===")
    for g, grp in fl.groupby("group"):
        n = len(grp)
        agree = grp["gemma_agree"].mean()
        cite = grp["gemma_cites_essential"].mean()
        overlap = grp["gemma_cites_overlap"].mean()
        ncited = grp["gemma_cited"].map(lambda s: len(s.split(";")) if s else 0).mean()
        print(f"  {g}: n={n} agree={agree:.2f} cites_essential={cite:.2f} "
              f"overlap={overlap:.2f} mean_cited={ncited:.2f}")
    print("\n=== Control (non-flipped) sets ===")
    if len(ctrl):
        print(f"  n={len(ctrl)} agree={ctrl['gemma_agree'].mean():.2f}")
        cited_counter: Counter = Counter()
        for s in ctrl["gemma_cited"]:
            for p in s.split(";") if s else ():
                cited_counter[p] += 1
        print(f"  cited distribution: {dict(cited_counter)}")
    unjoined = len(table) - len(out)
    if unjoined:
        print(f"  WARNING: {unjoined} sets missing from explain run")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("select", help="build ablation subset registry")
    p.add_argument("--benchmark-json", required=True, help="context-run JSON (siglip)")
    p.add_argument("--registry", required=True, help="full patch registry CSV")
    p.add_argument("--metadata", required=True, help="MIL metadata CSV with slide labels")
    p.add_argument("--out", required=True, help="output filtered registry CSV")
    p.add_argument("--upload-s3", default="")
    p.add_argument("--per-group", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_select)

    p = sub.add_parser("flips", help="detect flips, emit gemma explain registry")
    p.add_argument("--ablate-json", required=True, help="ablation-run JSON (siglip)")
    p.add_argument("--registry", required=True, help="full patch registry CSV")
    p.add_argument("--out", required=True, help="output filtered registry CSV")
    p.add_argument("--upload-s3", default="")
    p.add_argument("--n-control", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_flips)

    p = sub.add_parser("analyze", help="join gemma explain run with flip table")
    p.add_argument("--ablate-json", required=True, help="ablation-run JSON (siglip)")
    p.add_argument("--explain-json", required=True, help="explain-run JSON (med_gemma)")
    p.add_argument("--out", required=True, help="output agreement CSV")
    p.add_argument("--upload-s3", default="", help="optional MinIO target")
    p.set_defaults(func=cmd_analyze)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()