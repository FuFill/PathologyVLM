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

from scipy.stats import chi2, fisher_exact

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


def _set_key(rec: dict) -> tuple:
    return (
        rec.get("dataset", ""),
        rec.get("slide_id", ""),
        rec.get("selection_source", ""),
        rec.get("context_set", ""),
        int(rec.get("random_seed") or 0),
    )


def _flip_table(recs: list[dict]) -> dict[tuple, dict]:
    table: dict[tuple, dict] = {}
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
        table[_set_key(r)] = {
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


def _run_order(registry: pd.DataFrame) -> list[dict]:
    """Reproduce the benchmark's set order: groupby(keys, sort=False) over the
    registry (first-appearance order, duplicate keys merged), then top_k first."""
    df = registry.copy()
    if "random_seed" not in df.columns:
        df["random_seed"] = 0
    df["random_seed"] = df["random_seed"].fillna(0)
    keys = ["dataset", "slide_id", "selection_source", "context_set", "random_seed"]
    order: list[dict] = []
    seen: set = set()
    for _, grp in df.groupby(keys, sort=False):
        k = tuple(grp[keys].iloc[0])
        if k in seen:
            continue
        seen.add(k)
        order.append({
            "dataset": str(grp["dataset"].iloc[0]),
            "slide_id": str(grp["slide_id"].iloc[0]),
            "selection_source": str(grp["selection_source"].iloc[0]),
            "context_set": str(grp["context_set"].iloc[0]),
            "random_seed": int(grp["random_seed"].iloc[0]),
        })
    return [s for s in order if s["selection_source"] == "top_k"] + [
        s for s in order if s["selection_source"] != "top_k"
    ]


def _group_str(rec: dict) -> str:
    if rec["selection_source"] == "random":
        return f"random_seed_{int(rec['random_seed'])}|{rec['context_set']}"
    return f"{rec['selection_source']}|{rec['context_set']}"


def _log_lines_answers(log: str, total: int, ablate: bool) -> dict[int, dict]:
    """Parse 'RAW: <raw>  -> <ans>' lines for each set index (1-based)."""
    out: dict[int, dict] = {}
    lines = log.splitlines()
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        m = re.match(
            rf"^\s+\[(\d+)/{total}( ablate (P1|P2|P3|P1P2|P1P3|P2P3))?\] RAW: (.*)$",
            line,
        )
        if not m:
            continue
        idx, _, key, rest = m.groups()
        idx = int(idx)
        if (key and not ablate) or (not key and ablate):
            continue
        parts = rest.rsplit("  -> ", 1)
        raw = parts[0]
        ans = parts[1] if len(parts) == 2 and parts[1] in ("A", "B", "C") else ""
        score = None
        if i + 1 < len(lines):
            sm = re.search(r"score=(-?[\d.]+)", lines[i + 1])
            if sm:
                score = float(sm.group(1))
        out.setdefault(idx, {})[key or "full"] = {
            "raw": raw, "answer": ans, "score": score,
        }
    return out


HINT_TEXT = {"A": "TUMOR PRESENT", "B": "NO TUMOR", "C": "INSUFFICIENT"}
HINT_INVERT = {"A": "B", "B": "A", "C": "A"}


def cmd_hints(args: argparse.Namespace) -> None:
    with open(args.benchmark_json, encoding="utf-8") as f:
        data = json.load(f)
    recs = [r for r in data["models"]["med_siglip"]
            if r["dataset"] == "c17_native" and r["mode"] == "context"]
    answers = {_set_key(r): r["answer"] for r in recs}
    print(f"[hints] context answers: {len(answers)}")

    registry = pd.read_csv(args.registry)
    registry = registry[registry["dataset"] == "c17_native"].copy()
    if "random_seed" not in registry.columns:
        registry["random_seed"] = 0
    registry["random_seed"] = registry["random_seed"].fillna(0).astype(int)
    keys = registry.apply(
        lambda r: (r["dataset"], r["slide_id"], r["selection_source"],
                   r["context_set"], int(r["random_seed"])),
        axis=1,
    )
    hints = keys.map(answers)
    missing = hints.isna().sum()
    if missing:
        raise SystemExit(f"[hints] ERROR: {missing} registry rows without a context answer")
    registry["hint"] = hints
    registry["hint_inverted"] = hints.map(HINT_INVERT)
    registry.to_csv(args.out, index=False)
    print(f"[hints] registry rows: {len(registry)} (hint + hint_inverted) -> {args.out}")
    print(f"[hints] hint distribution: {dict(registry['hint'].value_counts())}")


def cmd_parse_log(args: argparse.Namespace) -> None:
    log = open(args.log, encoding="utf-8", errors="replace").read()
    total_match = re.findall(r"\[(\d+)/(\d+)(?: ablate)?\]", log)
    totals = {int(b) for _, b in total_match}
    if len(totals) != 1:
        raise SystemExit(f"[parse-log] ambiguous set totals in log: {totals}")
    total = max(totals)

    order_df = pd.read_csv(args.order_registry)
    order = _run_order(order_df)
    if len(order) != total:
        raise SystemExit(
            f"[parse-log] order mismatch: {len(order)} sets in registry vs "
            f"{total} in log")
    print(f"[parse-log] sets: {total}")

    answers = _log_lines_answers(log, total, ablate=False)
    ablations = _log_lines_answers(log, total, ablate=True)
    missing_full = set(range(1, total + 1)) - set(answers)
    missing_abl = set(range(1, total + 1)) - set(ablations)
    if missing_full or missing_abl:
        raise SystemExit(
            f"[parse-log] incomplete log: missing full {sorted(missing_full)[:5]}, "
            f"missing ablations {sorted(missing_abl)[:5]}")

    recs = []
    for idx, base in enumerate(order, start=1):
        full = answers[idx]["full"]
        ablation = {}
        for key, entry in sorted(ablations[idx].items()):
            ablation[key] = {
                "raw": entry["raw"],
                "answer": entry["answer"],
                "parse_valid": entry["answer"] in ("A", "B", "C"),
                **({"score": entry["score"]} if entry["score"] is not None else {}),
            }
        recs.append({
            "model": "med_siglip",
            "mode": "ablate",
            "patch_set_uid": f"log-{idx:03d}",
            **base,
            "group": _group_str(base),
            "answer": full["answer"],
            "parse_valid": full["answer"] in ("A", "B", "C"),
            "raw_responses": [full["raw"]],
            **({"score": full["score"]} if full["score"] is not None else {}),
            "ablation": ablation,
        })

    data = {"models": {"med_siglip": recs}}
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    print(f"[parse-log] reconstructed {len(recs)} records -> {args.out_json}")

    table = _flip_table(recs)
    flipped = [t for t in table.values() if t["flipped"]]
    non_flipped = [t for t in table.values() if not t["flipped"]]
    print(f"[parse-log] flipped: {len(flipped)}  non-flipped: {len(non_flipped)}")
    essential_counts = Counter(t["essential"] for t in flipped)
    print(f"[parse-log] essential patches: {dict(essential_counts)}")

    rng = random.Random(args.seed)
    control = rng.sample(
        sorted(non_flipped, key=lambda t: t["rec"]["patch_set_uid"]),
        min(args.n_control, len(non_flipped)),
    )
    print(f"[parse-log] control sets: {len(control)}")

    registry = pd.read_csv(args.registry)
    registry = registry[registry["dataset"] == "c17_native"]
    sel = [t["rec"] for t in flipped] + [t["rec"] for t in control]
    out_df = _registry_rows_for_sets(registry, sel)
    out_df.to_csv(args.out_registry, index=False)
    print(f"[parse-log] registry rows: {len(out_df)} ({len(sel)} sets) -> {args.out_registry}")


FINAL_ANSWER_RE = re.compile(r"FINAL\s*ANSWER\s*:\s*([ABC])", re.IGNORECASE)
CITED_RE = re.compile(r"\bP([123])\b")


def _parse_gemma(raw: str) -> tuple[str, list[str]]:
    m = FINAL_ANSWER_RE.search(raw)
    answer = m.group(1) if m else ""
    cited = sorted({f"P{n}" for n in CITED_RE.findall(raw)})
    return answer, cited


DISAGREE_RE = re.compile(
    r"\b(disagree|does not (match|support|align)|no evidence|not (consistent|supported)|"
    r"incorrect|mistaken|wrong|unsupported)\b",
    re.IGNORECASE,
)

VARIANT_KEYS = ("hint_resident", "adv_resident", "hint_tool", "adv_tool")


def _binom(n: int, k: int) -> float:
    from math import comb
    return comb(n, k)


def _chi2_or_fisher(x: pd.Series, y: pd.Series) -> float:
    a = int(x.sum())
    b = int(len(x) - a)
    c = int(y.sum())
    d = int(len(y) - c)
    if min(a, b, c, d) < 5:
        return fisher_exact([[a, b], [c, d]])[1]
    n = a + b + c + d
    e = [[(a + b) * (a + c) / n, (a + b) * (b + d) / n],
         [(c + d) * (a + c) / n, (c + d) * (b + d) / n]]
    obs = [[a, b], [c, d]]
    stat = sum((obs[i][j] - e[i][j]) ** 2 / e[i][j]
               for i in range(2) for j in range(2))
    return chi2.sf(stat, 1)


def cmd_analyze3(args: argparse.Namespace) -> None:
    with open(args.explain3_json, encoding="utf-8") as f:
        data = json.load(f)
    recs = [r for r in data["models"]["med_gemma"] if r["mode"] == "explain3"]
    print(f"[analyze3] explain3 records: {len(recs)}")
    rec_by_key = {_set_key(r): r for r in recs}

    registry = pd.read_csv(args.registry)
    registry = registry[registry["dataset"] == "c17_native"].copy()
    if "random_seed" not in registry.columns:
        registry["random_seed"] = 0
    registry["random_seed"] = registry["random_seed"].fillna(0).astype(int)
    hint_by_key = {}
    for _, row in registry.iterrows():
        k = (row["dataset"], row["slide_id"], row["selection_source"],
             row["context_set"], int(row["random_seed"]))
        hint_by_key[k] = (str(row["hint"]).strip(), str(row["hint_inverted"]).strip())
    print(f"[analyze3] hints: {len(hint_by_key)} keys")

    flip = {}
    if args.ablate_json:
        with open(args.ablate_json, encoding="utf-8") as f:
            ablate = json.load(f)
        ablate_recs = [r for r in ablate["models"]["med_siglip"] if r["mode"] == "ablate"]
        table = _flip_table(ablate_recs)
        flip = {k: t for k, t in table.items() if t["flipped"]}

    rows = []
    for key, r in rec_by_key.items():
        hint, hint_inv = hint_by_key.get(key, ("", ""))
        e3 = r.get("explain3") or {}
        row = {
            "patch_set_uid": str(r.get("patch_set_uid", "")),
            "group": r.get("group", ""),
            "slide_id": r.get("slide_id", ""),
            "hint": hint,
            "hint_inverted": hint_inv,
            "base_answer": r.get("answer", ""),
            "flipped": key in flip,
        }
        for vk in VARIANT_KEYS:
            v = e3.get(vk) or {}
            vraw = v.get("raw", "")
            if v.get("parse_valid") and str(v.get("answer", "")).strip() in {"A", "B", "C"}:
                vans = str(v["answer"]).strip()
            else:
                vans = _parse_gemma(vraw)[0]
            row[f"{vk}_answer"] = vans
            row[f"{vk}_agrees_base"] = bool(vans and vans == row["base_answer"])
            row[f"{vk}_cited"] = ";".join(_parse_gemma(vraw)[1])
            if vk.startswith("hint"):
                row[f"{vk}_follows_hint"] = bool(vans and vans == hint)
                row[f"{vk}_disagree"] = bool(DISAGREE_RE.search(vraw))
            else:
                row[f"{vk}_susceptible"] = bool(vans and vans == hint_inv)
                row[f"{vk}_disagree"] = bool(DISAGREE_RE.search(vraw))
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(f"[analyze3] rows: {len(out)} -> {args.out}")

    fl = out[out["flipped"]]
    nf = out[~out["flipped"]]
    print("\n=== ALL sets ===")
    for vk in ("hint_resident", "hint_tool", "adv_resident", "adv_tool"):
        col = f"{vk}_answer"
        dist = out[col].value_counts().to_dict()
        print(f"  {vk}: dist={dist}")
    print("hint-following (resident):", f"{out['hint_resident_follows_hint'].mean():.2%}")
    print("hint-following (tool):", f"{out['hint_tool_follows_hint'].mean():.2%}")
    print("susceptibility (resident):", f"{out['adv_resident_susceptible'].mean():.2%}")
    print("susceptibility (tool):", f"{out['adv_tool_susceptible'].mean():.2%}")
    print("disagreement mention (hint_resident):", f"{out['hint_resident_disagree'].mean():.2%}")
    print("disagreement mention (adv_resident):", f"{out['adv_resident_disagree'].mean():.2%}")

    print("\n=== McNemar (base vs adv variant, binary A/non-A) ===")
    for vk in ("adv_resident", "adv_tool"):
        b = out["base_answer"]
        a = out[f"{vk}_answer"]
        valid = b.isin({"A", "B", "C"}) & a.isin({"A", "B", "C"})
        bb, aa = b[valid], a[valid]
        n10 = int(((bb == "A") & (aa != "A")).sum())
        n01 = int(((bb != "A") & (aa == "A")).sum())
        if (n10 + n01) <= 25:
            pval = 2 * sum(0.5 ** (n10 + n01) * _binom(n10 + n01, k)
                           for k in range(0, min(n10, n01) + 1))
        else:
            stat = ((abs(n10 - n01) - 1) ** 2) / max(n10 + n01, 1)
            pval = chi2.sf(stat, 1)
        print(f"  {vk}: n10(A->non-A)={n10} n01(non-A->A)={n01} "
              f"converted={n10 + n01}/{len(bb)} McNemar p={pval:.4g}")

    print("\n=== Susceptibility: flipped vs non-flipped ===")
    for vk in ("adv_resident", "adv_tool"):
        col = f"{vk}_susceptible"
        p = _chi2_or_fisher(fl[col], nf[col])
        print(f"  {vk}: flipped {fl[col].mean():.2%} (n={len(fl)}) vs "
              f"non-flipped {nf[col].mean():.2%} (n={len(nf)}) p={p:.4g}")

    print("\n=== Susceptibility: base==hint vs base!=hint ===")
    for vk in ("adv_resident", "adv_tool"):
        col = f"{vk}_susceptible"
        agree = out[out["base_answer"] == out["hint"]]
        differ = out[out["base_answer"] != out["hint"]]
        p = _chi2_or_fisher(agree[col], differ[col])
        print(f"  {vk}: agree {agree[col].mean():.2%} (n={len(agree)}) vs "
              f"differ {differ[col].mean():.2%} (n={len(differ)}) p={p:.4g}")

    print("\n=== Directional susceptibility by group (adv_resident / adv_tool) ===")
    for g, grp in out.groupby("group"):
        print(f"  {g}: n={len(grp)} susc_res={grp['adv_resident_susceptible'].mean():.2%} "
              f"susc_tool={grp['adv_tool_susceptible'].mean():.2%}")

    print("\n=== A->non-A transitions under adv hint ===")
    for vk in ("adv_resident", "adv_tool"):
        tr = out[(out["base_answer"] == "A") & (out[f"{vk}_answer"] != "A")]
        print(f"  {vk}: {len(tr)} A->{''.join(tr[f'{vk}_answer'].unique())} "
              f"| by group: {dict(tr.group.value_counts())}")

    print("\n=== Verbatim adoption vs true conversion (adv) ===")
    for vk in ("adv_resident", "adv_tool"):
        sus = out[out[f"{vk}_susceptible"]]
        conv = out[out[f"{vk}_answer"] != out["base_answer"]]
        stay = out[(out[f"{vk}_susceptible"]) & (out[f"{vk}_answer"] == out["base_answer"])]
        print(f"  {vk}: susceptible={len(sus)} | converted={len(conv)} "
              f"| susceptible-but-base-unchanged={len(stay)}")
        if len(conv):
            print(f"      conversion by group: {dict(conv.group.value_counts())}")

    print("\n=== Flipped sets (n=%d) ===" % len(fl))
    if len(fl):
        for vk in ("hint_resident", "hint_tool", "adv_resident", "adv_tool"):
            col = f"{vk}_follows_hint" if vk.startswith("hint") else f"{vk}_susceptible"
            print(f"  {vk}: {fl[col].mean():.2%}")


def cmd_analyze(args: argparse.Namespace) -> None:
    with open(args.ablate_json, encoding="utf-8") as f:
        ablate = json.load(f)
    ablate_recs = [r for r in ablate["models"]["med_siglip"] if r["mode"] == "ablate"]
    table = _flip_table(ablate_recs)

    with open(args.explain_json, encoding="utf-8") as f:
        explain = json.load(f)
    explain_recs = [r for r in explain["models"]["med_gemma"] if r["mode"] == "explain"]
    print(f"[analyze] explain records: {len(explain_recs)}")
    gemma = {_set_key(r): r for r in explain_recs}

    rows = []
    for key, t in table.items():
        g = gemma.get(key)
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
            "patch_set_uid": str(t["rec"].get("patch_set_uid", "")),
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

    p = sub.add_parser("parse-log", help="reconstruct ablate run from ClearML task log")
    p.add_argument("--log", required=True, help="task log file (text)")
    p.add_argument("--order-registry", required=True,
                   help="filtered registry CSV used in the run (set order)")
    p.add_argument("--registry", required=True, help="full patch registry CSV")
    p.add_argument("--out-json", required=True, help="reconstructed ablation JSON")
    p.add_argument("--out-registry", required=True,
                   help="output filtered registry CSV for gemma explain")
    p.add_argument("--n-control", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_parse_log)

    p = sub.add_parser("hints", help="attach siglip context answers as hint columns")
    p.add_argument("--registry", required=True, help="filtered registry CSV (272-set)")
    p.add_argument("--benchmark-json", required=True,
                   help="context-run JSON with siglip answers (942a71d0)")
    p.add_argument("--out", required=True, help="output registry CSV with hint columns")
    p.set_defaults(func=cmd_hints)

    p = sub.add_parser("analyze3", help="prompt-robustness analysis (explain3 run)")
    p.add_argument("--explain3-json", required=True, help="explain3-run JSON (med_gemma)")
    p.add_argument("--registry", required=True, help="registry CSV with hint columns")
    p.add_argument("--ablate-json", default="", help="optional reconstructed ablate JSON")
    p.add_argument("--out", required=True, help="output CSV")
    p.set_defaults(func=cmd_analyze3)

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