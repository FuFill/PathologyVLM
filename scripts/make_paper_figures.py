"""Generate paper figures (Fig 1 pipeline, Fig 3 importance combo, Fig 4 hallucination).

Outputs PDF files into paper/figs/ plus a small JSON of computed numbers.

Sources:
  - Frozen CAMELYON17 VLM run (context mode, med_siglip) for Fig 4.
  - MIL faithfulness tables from prov_gigapath analysis reports (hardcoded from
    results/analysis/h1_diverse_vs_standard.md, h4_c17_binary.md, h5_cross_dataset.md).
  - VLM ablation flip statistics recomputed via scripts/explain_ablation.py
    _flip_table from the reconstructed ablate runs (C17 + C16).
"""

from __future__ import annotations

import collections
import importlib.util
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(ROOT, "paper", "figs")
os.makedirs(FIGS, exist_ok=True)

FROZEN_C17 = r"C:\Users\Matvey\Downloads\c17_vlm_benchmark_942a71d0105d4b51acd8809fd2197bc9.json"
ABLATE_C17 = r"C:\Users\Matvey\AppData\Local\Temp\opencode\ablate_reconstructed.json"
ABLATE_C16 = r"C:\Users\Matvey\AppData\Local\Temp\opencode\ablate_reconstructed_c16.json"

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 150,
})

COL_A = "#b2182b"
COL_B = "#2166ac"
COL_C = "#878787"


def load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fig4_hallucination() -> dict:
    """Answer composition per visual context regime, frozen C17 run."""
    d = json.load(open(FROZEN_C17, encoding="utf-8"))
    recs = d["models"]["med_siglip"]
    agg = collections.defaultdict(lambda: {"A": 0, "B": 0, "C": 0})
    for r in recs:
        g = f"{r['selection_source']}|{r['context_set']}"
        agg[g][r.get("answer")] += 1
    out = {}
    for g, c in sorted(agg.items()):
        n = sum(c.values())
        out[g] = {k: v / n for k, v in c.items()} | {"n": n}

    regimes = ["oracle_tumor", "oracle_non_tumor", "hard_negative", "random", "top_k"]
    variants = ["standard", "diverse"]

    # random: average over seeds within variant
    rnd = collections.defaultdict(lambda: {"A": [], "B": [], "C": []})
    seed_counts = collections.defaultdict(collections.Counter)
    for r in recs:
        if r["selection_source"] != "random":
            continue
        key = (r["context_set"], r.get("random_seed", 0))
        seed_counts[key][r.get("answer")] += 1
    for (ctx, seed), cnt in seed_counts.items():
        n = sum(cnt.values())
        for k in "ABC":
            rnd[ctx][k].append(cnt[k] / n)

    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    xticks, xlabels, pos = [], [], 0.0
    for ri, reg in enumerate(regimes):
        for vi, var in enumerate(variants):
            if reg == "random":
                share = {k: float(np.mean(rnd[var][k])) for k in "ABC"}
                label = f"Random ({var})"
            else:
                g = f"{reg}|{var}"
                c = agg[g]
                n = sum(c.values())
                share = {k: c[k] / n for k in "ABC"}
                label = f"{reg} ({var})"
            bottom = 0.0
            for k, col in (("A", COL_A), ("B", COL_B), ("C", COL_C)):
                v = share[k]
                ax.bar(pos, v, width=0.82, bottom=bottom, color=col,
                       edgecolor="white", linewidth=0.4,
                       label=k if (ri == 0 and vi == 0) else None)
                if v > 0.07:
                    ax.text(pos, bottom + v / 2, k, ha="center", va="center",
                            color="white", fontsize=7, fontweight="bold")
                bottom += v
            xticks.append(pos)
            short = {"oracle_tumor": "Oracle\ntumour", "oracle_non_tumor": "Oracle\nnon-tum.",
                     "hard_negative": "Hard\nneg.", "random": "Random", "top_k": "Top-k"}[reg]
            suffix = "" if len(variants) == 1 else ("\ns" if var == "standard" else "\nd")
            xlabels.append(short + suffix)
            pos += 1.0
        pos += 0.55
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)
    ax.set_ylabel("Share of answers")
    ax.set_ylim(0, 1.02)
    ax.spines[["top", "right"]].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (COL_A, COL_B, COL_C)]
    ax.legend(handles, ["A - tumour present", "B - no tumour", "C - insufficient evidence"],
              loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=3, frameon=False)
    fig.tight_layout()
    p = os.path.join(FIGS, "fig_hallucination.pdf")
    fig.savefig(p, bbox_inches="tight")
    fig.savefig(p.replace(".pdf", ".png"), bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"[fig4] -> {p}")
    return out


MIL_FAITHFULNESS_STD = {
    # model: [keep-drop at k=1, k=3, k=5]  (prov_gigapath H5 Table 5.3, std mode)
    "C16 native":   [-0.386, -0.352, -0.326],
    "C17->C16":     [-0.125, -0.082, -0.071],
    "C17 native":   [-0.081, -0.059, -0.050],
    "C16->C17":     [+0.305, +0.292, +0.285],
}
REMOVE_DROP_NATIVE = {
    # remove-drop std (H1 T1.3 C16 native; H4 T4.1 C17 native)
    "C16 native":   [0.012, 0.035, 0.052],
    "C17 native":   [0.028, 0.052, 0.062],
}


def flip_stats(path: str) -> dict:
    ea = load_module("ea_" + os.path.basename(path)[:6], os.path.join(ROOT, "scripts", "explain_ablation.py"))
    d = json.load(open(path, encoding="utf-8"))
    recs = d["models"]["med_siglip"] if "models" in d else d
    if isinstance(recs, dict):
        recs = list(recs.values())
    table = ea._flip_table(recs)
    grp = collections.defaultdict(lambda: [0, 0])
    ess_singles = 0
    for t in table.values():
        g = t["rec"]["selection_source"]
        grp[g][1] += 1
        if t["flipped"]:
            grp[g][0] += 1
            if "|" not in t["essential"]:
                ess_singles += 1
    flipped = sum(v[0] for v in grp.values())
    return {
        "groups": {g: {"flipped": f, "n": n} for g, (f, n) in grp.items()},
        "total_n": len(table),
        "total_flipped": flipped,
        "essential_single": ess_singles,
    }


def fig3_importance() -> dict:
    c17 = flip_stats(ABLATE_C17)
    c16 = flip_stats(ABLATE_C16)

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.2, 2.9))

    # Panel A: MIL faithfulness
    ks = [1, 3, 5]
    styles = {
        "C16 native": ("#762a83", "o", "-"),
        "C17->C16":   ("#1b7837", "s", "--"),
        "C17 native": ("#2166ac", "^", "-"),
        "C16->C17":   ("#b2182b", "D", ":"),
    }
    for model, vals in MIL_FAITHFULNESS_STD.items():
        col, mk, ls = styles[model]
        axa.plot(ks, vals, marker=mk, linestyle=ls, color=col, label=model,
                 markersize=4.5, linewidth=1.4)
    axa.axhline(0, color="black", linewidth=0.8)
    axa.set_xticks(ks)
    axa.set_xlabel("$k$ (selected patches)")
    axa.set_ylabel("Keep-drop ($p_{orig}-p_{keep}$)")
    axa.set_title("A  MIL faithfulness (std top-$k$)", loc="left")
    axa.legend(frameon=False, fontsize=7, loc="lower right")
    axa.spines[["top", "right"]].set_visible(False)

    # Panel B: VLM ablation flip rates by selection group
    groups = ["top_k", "hard_negative", "oracle_non_tumor", "oracle_tumor"]
    labels = ["Top-k", "Hard neg.", "Oracle\nnon-tumour", "Oracle\ntumour"]
    w = 0.38
    xs = np.arange(len(groups))
    c17r = [100 * c17["groups"][g]["flipped"] / c17["groups"][g]["n"] for g in groups]
    c16r = [100 * c16["groups"][g]["flipped"] / c16["groups"][g]["n"] for g in groups]
    b1 = axb.bar(xs - w / 2, c17r, width=w, color="#762a83", label="C17 (70/272)")
    b2 = axb.bar(xs + w / 2, c16r, width=w, color="#e08214", label="C16 (56/210)")
    for bars in (b1, b2):
        for rect in bars:
            h = rect.get_height()
            axb.text(rect.get_x() + rect.get_width() / 2, h + 1.0, f"{h:.0f}",
                     ha="center", va="bottom", fontsize=7)
    axb.set_xticks(xs)
    axb.set_xticklabels(labels, fontsize=7.5)
    axb.set_ylabel("Verdict flips under patch removal (%)")
    axb.set_ylim(0, 62)
    axb.set_title("B  VLM answer stability (single patch removed)", loc="left")
    axb.legend(frameon=False, fontsize=7, title=None)
    axb.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    p = os.path.join(FIGS, "fig_importance.pdf")
    fig.savefig(p, bbox_inches="tight")
    fig.savefig(p.replace(".pdf", ".png"), bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"[fig3] -> {p}")
    stats = {"c17": c17, "c16": c16}
    sp = os.path.join(FIGS, "flip_stats.json")
    json.dump(stats, open(sp, "w"), indent=1)
    print(f"[fig3] stats -> {sp}")
    return stats


def fig1_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 2.4))
    ax.axis("off")

    def box(x, y, w, h, text, fc, fontsize=8.5):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor="#333333",
                                   linewidth=0.9, zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize, zorder=3)

    def arrow(x0, x1, y):
        ax.annotate("", xy=(x1, y), xytext=(x0, y),
                    arrowprops=dict(arrowstyle="-|>", color="#333333", linewidth=1.2))

    y0, h0 = 0.45, 0.34
    box(0.01, y0, 0.115, h0, "WSI\n(CAMELYON\n16 / 17)", "#f4f4f4")
    box(0.165, y0, 0.135, h0, "GigaPath tile\nencoder\n(ViT embeddings)", "#dbe9f6")
    box(0.34, y0, 0.135, h0, "ABMIL\nattention MIL\n(slide-level label)", "#dbe9f6")
    box(0.515, y0, 0.13, h0, "Top-$k$ patches\n(evidence)", "#fde0dd")
    box(0.685, y0, 0.145, h0, "VLM\n(MedSigLIP /\nMedGemma)", "#e5f5e0")
    box(0.87, y0, 0.12, h0, "Verdict A/B/C\n+ explanation", "#ffffcc")
    for xa, xb in ((0.125, 0.165), (0.30, 0.34), (0.475, 0.515), (0.645, 0.685), (0.83, 0.87)):
        arrow(xa, xb, y0 + h0 / 2)

    y1 = 0.08
    box(0.24, y1, 0.235, 0.20, "Mask localization\n(hit rate, overlap)", "#fff7ec", fontsize=7.5)
    box(0.50, y1, 0.19, 0.20, "Patch ablation\n(verdict flips)", "#fff7ec", fontsize=7.5)
    box(0.715, y1, 0.27, 0.20, "Prompt robustness\n(truthful vs adversarial hints)", "#fff7ec", fontsize=7.5)
    ax.annotate("", xy=(0.357, y1 + 0.20), xytext=(0.58, y0),
                arrowprops=dict(arrowstyle="-", color="#999999", linewidth=0.8,
                                connectionstyle="arc3,rad=0.25"))
    ax.annotate("", xy=(0.595, y1 + 0.20), xytext=(0.60, y0),
                arrowprops=dict(arrowstyle="-", color="#999999", linewidth=0.8,
                                connectionstyle="arc3,rad=-0.15"))
    ax.annotate("", xy=(0.85, y1 + 0.20), xytext=(0.72, y0),
                arrowprops=dict(arrowstyle="-", color="#999999", linewidth=0.8,
                                connectionstyle="arc3,rad=-0.25"))

    fig.tight_layout()
    p = os.path.join(FIGS, "fig_pipeline.pdf")
    fig.savefig(p, bbox_inches="tight")
    fig.savefig(p.replace(".pdf", ".png"), bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"[fig1] -> {p}")


def main() -> None:
    which = sys.argv[1:] or ["pipeline", "hallucination", "importance"]
    if "pipeline" in which:
        fig1_pipeline()
    if "hallucination" in which:
        s = fig4_hallucination()
        json.dump(s, open(os.path.join(FIGS, "hallucination_shares.json"), "w"), indent=1)
    if "importance" in which:
        fig3_importance()


if __name__ == "__main__":
    main()
