#!/usr/bin/env python
"""LaTeX-optimised PDF versions of the MGnify manuscript figures.

Re-authors every figure that has a committed generator (07_manuscript.py,
08_weightspace.py, plot_cir_mgnify_figs.py, 10b_sibling_ladder_analyze.py) at
manuscript scale: vector PDF, embedded TrueType fonts (pdf.fonttype=42), consistent
9/8/7.5 pt type, tight margins, and figure widths chosen for a single-column
text width (~5.5 in). Reads the run's on-disk artefacts ONLY (no model reload / no
sweep), all via ABSOLUTE paths so it is immune to cwd changes from env_setup.sh.

    python scripts/make_manuscript_pdfs.py      # writes *.pdf into the run's manuscript/

NOT regenerated: fig_weightspace_l2_cir.pdf (no committed generator).
Recommended \\includegraphics widths are printed per figure at the end.
"""
import glob
import json
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Run root holds a results/ dir. Defaults to the MGnify run; pass a different root as
# argv[1] (e.g. ROOT itself for the top-level PF00106 run), and an optional
# comma-separated figure list as argv[2] (default: all figures with data).
RUN = f"{ROOT}/runs/MGYF18690_metagenomic"
RES = f"{RUN}/results"
OUT = f"{RES}/manuscript"
sys.path.insert(0, ROOT)

# ---- LaTeX-friendly global style ----------------------------------------- #
# Keep the default DejaVu Sans (covers θ ‖ α τ → × ₀); only tune sizes/weights.
plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,          # embed editable TrueType, not Type-3
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "axes.linewidth": 0.7, "lines.linewidth": 1.6, "lines.markersize": 4.5,
    "legend.frameon": False, "figure.dpi": 150,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "axes.titlepad": 4.0,
})

TEXTW = 5.5   # single-column text width (in)
COL = 3.35    # a "half"/column figure (place at ~0.5\textwidth or \linewidth in a subfig)
STYLE = {"A": ("tab:gray", "o", "A (base)"),
         "A_prime": ("tab:blue", "s", "A′ (control)"),
         "B": ("tab:red", "^", "B (task-blocked)")}
_widths = {}


def _save(fig, name, rec_width):
    p = f"{OUT}/{name}.pdf"
    fig.savefig(p)
    plt.close(fig)
    _widths[name] = rec_width
    print(f"  wrote {name}.pdf")


# --------------------------------------------------------------------------- #
# Data loaders (absolute paths)
# --------------------------------------------------------------------------- #
def load_sweep():
    KEY = ["model", "recipe", "lora_rank", "budget_n", "seed", "adversary_data", "target_family"]
    df = pd.concat([pd.read_csv(f) for f in glob.glob(f"{RES}/sweep/sweep_shard*_of*.csv")],
                   ignore_index=True)
    return df.drop_duplicates(subset=KEY, keep="last").reset_index(drop=True)


def load_json(p):
    return json.load(open(p))


# --------------------------------------------------------------------------- #
# 1. Acquisition curve (sweep; lora r8, clean, F)
# --------------------------------------------------------------------------- #
def fig_acquisition(df, tau):
    sub = df[(df.recipe == "lora") & (df.lora_rank == 8) &
             (df.adversary_data == "clean") & (df.target_family == "F")]
    fig, ax = plt.subplots(figsize=(COL, 2.7))
    for model, (c, mk, lab) in STYLE.items():
        m = sub[sub.model == model]
        if not len(m):
            continue
        b = sorted(m.budget_n.unique())
        med = [m[m.budget_n == x]["best_target_ppl"].median() for x in b]
        ax.plot(b, med, color=c, marker=mk, label=lab)
    ax.axhline(tau, color="black", ls="--", lw=1, label=f"τ = {tau:.1f}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("attacker budget $n$ (examples)")
    ax.set_ylabel("held-out F pseudo-PPL")
    ax.legend(loc="upper right", handlelength=1.6)
    _save(fig, "fig_acquisition", "\\linewidth (or 0.5\\textwidth as a subfigure)")


# --------------------------------------------------------------------------- #
# 2. Static block vs durability (summary n=0 + multipliers)
# --------------------------------------------------------------------------- #
def fig_static_vs_durable(mult, summ):
    fn0 = summ["forget_ppl_n0"]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(TEXTW, 2.55))
    # left: static n=0 forget PPL
    names, cols = ["A", "A_prime", "B"], ["tab:gray", "tab:blue", "tab:red"]
    vals = [fn0[n] for n in names]
    axL.bar(range(3), vals, color=cols, width=0.7)
    axL.set_yscale("log"); axL.set_xticks(range(3)); axL.set_xticklabels(["A", "A′", "B"])
    axL.set_ylabel("$n{=}0$ forget PPL on F")
    axL.set_title("Static: B unlearns F (%.0f× base)" % summ["n0_forget_ratio_B_over_A"])
    for i, v in enumerate(vals):
        axL.text(i, v, f"{v:.0f}" if v > 100 else f"{v:.1f}", ha="center", va="bottom", fontsize=7.5)
    axL.set_ylim(top=vals[2] * 6)
    # right: multiplier B/A' by recipe, clean vs homolog
    recs = [("full_ft", None, "full-FT"), ("lora", 8.0, "LoRA r8"), ("lora", 32.0, "LoRA r32")]
    x, w = range(len(recs)), 0.38

    def mv(recipe, rank, adv):
        q = mult[(mult.recipe == recipe) & (mult.adversary_data == adv) & (mult.target_family == "F")]
        if recipe == "lora":
            q = q[q.lora_rank == rank]
        return q["mult_B_over_Aprime"].iloc[0] if len(q) else float("nan")

    axR.bar([i - w / 2 for i in x], [mv(r, k, "clean") for r, k, _ in recs], w, label="clean", color="tab:red")
    axR.bar([i + w / 2 for i in x], [mv(r, k, "homolog_augmented") for r, k, _ in recs], w,
            label="homolog", color="darkorange")
    axR.axhline(summ.get("headline_min", 10), color="black", ls="--", lw=1, label="headline 10×")
    axR.axhline(summ.get("durability_min", 3), color="gray", ls=":", lw=1, label="durability 3×")
    axR.set_xticks(list(x)); axR.set_xticklabels([l for _, _, l in recs])
    axR.set_ylabel("multiplier  B / A′")
    axR.set_title("Durability: attacker re-learns (~1×)")
    axR.legend(loc="upper left", ncol=1)
    _save(fig, "fig_static_vs_durable", "\\textwidth")


# --------------------------------------------------------------------------- #
# 3. Retain-only relearning (sweep)
# --------------------------------------------------------------------------- #
def fig_retain_only(df, tau, summ):
    a_floor = summ["forget_ppl_n0"]["A"]

    def sel(adv, model, recipe="full_ft"):
        return df[(df.recipe == recipe) & (df.adversary_data == adv) &
                  (df.target_family == "F") & (df.model == model)]

    def curve(s):
        b = sorted(s.budget_n.unique())
        return b, [s[s.budget_n == x]["best_target_ppl"].median() for x in b]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(TEXTW, 2.7))
    for model, (c, mk, lab) in STYLE.items():
        m = sel("retain_only", model)
        if not len(m):
            continue
        b, med = curve(m)
        axL.plot(b, med, color=c, marker=mk, label=lab)
    axL.axhline(tau, color="black", ls="--", lw=1, label=f"τ = {tau:.1f}")
    axL.set_xscale("log"); axL.set_yscale("log")
    axL.set_xlabel("retain-set examples $n$ (zero F)")
    axL.set_ylabel("held-out F pseudo-PPL")
    axL.set_title("Retain-only: F stays dormant")
    axL.legend(loc="best", handlelength=1.6)
    for adv, (c, mk, lab) in {"clean": ("tab:red", "^", "direct-F"),
                              "retain_only": ("tab:green", "o", "retain-only")}.items():
        m = sel(adv, "B")
        if not len(m):
            continue
        b, med = curve(m)
        axR.plot(b, med, color=c, marker=mk, label=lab)
    axR.axhline(tau, color="black", ls="--", lw=1, label=f"τ = {tau:.1f}")
    axR.axhline(a_floor, color="tab:gray", ls=":", lw=1, label=f"A floor = {a_floor:.1f}")
    axR.set_xscale("log"); axR.set_yscale("log")
    axR.set_xlabel("attacker budget $n$")
    axR.set_title("B: needs F-specific data")
    axR.legend(loc="best", handlelength=1.6)
    _save(fig, "fig_retain_only", "\\textwidth")


# --------------------------------------------------------------------------- #
# 4. Per-layer weight movement (weightspace.json)
# --------------------------------------------------------------------------- #
def fig_weightspace_l2(ws):
    l2B, l2A = ws["l2"]["B"], ws["l2"].get("A_prime")

    def order(g):
        return (0, int(g.split("_")[1])) if g.startswith("layer_") else (1, g)

    groups = sorted([g for g in l2B if g != "__total__"], key=order)
    relB = [l2B[g]["rel"] for g in groups]
    fig, ax = plt.subplots(figsize=(TEXTW, 2.6))
    x = range(len(groups))
    if l2A:
        relA = [l2A.get(g, {}).get("rel", float("nan")) for g in groups]
        ax.bar([i - 0.2 for i in x], relB, 0.4, color="tab:red", label="B (task-blocked)")
        ax.bar([i + 0.2 for i in x], relA, 0.4, color="tab:blue", label="A′ (control)")
    else:
        ax.bar(list(x), relB, 0.6, color="tab:red", label="B (task-blocked)")
    ax.set_xticks(list(x))
    ax.set_xticklabels([g.replace("layer_", "L").replace("_", " ") for g in groups],
                       rotation=55, ha="right")
    ax.set_ylabel("‖θ − θ₀‖ / ‖θ₀‖")
    bt, at = l2B["__total__"]["rel"], (l2A["__total__"]["rel"] if l2A else float("nan"))
    ax.set_title(f"Per-layer weight movement (total B/A′ = {bt/at:.2f}×)")
    ax.legend(loc="upper right")
    _save(fig, "fig_weightspace_l2", "\\textwidth")


# --------------------------------------------------------------------------- #
# 5. LMC barrier (weightspace.json)
# --------------------------------------------------------------------------- #
def fig_lmc_barrier(ws):
    lmc = ws["lmc_B"]
    a = lmc["alphas"]
    fig, ax = plt.subplots(figsize=(COL, 2.7))
    ax.plot(a, lmc["f_ppl"], color="tab:red", marker="^", label="F-test PPL")
    if "benign_ppl" in lmc:
        ax.plot(a, lmc["benign_ppl"], color="tab:blue", marker="o", label="benign PPL")
    ax.set_yscale("log")
    ax.set_xlabel("interpolation α  (θ₀ → θ_B)")
    ax.set_ylabel("pseudo-PPL (log)")
    mono = lmc.get("f_monotonic", False)
    flat = lmc.get("benign_barrier_nats", 1.0) < 0.5
    tag = "no barrier" if (mono and flat) else f"barrier {lmc.get('f_barrier_nats', 0):.2f} nats"
    ax.set_title(f"LMC θ₀→θ_B: {tag}")
    ax.legend(loc="upper left", handlelength=1.6)
    _save(fig, "fig_lmc_barrier", "\\linewidth (or 0.5\\textwidth)")


# --------------------------------------------------------------------------- #
# 6. CIR efficacy vs disruption frontier (cir_frontier/knob B_compute.json)
# --------------------------------------------------------------------------- #
def _cir_cells(pattern):
    rows = []
    for p in sorted(glob.glob(pattern)):
        d = load_json(p)
        fx, dr = d.get("forget_ppl_ratio_B_over_base"), d.get("benign_drift_rel")
        if fx is not None and dr is not None:
            rows.append((dr * 100.0, fx))
    return rows


def fig_cir_frontier():
    frontier = _cir_cells(f"{RES}/cir_frontier/*/B_compute.json")
    knob = _cir_cells(f"{RES}/cir_knob/*/B_compute.json")
    fig, ax = plt.subplots(figsize=(TEXTW, 3.0))
    for rows, c, mk, lab in [(frontier, "tab:blue", "o", "frontier (rate × termination)"),
                             (knob, "tab:orange", "s", "knob (mode × n_pcs × layers)")]:
        if rows:
            ax.scatter([r[0] for r in rows], [r[1] for r in rows], c=c, marker=mk, s=32,
                       edgecolor="k", linewidth=0.4, label=lab, zorder=3)
    ax.axvline(2.0, color="green", ls="--", lw=1.1, label="utility gate (2%)")
    ax.axhline(2.0, color="gray", ls=":", lw=1.1, label='"unlearned" (2×)')
    # Annotated values are computed from the run's files, never typed in.
    cells = frontier + knob
    ceiling = max((r[1] for r in cells), default=None)   # None when the run has no CIR cells
    inside = [r for r in cells if r[0] <= 2.0]            # inside the 2% utility gate
    if inside:
        bx, by = max(inside, key=lambda r: r[1])
        ax.annotate(f"best util-preserving\n{by:.2f}× @ {bx:.2f}%", xy=(bx, by),
                    xytext=(bx + 0.43, by + 0.30), fontsize=7, color="darkred",
                    arrowprops=dict(arrowstyle="->", color="darkred", lw=0.8))
    summary_path = f"{RES}/summary.json"
    mlac = load_json(summary_path).get("n0_forget_ratio_B_over_A") if os.path.exists(summary_path) else None
    if mlac is not None:
        ax.text(5.3, 2.95, f"mlac_fo: {mlac:,.0f}× (off-scale ↑)", fontsize=7.5, color="tab:red",
                ha="right", va="top")
    ax.set_xlabel("benign utility drift (%)  →  more disruption")
    ax.set_ylabel("static forget  (B / base ×)")
    ax.set_title("CIR efficacy vs disruption"
                 + (f": ceiling ≈ {ceiling:.1f}× (no durable block)" if ceiling is not None else ""))
    ax.set_ylim(0.9, 3.0); ax.set_xlim(0.3, 5.4)
    ax.legend(fontsize=7, loc="lower right")
    _save(fig, "fig_cir_frontier", "\\textwidth")


# --------------------------------------------------------------------------- #
# 7. O(F,R) subspace overlap (cir_overlap.json)
# --------------------------------------------------------------------------- #
def fig_cir_overlap():
    d = load_json(f"{RES}/cir_overlap.json")
    pfam_path = f"{ROOT}/results/cir_overlap.json"
    pfam = load_json(pfam_path)["aggregate"]["removable_topk"] if os.path.exists(pfam_path) else None
    layers = sorted(d["per_layer"], key=lambda x: int(x))
    rem = [d["per_layer"][L]["removable_topk"] for L in layers]
    ov = [d["per_layer"][L]["O_topk"] for L in layers]
    agg = d["aggregate"]["removable_topk"]
    fig, ax = plt.subplots(figsize=(TEXTW, 2.9))
    x = range(len(layers))
    ax.bar([i - 0.2 for i in x], rem, 0.4, color="tab:green", label="removable (F-distinctive)")
    ax.bar([i + 0.2 for i in x], ov, 0.4, color="tab:gray", label="O$_{topk}$ (F inside R)")
    if pfam is not None:
        ax.axhline(pfam, color="tab:red", ls="--", lw=1.2, label=f"Pfam removable = {pfam:.2f}")
    ax.set_xticks(list(x)); ax.set_xticklabels([f"L{int(L):02d}" for L in layers])
    ax.set_ylabel("fraction of F's top-24 variance")
    ax.set_ylim(0, 1.0)
    ax.set_title(f"O(F,R): a distinctive subspace exists (removable = {agg:.2f})")
    ax.legend(loc="upper right", ncol=2)
    _save(fig, "fig_cir_overlap", "\\textwidth")


# --------------------------------------------------------------------------- #
# 8/9. Sibling ladder — re-authored at manuscript width (logic ported from
#      src/analysis/plots.py so fonts match the rest of the figure set)
# --------------------------------------------------------------------------- #
def _load_ladder():
    files = glob.glob(f"{RES}/sibling_ladder/sweep_shard*_of*.csv")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    key = ["model", "recipe", "lora_rank", "budget_n", "seed", "rung_acc"]
    return df.drop_duplicates(subset=key, keep="last").reset_index(drop=True)


def fig_sibling_ladder(df, tau, floor):
    """F-PPL vs distance to F, two stacked panels (identity, then cross-HMM bits)."""
    naive_B = float(df[df.model == "B"]["naive_target_ppl"].median())
    budget = int(max(df.budget_n.unique()))
    d = df[df.budget_n == budget]
    style = {("B", "lora"): ("tab:red", "^", "-"), ("B", "full_ft"): ("tab:red", "^", "--"),
             ("A_prime", "lora"): ("tab:blue", "s", "-"), ("A_prime", "full_ft"): ("tab:blue", "s", "--")}
    fig, axes = plt.subplots(2, 1, figsize=(3.6, 4.9), sharey=True)
    for ax, (xcol, xlabel, xlog, invert) in zip(axes, [
            ("mean_id_all", "mean identity to F (→ distance)", False, True),
            ("hmm_bits_mean", "cross-HMM bitscore to F (log)", True, False)]):
        for (model, recipe), sub in d.groupby(["model", "recipe"]):
            g = sub.groupby(xcol)["best_target_ppl"].median().sort_index()
            c, mk, ls = style.get((model, recipe), ("tab:gray", "o", "-"))
            ax.plot(g.index, g.values, color=c, marker=mk, linestyle=ls, label=f"{model}/{recipe}")
        ax.axhline(tau, color="black", ls=":", lw=1, label=f"τ={tau:.1f}")
        ax.axhline(naive_B, color="tab:red", ls=":", lw=1, alpha=0.6, label=f"B naive={naive_B:.0f}")
        ax.set_yscale("log")
        if xlog:
            ax.set_xscale("symlog")
        if invert:
            ax.invert_xaxis()
        ax.set_xlabel(xlabel); ax.set_ylabel("F-PPL (log)")
    axes[0].set_title(f"Sibling reawakening (n={budget})", fontsize=9)
    axes[0].legend(fontsize=6, ncol=2, handlelength=1.4)
    _save(fig, "fig_sibling_ladder", "0.55\\textwidth (tall) or as a subfigure")


def fig_sibling_ladder_budget(df):
    """Reawakening of F (%) as a rung × budget heatmap, one panel per recipe."""
    naive_B = float(df[df.model == "B"]["naive_target_ppl"].median())
    aprime_level = float(df[df.model == "A_prime"]["best_target_ppl"].median())
    d = df[df.model == "B"]
    has_band = "band" in d.columns
    cols = ["rung_acc", "rung_id", "mean_id_all", "hmm_bits_mean"] + (["band"] if has_band else [])
    meta = d[cols].drop_duplicates().copy()
    band_rank = {"close": 0, "mid": 1, "background": 2, "far": 2}
    meta["_br"] = (meta["band"].map(lambda b: band_rank.get(b, 1)) if has_band
                   else (meta["hmm_bits_mean"] <= 0).astype(int) * 2)
    meta = meta.sort_values(["_br", "mean_id_all", "hmm_bits_mean"], ascending=[True, False, False])
    rungs = list(meta["rung_acc"])

    def ylab(r):
        bg = (getattr(r, "band", "") in ("background", "far")) if has_band else (r.hmm_bits_mean <= 0)
        return f"{r.rung_id}\nbackgnd" if bg else f"{r.rung_id}\n{r.mean_id_all * 100:.0f}% id"
    ylabels = [ylab(r) for r in meta.itertuples()]
    n_detect = int((meta["_br"] < 2).sum())
    budgets = sorted(d["budget_n"].unique())
    denom = math.log(naive_B) - math.log(aprime_level)

    def rec(p):
        return max(0.0, min(1.0, (math.log(naive_B) - math.log(max(p, aprime_level))) / denom)) * 100

    recipes = ("lora", "full_ft")
    fig, axes = plt.subplots(1, 2, figsize=(TEXTW, 2.9), squeeze=False)
    im = None
    for k, (ax, recipe) in enumerate(zip(axes[0], recipes)):
        sub = d[d["recipe"] == recipe]
        import numpy as np
        M = np.full((len(rungs), len(budgets)), np.nan)
        for i, acc in enumerate(rungs):
            g = sub[sub["rung_acc"] == acc].groupby("budget_n")["best_target_ppl"].median()
            for j, b in enumerate(budgets):
                if b in g.index:
                    M[i, j] = rec(g[b])
        im = ax.imshow(M, cmap="Reds", vmin=0, vmax=100, aspect="auto")
        ax.set_xticks(range(len(budgets))); ax.set_xticklabels(budgets, fontsize=7)
        ax.set_yticks(range(len(rungs)))
        ax.set_yticklabels(ylabels if k == 0 else [""] * len(rungs), fontsize=6.5)
        ax.set_xlabel("attacker budget n", fontsize=8)
        ax.set_title(f"B / {recipe}", fontsize=8.5)
        for i in range(len(rungs)):
            for j in range(len(budgets)):
                if not math.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center",
                            color="white" if M[i, j] > 55 else "black", fontsize=6.5)
        if 0 < n_detect < len(rungs):
            ax.axhline(n_detect - 0.5, color="black", lw=1.4)
    fig.suptitle("Reawakening of F (%): sibling distance × budget (no F in training)", fontsize=8.5)
    if im is not None:
        fig.colorbar(im, ax=axes[0].tolist(), fraction=0.03, pad=0.02, label="recovery (%)")
    _save(fig, "fig_sibling_ladder_budget", "\\textwidth")


# --------------------------------------------------------------------------- #
FIGS_ALL = ["acquisition", "static_vs_durable", "retain_only", "weightspace_l2", "lmc_barrier",
            "cir_frontier", "cir_overlap", "sibling_ladder", "sibling_ladder_budget"]


def main():
    global RUN, RES, OUT
    if len(sys.argv) > 1:
        RUN = os.path.abspath(sys.argv[1]); RES = f"{RUN}/results"; OUT = f"{RES}/manuscript"
    os.makedirs(OUT, exist_ok=True)
    want = (sys.argv[2].split(",") if len(sys.argv) > 2 and sys.argv[2] not in ("all", "") else FIGS_ALL)
    print(f"Reading {RES}\nFigures: {', '.join(want)}")

    thr = load_json(f"{RES}/thresholds.json")
    tau, floor = thr["tau"], thr.get("ppl_floor")
    df = load_sweep() if ({"acquisition", "retain_only"} & set(want)) else None
    ws = load_json(f"{RES}/weightspace.json") if ({"weightspace_l2", "lmc_barrier"} & set(want)) else None

    print("Writing LaTeX-optimised PDFs:")
    if "acquisition" in want:
        fig_acquisition(df, tau)
    if "static_vs_durable" in want:
        fig_static_vs_durable(pd.read_csv(f"{RES}/multipliers.csv"), load_json(f"{RES}/summary.json"))
    if "retain_only" in want:
        fig_retain_only(df, tau, load_json(f"{RES}/summary.json"))
    if "weightspace_l2" in want:
        fig_weightspace_l2(ws)
    if "lmc_barrier" in want:
        fig_lmc_barrier(ws)
    if "cir_frontier" in want:
        fig_cir_frontier()
    if "cir_overlap" in want:
        fig_cir_overlap()
    if {"sibling_ladder", "sibling_ladder_budget"} & set(want):
        lad = _load_ladder()
        if "sibling_ladder" in want:
            fig_sibling_ladder(lad, tau, floor)
        if "sibling_ladder_budget" in want:
            fig_sibling_ladder_budget(lad)

    print("\nRecommended \\includegraphics widths:")
    for name, w in _widths.items():
        print(f"  {name}.pdf   ->  {w}")


if __name__ == "__main__":
    main()
