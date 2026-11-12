#!/usr/bin/env python
"""Reconstruct the two PF00106 full-chain recovery figures as TRUE VECTOR PDFs.

Replaces the raster-wrapped PDFs in results/manuscript/figures/ (which were PNGs in a
PDF container) by re-plotting from the run's on-disk data:

  recovery-to-baseline.pdf  — TAR (B) reversal: recovery fraction R(c) vs attacker FLOPs,
                              and held-out F-PPL vs n, for full-FT / LoRA r32 / LoRA r8.
  tar-vs-cir-recovery.pdf   — TAR vs CIR-peak vs CIR-lowdis on identical fullchain data
                              (LoRA r8): durability is insensitive to the static magnitude.

Metric (log-scale recovery to base competence):
    R(c) = (ln L_B − ln L(c)) / (ln L_B − ln L_A)
with L_A = base-A F-PPL, L_B = safeguard static F-PPL, L(c) = attacked F-PPL at compute c.
Reference values come from results/recovery_references.json (the figure's own source).
Vector output, embedded fonts, sized for a single-column text width.
"""
import glob
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN = f"{ROOT}/runs/PF00106_fullchain"
OUT = f"{ROOT}/results/manuscript/figures"
os.makedirs(OUT, exist_ok=True)

RED, NAVY, TEAL = "#C1272D", "#22406E", "#2AA198"
GREY = "#666666"
plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8,
    "axes.linewidth": 0.7, "lines.linewidth": 1.7, "lines.markersize": 4.5,
    "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
KEY = ["model", "recipe", "lora_rank", "budget_n", "seed", "adversary_data", "target_family"]


def load_sweep(run):
    df = pd.concat([pd.read_csv(f) for f in glob.glob(f"{run}/results/sweep/*.csv")], ignore_index=True)
    return df.drop_duplicates(subset=KEY, keep="last").reset_index(drop=True)


def rec_fraction(ppl, LB, LA):
    return (math.log(LB) - math.log(ppl)) / (math.log(LB) - math.log(LA))


def curve(df, recipe, rank):
    """B, clean, F: per-budget median (ppl, flops); returns sorted-by-n arrays."""
    s = df[(df.model == "B") & (df.adversary_data == "clean") & (df.target_family == "F")
           & (df.recipe == recipe)]
    if recipe == "lora":
        s = s[s.lora_rank == rank]
    g = s.groupby("budget_n").agg(ppl=("best_target_ppl", "median"),
                                  fl=("flops_total", "median")).reset_index().sort_values("budget_n")
    return g


def base_curve(df, recipe, rank):
    s = df[(df.model == "A") & (df.adversary_data == "clean") & (df.target_family == "F")
           & (df.recipe == recipe)]
    if recipe == "lora":
        s = s[s.lora_rank == rank]
    return s.groupby("budget_n")["best_target_ppl"].median().reset_index().sort_values("budget_n")


def r_reflines(ax, LA, refs=None):
    ax.axhline(1.0, color=GREY, lw=0.9)
    ax.axhline(0.95, color=GREY, ls="--", lw=0.9)
    ax.axhline(0.90, color=GREY, ls=":", lw=0.9)
    x0 = ax.get_xlim()[0]
    ax.text(x0, 1.005, " R=1.00 (base)", fontsize=6.3, color=GREY, va="bottom")
    ax.text(x0, 0.955, " R=0.95", fontsize=6.3, color=GREY, va="bottom")
    ax.text(x0, 0.905, " R=0.90", fontsize=6.3, color=GREY, va="bottom")


# --------------------------------------------------------------------------- #
def fig_recovery_to_baseline():
    df = load_sweep(RUN)
    refs = json.load(open(f"{RUN}/results/recovery_references.json"))
    LA, LB = refs["PPL_base_A"], refs["PPL_safeguard_B"]
    series = [("full_ft", None, "full fine-tune", RED, "o"),
              ("lora", 32.0, "LoRA r=32", NAVY, "s"),
              ("lora", 8.0, "LoRA r=8", TEAL, "^")]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.1, 2.95))
    for recipe, rank, lab, c, mk in series:
        g = curve(df, recipe, rank)
        R = [rec_fraction(p, LB, LA) for p in g.ppl]
        axA.plot(g.fl, R, color=c, marker=mk, label=lab)
    axA.set_xscale("log")
    axA.set_xlabel("Attacker compute — FLOPs")
    axA.set_ylabel(r"Recovery fraction  $R(c)=\frac{L_B-L(c)}{L_B-L_A}$")
    axA.set_title("A) Reversal of the safeguard vs attack compute")
    axA.set_ylim(0.4, 1.03)
    r_reflines(axA, LA)
    axA.legend(loc="lower right")

    for recipe, rank, lab, c, mk in series:
        g = curve(df, recipe, rank)
        axB.plot(g.budget_n, g.ppl, color=c, marker=mk, lw=1.7, label=f"B — {lab}")
        b = base_curve(df, recipe, rank)
        if len(b):
            axB.plot(b.budget_n, b.best_target_ppl, color=c, ls="--", lw=1.0, alpha=0.55)
    axB.axhline(LB, color=RED, ls=":", lw=1.0)
    axB.text(axB.get_xlim()[0] * 1.4 if axB.get_xlim()[0] > 0 else 16, LB * 0.62,
             f"static block (no attack) = {LB:,.0f}", fontsize=6.6, color=RED, va="top")
    axB.axhline(LA, color="black", lw=0.8)
    axB.text(70, LA * 1.05, f"base competence {LA:.2f}", fontsize=6.6, color="black", va="bottom")
    axB.set_xscale("log"); axB.set_yscale("log")
    axB.set_xlabel("Attacker examples  n")
    axB.set_ylabel("Held-out F perplexity (best epoch)")
    axB.set_title("B) B (solid) collapses onto base A (dashed)")
    axB.legend(loc="upper right")

    fig.suptitle("PF00106 (full-length) task block — recovery-to-baseline durability",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    p = f"{OUT}/recovery-to-baseline.pdf"
    fig.savefig(p); plt.close(fig)
    print(f"  wrote {os.path.basename(p)}  (L_A={LA:.3f}, L_B={LB:,.0f}, static={refs['static_suppression_x']:,.0f}×)")


# --------------------------------------------------------------------------- #
def fig_tar_vs_cir():
    tar = load_sweep(RUN)
    peak = load_sweep(f"{RUN}/cir_peak")
    lowd = load_sweep(f"{RUN}/cir_lowdis")
    refs = json.load(open(f"{RUN}/results/recovery_references.json"))
    LA = refs["PPL_base_A"]
    LB_tar = refs["PPL_safeguard_B"]
    LB_peak = json.load(open(f"{RUN}/cir_peak/results/summary.json"))["forget_ppl_n0"]["B"]
    LB_low = json.load(open(f"{RUN}/cir_lowdis/results/summary.json"))["forget_ppl_n0"]["B"]
    arms = [("TAR", tar, LB_tar, RED, "o"),
            ("CIR-peak", peak, LB_peak, NAVY, "s"),
            ("CIR-lowdis", lowd, LB_low, TEAL, "^")]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.1, 2.95))
    for name, df, LB, c, mk in arms:
        g = curve(df, "lora", 8.0)
        R = [rec_fraction(p, LB, LA) for p in g.ppl]
        axA.plot(g.fl, R, color=c, marker=mk, label=f"{name}  (static {LB/LA:.0f}×)")
    axA.set_xscale("log")
    axA.set_xlabel("Attacker compute — FLOPs (LoRA r8)")
    axA.set_ylabel(r"Recovery fraction  $R(c)=\frac{L_B-L(c)}{L_B-L_A}$")
    axA.set_title("A) Both safeguards reverse at ~the same attack compute")
    axA.set_ylim(0.47, 1.03)
    axA.axhline(1.0, color=GREY, lw=0.9)
    axA.axhline(0.95, color=GREY, ls="--", lw=0.9)
    axA.axhline(0.90, color=GREY, ls=":", lw=0.9)
    x0 = axA.get_xlim()[0]
    axA.text(x0, 0.955, " R=0.95", fontsize=6.3, color=GREY, va="bottom")
    axA.text(x0, 0.905, " R=0.90", fontsize=6.3, color=GREY, va="bottom")
    axA.legend(loc="lower right")

    for name, df, LB, c, mk in arms:
        g = curve(df, "lora", 8.0)
        axB.plot(g.budget_n, g.ppl, color=c, marker=mk, label=name)
    axB.axhline(LA, color="black", lw=0.8)
    axB.text(70, LA * 1.04, f"base competence {LA:.2f}", fontsize=6.6, color="black", va="bottom")
    axB.set_xscale("log"); axB.set_yscale("log")
    axB.set_xlabel("Attacker examples  n")
    axB.set_ylabel("Held-out F perplexity (best epoch)")
    axB.set_title("B) All three collapse onto base by ~64 examples")
    axB.legend(loc="upper right")

    fig.suptitle("TAR vs CIR on identical fullchain PF00106 data — durability insensitive "
                 "to static magnitude", fontsize=10, y=1.02)
    fig.tight_layout()
    p = f"{OUT}/tar-vs-cir-recovery.pdf"
    fig.savefig(p); plt.close(fig)
    print(f"  wrote {os.path.basename(p)}  (L_B: TAR={LB_tar:,.0f} peak={LB_peak:.0f} lowdis={LB_low:.0f})")


if __name__ == "__main__":
    print("Reconstructing recovery figures as vector PDFs:")
    fig_recovery_to_baseline()
    fig_tar_vs_cir()
    print(f"→ {OUT}")
