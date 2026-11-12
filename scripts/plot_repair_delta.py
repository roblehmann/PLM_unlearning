"""Standalone figure for the Phase-8d repair-delta analysis (reads cir_repair.json, no GPU).

Two panels tell the story:
  (left)  absolute ‖ΔW‖ by subspace — CIR's edit is large; the attacker's repair is ~9×
          smaller and nearly orthogonal to it (cos≈0, reversal≈0): a bypass, not a rewind.
  (right) each delta's F-distinctive vs R-shared L2 share — CIR's edit is F-distinctive-
          heavy (0.43 vs 0.15); the repair is balanced (0.33 vs 0.31), i.e. it leans ~2.6×
          more on the shared directions the safeguard had to spare.

    python scripts/plot_repair_delta.py [op=cir35m_peak]
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
op = "cir35m_peak"
for a in sys.argv[1:]:
    if a.startswith("op="):
        op = a.split("=", 1)[1]

rec = json.load(open(os.path.join(ROOT, "results", op, "cir_repair.json")))
C = rec["cir_update"]["fvsr"]["__total__"]
X = rec["repair_delta"]["fvsr"]["__total__"]
al = rec["alignment_vs_cir_update"]["targeted_dense"]
atk = rec["attack"]

C_CIR, C_ATK = "#0072B2", "#D55E00"          # CVD-safe (Okabe-Ito): edit vs repair
cats = ["F-distinctive", "R-shared", "rest"]
cir_abs = [C["f_distinctive_l2"], C["r_shared_l2"], C["rest_l2"]]
atk_abs = [X["f_distinctive_l2"], X["r_shared_l2"], X["rest_l2"]]

fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.6, 4.5))
w = 0.38
xs = range(len(cats))

# ---- left: absolute L2 by subspace ----
for i, (cv, av) in enumerate(zip(cir_abs, atk_abs)):
    axL.bar(i - w / 2, cv, w, color=C_CIR, zorder=3)
    axL.bar(i + w / 2, av, w, color=C_ATK, zorder=3)
    axL.text(i - w / 2, cv + 0.4, f"{cv:.1f}", ha="center", va="bottom", fontsize=8, color="#333")
    axL.text(i + w / 2, av + 0.4, f"{av:.1f}", ha="center", va="bottom", fontsize=8, color="#333")
axL.set_xticks(list(xs))
axL.set_xticklabels(cats)
axL.set_ylabel("‖ΔW‖ over targeted layers  (Frobenius)")
axL.set_title("Where the weight movement lives\n(repair is ~9× smaller)", fontsize=10)
axL.grid(True, axis="y", lw=0.4, color="#e6e6e6", zorder=0)
for s in ("top", "right"):
    axL.spines[s].set_visible(False)

# ---- right: F-distinctive vs R-shared share of each delta's total ----
cir_share = [C["f_distinctive_l2"] / C["total_l2"], C["r_shared_l2"] / C["total_l2"]]
atk_share = [X["f_distinctive_l2"] / X["total_l2"], X["r_shared_l2"] / X["total_l2"]]
sc = ["F-distinctive", "R-shared"]
for i, (cv, av) in enumerate(zip(cir_share, atk_share)):
    axR.bar(i - w / 2, cv, w, color=C_CIR, zorder=3)
    axR.bar(i + w / 2, av, w, color=C_ATK, zorder=3)
    axR.text(i - w / 2, cv + 0.008, f"{cv:.2f}", ha="center", va="bottom", fontsize=8, color="#333")
    axR.text(i + w / 2, av + 0.008, f"{av:.2f}", ha="center", va="bottom", fontsize=8, color="#333")
axR.set_xticks(range(len(sc)))
axR.set_xticklabels(sc)
axR.set_ylabel("share of ‖ΔW‖  (L2 fraction of total)")
axR.set_title("Repair leans on the spared shared directions\n(R:F  0.94  vs  CIR's 0.36  →  ~2.6×)", fontsize=10)
axR.grid(True, axis="y", lw=0.4, color="#e6e6e6", zorder=0)
for s in ("top", "right"):
    axR.spines[s].set_visible(False)

handles = [Line2D([0], [0], marker="s", ls="none", color=C_CIR, ms=10, label="Δ_CIR  (defender edit, θ_B−θ₀)"),
           Line2D([0], [0], marker="s", ls="none", color=C_ATK, ms=10, label="Δ_attack  (repair, θ_attacked−θ_B)")]
fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))
fig.suptitle(f"The attacker BYPASSES CIR, it does not reverse it   "
             f"(cos = {al['cos']:+.2f},  reversal ≈ {al['reversal_fraction']:.2f};  "
             f"F-PPL {atk['naive_target_ppl']:.0f} → {atk['best_target_ppl']:.1f})",
             fontsize=10.5, y=1.0)
fig.tight_layout(rect=(0, 0.06, 1, 0.96))
out = os.path.join(ROOT, "results", op, "plots", "fig_repair_delta.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=140, bbox_inches="tight")
print("wrote", out)
