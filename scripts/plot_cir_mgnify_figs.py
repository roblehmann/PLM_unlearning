#!/usr/bin/env python
"""Two CIR figures for the MGnify run, from on-disk JSON (matplotlib, CPU):
  fig_cir_frontier.png  — efficacy (forget x) vs disruption (benign drift %) for all 24 CIR
                          operating points (frontier + knob sweeps); gate lines at 2% drift
                          and 2x forget; mlac_fo annotated off-scale. All annotated values are
                          computed from the run's JSON, none are typed in.
  fig_cir_overlap.png   — O(F,R): per-layer removable_topk / O_topk for MGnify, with the Pfam
                          aggregate removable_topk as a reference line (distinctive subspace exists).
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN = f"{ROOT}/runs/MGYF18690_metagenomic"
OUT = f"{RUN}/results/manuscript"
os.makedirs(OUT, exist_ok=True)


def _cells(pattern):
    rows = []
    for p in sorted(glob.glob(pattern)):
        d = json.load(open(p))
        fx, dr = d.get("forget_ppl_ratio_B_over_base"), d.get("benign_drift_rel")
        if fx is not None and dr is not None:
            rows.append((dr * 100.0, fx, os.path.basename(os.path.dirname(p)),
                         d.get("forget_ppl_base"), d.get("forget_ppl_B")))
    return rows


UTILITY_GATE_PCT = 2.0   # benign-drift gate drawn on the frontier figure (%)


def frontier_stats(cells, summary_path):
    """Values annotated on the frontier figure, derived from the run's own files.

    ``ceiling``  largest static forget ratio over all CIR operating points
    ``best``     the highest-forget point inside the utility gate (or None)
    ``mlac``     static forget ratio of the task-blocked model, from summary.json (or None)
    """
    ceiling = max(r[1] for r in cells)
    inside = [r for r in cells if r[0] <= UTILITY_GATE_PCT]
    best = max(inside, key=lambda r: r[1]) if inside else None
    mlac = None
    if os.path.exists(summary_path):
        mlac = json.load(open(summary_path)).get("n0_forget_ratio_B_over_A")
    return ceiling, best, mlac


def fig_frontier():
    frontier = _cells(f"{RUN}/results/cir_frontier/*/B_compute.json")
    knob = _cells(f"{RUN}/results/cir_knob/*/B_compute.json")
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for rows, c, mk, lab in [(frontier, "tab:blue", "o", "frontier (rate × termination)"),
                             (knob, "tab:orange", "s", "knob (subspace_mode × n_pcs × layers)")]:
        if rows:
            ax.scatter([r[0] for r in rows], [r[1] for r in rows], c=c, marker=mk, s=45,
                       edgecolor="k", linewidth=0.4, label=lab, zorder=3)
    # gates
    ax.axvline(2.0, color="green", ls="--", lw=1.2, label="utility gate (2% drift)")
    ax.axhline(2.0, color="gray", ls=":", lw=1.2, label='"unlearned" (2× forget)')
    ax.set_xlabel("benign utility drift  (%)  →  more disruption")
    ax.set_ylabel("static forget  (B / base ×)  →  more deletion")
    ceiling, best, mlac = frontier_stats(frontier + knob, f"{RUN}/results/summary.json")
    ax.set_title(f"CIR efficacy vs disruption (metagenomic family): ceiling ≈ {ceiling:.1f}× — no meaningful block")
    note = f"CIR ceiling ≈ {ceiling:.1f}× anywhere"
    if best is not None:
        note += f";\nbest utility-preserving point only {best[1]:.2f}×"
        if best[3] is not None and best[4] is not None:
            note += f" (F-PPL {best[3]:.0f}→{best[4]:.0f})"
        ax.annotate(f"best util-preserving\n{best[1]:.2f}× @ {best[0]:.2f}%", xy=(best[0], best[1]),
                    xytext=(best[0] + 0.33, best[1] + 0.26), fontsize=7.5, color="darkred",
                    arrowprops=dict(arrowstyle="->", color="darkred", lw=0.8))
    ax.text(0.35, 2.72, note, fontsize=8, color="darkred", ha="left", va="top")
    ax.set_ylim(0.9, 3.0)
    ax.set_xlim(0.3, 5.4)
    if mlac is not None:
        ax.text(5.25, 2.96, f"mlac_fo: {mlac:,.0f}×  (≈{round(mlac / ceiling, -2):,.0f}× higher, off-scale ↑)",
                fontsize=8, color="tab:red", ha="right", va="top")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    p = f"{OUT}/fig_cir_frontier.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print("wrote", p, f"({len(frontier)} frontier + {len(knob)} knob cells; "
          f"max forget = {max([r[1] for r in frontier + knob]):.2f}×)")


def fig_overlap():
    d = json.load(open(f"{RUN}/results/cir_overlap.json"))
    pfam = json.load(open(f"{ROOT}/results/cir_overlap.json"))["aggregate"]["removable_topk"]
    layers = sorted(d["per_layer"], key=lambda x: int(x))
    rem = [d["per_layer"][L]["removable_topk"] for L in layers]
    ov = [d["per_layer"][L]["O_topk"] for L in layers]
    agg = d["aggregate"]["removable_topk"]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    x = range(len(layers))
    ax.bar([i - 0.2 for i in x], rem, 0.4, color="tab:green",
           label="removable_topk (F distinctive, outside R)")
    ax.bar([i + 0.2 for i in x], ov, 0.4, color="tab:gray",
           label="O_topk (F inside R's subspace)")
    ax.axhline(pfam, color="tab:red", ls="--", lw=1.3,
               label=f"Pfam removable_topk = {pfam:.2f} (CIR blocked there)")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"L{int(L):02d}" for L in layers])
    ax.set_ylabel("fraction of F's top-24 activation variance")
    ax.set_ylim(0, 1.0)
    ax.set_title(f"O(F,R): a large distinctive subspace EXISTS on MGnify — "
                 f"removable_topk = {agg:.2f} ≈ Pfam {pfam:.2f}")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    p = f"{OUT}/fig_cir_overlap.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print("wrote", p, f"(MGnify removable_topk={agg:.3f} vs Pfam {pfam:.3f})")


if __name__ == "__main__":
    fig_frontier()
    fig_overlap()
