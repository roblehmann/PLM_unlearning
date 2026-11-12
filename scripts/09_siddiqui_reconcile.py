"""Phase 9 — the Siddiqui reconciliation, plotted.

The finding "displacement, high or low, doesn't buy durability" reads as a flat
contradiction of the "L2 predicts tamper-resistance" result (Siddiqui et al.). It is not —
it *refines* it. This phase foregrounds the reconciliation by plotting the durability
multiplier M against BOTH global weight displacement AND the LMC barrier height, across
every operating point of both safeguards (first-order TAR, CIR lowdis, CIR peak).

The honest, defensible reading the figure supports:
  * Global L2 does NOT predict M on its own — CIR **peak** has the LARGEST global
    displacement of any point yet M≈1, a direct counterexample to the raw-L2 heuristic.
  * L2 predicts durability only insofar as it reflects LEAVING THE BASIN. Every point here
    sits at a near-zero LMC barrier (all linearly connected to θ₀), and every point is
    reversible (M≈1). Barrier — not raw L2 — is the operative variable.
  * ONE-SIDEDNESS (stated on the figure): we have shown no-barrier ⇒ reversible, but we
    have NO positive point with a real barrier, so we cannot yet show barrier ⇒ durable.
    The high-barrier / high-M quadrant is unpopulated, not refuted.

Pure JSON/CSV + matplotlib (no torch, no GPU). Reads each op's weightspace.json (global L2
of B and A′, LMC f_barrier_nats) and multipliers.csv (M = mult_B_over_Aprime over the
clean-F recipes). Writes results/plots/siddiqui_reconcile.png + results/siddiqui_reconcile.csv.

    python scripts/09_siddiqui_reconcile.py
"""
import csv
import os

import _bootstrap  # noqa: F401
import hydra

# method -> CVD-safe categorical hue (Okabe-Ito blue / vermillion) + marker (identity is
# never color-alone). Assigned in fixed order, never cycled.
METHOD_STYLE = {
    "TAR": {"color": "#0072B2", "label": "first-order TAR"},
    "CIR": {"color": "#D55E00", "label": "CIR"},
}
# one operating point per row: (label, method, results-subdir relative to paths.results)
DEFAULT_OPS = [
    ("TAR", "TAR", ""),
    ("CIR lowdis", "CIR", "cir35m_lowdis"),
    ("CIR peak", "CIR", "cir35m_peak"),
]
MARKERS = ["o", "s", "^", "D", "v", "P"]


def _clean_F_multipliers(csv_path):
    """M = mult_B_over_Aprime for every clean-F recipe; returns (headline_full_ft, lo, hi)."""
    vals, full_ft = [], None
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row["adversary_data"] == "clean" and row["target_family"] == "F":
                m = float(row["mult_B_over_Aprime"])
                vals.append(m)
                if row["recipe"] == "full_ft":
                    full_ft = m
    if not vals:
        return None, None, None
    return (full_ft if full_ft is not None else max(vals)), min(vals), max(vals)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    _bootstrap.banner("PHASE 9 — SIDDIQUI RECONCILIATION (M vs global L2 and LMC barrier)")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from src.utils.io import ensure_dir, load_json

    rroot = cfg.paths.results
    ops_cfg = cfg.get("ops", None)
    ops = DEFAULT_OPS
    if ops_cfg:  # optional override: +ops='TAR:TAR:,CIR peak:CIR:cir35m_peak'
        ops = [tuple(x.split(":")) for x in (ops_cfg if isinstance(ops_cfg, str) else ",".join(ops_cfg)).split(",")]

    rows = []
    for i, (label, method, sub) in enumerate(ops):
        rdir = os.path.join(rroot, sub) if sub else rroot
        ws_p = os.path.join(rdir, "weightspace.json")
        mult_p = os.path.join(rdir, "multipliers.csv")
        if not (os.path.exists(ws_p) and os.path.exists(mult_p)):
            print(f"  skip {label}: missing weightspace.json or multipliers.csv in {rdir}")
            continue
        ws = load_json(ws_p)
        l2 = ws["l2"]
        B_rel = l2["B"]["__total__"]["rel"]
        Ap_rel = l2.get("A_prime", {}).get("__total__", {}).get("rel", float("nan"))
        barrier = ws.get("lmc_B", {}).get("f_barrier_nats", float("nan"))
        mono = ws.get("lmc_B", {}).get("f_monotonic", None)
        m_full, m_lo, m_hi = _clean_F_multipliers(mult_p)
        rows.append({
            "label": label, "method": method, "marker": MARKERS[i % len(MARKERS)],
            "M_full_ft": m_full, "M_lo": m_lo, "M_hi": m_hi,
            "global_L2_B_rel": B_rel, "global_L2_Aprime_rel": Ap_rel,
            "L2_ratio_B_over_Aprime": (B_rel / Ap_rel) if Ap_rel and Ap_rel == Ap_rel else float("nan"),
            "lmc_barrier_nats": barrier, "lmc_monotonic": mono,
        })

    if not rows:
        raise SystemExit("No operating points found — expected weightspace.json + multipliers.csv per op.")

    ensure_dir(rroot)
    csv_out = os.path.join(rroot, "siddiqui_reconcile.csv")
    with open(csv_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- report ----
    print(f"{'operating point':<14}{'method':<7}{'M(fullFT)':>10}{'M range':>16}"
          f"{'L2_B(rel)':>11}{'B/A′':>7}{'barrier(nats)':>14}{'mono':>6}")
    for r in rows:
        print(f"{r['label']:<14}{r['method']:<7}{r['M_full_ft']:>10.3f}"
              f"{('[%.3f,%.3f]' % (r['M_lo'], r['M_hi'])):>16}{r['global_L2_B_rel']:>11.4f}"
              f"{r['L2_ratio_B_over_Aprime']:>7.2f}{r['lmc_barrier_nats']:>14.3f}{str(r['lmc_monotonic']):>6}")

    # ---- figure: two panels, M vs global L2 (left) and M vs LMC barrier (right) ----
    # The operative fact: every θ₀→θ_B path is MONOTONIC (no return ridge) ⇒ all reversible ⇒
    # M≈1. So neither scalar predicts M in-sample: raw L2 varies ~8× (peak largest), barrier
    # height varies ~3× (TAR largest, yet lowest M — it is convexity of a monotone path across
    # a huge endpoint gap, not a ridge). The durable band (M≥3) is entirely unpopulated.
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.6, 4.6))
    ymax = max(3.6, max(r["M_hi"] for r in rows) * 1.15)
    all_mono = all(bool(r["lmc_monotonic"]) for r in rows)

    def panel(ax, xkey, xlabel, logx, note_mono=False):
        # shade the unpopulated durability-pass band (M≥3) — the same on both panels
        ax.axhspan(3.0, ymax, color="#f0e0d0", alpha=0.55, zorder=0)
        for r in rows:
            c = METHOD_STYLE[r["method"]]["color"]
            x = r[xkey]
            ax.errorbar(x, r["M_full_ft"], yerr=[[r["M_full_ft"] - r["M_lo"]], [r["M_hi"] - r["M_full_ft"]]],
                        fmt=r["marker"], ms=9, color=c, ecolor=c, elinewidth=1.4, capsize=3,
                        mec="white", mew=1.0, zorder=5)
            lab = r["label"] + ("  (mono)" if note_mono and r["lmc_monotonic"] else "")
            ax.annotate(lab, (x, r["M_hi"]), textcoords="offset points", xytext=(0, 7),
                        ha="center", fontsize=8, color="#333333")
        ax.axhline(3.0, ls="--", lw=1.2, color="#a06a30", zorder=2)
        ax.axhline(1.0, ls=":", lw=1.0, color="#999999", zorder=1)
        ax.text(0.02, 3.08, "durability pass (M≥3) — UNPOPULATED", transform=ax.get_yaxis_transform(),
                fontsize=7.5, color="#a06a30", va="bottom")
        ax.text(0.02, 1.03, "no adaptation cost (M=1)", transform=ax.get_yaxis_transform(),
                fontsize=7.5, color="#999999", va="bottom")
        if logx:
            ax.set_xscale("log")
        ax.set_ylim(0.7, ymax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("durability multiplier  M = n×(B) / n×(A′)")
        ax.grid(True, lw=0.4, color="#e6e6e6", zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    panel(axL, "global_L2_B_rel", "global weight displacement  ‖θ_B − θ₀‖ / ‖θ₀‖", logx=True)
    axL.set_title("M vs global L2  —  raw L2 does not predict M\n(CIR peak: largest L2, still M≈1)", fontsize=9.5)

    panel(axR, "lmc_barrier_nats", "LMC barrier to base  (nats)  —  all paths monotonic", logx=False, note_mono=True)
    axR.set_title("M vs LMC barrier  —  height doesn't track M either\n(all monotonic ⇒ no ridge ⇒ in-basin)", fontsize=9.5)
    axR.set_xlim(-0.03, max(r["lmc_barrier_nats"] for r in rows) * 1.25)

    handles = [Line2D([0], [0], marker="o", ls="none", color=METHOD_STYLE[m]["color"],
                      mec="white", ms=9, label=METHOD_STYLE[m]["label"]) for m in ("TAR", "CIR")]
    note = ("One-sidedness: every point is monotonically connected to θ₀ (in-basin) and reversible — "
            "we show in-basin ⇒ reversible, but have NO non-monotonic (ridge) point, so cannot yet show "
            "leaving-the-basin ⇒ durable.")
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.01))
    fig.text(0.5, 0.055, note, ha="center", fontsize=7.6, color="#8a5a2b", wrap=True)
    fig.suptitle("Reconciling Siddiqui: L2 predicts durability only insofar as it reflects leaving the "
                 "basin — none of our points do", fontsize=10.5, y=1.0)
    fig.tight_layout(rect=(0, 0.10, 1, 0.97))
    ensure_dir(os.path.join(rroot, "plots"))
    png = os.path.join(rroot, "plots", "siddiqui_reconcile.png")
    fig.savefig(png, dpi=140, bbox_inches="tight")
    print(f"\nplot  -> {png}\ntable -> {csv_out}")
    l2span = max(r["global_L2_B_rel"] for r in rows) / min(r["global_L2_B_rel"] for r in rows)
    print(f"\nReconciliation: global L2 spans ~{l2span:.0f}× (CIR peak largest) and barrier height spans "
          f"~{max(r['lmc_barrier_nats'] for r in rows)/min(r['lmc_barrier_nats'] for r in rows):.0f}× "
          f"(TAR largest), yet M stays ≈1 throughout. All {len(rows)} points are monotonically connected "
          f"to θ₀ (no return ridge){' ' if all_mono else ' [CHECK: not all monotonic] '}⇒ reversible. "
          "Neither scalar predicts M; the operative structural fact is monotonic in-basin connectivity. "
          "This refines Siddiqui (leaving the basin is what L2 must proxy) without a positive point to "
          "test the converse.")


if __name__ == "__main__":
    main()
