"""Plots for Phase 7: acquisition curves, the utility-vs-cost frontier, and the
durability table-as-bars. Headless (Agg) so it runs on compute nodes."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MODEL_STYLE = {
    "A": ("tab:gray", "o", "A (base)"),
    "A_prime": ("tab:blue", "s", "A' (compute-matched control)"),
    "B": ("tab:red", "^", "B (task-blocked)"),
}


def _frange(start, stop, step):
    out, v = [], start
    while v < stop:
        out.append(v)
        v += step
    return out


def _ensure(path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def plot_acquisition_curves(df, tau: float, out_path, ppl_col: str = "best_target_ppl", title: str = "") -> str:
    """Held-out target-family pseudo-perplexity vs attacker budget n, per model.
    Median over seeds (line) + per-seed points; horizontal τ line."""
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for model, (color, marker, label) in MODEL_STYLE.items():
        m = df[df["model"] == model]
        if len(m) == 0:
            continue
        budgets = sorted(m["budget_n"].unique())
        med = [m[m["budget_n"] == b][ppl_col].median() for b in budgets]
        ax.plot(budgets, med, color=color, marker=marker, label=label, linewidth=2)
        ax.scatter(m["budget_n"], m[ppl_col], color=color, alpha=0.25, s=14)
    ax.axhline(tau, color="black", linestyle="--", linewidth=1, label=f"τ = {tau:.3f}")
    ax.set_xscale("log")
    ax.set_xlabel("attacker budget n (fine-tuning examples)")
    ax.set_ylabel("held-out target-family pseudo-perplexity")
    if title:
        ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = _ensure(out_path)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return str(out)


def plot_all_conditions(df, tau: float, out_dir, ppl_col: str = "best_target_ppl") -> list:
    """One acquisition-curve figure per condition (recipe × rank × adversary × family)."""
    out_dir = Path(out_dir)
    cols = [c for c in ["recipe", "lora_rank", "adversary_data", "target_family"] if c in df.columns]
    paths = []
    for cond, sub in df.groupby(cols, dropna=False):
        cond = cond if isinstance(cond, tuple) else (cond,)
        tag = "__".join(f"{c}-{v}" for c, v in zip(cols, cond))
        title = ", ".join(f"{c}={v}" for c, v in zip(cols, cond))
        paths.append(plot_acquisition_curves(sub, tau, out_dir / f"acq_{tag}.png", ppl_col, title))
    return paths


def plot_utility_frontier(multiplier_df, benign_ppl: Dict[str, float], out_path, util_tol: Optional[float] = None) -> str:
    """Utility (benign PPL retention of B vs A') against the headline multiplier.
    The safeguard is only interesting where utility ≈ A' AND the multiplier is high."""
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ppl_b, ppl_ap = benign_ppl.get("B"), benign_ppl.get("A_prime")
    retention = abs(ppl_b - ppl_ap) / ppl_ap if (ppl_b and ppl_ap) else None
    m = multiplier_df[(multiplier_df.get("target_family") == "F") & (multiplier_df.get("adversary_data") == "clean")] \
        if {"target_family", "adversary_data"}.issubset(multiplier_df.columns) else multiplier_df
    for _, r in m.iterrows():
        x = r.get("mult_B_over_Aprime")
        if x is None or retention is None:
            continue
        censored = bool(r.get("mult_B_over_Aprime_censored", False))
        ax.scatter(x, retention, marker=">" if censored else "o",
                   color="tab:red" if censored else "tab:blue", s=60)
    if retention is not None:
        ax.axhline(retention, color="gray", linestyle=":", linewidth=1)
    if util_tol is not None:
        ax.axhline(util_tol, color="green", linestyle="--", linewidth=1, label=f"utility tol = {util_tol}")
    ax.set_xlabel("adaptation-cost multiplier  B / A'   (> = right-censored lower bound)")
    ax.set_ylabel("benign PPL retention |B−A'|/A'")
    ax.set_title("Utility-vs-cost frontier")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = _ensure(out_path)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return str(out)


def plot_layer_l2(l2_B, l2_Aprime, out_path) -> str:
    """Per-layer relative weight movement ‖Δθ‖/‖θ₀‖, B vs the A′ control."""
    def _order(g):
        return (0, int(g.split("_")[1])) if g.startswith("layer_") else (1, g)

    groups = sorted([g for g in l2_B if g != "__total__"], key=_order)
    relB = [l2_B[g]["rel"] for g in groups]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = list(range(len(groups)))
    if l2_Aprime:
        relA = [l2_Aprime.get(g, {}).get("rel", float("nan")) for g in groups]
        ax.bar([i - 0.2 for i in x], relB, 0.4, label="B (task-blocked)", color="tab:red")
        ax.bar([i + 0.2 for i in x], relA, 0.4, label="A′ (compute-matched control)", color="tab:blue")
    else:
        ax.bar(x, relB, color="tab:red", label="B (task-blocked)")
    ax.set_xticks(x)
    ax.set_xticklabels([g.replace("layer_", "L") for g in groups], rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("relative ‖θ − θ₀‖ / ‖θ₀‖")
    ax.set_title("Weight movement from base θ₀, per layer")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = _ensure(out_path)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return str(out)


def plot_lmc_barrier(lmc, out_path) -> str:
    """LMC path θ₀→θ_B: F-test and benign pseudo-perplexity on one shared log axis vs α.
    A single axis (no zoomed twin) so the flat benign line reads as flat. ``no barrier``
    is keyed off F-path monotonicity (a ridge requires an interior point above both
    endpoints), not an absolute nat threshold."""
    a = lmc["alphas"]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(a, lmc["f_ppl"], color="tab:red", marker="^", lw=2, label="F-test PPL")
    if "benign_ppl" in lmc:
        ax.plot(a, lmc["benign_ppl"], color="tab:blue", marker="o", lw=2, label="benign PPL (≈ flat)")
    ax.set_yscale("log")
    ax.set_xlabel("interpolation  α   (θ₀ → θ_B)")
    ax.set_ylabel("pseudo-perplexity (log)")
    fb = lmc.get("f_barrier_nats", float("nan"))
    mono = lmc.get("f_monotonic", False)
    benign_flat = lmc.get("benign_barrier_nats", 1.0) < 0.5
    no_barrier = mono and benign_flat
    tag = ("no barrier (F monotonic, benign flat)"
           if no_barrier else f"F-barrier = {fb:.2f} nats")
    ax.set_title(f"LMC θ₀ → θ_B:  {tag}")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    out = _ensure(out_path)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return str(out)


def plot_ladder_curve(df, tau: float, out_path, ppl_col: str = "best_target_ppl",
                      budget: Optional[int] = None, floor: Optional[float] = None,
                      naive_B: Optional[float] = None) -> str:
    """Sibling-ladder leakage curve (Phase 10): held-out FORGET-family pseudo-perplexity
    (F-PPL) vs the rung's homology distance to F, one line per (model, recipe).

    Two panels sharing the log F-PPL y-axis: (top) F-PPL vs mean sequence identity to F
    (%, decreasing left→right = increasing distance); (bottom) F-PPL vs cross-HMM
    bitscore (log; the axis that orders the rungs past the identity-detection floor). At
    ``budget`` (default: the max budget in df) each point is the median over seeds. τ,
    A's floor and B's pre-attack naive F-PPL are drawn as reference lines. A curve that
    sits below τ at close rungs and rises back above it at distant rungs is the
    reawaken-by-proximity signal; where it crosses τ is the distance at which it stops."""
    import numpy as np

    if budget is None:
        budget = int(max(df["budget_n"].unique()))
    d = df[df["budget_n"] == budget]

    fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharey=True)
    x_specs = [
        ("mean_id_all", "mean identity to F (fraction; → distance)", False, True),
        ("hmm_bits_mean", "cross-HMM bitscore to F (log)", True, False),
    ]
    style = {
        ("B", "lora"): ("tab:red", "^", "-"),
        ("B", "full_ft"): ("tab:red", "^", "--"),
        ("A_prime", "lora"): ("tab:blue", "s", "-"),
        ("A_prime", "full_ft"): ("tab:blue", "s", "--"),
    }
    for ax, (xcol, xlabel, xlog, invert) in zip(axes, x_specs):
        for (model, recipe), sub in d.groupby(["model", "recipe"]):
            g = sub.groupby(xcol)[ppl_col].median().sort_index()
            color, marker, ls = style.get((model, recipe), ("tab:gray", "o", "-"))
            ax.plot(g.index, g.values, color=color, marker=marker, linestyle=ls,
                    lw=2, label=f"{model} / {recipe}")
        ax.axhline(tau, color="black", linestyle=":", lw=1, label=f"τ = {tau:.2f}")
        if floor is not None:
            ax.axhline(floor, color="green", linestyle=":", lw=1, label=f"A floor = {floor:.2f}")
        if naive_B is not None:
            ax.axhline(naive_B, color="tab:red", linestyle=":", lw=1, alpha=0.6,
                       label=f"B naive = {naive_B:.0f}")
        ax.set_yscale("log")
        if xlog:
            ax.set_xscale("symlog")
        if invert:
            ax.invert_xaxis()  # high identity (close) on the left
        ax.set_xlabel(xlabel)
        ax.set_ylabel("F-PPL on F_test (log)")
    axes[0].set_title(f"Sibling-ladder relearning: F-PPL vs distance to F (budget n={budget})")
    axes[0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    out = _ensure(out_path)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return str(out)


def plot_ladder_budget_heatmap(df, out_path, naive_B: float, aprime_level: float,
                               model: str = "B", recipes=("lora", "full_ft")) -> str:
    """Reawakening of F (%) as a distance × budget grid, for the blocked model.

    Rows = ladder rungs (closest→farthest to F, top→bottom); columns = attacker budget
    n; cell = log-scale recovery of F-PPL from the blocked baseline (naive_B) back toward
    the never-blocked level (aprime_level), 0–100%. One panel per recipe. Makes the
    moving boundary explicit: the reawakened (deep-red) corner spreads down-and-right as
    budget grows, so *how related* the data must be depends on *how much* of it there is."""
    import math

    import numpy as np

    d = df[df["model"] == model]
    cols = ["rung_acc", "rung_id", "mean_id_all", "hmm_bits_mean"]
    has_band = "band" in d.columns
    if has_band:
        cols = cols + ["band"]
    meta = d[cols].drop_duplicates().copy()
    # group DETECTABLE homologs (close/mid, HMM bits > 0) above BACKGROUND (no HMM hit),
    # each block sorted closest→farthest — so identity, which is unreliable for background
    # families, doesn't interleave them with real siblings.
    band_rank = {"close": 0, "mid": 1, "background": 2, "far": 2}
    meta["_br"] = (meta["band"].map(lambda b: band_rank.get(b, 1)) if has_band
                   else (meta["hmm_bits_mean"] <= 0).astype(int) * 2)
    meta = meta.sort_values(["_br", "mean_id_all", "hmm_bits_mean"], ascending=[True, False, False])
    rungs = list(meta["rung_acc"])

    def _ylab(r):
        bg = (getattr(r, "band", "") in ("background", "far")) if has_band else (r.hmm_bits_mean <= 0)
        return f"{r.rung_id}\nbackground" if bg else f"{r.rung_id}\n{r.mean_id_all * 100:.0f}% id"
    ylabels = [_ylab(r) for r in meta.itertuples()]
    n_detect = int((meta["_br"] < 2).sum())  # boundary row between detectable and background
    budgets = sorted(d["budget_n"].unique())
    denom = math.log(naive_B) - math.log(aprime_level)

    def rec(p):
        return max(0.0, min(1.0, (math.log(naive_B) - math.log(max(p, aprime_level))) / denom)) * 100

    fig, axes = plt.subplots(1, len(recipes), figsize=(4.2 * len(recipes) + 0.8, 4.6), squeeze=False)
    im = None
    for k, (ax, recipe) in enumerate(zip(axes[0], recipes)):
        sub = d[d["recipe"] == recipe]
        M = np.full((len(rungs), len(budgets)), np.nan)
        for i, acc in enumerate(rungs):
            g = sub[sub["rung_acc"] == acc].groupby("budget_n")["best_target_ppl"].median()
            for j, b in enumerate(budgets):
                if b in g.index:
                    M[i, j] = rec(g[b])
        im = ax.imshow(M, cmap="Reds", vmin=0, vmax=100, aspect="auto")
        ax.set_xticks(range(len(budgets)))
        ax.set_xticklabels(budgets)
        ax.set_yticks(range(len(rungs)))
        ax.set_yticklabels(ylabels if k == 0 else [""] * len(rungs), fontsize=8)
        ax.set_xlabel("attacker budget n")
        ax.set_title(f"{model} / {recipe}")
        for i in range(len(rungs)):
            for j in range(len(budgets)):
                if not math.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center",
                            color="white" if M[i, j] > 55 else "black", fontsize=8)
        if 0 < n_detect < len(rungs):  # divide detectable homologs (above) from background (below)
            ax.axhline(n_detect - 0.5, color="black", lw=1.6)
    fig.suptitle("Reawakening of F (%) — sibling distance × attacker budget (no F in training)\n"
                 "detectable homologs (top) vs background/unrelated families (below the line)", fontsize=10)
    if im is not None:
        fig.colorbar(im, ax=axes[0].tolist(), fraction=0.03, pad=0.02,
                     label="recovery toward unblocked (%)")
    out = _ensure(out_path)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(out)


def plot_ladder_band_profile(df, out_path, tau: float, naive_B: float, aprime_level: float,
                             mean_len: Optional[Dict[str, float]] = None,
                             ppl_col: str = "best_target_ppl") -> str:
    """Band-aware ladder figure: the honest replacement for ``plot_ladder_curve``.

    ``plot_ladder_curve`` puts the rungs on a numeric identity axis, which misleads on a
    metagenomic ladder: a *background* family carrying a noisy profile-identity label
    (e.g. 25%) lands between genuine homologs and draws a spurious spike-and-recover
    zigzag, while every background rung collapses onto bitscore 0 in the second panel.
    Here the rungs are CATEGORICAL, ordered detectable-first (by cross-HMM bits) with the
    background block separated and shaded — the same convention as the budget heatmap.

    (top) held-out F-PPL per rung, one line per attacker budget for B, with A' at the top
    budget as the never-blocked control, and the blocked baseline / unblocked level / τ as
    reference lines. (bottom, when ``mean_len`` maps rung_acc -> mean corpus sequence
    length) recovery vs the number of TRAINING RESIDUES the attacker actually saw, which
    removes the confound that background families here have much shorter members: if the
    detectable rungs sit at ~100% recovery on FEWER residues than a background rung uses
    to reach a fraction of that, the contrast cannot be a token-budget artifact."""
    import matplotlib.transforms as mtransforms
    import numpy as np

    d = df[df["model"].isin(["B", "A_prime"])]
    band_rank = {"close": 0, "mid": 1, "background": 2, "far": 2}
    meta = (d[["rung_acc", "rung_id", "hmm_bits_mean", "band"]].drop_duplicates().copy())
    meta["band_rank"] = meta["band"].map(lambda b: band_rank.get(b, 1))
    meta = meta.sort_values(["band_rank", "hmm_bits_mean"], ascending=[True, False]).reset_index(drop=True)
    order = list(meta["rung_acc"])
    x = list(range(len(order)))
    n_detect = int((meta["band_rank"] < 2).sum())
    budgets = sorted(d["budget_n"].unique())

    two = mean_len is not None
    fig, axes = plt.subplots(2 if two else 1, 1, figsize=(9, 8.6 if two else 4.8), squeeze=False)
    ax = axes[0][0]
    blend = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)

    if n_detect < len(order):
        ax.axvspan(n_detect - 0.5, len(order) - 0.5, color="0.92", zorder=0)
        ax.axvline(n_detect - 0.5, color="black", lw=1.2, zorder=3)
        ax.text((n_detect + len(order) - 1) / 2, 0.02, "background / unrelated families",
                transform=blend, ha="center", fontsize=8, color="0.35")
        ax.text((n_detect - 1) / 2, 0.02, "detectable homologs",
                transform=blend, ha="center", fontsize=8, color="0.35")

    reds = plt.cm.Reds(np.linspace(0.45, 0.95, len(budgets)))
    for color, n in zip(reds, budgets):
        y = [d[(d.model == "B") & (d.budget_n == n) & (d.rung_acc == a)][ppl_col].median() for a in order]
        ax.plot(x, y, marker="^", color=color, lw=2, label=f"B, n={int(n)}", zorder=4)
    y_ap = [d[(d.model == "A_prime") & (d.budget_n == budgets[-1]) & (d.rung_acc == a)][ppl_col].median()
            for a in order]
    ax.plot(x, y_ap, marker="s", color="tab:blue", lw=2, ls="--",
            label=f"A' (never blocked), n={int(budgets[-1])}", zorder=4)

    ax.axhline(naive_B, color="tab:red", ls=":", lw=1, alpha=0.7, label=f"B blocked = {naive_B:,.0f}")
    ax.axhline(aprime_level, color="tab:blue", ls=":", lw=1, alpha=0.7, label=f"unblocked ≈ {aprime_level:.1f}")
    ax.axhline(tau, color="black", ls=":", lw=1, label=f"τ = {tau:.2f}")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r.rung_id}\n{'background' if r.band_rank >= 2 else f'{r.hmm_bits_mean:.0f} bits'}"
                        for r in meta.itertuples()], fontsize=8)
    ax.set_ylabel("F-PPL on F_test (log)")
    ax.set_title("Sibling-ladder reawakening: detectable homologs vs background families\n"
                 "(B is fine-tuned on the rung, never on F)")
    ax.legend(fontsize=7, ncol=2, loc="center left", framealpha=0.9)

    if two:
        ax2 = axes[1][0]
        denom = math.log(naive_B) - math.log(aprime_level)
        for a in order:
            row = meta[meta.rung_acc == a].iloc[0]
            bg = row["band_rank"] >= 2
            xs, ys = [], []
            for n in budgets:
                p = d[(d.model == "B") & (d.budget_n == n) & (d.rung_acc == a)][ppl_col].median()
                if not (p == p):
                    continue
                xs.append(n * mean_len.get(a, float("nan")))
                ys.append(max(0.0, min(1.0, (math.log(naive_B) - math.log(max(p, aprime_level))) / denom)) * 100)
            ax2.plot(xs, ys, marker="o", lw=1.6,
                     color=("0.55" if bg else "tab:red"), ls=("--" if bg else "-"), zorder=3)
            # Only the background rungs are labelled: the detectable ones all pile up at ~100%,
            # where per-point labels would collide into an unreadable smear.
            if bg and xs:
                ax2.annotate(str(row["rung_id"]), (xs[-1], ys[-1]), fontsize=8,
                             xytext=(6, 7), textcoords="offset points", va="center", color="0.35",
                             bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=0.6))
        ax2.axhline(50, color="black", ls=":", lw=1)
        ax2.text(0.995, 0.53, "reawakening threshold (50%)", transform=ax2.transAxes,
                 ha="right", fontsize=7, color="0.3")
        from matplotlib.lines import Line2D
        ax2.legend(handles=[Line2D([], [], color="tab:red", marker="o", lw=1.6,
                                   label=f"detectable homolog rungs ({n_detect})"),
                            Line2D([], [], color="0.55", marker="o", lw=1.6, ls="--",
                                   label=f"background families ({len(order) - n_detect})")],
                   fontsize=8, loc="center left")
        ax2.set_xscale("log")
        ax2.set_xlabel("training residues the attacker saw (budget n × mean corpus length, log)")
        ax2.set_ylabel("recovery of F (%)")
        ax2.set_ylim(-4, 104)
        ax2.grid(alpha=0.3)
        ax2.set_title("Token-matched view: the detectable rungs saturate on FEWER residues than the "
                      "background rungs consume", fontsize=9)

    fig.tight_layout()
    out = _ensure(out_path)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return str(out)


def plot_durability(multiplier_df, out_path) -> str:
    """Bar chart of the headline multiplier across adversary_data × target_family."""
    if not {"adversary_data", "target_family"}.issubset(multiplier_df.columns):
        return ""
    fig, ax = plt.subplots(figsize=(6, 4.2))
    groups = multiplier_df.groupby(["target_family", "adversary_data"])["mult_B_over_Aprime"].median()
    labels = [f"{tf}/{ad}" for (tf, ad) in groups.index]
    ax.bar(range(len(groups)), groups.values, color="tab:red")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("median multiplier B / A'")
    ax.set_title("Durability: multiplier by target family × adversary data")
    fig.tight_layout()
    out = _ensure(out_path)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return str(out)


# Adjacent categorical slots 3 and 4 (aqua / amber) of the validated default palette:
# the pair clears the CVD and normal-vision separation gates in light mode. Deliberately
# NOT the blue/red of MODEL_STYLE — those are reserved paper-wide for the A'/B arms, and
# reusing them for the adversary axis would make one hue mean two things. Marker shape
# repeats the distinction, so identity is never carried by colour alone.
ADVERSARY_STYLE = {
    "clean": ("#1baf7a", "o", "clean $F$"),
    "homolog_augmented": ("#eda100", "^", "homolog-augmented $F$"),
}


def _cond_label(recipe: str, rank) -> str:
    if recipe == "full_ft":
        return "full fine-tune"
    return f"LoRA r={int(rank)}" if rank is not None else "LoRA"


def _recipe_conditions(d):
    """(recipe, rank) pairs present, full fine-tune first then ascending rank. Rank is
    normalised to None for full_ft — it arrives as NaN, which is never equal to itself,
    so leaving it raw would split one recipe into a separate condition per row."""
    seen = {(r.recipe, None if r.lora_rank != r.lora_rank else float(r.lora_rank))
            for r in d.itertuples()}
    return sorted(seen, key=lambda c: (c[0] != "full_ft", c[1] if c[1] is not None else 0))


def _cond_rows(d, recipe, rank):
    m = d[d.recipe == recipe]
    return m[m.lora_rank.isna()] if rank is None else m[m.lora_rank == rank]


def plot_per_seed_multipliers(per_seed_df, out_path, gates=(3.0, 10.0), target_family: str = "F") -> str:
    """Paired per-seed durability multipliers: every seed shown as its own point, with
    the median and the min-max range, alongside the per-seed crossing budgets the ratio
    is built from. The left panel is the endpoint; the right panel is why the ratio is
    tighter than its parts — the seed's data draw moves B and A' together, so it cancels.
    """
    d = per_seed_df[per_seed_df["target_family"] == target_family].copy()
    d = d.dropna(subset=["mult_B_over_Aprime"])
    conds = _recipe_conditions(d)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 4.2),
                                   gridspec_kw={"width_ratios": [1.35, 1]})

    rows, labels = [], []
    for i, (recipe, rank) in enumerate(conds):
        for j, adv in enumerate(("clean", "homolog_augmented")):
            rows.append((recipe, rank, adv, -(2.2 * i + j)))
            labels.append(_cond_label(recipe, rank) if j == 0 else "")

    # ---- left: paired multipliers, one point per seed ---------------------- #
    vals_all = d["mult_B_over_Aprime"]
    lo, hi = float(vals_all.min()), float(vals_all.max())
    span = max(hi, 1.0) - min(lo, 1.0)
    x_text = max(hi, 1.0) + 0.10 * span          # one fixed column for the annotations
    axL.set_xlim(min(lo, 1.0) - 0.10 * span, x_text + 0.42 * span)
    axL.axvline(1.0, color="black", linestyle="--", linewidth=1, zorder=1)

    for recipe, rank, adv, y in rows:
        g = _cond_rows(d, recipe, rank)
        v = sorted(g[g.adversary_data == adv]["mult_B_over_Aprime"].tolist())
        if not v:
            continue
        color, marker, _ = ADVERSARY_STYLE[adv]
        axL.plot([v[0], v[-1]], [y, y], color=color, linewidth=2, alpha=0.35,
                 solid_capstyle="round", zorder=2)
        axL.scatter(v, [y] * len(v), color=color, marker=marker, s=44,
                    edgecolors="white", linewidths=0.8, zorder=3)
        med = v[len(v) // 2] if len(v) % 2 else 0.5 * (v[len(v) // 2 - 1] + v[len(v) // 2])
        axL.plot([med, med], [y - 0.3, y + 0.3], color=color, linewidth=2.2, zorder=4)
        axL.annotate(f"{med:.2f}  [{v[0]:.2f}, {v[-1]:.2f}]", (x_text, y), va="center",
                     fontsize=8, color="#4a4a4a")

    ys = [y for *_, y in rows]
    axL.set_yticks(ys)
    axL.set_yticklabels(labels, fontsize=9)
    axL.set_ylim(min(ys) - 0.9, 1.9)
    axL.set_xlabel("adaptation-cost multiplier  $M = n_{cross}(B)\\,/\\,n_{cross}(A')$")
    axL.set_title("Per-seed paired multipliers (median, min-max)", fontsize=10, loc="left")
    gate_txt = " and ".join(f"{g:g}x" for g in gates)
    axL.annotate(f"{gate_txt} durability gates lie far off-scale to the right",
                 xy=(0.02, 0.02), xycoords="axes fraction", fontsize=8, color="#6a6a6a")
    for spine in ("top", "right"):
        axL.spines[spine].set_visible(False)
    axL.grid(axis="x", color="#e6e6e6", linewidth=0.8, zorder=0)
    axL.set_axisbelow(True)
    ticks = [t for t in axL.get_xticks() if t <= max(hi, 1.0) + 0.02 * span]
    axL.set_xticks(ticks)
    handles = [plt.Line2D([], [], color=c, marker=m, linestyle="none", label=lab)
               for c, m, lab in ADVERSARY_STYLE.values()]
    axL.legend(handles=handles, fontsize=8, loc="upper left", frameon=False,
               ncol=2, borderaxespad=0.2)

    # ---- right: the paired crossing budgets behind those ratios ------------ #
    # Log scale: the recipes sit at different absolute budgets, and what matters is
    # relative spread, which a log axis renders at the same size in every group.
    clean = d[d.adversary_data == "clean"]
    for recipe, rank in conds:
        g = _cond_rows(clean, recipe, rank)
        for _, r in g.iterrows():
            axR.plot([0, 1], [r["n_cross_A_prime"], r["n_cross_B"]], color="#9a9a9a",
                     linewidth=1, alpha=0.75, zorder=2)
            axR.scatter([0, 1], [r["n_cross_A_prime"], r["n_cross_B"]],
                        color=[MODEL_STYLE["A_prime"][0], MODEL_STYLE["B"][0]],
                        s=32, zorder=3, edgecolors="white", linewidths=0.7)
        if len(g):
            axR.annotate(_cond_label(recipe, rank), (1, g["n_cross_B"].median()),
                         xytext=(12, 0), textcoords="offset points", va="center",
                         fontsize=8, color="#4a4a4a")
    axR.set_yscale("log")
    axR.set_xticks([0, 1])
    axR.set_xticklabels(["$A'$ (control)", "$B$ (blocked)"], fontsize=9)
    axR.set_xlim(-0.3, 1.95)
    axR.set_ylabel("crossing budget $n_{cross}$ (examples, log)")
    axR.set_title("Same seed, both arms (clean $F$)", fontsize=10, loc="left")
    ylo, yhi = clean[["n_cross_A_prime", "n_cross_B"]].min().min(), clean[["n_cross_A_prime", "n_cross_B"]].max().max()
    step = 10 ** math.floor(math.log10(yhi - ylo))
    yticks = [t for t in _frange(step * math.floor(ylo / step), yhi + step, step) if ylo * 0.97 <= t <= yhi * 1.03]
    axR.set_yticks(yticks)
    axR.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    axR.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    for spine in ("top", "right"):
        axR.spines[spine].set_visible(False)
    axR.grid(axis="y", color="#e6e6e6", linewidth=0.8, zorder=0)
    axR.set_axisbelow(True)

    fig.tight_layout()
    p = _ensure(out_path)
    fig.savefig(p, dpi=200, bbox_inches="tight")
    fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return str(p)
