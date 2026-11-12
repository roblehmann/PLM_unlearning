"""Aggregate T0-T3 into results/{summary.json, RESULTS.md, fig_*.png}.

CPU only, and tolerant of missing phases (it reports which are absent), so it is safe to
re-run at any point to see the state of the experiment.

Beyond collecting numbers it runs the experiment's internal consistency checks — the ones
that would otherwise be assumed:

* T2's forward ladder at k=0 IS T1's ``H_head`` at H1, and the reverse at k=0 IS
  ``H_trunk``. They are computed by different code paths; if they disagree, the splice is
  not deterministic and nothing here is trustworthy.
* T3's pre-attack PPPL at seed 0 must equal T0/T1's B, since both score the same corpus.

    sbatch experiments/5_readout_vs_representation/aggregate.sbatch
    python experiments/5_readout_vs_representation/aggregate.py --arms mgnify
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import common as C

# Okabe-Ito, the standard colour-vision-deficiency-safe qualitative set, and already this
# repo's palette (experiments/4/aggregate.py). Assigned to arms in a FIXED order and never
# cycled; #E69F00 is deliberately left out because it sits close to #D55E00 under
# deuteranopia. Arms are also positionally ordered and directly labelled, so identity is
# never carried by colour alone.
ARM_COLOR = {
    "mgnify": "#D55E00",
    "pfam_fullchain": "#0072B2",
    "pfam_legacy": "#009E73",
    "pfam_domain": "#CC79A7",
    "pf00107": "#56B4E9",
}
SERIES_COLOR = {
    "H_head": "#0072B2", "H_trunk": "#D55E00",
    "head_only": "#0072B2", "trunk_only": "#D55E00", "full_ft": "#009E73",
}
NEUTRAL = "#666666"
GRID = "#D9D9D9"

PHASES = {"t0": "t0_sanity.json", "t1": "t1_hybrids.json",
          "t2": "t2_depth_ladder.json", "t3": "t3_partial_attack.json"}
CONSISTENCY_TOL = 1e-3


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arms", nargs="+", default=C.ARM_ORDER, choices=C.ARM_ORDER)
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Collect
# --------------------------------------------------------------------------- #
def collect(arm) -> dict:
    out = {"key": arm.key, "label": arm.label, "have": {}}
    for phase, fname in PHASES.items():
        p = arm.results / fname
        out["have"][phase] = p.exists()
        out[phase] = C.load_json(p) if p.exists() else None
    return out


def consistency(a: dict) -> list:
    """Cross-phase checks. Returns a list of {check, ok, detail}."""
    checks = []
    t1, t2, t3 = a.get("t1"), a.get("t2"), a.get("t3")

    if t1 and t2:
        h1 = t1["hybrids"][C.HEADLINE_HEAD_SET]
        for direction, row_name in (("forward", "H_head"), ("reverse", "H_trunk")):
            lad = {r["k"]: r for r in (t2["ladders"].get(direction) or [])}
            if 0 in lad and h1.get(row_name):
                x, y = lad[0]["pppl_F"], h1[row_name]["pppl_F"]
                rel = abs(x - y) / y if y else float("inf")
                checks.append({
                    "check": f"T2 {direction} k=0 == T1 {row_name} (H1)",
                    "ok": rel <= CONSISTENCY_TOL,
                    "detail": f"{x:.6f} vs {y:.6f} (rel {rel:.2e})",
                })

    if t1 and t3:
        b = t1["donors"]["B"]["F"]
        seed0 = [r for r in t3["rows"] if r["seed"] == 0]
        if seed0:
            x = seed0[0]["naive_target_ppl"]
            rel = abs(x - b) / b if b else float("inf")
            checks.append({
                "check": "T3 pre-attack (seed 0) == T1 B",
                "ok": rel <= CONSISTENCY_TOL,
                "detail": f"{x:.6f} vs {b:.6f} (rel {rel:.2e})",
            })

    if a.get("t0"):
        checks.append({"check": "T0 gate", "ok": bool(a["t0"]["gate_pass"]),
                       "detail": "reproduction + identity splice + key accounting"})
    return checks


def verdicts(arms: dict) -> dict:
    """The experiment-level reading, across arms."""
    rows = []
    for k, a in arms.items():
        t1 = a.get("t1")
        if not t1:
            continue
        h = t1["hybrids"][C.HEADLINE_HEAD_SET]
        rows.append({
            "arm": k, "label": t1["label"],
            "static_block_ratio": t1["static_block_ratio"],
            "block_nats": t1["block_nats"],
            "frac_block_H_head": h["H_head"]["frac_block"],
            "frac_block_H_trunk": h["H_trunk"]["frac_block"],
            "frac_block_A_head": h["A_head"]["frac_block"],
            "frac_block_A_trunk": h["A_trunk"]["frac_block"],
            "benign_drift_H_head": h["H_head"]["benign_drift"],
            "benign_drift_H_trunk": h["H_trunk"]["benign_drift"],
            "frac_params_from_head": h["H_head"]["frac_params_from_head"],
            "call": t1["headline"]["call"],
            "reason": t1["headline"]["reason"],
        })
    calls = [r["call"] for r in rows]
    heads = [r["frac_block_H_head"] for r in rows if r["frac_block_H_head"] is not None]
    overall = "insufficient data"
    if calls:
        uniq = set(calls)
        if len(uniq) == 1:
            overall = f"{calls[0]} on all {len(calls)} arms"
        else:
            overall = "mixed: " + ", ".join(f"{c}x{calls.count(c)}" for c in sorted(uniq))
    return {
        "per_arm": rows,
        "overall_call": overall,
        "frac_block_H_head_range": ([min(heads), max(heads)] if heads else None),
        "frac_block_H_head_mean": (sum(heads) / len(heads) if heads else None),
        "dose_response_flat": (bool(max(heads) - min(heads) <= 0.15) if len(heads) > 1 else None),
    }


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def style_axis(ax, ygrid: bool = True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(NEUTRAL)
    ax.spines["bottom"].set_color(NEUTRAL)
    ax.tick_params(colors=NEUTRAL, labelsize=8)
    if ygrid:
        ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _short(label: str) -> str:
    return label.split("—")[0].strip()


def main_figure(arms: dict, out_path: Path) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    keys = [k for k in C.ARM_ORDER if k in arms and arms[k].get("t1")]
    if not keys:
        return ""
    labels = [_short(arms[k]["t1"]["label"]) for k in keys]
    h1 = {k: arms[k]["t1"]["hybrids"][C.HEADLINE_HEAD_SET] for k in keys}

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.4))
    fig.suptitle("Experiment 5 — is the task-block a readout edit or a representational one?",
                 fontsize=12, y=0.98)

    # A. frac_block, the headline. Two series -> legend present; controls drawn as
    # neutral markers so they read as reference, not as a third category.
    ax = axes[0][0]
    x = np.arange(len(keys))
    w = 0.36
    for i, name in enumerate(("H_head", "H_trunk")):
        vals = [(h1[k][name]["frac_block"] or 0.0) for k in keys]
        ax.bar(x + (i - 0.5) * w, vals, w * 0.92, color=SERIES_COLOR[name],
               label={"H_head": r"$\theta_0$ trunk + B head",
                      "H_trunk": r"B trunk + $\theta_0$ head"}[name])
    for name, mk in (("A_head", "o"), ("A_trunk", "s")):
        vals = [h1[k][name]["frac_block"] for k in keys]
        ax.plot(x, [v if v is not None else np.nan for v in vals], mk, color=NEUTRAL,
                markersize=5, linestyle="none", label=f"A' control ({name})")
    ax.axhline(1.0, color=NEUTRAL, linewidth=1.0, linestyle="--")
    ax.axhline(0.0, color=NEUTRAL, linewidth=0.8)
    ax.text(len(keys) - 0.4, 1.0, "B", fontsize=8, color=NEUTRAL, va="bottom", ha="right")
    ax.text(len(keys) - 0.4, 0.0, r"$\theta_0$", fontsize=8, color=NEUTRAL, va="bottom", ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=18, ha="right")
    ax.set_ylabel("fraction of the block reproduced (log space)", fontsize=9)
    ax.set_title("A. Which half carries the block?", fontsize=10, loc="left")
    ax.legend(fontsize=7.5, frameon=False, ncol=2)
    style_axis(ax)

    # B. Selectivity — the falsifier. A hybrid that blocks F but wrecks benign is broken.
    ax = axes[0][1]
    for i, name in enumerate(("H_head", "H_trunk")):
        vals = [(h1[k][name]["benign_drift"] or 0.0) for k in keys]
        ax.bar(x + (i - 0.5) * w, vals, w * 0.92, color=SERIES_COLOR[name],
               label={"H_head": r"$\theta_0$ trunk + B head",
                      "H_trunk": r"B trunk + $\theta_0$ head"}[name])
    b_drift = [arms[k]["t1"]["donors"]["B"]["benign"] / arms[k]["t1"]["donors"]["theta0"]["benign"] - 1
               for k in keys]
    ax.plot(x, b_drift, "D", color=NEUTRAL, markersize=5, linestyle="none",
            label="B itself (selective)")
    ax.axhspan(-0.10, 0.10, color=GRID, alpha=0.55, zorder=0)
    ax.text(-0.45, 0.10, "selectivity budget", fontsize=7.5, color=NEUTRAL, va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=18, ha="right")
    ax.set_ylabel("benign PPPL drift vs $\\theta_0$", fontsize=9)
    ax.set_title("B. Is the hybrid selective, or just broken?", fontsize=10, loc="left")
    ax.legend(fontsize=7.5, frameon=False)
    style_axis(ax)

    # C. Head-only alpha path — dose response in the head alone.
    ax = axes[1][0]
    any_alpha = False
    for k in keys:
        ap = arms[k]["t1"].get("alpha_path")
        if not ap:
            continue
        any_alpha = True
        ax.plot(ap["alphas"], ap["F_ppl"], "-o", color=ARM_COLOR[k], markersize=4,
                linewidth=2, label=_short(arms[k]["t1"]["label"]))
    if any_alpha:
        ax.set_yscale("log")
        ax.set_xlabel(r"$\alpha$   ($\theta_0$ head $\rightarrow$ B head, trunk fixed at $\theta_0$)",
                      fontsize=9)
        ax.set_ylabel("F pseudo-perplexity", fontsize=9)
        ax.legend(fontsize=7.5, frameon=False)
    else:
        ax.text(0.5, 0.5, "alpha path not run", ha="center", va="center",
                fontsize=9, color=NEUTRAL, transform=ax.transAxes)
    ax.set_title("C. Interpolating the head alone", fontsize=10, loc="left")
    style_axis(ax)

    # D. Dose-response: does the readout fraction depend on how hard the block is?
    ax = axes[1][1]
    for k in keys:
        r = arms[k]["t1"]["static_block_ratio"]
        v = h1[k]["H_head"]["frac_block"]
        if r is None or v is None:
            continue
        ax.plot([r], [v], "o", color=ARM_COLOR[k], markersize=9)
        ax.annotate(_short(arms[k]["t1"]["label"]), (r, v), textcoords="offset points",
                    xytext=(8, 4), fontsize=7.5, color=NEUTRAL)
    ax.set_xscale("log")
    ax.axhline(1.0, color=NEUTRAL, linewidth=1.0, linestyle="--")
    ax.axhline(0.0, color=NEUTRAL, linewidth=0.8)
    ax.set_xlabel("static block, B / $\\theta_0$ F-PPPL", fontsize=9)
    ax.set_ylabel(r"frac_block($\theta_0$ trunk + B head)", fontsize=9)
    ax.set_title("D. Flat across two orders of magnitude of block strength?",
                 fontsize=10, loc="left")
    style_axis(ax)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return str(out_path)


def ladder_figure(arms: dict, out_path: Path) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = [k for k in C.ARM_ORDER if k in arms and arms[k].get("t2")]
    if not keys:
        return ""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    fig.suptitle("Experiment 5, T2 — how deep does the block go?", fontsize=12)

    for ax, direction, title in (
        (axes[0], "forward", r"Forward: $\theta_0$ + B's head + B's last $k$ layers"),
        (axes[1], "reverse", r"Reverse: B + $\theta_0$'s head + $\theta_0$'s last $k$ layers"),
    ):
        for k in keys:
            rows = arms[k]["t2"]["ladders"].get(direction) or []
            xs = [r["k"] for r in rows]
            ys = [r["frac_block"] for r in rows]
            ax.plot(xs, ys, "-o", color=ARM_COLOR[k], markersize=4.5, linewidth=2,
                    label=_short(arms[k]["t2"]["label"]))
        ax.axhline(1.0, color=NEUTRAL, linewidth=1.0, linestyle="--")
        ax.axhline(0.0, color=NEUTRAL, linewidth=0.8)
        ax.set_xlabel("$k$ encoder layers transferred (from the top)", fontsize=9)
        ax.set_ylabel("fraction of the block reproduced", fontsize=9)
        ax.set_title(title, fontsize=10, loc="left")
        style_axis(ax)
    axes[0].legend(fontsize=7.5, frameon=False)

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return str(out_path)


def t3_figure(arms: dict, out_path: Path) -> str:
    """Small multiples — one panel per arm, so 15 series never share one axes."""
    import statistics

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = [k for k in C.ARM_ORDER if k in arms and arms[k].get("t3")]
    if not keys:
        return ""
    fig, axes = plt.subplots(1, len(keys), figsize=(3.1 * len(keys), 4.0), squeeze=False)
    fig.suptitle("Experiment 5, T3 — does relearning only the head reverse the block?",
                 fontsize=12)

    for ax, k in zip(axes[0], keys):
        t3 = arms[k]["t3"]
        budgets = t3["budgets"]
        for mode in t3["modes"]:
            ys = []
            for n in budgets:
                v = [r["best_target_ppl"] for r in t3["rows"]
                     if r["mode"] == mode and r["budget_n"] == n]
                ys.append(statistics.median(v) if v else float("nan"))
            ax.plot(budgets, ys, "-o", color=SERIES_COLOR[mode], markersize=5,
                    linewidth=2, label=mode.replace("_", " "))
        base = t3.get("full_ft_baseline_from_sweep") or {}
        ys = [base.get(str(n)) for n in budgets]
        if any(v is not None for v in ys):
            ax.plot(budgets, [v if v is not None else float("nan") for v in ys], "--s",
                    color=SERIES_COLOR["full_ft"], markersize=5, linewidth=2,
                    label="full fine-tune (sweep)")
        t1 = arms[k].get("t1")
        if t1:
            ax.axhline(t1["donors"]["theta0"]["F"], color=NEUTRAL, linewidth=1.0,
                       linestyle=":")
            ax.text(budgets[0], t1["donors"]["theta0"]["F"], r" unblocked $\theta_0$",
                    fontsize=7.5, color=NEUTRAL, va="bottom")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("F examples (n)", fontsize=9)
        ax.set_title(_short(t3["label"]), fontsize=9.5, loc="left")
        style_axis(ax)
    axes[0][0].set_ylabel("best F pseudo-perplexity after attack", fontsize=9)
    axes[0][0].legend(fontsize=7.5, frameon=False)

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return str(out_path)


# --------------------------------------------------------------------------- #
# Write-up
# --------------------------------------------------------------------------- #
def _f(v, spec="+.3f"):
    return "n/a" if v is None or (isinstance(v, float) and not math.isfinite(v)) else format(v, spec)


def results_md(arms: dict, v: dict, checks: dict, figs: list) -> str:
    L = ["# Experiment 5 — readout or representation?", "",
         "*Generated by `aggregate.py`; nothing here is typed by hand.*", "",
         f"**Overall: {v['overall_call']}.**", ""]

    if v["frac_block_H_head_range"]:
        lo, hi = v["frac_block_H_head_range"]
        L += [f"Across {len(v['per_arm'])} arms, the head alone "
              f"(`H1`, {v['per_arm'][0]['frac_params_from_head']:.2%} of parameters) reproduces "
              f"**{lo:+.1%} to {hi:+.1%}** of the block.", ""]

    L += ["## Headline (head set H1)", "",
          "| arm | static block | H_head | H_trunk | A_head | A_trunk | benign drift (H_head) | call |",
          "|---|---|---|---|---|---|---|---|"]
    for r in v["per_arm"]:
        L.append(f"| {r['label']} | {r['static_block_ratio']:.0f}× | "
                 f"**{_f(r['frac_block_H_head'])}** | **{_f(r['frac_block_H_trunk'])}** | "
                 f"{_f(r['frac_block_A_head'])} | {_f(r['frac_block_A_trunk'])} | "
                 f"{_f(r['benign_drift_H_head'])} | {r['call']} |")
    L += ["",
          "`frac_block` is the fraction of the block reproduced **in log space**: 0 at θ₀, 1 at B. "
          "`A_head`/`A_trunk` are the same splices performed with the compute-matched control A′, "
          "which carries no block — they must sit at ~0, or splicing itself is damaging and the "
          "B numbers are confounded.", ""]

    # Head sets
    L += ["## The head-set ladder", "",
          "The output projection is tied to the input embedding, so \"where the head ends\" is "
          "ambiguous; all three cuts are reported rather than one being chosen.", "",
          "| arm | " + " | ".join(f"{hs} H_head / H_trunk" for hs in C.HEAD_SET_ORDER) + " |",
          "|---|" + "---|" * len(C.HEAD_SET_ORDER)]
    for k, a in arms.items():
        if not a.get("t1"):
            continue
        cells = []
        for hs in C.HEAD_SET_ORDER:
            h = a["t1"]["hybrids"][hs]
            cells.append(f"{_f(h['H_head']['frac_block'])} / {_f(h['H_trunk']['frac_block'])}")
        L.append(f"| {_short(a['t1']['label'])} | " + " | ".join(cells) + " |")
    for hs in C.HEAD_SET_ORDER:
        L.append(f"")
        L.append(f"* **{hs}** — {C.HEAD_SET_DOC[hs]}")
    L.append("")

    # Displacement cross-check
    L += ["## Cross-check against weight-space displacement", "",
          "| arm | B `lm_head` rel ‖Δθ‖ | B total rel ‖Δθ‖ | frac_block(H_head) |",
          "|---|---|---|---|"]
    for k, a in arms.items():
        t1 = a.get("t1")
        if not t1:
            continue
        d = t1.get("lm_head_displacement") or {}
        h = t1["hybrids"][C.HEADLINE_HEAD_SET]["H_head"]["frac_block"]
        L.append(f"| {_short(t1['label'])} | {_f(d.get('B_lm_head_rel'), '.4f')} | "
                 f"{_f(d.get('B_total_rel'), '.4f')} | {_f(h)} |")
    L += ["", "A large functional fraction carried by a group whose *norm* displacement is at or "
              "below the model average is the sharpest form of the finding: the block is not where "
              "the weights moved most.", ""]

    # T2
    if any(a.get("t2") for a in arms.values()):
        L += ["## T2 — depth ladder", "",
              "| arm | direction | " + " | ".join(f"k={k}" for k in C.K_LADDER) + " |",
              "|---|---|" + "---|" * len(C.K_LADDER)]
        for k, a in arms.items():
            if not a.get("t2"):
                continue
            for direction in ("forward", "reverse"):
                rows = {r["k"]: r["frac_block"] for r in (a["t2"]["ladders"].get(direction) or [])}
                if not rows:
                    continue
                cells = " | ".join(_f(rows.get(kk)) for kk in C.K_LADDER)
                L.append(f"| {_short(a['t2']['label'])} | {direction} | {cells} |")
        L.append("")
        for k, a in arms.items():
            res = (a.get("t2") or {}).get("encoder_residual")
            if res:
                L.append(f"* {_short(a['t2']['label'])}: transferring every encoder layer plus the "
                         f"head reaches {res['frac_block']:+.3f} of the block, leaving "
                         f"{res['missing_frac_block']:+.3f} outside the encoder stack "
                         f"(input embedding / final LayerNorm).")
        L.append("")

    # T3
    if any(a.get("t3") for a in arms.values()):
        import statistics

        L += ["## T3 — relearning only the head", "",
              "Median `best_target_ppl` over seeds. `full fine-tune` is read from the main "
              "sweep on disk, under the identical protocol.", "",
              "| arm | mode | " + " | ".join(f"n={n}" for n in C.T3_BUDGETS) + " |",
              "|---|---|" + "---|" * len(C.T3_BUDGETS)]
        for k, a in arms.items():
            t3 = a.get("t3")
            if not t3:
                continue
            for mode in t3["modes"]:
                cells = []
                for n in C.T3_BUDGETS:
                    vals = [r["best_target_ppl"] for r in t3["rows"]
                            if r["mode"] == mode and r["budget_n"] == n]
                    cells.append(f"{statistics.median(vals):.3f}" if vals else "n/a")
                L.append(f"| {_short(t3['label'])} | {mode} | " + " | ".join(cells) + " |")
            base = t3.get("full_ft_baseline_from_sweep") or {}
            cells = [(f"{base[str(n)]:.3f}" if base.get(str(n)) is not None else "n/a")
                     for n in C.T3_BUDGETS]
            L.append(f"| {_short(t3['label'])} | full fine-tune (sweep) | " + " | ".join(cells) + " |")
        L.append("")

    # Consistency
    L += ["## Consistency checks", "", "| arm | check | result | detail |", "|---|---|---|---|"]
    for k, cs in checks.items():
        for c in cs:
            L.append(f"| {k} | {c['check']} | {'PASS' if c['ok'] else '**FAIL**'} | {c['detail']} |")
    L.append("")

    if figs:
        L += ["## Figures", ""] + [f"![{Path(f).stem}]({Path(f).name})" for f in figs if f] + [""]

    L += ["## Caveats", "",
          "* Under `H1`/`H2` the output projection is tied to the input embedding and therefore "
          "follows the **trunk** donor; `H3` is the variant that unties it. Read the ladder, not "
          "one cut.",
          "* The splice is licensed by the LMC result (θ₀ and θ_B in one basin, monotone path, "
          "~0.3-nat barrier). It would not be meaningful between models in different basins.",
          "* `pfam_legacy` and `pfam_domain` have known data/headroom problems that make their "
          "durability multiplier unreliable. Only their **static** block is used here, which is "
          "well defined.",
          ""]
    return "\n".join(L)


def main() -> int:
    args = parse_args()
    C.banner("AGGREGATE — experiment 5")
    arms = {k: collect(C.get_arm(k)) for k in args.arms}
    for k, a in arms.items():
        print(f"  {k}: " + ", ".join(f"{p}={'ok' if ok else 'MISSING'}"
                                     for p, ok in a["have"].items()))

    checks = {k: consistency(a) for k, a in arms.items()}
    failed = [(k, c) for k, cs in checks.items() for c in cs if not c["ok"]]
    for k, c in failed:
        print(f"  CONSISTENCY FAIL [{k}] {c['check']}: {c['detail']}")

    v = verdicts(arms)
    figs = [
        main_figure(arms, C.RESULTS / "fig_readout_vs_representation.png"),
        ladder_figure(arms, C.RESULTS / "fig_depth_ladder.png"),
        t3_figure(arms, C.RESULTS / "fig_t3_partial_attack.png"),
    ]
    figs = [f for f in figs if f]
    for f in figs:
        print(f"  wrote {f}")

    C.RESULTS.mkdir(parents=True, exist_ok=True)
    C.save_json({"arms": arms, "verdicts": v, "consistency": checks,
                 "head_set_docs": C.HEAD_SET_DOC, "figures": figs},
                C.RESULTS / "summary.json")
    (C.RESULTS / "RESULTS.md").write_text(results_md(arms, v, checks, figs))
    print(f"  wrote {C.RESULTS / 'RESULTS.md'}")

    C.banner("VERDICT")
    print(f"  {v['overall_call']}")
    if v["frac_block_H_head_range"]:
        lo, hi = v["frac_block_H_head_range"]
        print(f"  frac_block(theta_0 trunk + B head) across arms: {lo:+.3f} .. {hi:+.3f}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
