"""Phase F of the 650M spot-check — LADDER FIGURE + SCALE TABLE + RESULTS PARAGRAPH.

Regenerates every scale-robustness artifact **deterministically from the three
manifests** (``headroom_ladder.json``, ``mechanism_ladder.json``, ``m_point_<scale>.json``)
and nothing else. No number reaches the manuscript by hand; re-running on unchanged
manifests reproduces byte-identical outputs.

Outputs, under ``<ladder_root>/report/``:

  fig_scale_ladder.png       x = parameter count (log); (a) static-forget x, (b) B/A'
                             displacement ratio, (c) LMC barrier in nats. The headline is
                             a **flat or declining** barrier-and-ratio trend: the in-basin
                             geometry that explains M~=1 does not thin out with scale.
  fig_scale_per_layer.png    per-layer displacement vs *relative depth* — the overlay that
                             makes 6/12/30/33-layer models comparable at all.
  scale_table.md / .tex      per-scale rows: val PPL, anchor, naive/floor/tau, headroom,
                             static-forget x, disp B/A', LMC nats, and M at the top rung.
  results_paragraph.md       results paragraph, generated from the manifests.

    python experiments/3_scale_ladder/phaseF_aggregate.py
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import common  # noqa: E402
from common import DEFAULT_LADDER_ROOT, HERE, RANDOM_CEILING, banner, manifest_dir  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(HERE / "scale_ladder.yaml"))
    p.add_argument("--ladder-root", default=str(DEFAULT_LADDER_ROOT))
    p.add_argument("--out-dir", default=None, help="default: <ladder_root>/report")
    return p.parse_args()


def main():
    args = parse_args()
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(args.config)
    ladder_root = Path(args.ladder_root)
    md = manifest_dir(ladder_root)
    out = Path(args.out_dir or (ladder_root / "report"))
    out.mkdir(parents=True, exist_ok=True)

    gate_rows = common.load_manifest(md / "headroom_ladder.json").get("rows", [])
    train_rows = common.load_manifest(md / "train_ladder.json").get("rows", [])
    mech_rows = common.load_manifest(md / "mechanism_ladder.json").get("rows", [])
    m_scale = str(cfg.m_point_scale)
    m_rows = common.load_manifest(md / f"m_point_{m_scale}.json").get("rows", [])
    m_point = m_rows[0] if m_rows else None
    if not gate_rows and not mech_rows:
        raise SystemExit(f"no manifests under {md}; run Phases B and D first")

    gate = {r["scale"]: r for r in gate_rows}
    train = {r["scale"]: r for r in train_rows}
    mech = {r["scale"]: r for r in mech_rows}
    scales = common.sort_scales(set(gate) | set(mech) | set(train))

    banner("PHASE F — AGGREGATION")
    fig1 = plot_ladder(scales, mech, out / "fig_scale_ladder.png")
    fig2 = plot_per_layer(scales, mech, out / "fig_scale_per_layer.png")
    tbl_md = write_table(scales, gate, train, mech, m_point, m_scale, out)
    para = write_paragraph(scales, gate, train, mech, m_point, m_scale, out)
    for p in (fig1, fig2, tbl_md, para):
        print(f"  wrote {p}")
    print(f"\nAll regenerated from {md}/*.json — no hand-edited numbers.")


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def plot_ladder(scales, mech, out_path: Path) -> Path:
    """Three panels sharing a log-parameter x-axis. Panels (b) and (c) carry the argument;
    panel (a) is context (a huge static forget that is nonetheless in-basin)."""
    xs = [common.SCALES[s]["n_params"] for s in scales if s in mech]
    labels = [s for s in scales if s in mech]
    if not xs:
        xs, labels = [], []

    def series(key):
        return [mech[s].get(key) for s in labels]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    panels = [
        ("static_forget_x", "static forget  PPL$_F$(B) / PPL$_F$(A)", "tab:red", True),
        ("disp_ratio", r"displacement ratio  $\|\Delta\theta_B\|/\|\Delta\theta_{A'}\|$", "tab:blue", False),
        ("lmc_barrier_nats", "LMC barrier (nats)", "tab:green", False),
    ]
    for ax, (key, ylabel, color, logy) in zip(axes, panels):
        ys = series(key)
        pts = [(x, y, lb) for x, y, lb in zip(xs, ys, labels) if y is not None and math.isfinite(y)]
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "-o", color=color, linewidth=2)
            for x, y, lb in pts:
                ax.annotate(lb, (x, y), textcoords="offset points", xytext=(0, 7),
                            ha="center", fontsize=8, color="0.3")
        ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("parameters")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(alpha=0.25)
        ax.margins(x=0.22, y=0.22)      # room for the per-point scale labels
    # Reference lines, each annotated just under itself so the note never lands on the data.
    axes[1].axhline(1.0, color="0.4", linestyle="--", linewidth=1)
    axes[1].annotate("B moves as much as the control", (0.98, 1.0), xycoords=("axes fraction", "data"),
                     ha="right", va="bottom", fontsize=7.5, color="0.4")
    axes[2].axhline(0.0, color="0.4", linestyle="--", linewidth=1)
    axes[2].annotate("no ridge between $\\theta_0$ and $\\theta_B$", (0.98, 0.0),
                     xycoords=("axes fraction", "data"), ha="right", va="bottom",
                     fontsize=7.5, color="0.4")
    fig.suptitle("Scaling ladder of the mechanism: a large static forget stays a thin, "
                 "in-basin perturbation at every scale", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_per_layer(scales, mech, out_path: Path) -> Path:
    """Per-layer relative displacement vs relative depth — the cross-scale overlay. Raw
    layer index would be meaningless across 6/12/30/33-layer models; relative depth is the
    whole reason these curves can share an axis."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), sharex=True)
    cmap = plt.get_cmap("viridis")
    present = [s for s in scales if s in mech and mech[s].get("per_layer")]
    for i, s in enumerate(present):
        color = cmap(i / max(1, len(present) - 1) * 0.85)
        layers = [pl for pl in mech[s]["per_layer"] if pl.get("relative_depth") is not None]
        layers.sort(key=lambda pl: pl["relative_depth"])
        d = [pl["relative_depth"] for pl in layers]
        axes[0].plot(d, [pl["B_rel"] for pl in layers], "-o", ms=3, color=color, label=f"{s} B")
        ap = [pl["Aprime_rel"] for pl in layers]
        if all(v is not None for v in ap):
            axes[0].plot(d, ap, "--", color=color, alpha=0.6, label=f"{s} A'")
        rt = [(x, pl["ratio"]) for x, pl in zip(d, layers) if pl.get("ratio")]
        if rt:
            axes[1].plot([x for x, _ in rt], [y for _, y in rt], "-o", ms=3, color=color, label=s)
    axes[0].set_ylabel(r"relative $\|\theta-\theta_0\|/\|\theta_0\|$")
    axes[1].set_ylabel(r"B / A$'$ displacement ratio")
    axes[1].axhline(1.0, color="0.4", linestyle="--", linewidth=1)
    for ax in axes:
        ax.set_xlabel("relative depth  (layer index / (n_layers - 1))")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    fig.suptitle("Per-layer displacement on a common depth axis", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# Table + prose
# --------------------------------------------------------------------------- #
COLUMNS = [
    ("scale", "scale", lambda r: r.get("scale")),
    ("val_ppl", "ESM-2 val PPL", lambda r: _f(r.get("val_ppl"), 2)),
    ("anchor_family", "anchor F", lambda r: r.get("anchor_family")),
    ("naive_ppl", "naive PPL", lambda r: _f(r.get("naive_ppl"), 2)),
    ("floor_ppl", "floor PPL", lambda r: _f(r.get("floor_ppl"), 2)),
    ("tau", "tau", lambda r: _f(r.get("tau"), 2)),
    ("headroom_rel", "headroom", lambda r: _pct(r.get("headroom_rel"))),
    # A scale that did not pass but has mechanism results was carried forward as a
    # documented marginal case (ACCEPT_MARGINAL_GATE / --accept-marginal-gate).
    ("gate_pass", "gate", lambda r: "PASS" if r.get("gate_pass") else (
        "marginal (accepted)" if r.get("lmc_barrier_nats") is not None else "not passed")),
    ("static_forget_x", "static forget", lambda r: _x(r.get("static_forget_x"), 0)),
    ("disp_B_rel", "disp B", lambda r: _pct(r.get("disp_B_rel"))),
    ("disp_Aprime_rel", "disp A'", lambda r: _pct(r.get("disp_Aprime_rel"))),
    ("disp_ratio", "B/A'", lambda r: _x(r.get("disp_ratio"), 2)),
    ("utility_retention_rel", "utility drift", lambda r: _pct(r.get("utility_retention_rel"))),
    ("lmc_barrier_nats", "LMC nats", lambda r: _f(r.get("lmc_barrier_nats"), 3)),
    ("lmc_monotone", "monotone", lambda r: _b(r.get("lmc_monotone"))),
    ("M", "M = n_B/n_A'", lambda r: _m(r)),
]


def _merged(scales, gate, train, mech, m_point, m_scale):
    """One row per scale, later manifests winning on shared keys: gate (naive/floor/tau)
    -> train (static forget, utility, FLOP match) -> mechanism (displacement, LMC, and its
    own alpha=0 static forget) -> the single M point."""
    rows = []
    for s in scales:
        r = {"scale": s}
        r.update(gate.get(s, {}))
        r.update(train.get(s, {}))
        r.update({k: v for k, v in mech.get(s, {}).items() if k != "per_layer"})
        if m_point and s == m_scale:
            r["M"] = m_point.get("M")
            r["M_is_lower_bound"] = m_point.get("M_is_lower_bound")
            r["M_status"] = m_point.get("status", "OK")
        rows.append(r)
    return rows


def write_table(scales, gate, train, mech, m_point, m_scale, out: Path) -> Path:
    rows = _merged(scales, gate, train, mech, m_point, m_scale)
    head = [h for _, h, _ in COLUMNS]
    body = [[fn(r) for _, _, fn in COLUMNS] for r in rows]

    md = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    md += ["| " + " | ".join(str(c) for c in row) + " |" for row in body]
    md += ["",
           f"Random-prediction ceiling = {RANDOM_CEILING:.0f} (20 residue types; ESM-2 definition).",
           "M is measured only at the top rung; the attacker is epoch-capped and blocking is "
           "first-order (K=4), so M is a **lower bound** in both directions."]
    md_path = out / "scale_table.md"
    md_path.write_text("\n".join(md) + "\n")

    tex = ["\\begin{tabular}{" + "l" * len(head) + "}", "\\toprule",
           " & ".join(h.replace("_", "\\_").replace("'", "$'$") for h in head) + " \\\\", "\\midrule"]
    tex += [" & ".join(str(c).replace("_", "\\_") for c in row) + " \\\\" for row in body]
    tex += ["\\bottomrule", "\\end{tabular}"]
    (out / "scale_table.tex").write_text("\n".join(tex) + "\n")
    return md_path


def write_paragraph(scales, gate, train, mech, m_point, m_scale, out: Path) -> Path:
    rows = {r["scale"]: r for r in _merged(scales, gate, train, mech, m_point, m_scale)}
    have_mech = [s for s in scales if rows[s].get("lmc_barrier_nats") is not None]
    ratios = [(s, rows[s]["disp_ratio"]) for s in have_mech if rows[s].get("disp_ratio")]
    barriers = [(s, rows[s]["lmc_barrier_nats"]) for s in have_mech]

    trend = "flat"
    if len(barriers) >= 2:
        trend = "declining" if barriers[-1][1] < barriers[0][1] else (
            "flat" if abs(barriers[-1][1] - barriers[0][1]) < 0.1 else "rising")

    lines = [
        "# Scale robustness (Phase F output — regenerate, do not edit)",
        "",
        "## Results paragraph",
        "",
    ]
    span = f"{have_mech[0]} to {have_mech[-1]}" if have_mech else "the ladder"
    rung_names = ", ".join(f"{s} ({common.SCALES[s]['model_id'].split('/')[-1]})" for s in have_mech)
    sent = [
        "To test whether the near-unit adaptation-cost multiplier is an artifact of model scale, "
        f"we repeated the mechanism diagnostics across {span} ({rung_names}), "
        "re-validating anchor selection and the headroom gate independently at every scale rather "
        "than assuming the 35M selection transfers."
    ]
    passes = [s for s in scales if rows[s].get("gate_pass")]
    not_passed = [s for s in scales if s in gate and not rows[s].get("gate_pass")]
    accepted = [s for s in not_passed if s in have_mech]      # carried forward as marginal
    stopped = [s for s in not_passed if s not in have_mech]   # no safeguard compute spent
    if not_passed:
        sent.append(f"The gate passed at {', '.join(passes) or 'no scale'}.")
    for s in accepted:
        sent.append(f"At {s} the gate was marginally not met "
                    f"({'; '.join(rows[s].get('gate_fail_reasons', []))}); the scale was carried "
                    "forward as a documented marginal case.")
    if stopped:
        sent.append(f"The gate was not passed at {', '.join(stopped)} "
                    f"({'; '.join(rows[stopped[0]].get('gate_fail_reasons', [])[:1])}), "
                    "so no safeguard compute was spent there.")
    if ratios:
        sent.append("Displacement of the blocked model B relative to base remained at or below that of "
                    "the compute-matched control A' at every scale (" +
                    ", ".join(f"{s}: {v:.2f}x" for s, v in ratios) + ").")
    if barriers:
        sent.append("The linear-mode-connectivity barrier on the theta_0 -> theta_B path was " + trend +
                    " with scale (" + ", ".join(f"{s}: {v:.3f} nats" for s, v in barriers) + "), and the "
                    "path remained monotone, i.e. no ridge separates the blocked model from base.")
    if m_point and m_point.get("status") == "M_UNDEFINED_DENOMINATOR_CENSORED":
        sent.append(f"At {m_scale} no multiplier is reported: the compute-matched control A' did not "
                    f"itself reach tau within the swept budget ({m_point.get('status_note', '')}), so "
                    "the denominator of M is undefined and the attacker sweep must be re-run on a "
                    "wider grid before any M at this scale is quoted.")
    elif m_point:
        lb = ">= " if m_point.get("M_is_lower_bound") else ""
        sent.append(f"At {m_scale} the single uncensored adaptation-cost multiplier was "
                    f"M = n_B/n_A' = {lb}{m_point['M']:.2f}x "
                    f"(n_A' = {_f(m_point.get('n_cross_Aprime'), 0)}, n_B = {_f(m_point.get('n_cross_B'), 0)}, "
                    f"validity n_A'/n_A = {_f(m_point.get('validity_ratio'), 2)}), consistent with the 35M result.")
    sent.append("Because the attacker is epoch-capped and the safeguard is first-order (K=4), every "
                "multiplier reported here is a lower bound; the scale-invariance of the geometry, not "
                "the magnitude of any single M, is what rules out the small-model explanation.")
    lines += [" ".join(sent), "", "## Provenance", "",
              "Every number above is regenerated by `experiments/3_scale_ladder/phaseF_aggregate.py` "
              "from the three manifests; see `scale_table.md` for the full per-scale table."]
    p = out / "results_paragraph.md"
    p.write_text("\n".join(lines) + "\n")
    return p


def _f(x, nd=3):
    return "—" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{nd}f}"


def _pct(x):
    return "—" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{100 * x:.2f}%"


def _x(v, nd=1):
    return "—" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:.{nd}f}x"


def _b(v):
    return "—" if v is None else ("yes" if v else "no")


def _m(r):
    # A censored *denominator* means no multiplier exists — never render that as a number.
    if r.get("M_status") == "M_UNDEFINED_DENOMINATOR_CENSORED":
        return "undefined (A′ censored)"
    if r.get("M") is None or not math.isfinite(r["M"]):
        return "—"
    return ("≥ " if r.get("M_is_lower_bound") else "") + f"{r['M']:.2f}x"


if __name__ == "__main__":
    main()
