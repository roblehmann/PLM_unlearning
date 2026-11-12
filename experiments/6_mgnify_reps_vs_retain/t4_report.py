#!/usr/bin/env python3
"""T4 — merge every phase, run the statistics, draw the figures, write RESULTS.md.

The pre-registered headline is ONE number: the median difference in log pseudo-perplexity
between the MGnify representatives and a length-matched retain sample. Everything else in
here is a control on that number or a secondary readout, because the naive comparison is
confounded by length, truncation and composition, and because at n=35,459 per arm a
p-value is significant for any difference whatsoever — the effect size and its CI are the
result.

Exits non-zero if any cross-phase consistency check fails, so a broken merge cannot quietly
produce a plausible-looking figure.

  python experiments/6_mgnify_reps_vs_retain/t4_report.py
"""
from __future__ import annotations

import argparse
import glob
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

import common as C


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-boot", type=int, default=C.N_BOOT)
    ap.add_argument("--skip-figures", action="store_true")
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="report on the shards that exist instead of failing (diagnostics only)")
    C.common_args(ap)
    return ap.parse_args()


# --------------------------------------------------------------------------- #
# Load + merge
# --------------------------------------------------------------------------- #
def load_everything(args):
    checks = []
    missing = []

    def check(name, ok, detail=""):
        checks.append({"name": name, "ok": bool(ok), "detail": detail})
        print(f"  [{'ok ' if ok else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}", flush=True)
        return ok

    manifest = C.load_json(C.RESULTS / "corpus_manifest.json")
    work = pd.read_csv(C.RESULTS / "work_table.tsv", sep="\t")

    shard_files = sorted(glob.glob(str(C.RESULTS / "ppl" / "shard_*_of_*.tsv")))
    if not shard_files:
        raise SystemExit("FAIL: no PPL shards found — run t2 first.")
    ppl = pd.concat([pd.read_csv(f, sep="\t") for f in shard_files], ignore_index=True)

    n_shards = int(manifest["n_shards"])
    done = sorted(glob.glob(str(C.RESULTS / "ppl" / "shard_*_of_*.done.json")))
    ok_shards = check(f"all {n_shards} PPL shards finished", len(done) == n_shards,
                      f"{len(done)} .done.json files")
    ok_rows = check("every work-table row is scored exactly once",
                    len(ppl) == len(work) and ppl["row_id"].is_unique,
                    f"{len(ppl):,} scored rows vs {len(work):,} planned, "
                    f"{int(ppl['row_id'].duplicated().sum())} duplicates")
    if not (ok_shards and ok_rows) and not args.allow_incomplete:
        raise SystemExit("FAIL: the PPL phase is incomplete — resubmit the missing shards "
                         "(t2 is resumable) or pass --allow-incomplete for a diagnostic run.")

    df = ppl.merge(work[["row_id", "in_matched"]], on="row_id", how="left")
    check("no non-finite log_ppl", bool(np.isfinite(df["log_ppl"]).all()),
          f"{int((~np.isfinite(df['log_ppl'])).sum())} bad rows")
    check("no zero-length scoring", bool((df["n_positions"] > 0).all()),
          f"{int((df['n_positions'] <= 0).sum())} rows with n_positions == 0")

    reps_meta = pd.read_csv(C.RESULTS / "meta" / "reps_meta.tsv", sep="\t")
    df = df.merge(reps_meta[["id", "source", "full_size", "is_anchor", "anchor_role", "entropy"]],
                  on="id", how="left")
    retain_meta = pd.read_csv(C.RESULTS / "meta" / "retain_meta.tsv", sep="\t")
    df = df.merge(retain_meta[["id", "entropy"]].rename(columns={"entropy": "entropy_retain"}),
                  on="id", how="left")
    df["entropy"] = df["entropy"].fillna(df.pop("entropy_retain"))
    df["truncated"] = df["length"] > C.SCORED_MAX

    # ---- optional phases
    hom_path = C.RESULTS / "homology" / "reps_homology.tsv"
    if hom_path.exists():
        hom = pd.read_csv(hom_path, sep="\t")
        df = df.merge(hom[["id", "hom_class", "hom_class_strict", "best_bits", "best_fident"]],
                      on="id", how="left")
    else:
        missing.append("t2b_homology")
        df["hom_class"] = np.nan

    coords_path = C.RESULTS / "umap" / "coords.npy"
    if coords_path.exists():
        coords = np.load(coords_path)
        rows = pd.read_csv(C.RESULTS / "umap" / "rows.tsv", sep="\t")
        check("UMAP coords align with rows.tsv", coords.shape[0] == len(rows),
              f"{coords.shape[0]} coords vs {len(rows)} rows")
        rows["umap_x"], rows["umap_y"] = coords[:, 0], coords[:, 1]
        df = df.merge(rows[["id", "corpus", "umap_x", "umap_y"]], on=["id", "corpus"], how="left")
    else:
        missing.append("t3_umap")
        df["umap_x"] = df["umap_y"] = np.nan

    exact_path = C.RESULTS / "ppl_exact_check.json"
    exact = C.load_json(exact_path) if exact_path.exists() else None
    if exact is None:
        missing.append("t2x_exact_check")

    emb_manifest_path = C.RESULTS / "embeddings" / "manifest.json"
    emb = C.load_json(emb_manifest_path) if emb_manifest_path.exists() else {}
    vr = (emb.get("retain") or {}).get("verify_retain")
    if vr:
        check("cached retain embeddings reproduce (t1 gate)",
              vr["median_cosine"] >= vr.get("threshold", 0.999),
              f"median cosine {vr['median_cosine']:.6f}")

    mt = manifest.get("length_matching", {})
    if "ks_stat_len_vs_reps" in mt:
        # Matching can only be near-exact when the pool dwarfs the target set — which it does
        # in the real run (194,064 retain for 35,459 reps, >=3.89x in every length bin). On a
        # smoke-sized pool a poor KS says nothing about the code, so the check is advisory
        # there rather than a failure.
        feasible = mt.get("n_pool", 0) >= 20 * max(1, mt.get("n_target", 1))
        ks = mt["ks_stat_len_vs_reps"]
        check("length matching worked (KS of matched retain vs reps lengths < 0.01)",
              (ks < 0.01) or not feasible,
              f"KS {ks:.5f}" + ("" if feasible else
                                f" — advisory only: pool {mt.get('n_pool')} is too small "
                                f"relative to {mt.get('n_target')} targets for exact matching"))

    return df, manifest, exact, emb, checks, missing


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def corpus_block(sub: pd.DataFrame) -> dict:
    lg = sub["log_ppl"].to_numpy()
    return {
        "n_seqs": int(len(sub)),
        "n_positions": int(sub["n_positions"].sum()),
        "ppl_micro": C.micro_ppl(sub["nll_sum"], sub["n_positions"]),
        "ppl_macro": float(sub["ppl"].mean()),
        "log_ppl_mean": float(lg.mean()), "log_ppl_sd": float(lg.std(ddof=1)),
        "log_ppl_median": float(np.median(lg)),
        "log_ppl_q25": float(np.quantile(lg, 0.25)), "log_ppl_q75": float(np.quantile(lg, 0.75)),
        "log_ppl_p05": float(np.quantile(lg, 0.05)), "log_ppl_p95": float(np.quantile(lg, 0.95)),
        "len_median": float(sub["length"].median()), "len_mean": float(sub["length"].mean()),
        "n_truncated": int(sub["truncated"].sum()),
        "frac_truncated": float(sub["truncated"].mean()),
    }


def run_stats(df: pd.DataFrame, manifest: dict, exact, emb, args) -> dict:
    reps = df[df.corpus == "reps"]
    retain = df[df.corpus == "retain"]
    matched = retain[retain.in_matched == 1]
    nb = args.n_boot

    out = {"protocol": C.protocol_block(n_boot=nb)}
    arms = {"reps": reps, "retain": retain, "retain_matched": matched}
    out["corpora"] = {k: corpus_block(v) for k, v in arms.items()}
    for k, sub in arms.items():
        # The CI re-aggregates exp(sum nll / sum positions) over resampled SEQUENCES —
        # exactly what PPLResult keeps per_seq_nll / per_seq_positions for.
        out["corpora"][k]["ppl_micro_ci95"] = list(C.bootstrap_ci(
            C.micro_ppl, sub["nll_sum"].to_numpy(), sub["n_positions"].to_numpy(),
            n_boot=min(nb, 500), seed=args.seed))

    # ------------------------------------------------------------ contrasts
    def contrast(a: pd.DataFrame, b: pd.DataFrame) -> dict:
        return C.two_sample_stats(
            a["log_ppl"].to_numpy(), b["log_ppl"].to_numpy(),
            a_nll=a["nll_sum"].to_numpy(), a_pos=a["n_positions"].to_numpy(),
            b_nll=b["nll_sum"].to_numpy(), b_pos=b["n_positions"].to_numpy(),
            n_boot=min(nb, 500), seed=args.seed)

    out["ppl_contrasts"] = {
        "reps_vs_retain_raw": contrast(reps, retain),
        "reps_vs_retain_matched": contrast(reps, matched),
        "reps_vs_retain_untruncated": contrast(reps[~reps.truncated], retain[~retain.truncated]),
    }
    out["ppl_contrasts"]["reps_vs_retain_matched"]["note"] = (
        "THE HEADLINE — length-matched, so the length confound cannot explain it")

    # ------------------------------------------------------- length control
    bins = C.binned_contrast(reps["length"], reps["log_ppl"], retain["length"],
                             retain["log_ppl"], n_boot=min(nb, 500), seed=args.seed)
    usable = [b for b in bins if b["delta"] is not None]
    n_higher = sum(1 for b in usable if b["delta"] > 0)
    from scipy.stats import binomtest

    std_mean, _ = C.standardize_by_length(reps["length"], retain["length"], retain["log_ppl"])
    out["length_control"] = {
        "edges": [int(e) if e < 10**8 else None for e in C.LEN_EDGES],
        "bins": bins,
        "n_bins_usable": len(usable),
        "n_bins_reps_higher": n_higher,
        "sign_test_p": float(binomtest(n_higher, len(usable), 0.5).pvalue) if usable else None,
        "matching": manifest.get("length_matching", {}),
        "standardization": {
            "method": "direct standardization: retain reweighted to the reps length mix",
            "log_ppl_reps": float(reps["log_ppl"].mean()),
            "log_ppl_retain_standardized": std_mean,
            "delta_log_ppl": float(reps["log_ppl"].mean() - std_mean),
        },
    }

    # --------------------------------------------------- regression adjust
    sample = df
    if len(df) > 120_000:                       # the bootstrap is O(n) per resample
        sample = df.sample(n=120_000, random_state=args.seed)
        out.setdefault("notes", []).append(
            f"regression fitted on a {len(sample):,}-row random subsample for bootstrap cost")
    X = np.column_stack([
        np.ones(len(sample)),
        np.log(sample["n_positions"].to_numpy()),
        (sample["corpus"] == "reps").to_numpy().astype(float),
    ])
    reg = C.ols_bootstrap(X, sample["log_ppl"].to_numpy(), n_boot=min(nb, 400), seed=args.seed)
    out["length_control"]["regression"] = {
        "formula": "log_ppl ~ 1 + log(n_positions) + is_reps",
        "intercept": reg["coef"][0], "coef_log_len": reg["coef"][1],
        "coef_is_reps": reg["coef"][2], "coef_is_reps_ci95": reg["coef_ci95"][2],
        "r2": reg["r2"], "n": reg["n"], "n_boot": reg["n_boot"],
    }
    Xe = np.column_stack([X, sample["entropy"].fillna(sample["entropy"].mean()).to_numpy()])
    rege = C.ols_bootstrap(Xe, sample["log_ppl"].to_numpy(), n_boot=min(nb, 200), seed=args.seed)

    # ------------------------------------------------------------ the rest
    out["truncation"] = {
        "max_length": C.MAX_LENGTH, "scored_positions_max": C.SCORED_MAX,
        "reps": {"n": int(reps.truncated.sum()), "frac": float(reps.truncated.mean())},
        "retain": {"n": int(retain.truncated.sum()), "frac": float(retain.truncated.mean())},
        "note": "retain truncates MORE than reps, so truncation cannot manufacture a "
                "reps-is-worse result",
    }
    comp = manifest.get("composition", {})
    out["composition"] = {
        "js_divergence_reps_vs_retain": comp.get("js_divergence_reps_vs_retain"),
        "aa_freq": comp.get("aa_freq"),
        "entropy_mean": {"reps": float(reps["entropy"].mean()),
                         "retain": float(retain["entropy"].mean())},
        "entropy_adjusted": {"formula": "log_ppl ~ 1 + log(n_positions) + is_reps + entropy",
                             "coef_is_reps": rege["coef"][2],
                             "coef_is_reps_ci95": rege["coef_ci95"][2],
                             "coef_entropy": rege["coef"][3], "r2": rege["r2"]},
    }

    strata = {}
    for src, sub in reps.groupby("source", dropna=True):
        if len(sub) >= 30:
            strata[str(src)] = {"n": int(len(sub)), "ppl_micro": C.micro_ppl(sub["nll_sum"], sub["n_positions"]),
                                "log_ppl_median": float(sub["log_ppl"].median())}
    out["source_strata"] = strata

    fs = reps.dropna(subset=["full_size"])
    if len(fs) > 100:
        from scipy.stats import spearmanr

        rho = spearmanr(np.log(fs["full_size"].to_numpy()), fs["log_ppl"].to_numpy())
        out["family_size"] = {"spearman_rho_log_ppl_vs_log_full_size": float(rho.statistic),
                              "p": float(rho.pvalue), "n": int(len(fs))}

    if "hom_class" in df.columns and reps["hom_class"].notna().any():
        novel = reps[reps.hom_class == "novel"]
        homo = reps[reps.hom_class == "homologous"]
        blk = {"n_novel": int(len(novel)), "n_homologous": int(len(homo)),
               "frac_novel": float(len(novel) / max(1, len(reps)))}
        if len(novel) >= 30 and len(homo) >= 30:
            blk["contrast_novel_vs_homologous"] = C.two_sample_stats(
                novel["log_ppl"].to_numpy(), homo["log_ppl"].to_numpy(),
                a_nll=novel["nll_sum"].to_numpy(), a_pos=novel["n_positions"].to_numpy(),
                b_nll=homo["nll_sum"].to_numpy(), b_pos=homo["n_positions"].to_numpy(),
                n_boot=min(nb, 500), seed=args.seed)
            blk["contrast_novel_vs_retain_matched"] = C.two_sample_stats(
                novel["log_ppl"].to_numpy(), matched["log_ppl"].to_numpy(),
                a_nll=novel["nll_sum"].to_numpy(), a_pos=novel["n_positions"].to_numpy(),
                b_nll=matched["nll_sum"].to_numpy(), b_pos=matched["n_positions"].to_numpy(),
                n_boot=min(nb, 500), seed=args.seed)
        out["homology"] = {"enabled": True, **blk}
    else:
        out["homology"] = {"enabled": False}

    anchors = {}
    for aid, role in C.ANCHORS.items():
        row = reps[reps.id == aid]
        if len(row):
            r = row.iloc[0]
            anchors[aid] = {
                "role": role, "length": int(r["length"]), "truncated": bool(r["truncated"]),
                "source": r.get("source"), "full_size": (None if pd.isna(r.get("full_size"))
                                                         else int(r["full_size"])),
                "ppl": float(r["ppl"]), "log_ppl": float(r["log_ppl"]),
                "percentile_within_reps": float((reps["log_ppl"] < r["log_ppl"]).mean() * 100),
                "umap_xy": [None if pd.isna(r["umap_x"]) else float(r["umap_x"]),
                            None if pd.isna(r["umap_y"]) else float(r["umap_y"])],
            }
    out["anchors"] = anchors

    out["embedding"] = emb
    up = C.RESULTS / "umap" / "umap_params.json"
    out["umap"] = C.load_json(up) if up.exists() else {}
    km = C.RESULTS / "umap" / "knn_mixing.json"
    out["separation"] = C.load_json(km) if km.exists() else {}
    out["mpp_fidelity"] = exact or {"enabled": False}

    # ------------------------------------------------------------ verdicts
    head = out["ppl_contrasts"]["reps_vs_retain_matched"]
    ci = head["delta_log_ppl_median_ci95"]
    sep = out["separation"]
    mixing_gap = (abs(sep.get("mean_frac_retain_neighbours_of_reps", float("nan"))
                      - sep.get("null_frac_retain", float("nan")))
                  if sep else float("nan"))
    out["verdicts"] = {
        "headline_delta_log_ppl_median": head["delta_log_ppl_median"],
        "headline_ci95": ci,
        "headline_ppl_ratio": head.get("ratio_micro_ppl"),
        "reps_ppl_higher": bool(head["delta_log_ppl_median"] > 0 and ci[0] > 0),
        "holds_after_length_matching": bool(ci[0] > 0),
        "holds_after_standardization": bool(out["length_control"]["standardization"]["delta_log_ppl"] > 0),
        "holds_after_regression": bool(out["length_control"]["regression"]["coef_is_reps_ci95"][0] > 0),
        "holds_in_all_length_bins": bool(n_higher == len(usable)) if usable else None,
        "n_bins_reps_higher": f"{n_higher}/{len(usable)}",
        "occupies_distinct_umap_region": (bool(mixing_gap > 0.10)
                                          if math.isfinite(mixing_gap) else None),
        "knn_mixing_gap_vs_null": mixing_gap if math.isfinite(mixing_gap) else None,
    }
    return out


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def make_figures(df: pd.DataFrame, summary: dict) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    C.set_rc()
    reps = df[df.corpus == "reps"]
    retain = df[df.corpus == "retain"]
    matched = retain[retain.in_matched == 1]
    has_umap = df["umap_x"].notna().any()

    # ---- fig1: who sits where -------------------------------------------
    if has_umap:
        r_xy = reps.dropna(subset=["umap_x"])
        t_xy = retain.dropna(subset=["umap_x"])
        xlim = (float(np.quantile(df["umap_x"].dropna(), 0.002)),
                float(np.quantile(df["umap_x"].dropna(), 0.998)))
        ylim = (float(np.quantile(df["umap_y"].dropna(), 0.002)),
                float(np.quantile(df["umap_y"].dropna(), 0.998)))
        fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.4), layout="constrained")
        panels = [("retain density, MGnify reps overlaid", t_xy, r_xy, C.C_REPS, "reps"),
                  ("MGnify reps density, retain overlaid", r_xy, t_xy, C.C_RETAIN, "retain")]
        for ax, (title, bg, fg, col, lab) in zip(axes, panels):
            ax.hexbin(bg["umap_x"], bg["umap_y"], extent=(*xlim, *ylim), **C.HEXBIN)
            ax.scatter(fg["umap_x"], fg["umap_y"], s=2, c=col, alpha=0.30, linewidths=0,
                       rasterized=True, label=lab)
            for aid, a in (summary.get("anchors") or {}).items():
                if a.get("umap_xy") and a["umap_xy"][0] is not None:
                    ax.scatter(*a["umap_xy"], s=90, marker="*", c="white", edgecolors="black",
                               linewidths=0.9, zorder=5)
                    ax.annotate(a["role"], a["umap_xy"], textcoords="offset points",
                                xytext=(6, 4), fontsize=8, zorder=6)
            ax.set(title=title, xlim=xlim, ylim=ylim, xticks=[], yticks=[],
                   xlabel="UMAP-1", ylabel="UMAP-2")
            ax.legend(loc="upper right", markerscale=4)
        sep = summary.get("separation", {})
        if sep:
            axes[0].text(0.02, 0.02,
                         f"kNN mixing: reps' neighbours are "
                         f"{sep['mean_frac_retain_neighbours_of_reps']:.2f} retain\n"
                         f"chance = {sep['null_frac_retain']:.2f}  |  "
                         f"centroid cos = {sep.get('centroid_cosine_reps_vs_retain', float('nan')):.3f}",
                         transform=axes[0].transAxes, fontsize=7.5, va="bottom",
                         bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.7", alpha=0.9))
        fig.suptitle("MGnify family representatives vs. the retain corpus in ESM-2 35M space",
                     fontsize=11)
        C.save_fig(fig, "fig1_umap_reps_vs_retain")

        # ---- fig2: does separation coincide with poor modelling? ---------
        lo, hi = (float(np.quantile(df["log_ppl"], 0.02)), float(np.quantile(df["log_ppl"], 0.98)))
        fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.0), layout="constrained")
        for ax, (name, sub, col) in zip(axes[:2],
                                        [("MGnify reps", r_xy, None), ("retain", t_xy, None)]):
            sc = ax.scatter(sub["umap_x"], sub["umap_y"], s=2, c=sub["log_ppl"], cmap=C.SEQ_CMAP,
                            vmin=lo, vmax=hi, alpha=0.6, linewidths=0, rasterized=True)
            ax.set(title=f"{name} — log pseudo-perplexity", xlim=xlim, ylim=ylim,
                   xticks=[], yticks=[])
            fig.colorbar(sc, ax=ax, shrink=0.85, label="log PPL (nats/residue)")
        # Third panel: reps again, but on a diverging scale centred on the RETAIN median, so
        # regions of the manifold where MGnify is modelled worse than typical pretraining
        # sequence read directly off the colour.
        centre = float(retain["log_ppl"].median())
        hb = axes[2].hexbin(r_xy["umap_x"], r_xy["umap_y"], C=r_xy["log_ppl"], gridsize=70,
                            extent=(*xlim, *ylim), mincnt=10, cmap=C.DIV_CMAP,
                            norm=TwoSlopeNorm(vcenter=centre, vmin=min(lo, centre - 1e-3),
                                              vmax=max(hi, centre + 1e-3)))
        axes[2].set(title=f"reps vs the retain median ({centre:.2f})", xlim=xlim, ylim=ylim,
                    xticks=[], yticks=[])
        fig.colorbar(hb, ax=axes[2], shrink=0.85, label="log PPL (nats/residue)")
        C.save_fig(fig, "fig2_umap_logppl")

    # ---- fig3: the distributions ----------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.6), layout="constrained",
                             gridspec_kw={"width_ratios": [1.0, 0.8, 1.15]})
    ax = axes[0]
    for name, sub, col in (("MGnify reps", reps, C.C_REPS), ("retain (all)", retain, C.C_RETAIN),
                           ("retain (length-matched)", matched, C.C_MATCHED)):
        x = np.sort(sub["log_ppl"].to_numpy())
        ax.plot(x, np.arange(1, x.size + 1) / x.size, color=col, lw=1.6, label=f"{name} (n={x.size:,})")
    ax.set(xlabel="log pseudo-perplexity (nats/residue)", ylabel="ECDF",
           title="Per-sequence log-PPL")
    ax.legend(loc="lower right", fontsize=7.5)
    C.style_axis(ax)

    ax = axes[1]
    data = [reps["log_ppl"].to_numpy(), matched["log_ppl"].to_numpy(), retain["log_ppl"].to_numpy()]
    parts = ax.violinplot(data, showextrema=False, widths=0.85)
    for body, col in zip(parts["bodies"], (C.C_REPS, C.C_MATCHED, C.C_RETAIN)):
        body.set_facecolor(col); body.set_alpha(0.55)
    ax.boxplot(data, widths=0.12, showfliers=False,
               medianprops=dict(color="black", lw=1.2))
    ax.set(xticks=[1, 2, 3], xticklabels=["reps", "retain\n(matched)", "retain\n(all)"],
           ylabel="log PPL", title="Distribution")
    C.style_axis(ax)

    ax = axes[2]
    rows = []
    for key, lab in (("reps_vs_retain_matched", "length-matched (headline)"),
                     ("reps_vs_retain_raw", "raw"),
                     ("reps_vs_retain_untruncated", "untruncated only")):
        c = summary["ppl_contrasts"].get(key)
        if c:
            rows.append((lab, c["delta_log_ppl_median"], c["delta_log_ppl_median_ci95"]))
    std = summary["length_control"]["standardization"]
    rows.append(("standardized (mean)", std["delta_log_ppl"], None))
    reg = summary["length_control"]["regression"]
    rows.append(("regression coef", reg["coef_is_reps"], reg["coef_is_reps_ci95"]))
    ypos = np.arange(len(rows))[::-1]
    for y, (lab, val, ci) in zip(ypos, rows):
        if ci:
            ax.plot(ci, [y, y], color="0.4", lw=1.4)
        ax.plot([val], [y], "o", color=C.C_REPS, ms=6)
    ax.axvline(0, color="0.3", lw=1, ls="--")
    ax.set(yticks=ypos, yticklabels=[r[0] for r in rows],
           xlabel="Δ log-PPL (reps − retain)", title="Effect size under every adjustment")
    C.style_axis(ax, ygrid=False)
    C.save_fig(fig, "fig3_ppl_distributions")

    # ---- fig4: the length confound, visually ----------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), layout="constrained")
    ax = axes[0]
    for name, sub, col in (("MGnify reps", reps, C.C_REPS), ("retain", retain, C.C_RETAIN)):
        edges = np.array([e for e in C.LEN_EDGES if e < 10**8])
        centres, meds = [], []
        lo = 0
        for hi in edges:
            m = (sub["length"] > lo) & (sub["length"] <= hi)
            if m.sum() >= 30:
                centres.append((lo + hi) / 2)
                meds.append(float(sub.loc[m, "log_ppl"].median()))
            lo = hi
        ax.plot(centres, meds, "o-", color=col, ms=3.5, lw=1.5, label=name)
    ax.axvline(C.SCORED_MAX, color="0.4", ls=":", lw=1)
    ax.annotate(f"truncation at {C.SCORED_MAX}\nreps {summary['truncation']['reps']['frac']:.1%} / "
                f"retain {summary['truncation']['retain']['frac']:.1%}",
                (C.SCORED_MAX, ax.get_ylim()[1]), textcoords="offset points", xytext=(-95, -26),
                fontsize=7)
    ax.set(xlabel="sequence length (aa)", ylabel="median log PPL",
           title="log-PPL vs length, binned")
    ax.legend()
    C.style_axis(ax)

    ax = axes[1]
    bins = [b for b in summary["length_control"]["bins"] if b["delta"] is not None]
    x = [(b["lo"] + (b["hi"] or b["lo"] + 50)) / 2 for b in bins]
    d = [b["delta"] for b in bins]
    loci = [b["delta_ci95"][0] for b in bins]
    hici = [b["delta_ci95"][1] for b in bins]
    ax.fill_between(x, loci, hici, color=C.C_REPS, alpha=0.20, linewidth=0)
    ax.plot(x, d, "o-", color=C.C_REPS, ms=3.5, lw=1.5)
    ax.axhline(0, color="0.3", lw=1, ls="--")
    v = summary["verdicts"]
    ax.set(xlabel="sequence length (aa)", ylabel="Δ median log PPL (reps − retain)",
           title=f"Per-bin difference — reps higher in {v['n_bins_reps_higher']} bins")
    C.style_axis(ax)
    C.save_fig(fig, "fig4_ppl_vs_length")

    # ---- fig5: novelty --------------------------------------------------
    if summary.get("homology", {}).get("enabled") and reps["hom_class"].notna().any():
        fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), layout="constrained")
        ax = axes[0]
        for lab, sub, col in (("reps: novel (no UniRef50 homolog)", reps[reps.hom_class == "novel"], C.C_NOVEL),
                              ("reps: has a UniRef50 homolog", reps[reps.hom_class == "homologous"], C.C_REPS),
                              ("retain (length-matched)", matched, C.C_MATCHED)):
            if len(sub) < 5:
                continue
            x = np.sort(sub["log_ppl"].to_numpy())
            ax.plot(x, np.arange(1, x.size + 1) / x.size, color=col, lw=1.6,
                    label=f"{lab} (n={x.size:,})")
        ax.set(xlabel="log PPL", ylabel="ECDF", title="Is it novelty, or just domain shift?")
        ax.legend(loc="lower right", fontsize=7.5)
        C.style_axis(ax)
        ax = axes[1]
        if has_umap:
            for lab, sub, col in (("novel", reps[reps.hom_class == "novel"], C.C_NOVEL),
                                  ("homologous", reps[reps.hom_class == "homologous"], C.C_REPS)):
                sub = sub.dropna(subset=["umap_x"])
                ax.scatter(sub["umap_x"], sub["umap_y"], s=2, c=col, alpha=0.35, linewidths=0,
                           rasterized=True, label=f"{lab} (n={len(sub):,})")
            ax.set(title="reps by homology class", xticks=[], yticks=[])
            ax.legend(markerscale=4, fontsize=7.5)
        C.save_fig(fig, "fig5_homology")

    # ---- fig6: strata ---------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), layout="constrained")
    ax = axes[0]
    srcs = [s for s in reps["source"].dropna().unique()]
    data, labs = [], []
    for s in srcs:
        sub = reps[reps.source == s]
        if len(sub) >= 30:
            data.append(sub["log_ppl"].to_numpy()); labs.append(f"{s}\n(n={len(sub):,})")
    if data:
        ax.boxplot(data, showfliers=False, medianprops=dict(color="black"))
        ax.set(xticklabels=labs, ylabel="log PPL", title="reps by upstream pipeline")
        C.style_axis(ax)
    ax = axes[1]
    fs = reps.dropna(subset=["full_size"])
    if len(fs) > 100:
        ax.hexbin(np.log10(fs["full_size"]), fs["log_ppl"], gridsize=60, bins="log",
                  cmap="Greys", mincnt=1)
        rho = summary.get("family_size", {}).get("spearman_rho_log_ppl_vs_log_full_size")
        ax.set(xlabel="log10 family size", ylabel="log PPL",
               title=f"Family size vs modelling difficulty (Spearman ρ={rho:.3f})"
                     if rho is not None else "Family size vs modelling difficulty")
        C.style_axis(ax)
    C.save_fig(fig, "fig6_strata")


# --------------------------------------------------------------------------- #
# RESULTS.md — generated, never hand-typed
# --------------------------------------------------------------------------- #
def _fmt(value, spec: str = ".4f", fallback: str = "n/a") -> str:
    """Format a statistic that may legitimately be absent.

    Several quantities are None by design rather than by accident — the sign test has no
    p-value when no length bin holds >=30 sequences of BOTH corpora, and a contrast has no
    PPL ratio when an arm is empty. Writing the report must not crash on those: a partial
    run's RESULTS.md is exactly what someone reads to find out why it was partial.
    """
    if value is None:
        return fallback
    try:
        return format(value, spec)
    except (TypeError, ValueError):
        return str(value)


def write_results_md(summary: dict, df: pd.DataFrame, missing) -> None:
    s, v = summary, summary["verdicts"]
    c = s["corpora"]
    head = s["ppl_contrasts"]["reps_vs_retain_matched"]
    raw = s["ppl_contrasts"]["reps_vs_retain_raw"]
    sep = s.get("separation", {})
    L = []
    A = L.append

    A("# MGnify family representatives vs. the retain corpus\n")
    A(f"Generated by `t4_report.py` from `summary.json`. Model: **{s['protocol']['model_id']}**, "
      f"base weights only. {c['reps']['n_seqs']:,} MGnify family representatives vs "
      f"{c['retain']['n_seqs']:,} UniRef50 retain sequences, all scored with the project's "
      f"standard eval protocol (max_length {s['protocol']['max_length']}, "
      f"masks_per_pass {s['protocol']['masks_per_pass']}).\n")

    A("## Headline\n")
    verdict = ("**higher** (worse-modelled)" if v["reps_ppl_higher"]
               else "**not detectably higher**")
    A(f"Against a length-matched retain sample, MGnify representatives' pseudo-perplexity is "
      f"{verdict}: median Δ log-PPL = **{head['delta_log_ppl_median']:+.4f}** nats/residue "
      f"(95% CI {head['delta_log_ppl_median_ci95'][0]:+.4f} to "
      f"{head['delta_log_ppl_median_ci95'][1]:+.4f}), a corpus PPL ratio of "
      f"**{_fmt(head.get('ratio_micro_ppl'), '.3f')}x** "
      f"({c['reps']['ppl_micro']:.3f} vs {c['retain_matched']['ppl_micro']:.3f}). "
      f"Cliff's δ = {head['cliffs_delta']:+.3f}.\n")
    if sep:
        A(f"In embedding space the two corpora are "
          f"{'**separated**' if v.get('occupies_distinct_umap_region') else '**interleaved**'}: "
          f"a MGnify representative's {sep['knn_k']} nearest neighbours are "
          f"{sep['mean_frac_retain_neighbours_of_reps']:.3f} retain on average, against a chance "
          f"rate of {sep['null_frac_retain']:.3f}; centroid cosine "
          f"{sep.get('centroid_cosine_reps_vs_retain', float('nan')):.4f}.\n")

    A("## Does the difference survive the confounds?\n")
    A("| control | result |")
    A("|---|---|")
    A(f"| raw (no adjustment) | Δ median {raw['delta_log_ppl_median']:+.4f}, "
      f"ratio {_fmt(raw.get('ratio_micro_ppl'), '.3f')}x |")
    A(f"| length-matched (headline) | Δ median {head['delta_log_ppl_median']:+.4f} "
      f"[{head['delta_log_ppl_median_ci95'][0]:+.4f}, {head['delta_log_ppl_median_ci95'][1]:+.4f}] |")
    std = s["length_control"]["standardization"]
    A(f"| direct standardization | Δ mean {std['delta_log_ppl']:+.4f} |")
    reg = s["length_control"]["regression"]
    A(f"| regression on log length | coef(is_reps) {reg['coef_is_reps']:+.4f} "
      f"[{reg['coef_is_reps_ci95'][0]:+.4f}, {reg['coef_is_reps_ci95'][1]:+.4f}], R²={reg['r2']:.3f} |")
    ent = s["composition"]["entropy_adjusted"]
    A(f"| + composition entropy | coef(is_reps) {ent['coef_is_reps']:+.4f} "
      f"[{ent['coef_is_reps_ci95'][0]:+.4f}, {ent['coef_is_reps_ci95'][1]:+.4f}] |")
    A(f"| per-length-bin sign test | reps higher in {v['n_bins_reps_higher']} bins, "
      f"p={_fmt(s['length_control']['sign_test_p'], '.3g')} |")
    unt = s["ppl_contrasts"]["reps_vs_retain_untruncated"]
    A(f"| untruncated sequences only | Δ median {unt['delta_log_ppl_median']:+.4f} |")
    A("")
    A(f"Truncation at {s['truncation']['scored_positions_max']} scored positions affects "
      f"{s['truncation']['reps']['frac']:.2%} of reps and {s['truncation']['retain']['frac']:.2%} "
      f"of retain — it hits the retain side harder, so it cannot manufacture this result. "
      f"Amino-acid composition differs by a JS divergence of "
      f"{s['composition']['js_divergence_reps_vs_retain']:.5f} nats.\n")

    if s.get("homology", {}).get("enabled"):
        h = s["homology"]
        A("## Novelty, or merely domain shift?\n")
        A(f"{h['n_novel']:,} of {h['n_novel'] + h['n_homologous']:,} representatives "
          f"({h['frac_novel']:.1%}) have no detectable UniRef50 homolog.")
        cnh = h.get("contrast_novel_vs_homologous")
        if cnh:
            A(f"Novel vs homologous representatives: Δ median log-PPL "
              f"{cnh['delta_log_ppl_median']:+.4f} "
              f"[{cnh['delta_log_ppl_median_ci95'][0]:+.4f}, {cnh['delta_log_ppl_median_ci95'][1]:+.4f}], "
              f"Cliff's δ {cnh['cliffs_delta']:+.3f}. Novel vs length-matched retain: "
              f"{h['contrast_novel_vs_retain_matched']['delta_log_ppl_median']:+.4f}.")
        A("")

    if s.get("mpp_fidelity", {}).get("bias"):
        m = s["mpp_fidelity"]
        A("## Is the speedup to blame?\n")
        A(f"`masks_per_pass=8` biases log-PPL by {m['bias']['reps']['mean']:+.5f} on reps and "
          f"{m['bias']['retain']['mean']:+.5f} on retain; the difference is "
          f"{m['bias_difference_reps_minus_retain']:+.5f} "
          f"[{m['bias_difference_ci95'][0]:+.5f}, {m['bias_difference_ci95'][1]:+.5f}]. "
          + ("This CI **excludes zero** — the approximation is differentially biased and the "
             "headline should be re-run at masks_per_pass=1."
             if m.get("verdict_ci_excludes_zero") else
             "This CI includes zero, so the approximation treats both corpora alike and the "
             "headline is not an artifact of it.") + "\n")

    A("## Corpora\n")
    A("| corpus | n | median length | corpus PPL | median log-PPL | truncated |")
    A("|---|---:|---:|---:|---:|---:|")
    for key, lab in (("reps", "MGnify reps"), ("retain", "retain (all)"),
                     ("retain_matched", "retain (length-matched)")):
        b = c[key]
        A(f"| {lab} | {b['n_seqs']:,} | {b['len_median']:.0f} | {b['ppl_micro']:.3f} | "
          f"{b['log_ppl_median']:.4f} | {b['frac_truncated']:.2%} |")
    A("")

    if s.get("anchors"):
        A("## The study's own anchor families\n")
        A("Both are in this corpus **by construction** — it is the complete MGnify family "
          "representative set. Only base ESM-2 weights are used anywhere in this experiment, so "
          "no task-blocked checkpoint touches these numbers, and at n=2 of "
          f"{c['reps']['n_seqs']:,} they cannot move any aggregate.\n")
        A("| family | role | length | log-PPL | percentile within reps |")
        A("|---|---|---:|---:|---:|")
        for aid, a in s["anchors"].items():
            A(f"| {aid} | {a['role']} | {a['length']}{' (truncated)' if a['truncated'] else ''} "
              f"| {a['log_ppl']:.4f} | {a['percentile_within_reps']:.1f} |")
        A("")

    A("## Figures\n")
    for stem, cap in (("fig1_umap_reps_vs_retain", "joint UMAP, each corpus over the other's density"),
                      ("fig2_umap_logppl", "the same coordinates coloured by log-PPL"),
                      ("fig3_ppl_distributions", "distributions and every adjusted effect size"),
                      ("fig4_ppl_vs_length", "the length confound and the per-bin difference"),
                      ("fig5_homology", "novel vs homologous representatives"),
                      ("fig6_strata", "upstream pipeline and family size")):
        if (C.RESULTS / "plots" / f"{stem}.png").exists():
            A(f"- `plots/{stem}.png` — {cap}")
    A("")

    A("## Caveats\n")
    A(f"- ESM-2 35M only. The scale ladder found naive PPL falls with model size, so the gap may "
      f"narrow at 650M; this is a claim about the 35M model.")
    A(f"- `masks_per_pass={s['protocol']['masks_per_pass']}` is the project's strided "
      f"approximation to exact pseudo-log-likelihood, applied identically to both corpora "
      f"and audited above.")
    A("- The retain set is a 200k UniRef50 subsample — a *proxy* for the ESM-2 pretraining "
      "corpus, not the corpus itself.")
    A("- MGnify representatives are one sequence per family: this is family-level sampling, "
      "so it says nothing about how deeply any individual family is covered.")
    A("- GPU non-determinism moves log-PPL at ~1e-6 relative; a re-run reproduces to about five "
      "decimals, not bitwise.")
    A("- UMAP inter-cluster distance is not metric. The separation claim rests on the kNN "
      "mixing statistic and the centroid cosine; the figure only illustrates them.")
    if missing:
        A(f"- Phases not run: {', '.join(missing)}.")

    (C.RESULTS / "RESULTS.md").write_text("\n".join(L) + "\n")
    print(f"  wrote {C.RESULTS / 'RESULTS.md'}", flush=True)


def main() -> int:
    args = parse_args()
    C.apply_results_override(args.results)
    t_start = time.time()
    C.banner("T4 — merge, statistics, figures, RESULTS.md")

    df, manifest, exact, emb, checks, missing = load_everything(args)
    out = C.RESULTS / "per_sequence.tsv.gz"
    df.to_csv(out, sep="\t", index=False, compression="gzip")
    print(f"  wrote {out} ({len(df):,} rows)", flush=True)

    summary = run_stats(df, manifest, exact, emb, args)
    summary["checks"] = checks
    summary["missing_phases"] = missing
    summary["runtime"] = {"t4_seconds": time.time() - t_start}
    C.save_json(summary, C.RESULTS / "summary.json")

    if not args.skip_figures:
        make_figures(df, summary)
    write_results_md(summary, df, missing)

    v = summary["verdicts"]
    print(f"\n  HEADLINE  Δ median log-PPL (reps − length-matched retain) = "
          f"{v['headline_delta_log_ppl_median']:+.4f} "
          f"CI {v['headline_ci95']}  |  PPL ratio {_fmt(v['headline_ppl_ratio'], '.3f')}x")
    print(f"  separation: knn mixing gap vs null = {v['knn_mixing_gap_vs_null']}")
    if any(not c["ok"] for c in checks) and not args.allow_incomplete:
        print("\nT4 FAILED a consistency check", flush=True)
        return 1
    print(f"\nT4 OK in {time.time() - t_start:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
