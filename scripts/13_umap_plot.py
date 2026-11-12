"""UMAP of F / F' / ladder rungs / retain in ESM-2 embedding space.

Stage 2 of 2 (stage 1 is `scripts/12_embed_corpora.py`, which caches the embeddings).

What this is for. Every "distance from F" in the project is a SEQUENCE-space number
(mmseqs / HMM identity). This puts the same corpora in the model's REPRESENTATION space,
which is an independent read on the ladder's own metadata — most pointedly on MGnify rung
6 (MGYF0000026700), labelled `band: background` while carrying mean_id_all 0.25, higher
than rung 3.

Design notes (palette validated with the dataviz skill's checker, --pairs all / --ordinal,
all checks PASS):
  retain  greyscale hexbin density underlay — 194k points would bury everything as scatter,
          but they still shape the manifold, which is the point of a dense background
  F       #D55E00 vermillion; train = filled circle, test = open ring (same entity, so the
          same hue; the marker carries the split)
  F'      #0072B2 blue square
  rungs   single-hue green ramp, light->dark by rung order, triangles
The F/rung pair sits at tritan dE 6.3, inside the floor band that is legal only WITH
secondary encoding — hence the per-group direct labels at centroids and the distinct
marker shapes. Do not drop either.

    python scripts/13_umap_plot.py
    python scripts/13_umap_plot.py --n-neighbors 50 --min-dist 0.05
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import _bootstrap  # noqa: F401

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = _bootstrap.ROOT

# --------------------------------------------------------------------------- #
# Validated palette (see module docstring)
# --------------------------------------------------------------------------- #
C_F = "#D55E00"      # vermillion — forget family F
C_FPRIME = "#0072B2"  # blue — near family F'

# Single-hue green ramps, OKLab-spaced: monotone L, adjacent dL >= 0.06, light end
# clears 2:1 on the chart surface. Indexed by rung count.
RAMP = {
    8: ["#45C396", "#26AD81", "#00966D", "#007F5C", "#00694B", "#00533B", "#003E2B", "#002B1C"],
    6: ["#45C396", "#18A57A", "#008661", "#00684B", "#004B35", "#003021"],
}

ANCHOR_TITLE = {
    "mgnify": "MGnify MGYF0000018690\n(metagenomic, headline arm)",
    "PF00106": "Pfam PF00106 / adh_short\n(clan CL0063)",
    "PF00107": "Pfam PF00107 / ADH_zinc_N\n(clan CL0063)",
}
ANCHOR_ORDER = ["mgnify", "PF00106", "PF00107"]


def ramp_for(n: int) -> List[str]:
    """Nearest validated ramp, subsampled if the rung count is unusual."""
    if n in RAMP:
        return RAMP[n]
    base = RAMP[8]
    idx = np.linspace(0, len(base) - 1, n).round().astype(int)
    return [base[i] for i in idx]


# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #
def load_shards(emb_root: str) -> Tuple[np.ndarray, List[dict]]:
    """Concatenate every cached group into one matrix + a per-row metadata list."""
    manifest = json.load(open(os.path.join(emb_root, "manifest.json")))
    mats, rows = [], []
    for key in sorted(manifest):
        meta = manifest[key]
        anchor, label = key.split("/", 1)
        npy = os.path.join(emb_root, anchor, f"{label}.npy")
        ids_path = os.path.join(emb_root, anchor, f"{label}.tsv")
        if not os.path.exists(npy):
            print(f"  WARN: manifest lists {key} but {npy} is missing — skipping")
            continue
        X = np.load(npy).astype(np.float32)
        ids = [ln.strip() for ln in open(ids_path)][1:]
        if len(ids) != len(X):
            raise SystemExit(f"{key}: {len(ids)} ids vs {len(X)} rows — cache is inconsistent")
        mats.append(X)
        for i in ids:
            rows.append({"id": i, "anchor": anchor, "group": label, "role": meta["role"],
                         "homology": meta.get("homology"), "band": meta.get("band"),
                         "rung": meta.get("rung")})
        print(f"  loaded {key:32s} n={len(X):6d} dim={X.shape[1]}")
    if not mats:
        raise SystemExit(f"no embeddings found under {emb_root} — run scripts/12_embed_corpora.py first")
    return np.concatenate(mats, axis=0), rows


# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #
def _deoverlap(labels: List[dict], xlim, ylim, min_sep: float = 0.055,
               iters: int = 400) -> None:
    """Nudge centroid labels apart in place, in axes-fraction space.

    Groups that genuinely sit on top of each other — a rung landing inside F is exactly
    the thing this figure is meant to reveal — produce colliding labels. Moving the text
    slightly keeps both readable; a leader line back to the true centroid is drawn by the
    caller whenever a label travels far enough for its position to be misread.
    """
    if len(labels) < 2:
        return
    sx = (xlim[1] - xlim[0]) or 1.0
    sy = (ylim[1] - ylim[0]) or 1.0
    pos = np.array([[(l["x"] - xlim[0]) / sx, (l["y"] - ylim[0]) / sy] for l in labels])
    for _ in range(iters):
        moved = False
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                d = pos[j] - pos[i]
                dist = float(np.hypot(*d))
                if dist >= min_sep:
                    continue
                if dist < 1e-6:
                    # Exactly coincident: break the tie deterministically.
                    d = np.array([1.0, 0.6]) * (1 if (i + j) % 2 == 0 else -1)
                    dist = float(np.hypot(*d))
                push = (min_sep - dist) / 2.0 * (d / dist)
                pos[i] -= push
                pos[j] += push
                moved = True
        if not moved:
            break
    for l, p in zip(labels, pos):
        l["lx"] = xlim[0] + float(np.clip(p[0], 0.01, 0.99)) * sx
        l["ly"] = ylim[0] + float(np.clip(p[1], 0.01, 0.99)) * sy


def draw(coords: np.ndarray, rows: List[dict], out_png: str, out_pdf: str,
         anchors: List[str]) -> None:

    anchor_of = np.array([r["anchor"] for r in rows])
    group_of = np.array([r["group"] for r in rows])
    role_of = np.array([r["role"] for r in rows])

    retain_mask = role_of == "retain"
    retain_xy = coords[retain_mask]

    # Shared limits across panels, clipped so a few stragglers don't flatten the view.
    lo = np.percentile(coords, 0.2, axis=0)
    hi = np.percentile(coords, 99.8, axis=0)
    pad = 0.04 * (hi - lo)
    xlim, ylim = (lo[0] - pad[0], hi[0] + pad[0]), (lo[1] - pad[1], hi[1] + pad[1])

    fig, axes = plt.subplots(1, len(anchors), figsize=(5.0 * len(anchors), 5.4),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes)

    label_fx = [pe.withStroke(linewidth=2.4, foreground="white")]

    for ax, anchor in zip(axes, anchors):
        # --- retain density underlay (every panel, same corpus) ---
        if len(retain_xy):
            ax.hexbin(retain_xy[:, 0], retain_xy[:, 1], gridsize=170, bins="log",
                      cmap="Greys", mincnt=1, linewidths=0, alpha=0.55,
                      extent=(*xlim, *ylim), zorder=1)

        in_anchor = anchor_of == anchor
        rung_groups = sorted({g for g, m in zip(group_of, in_anchor) if m and g.startswith("L")},
                             key=lambda g: int(g.split("_")[0][1:]))
        ramp = ramp_for(len(rung_groups))
        labels: List[dict] = []

        # --- ladder rungs, light->dark by rung order ---
        for j, g in enumerate(rung_groups):
            m = in_anchor & (group_of == g)
            if not m.any():
                continue
            ax.scatter(coords[m, 0], coords[m, 1], s=7, c=ramp[j], marker="^",
                       linewidths=0, alpha=0.75, zorder=3)
            labels.append({"x": float(np.median(coords[m, 0])),
                           "y": float(np.median(coords[m, 1])),
                           "text": g.split("_")[0], "color": ramp[j], "size": 8})

        # --- F' ---
        m = in_anchor & (role_of == "F_prime")
        if m.any():
            ax.scatter(coords[m, 0], coords[m, 1], s=7, c=C_FPRIME, marker="s",
                       linewidths=0, alpha=0.70, zorder=4)
            labels.append({"x": float(np.median(coords[m, 0])),
                           "y": float(np.median(coords[m, 1])),
                           "text": "F'", "color": C_FPRIME, "size": 10})

        # --- F: train filled, test as open rings on top ---
        m = in_anchor & (role_of == "F_train")
        if m.any():
            ax.scatter(coords[m, 0], coords[m, 1], s=8, c=C_F, marker="o",
                       linewidths=0, alpha=0.70, zorder=5)
        m_te = in_anchor & (role_of == "F_test")
        if m_te.any():
            ax.scatter(coords[m_te, 0], coords[m_te, 1], s=14, facecolors="none",
                       edgecolors=C_F, marker="o", linewidths=0.6, alpha=0.95, zorder=5)
        if m.any() or m_te.any():
            mm = m | m_te
            labels.append({"x": float(np.median(coords[mm, 0])),
                           "y": float(np.median(coords[mm, 1])),
                           "text": "F", "color": C_F, "size": 11})

        # Direct labels are load-bearing here, not decoration: the F/rung hue pair sits
        # at tritan dE 6.3, which the palette checks allow only WITH secondary encoding.
        _deoverlap(labels, xlim, ylim)
        lead = 0.012 * max(xlim[1] - xlim[0], ylim[1] - ylim[0])
        for l in labels:
            if np.hypot(l["lx"] - l["x"], l["ly"] - l["y"]) > lead:
                ax.plot([l["x"], l["lx"]], [l["y"], l["ly"]], color=l["color"],
                        lw=0.6, alpha=0.55, zorder=6, solid_capstyle="round")
            ax.text(l["lx"], l["ly"], l["text"], fontsize=l["size"], fontweight="bold",
                    color=l["color"], ha="center", va="center", zorder=7,
                    path_effects=label_fx)

        ax.set_title(ANCHOR_TITLE.get(anchor, anchor), fontsize=9.5)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#cccccc")

    axes[0].set_ylabel("UMAP-2", fontsize=9, color="#555555")
    for ax in axes:
        ax.set_xlabel("UMAP-1", fontsize=9, color="#555555")

    # Legend: identity is never colour-alone — marker shape and the direct labels repeat it.
    handles = [
        Line2D([], [], marker="o", ls="", ms=6, mfc=C_F, mec=C_F, label="F (train)"),
        Line2D([], [], marker="o", ls="", ms=6, mfc="none", mec=C_F, label="F (held-out test)"),
        Line2D([], [], marker="s", ls="", ms=6, mfc=C_FPRIME, mec=C_FPRIME, label="F' (near family)"),
        Line2D([], [], marker="^", ls="", ms=6, mfc=RAMP[8][1], mec=RAMP[8][1],
               label="ladder rungs L0→Ln (light→dark)"),
        Line2D([], [], marker="h", ls="", ms=6, mfc="#999999", mec="#999999",
               label="retain corpus, 194k (density)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8.5,
               frameon=False, bbox_to_anchor=(0.5, -0.005))

    fig.suptitle("Forget family, near family and sibling ladder in ESM-2 35M embedding space",
                 fontsize=11.5, y=0.99)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    for path in (out_png, out_pdf):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nwrote {out_png}\nwrote {out_pdf}")


def write_table(coords: np.ndarray, X: np.ndarray, rows: List[dict], out_tsv: str) -> None:
    """Per-group centroid summary — the table view behind the figure, and the
    quantitative form of the question the plot asks.

    `cos_to_F` is computed in the ORIGINAL embedding space, not on the 2-D coordinates:
    UMAP distances between well-separated clusters are not metric, so reading "how far
    is this rung from F" off the picture alone would be a mistake. The 2-D centroid
    distance is reported beside it precisely so the two can be compared.
    """
    anchor_of = np.array([r["anchor"] for r in rows])
    group_of = np.array([r["group"] for r in rows])
    role_of = np.array([r["role"] for r in rows])

    Xn = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-8, None)

    lines = ["anchor\tgroup\trole\tn\thomology\tband\tcos_to_F_centroid\tumap_dist_to_F"]
    for anchor in sorted(set(anchor_of) - {"shared"}):
        fm = (anchor_of == anchor) & (role_of == "F_train")
        if not fm.any():
            continue
        f_dir = Xn[fm].mean(axis=0)
        f_dir /= np.clip(np.linalg.norm(f_dir), 1e-8, None)
        f_xy = np.median(coords[fm], axis=0)

        groups = [g for g in dict.fromkeys(group_of[anchor_of == anchor])]
        for g in groups:
            m = (anchor_of == anchor) & (group_of == g)
            meta = next(r for r, mm in zip(rows, m) if mm)
            cen = Xn[m].mean(axis=0)
            cen /= np.clip(np.linalg.norm(cen), 1e-8, None)
            cos = float(cen @ f_dir)
            d2 = float(np.linalg.norm(np.median(coords[m], axis=0) - f_xy))
            hom = "" if meta["homology"] is None else f"{meta['homology']:.3f}"
            lines.append(f"{anchor}\t{g}\t{meta['role']}\t{int(m.sum())}\t{hom}\t"
                         f"{meta['band'] or ''}\t{cos:.4f}\t{d2:.2f}")

    os.makedirs(os.path.dirname(out_tsv), exist_ok=True)
    open(out_tsv, "w").write("\n".join(lines) + "\n")
    print(f"wrote {out_tsv}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emb", default="results/umap/embeddings")
    ap.add_argument("--out", default="results/umap")
    ap.add_argument("--pca-dim", type=int, default=50,
                    help="PCA before UMAP — standard for ~240k points, and much faster "
                         "than running UMAP on the raw 480-d vectors.")
    ap.add_argument("--n-neighbors", type=int, default=30)
    ap.add_argument("--min-dist", type=float, default=0.1)
    ap.add_argument("--metric", default="cosine")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    emb_root = os.path.join(ROOT, args.emb)
    out_root = os.path.join(ROOT, args.out)
    os.makedirs(out_root, exist_ok=True)

    _bootstrap.banner(f"13_umap_plot | n_neighbors={args.n_neighbors} "
                      f"min_dist={args.min_dist} metric={args.metric}")

    X, rows = load_shards(emb_root)
    print(f"\nunion: {X.shape[0]:,} sequences x {X.shape[1]} dims")

    from sklearn.decomposition import PCA
    import umap

    n_comp = min(args.pca_dim, X.shape[1], X.shape[0])
    pca = PCA(n_components=n_comp, random_state=args.seed)
    Xp = pca.fit_transform(X)
    print(f"PCA -> {n_comp} dims ({pca.explained_variance_ratio_.sum():.1%} variance retained)")

    # ONE global fit over the union, so the three panels share a coordinate frame and
    # are directly comparable; retain is embedded once and reused across panels.
    reducer = umap.UMAP(n_neighbors=args.n_neighbors, min_dist=args.min_dist,
                        metric=args.metric, random_state=args.seed, verbose=True)
    coords = reducer.fit_transform(Xp)
    print(f"UMAP -> {coords.shape}")

    np.save(os.path.join(out_root, "coords.npy"), coords.astype(np.float32))
    json.dump({"n_neighbors": args.n_neighbors, "min_dist": args.min_dist,
               "metric": args.metric, "pca_dim": n_comp, "seed": args.seed,
               "n_points": int(X.shape[0])},
              open(os.path.join(out_root, "umap_params.json"), "w"), indent=2)

    present = [a for a in ANCHOR_ORDER if any(r["anchor"] == a for r in rows)]
    draw(coords, rows, os.path.join(out_root, "plots", "umap_families.png"),
         os.path.join(out_root, "plots", "umap_families.pdf"), present)
    write_table(coords, X, rows, os.path.join(out_root, "group_centroids.tsv"))


if __name__ == "__main__":
    # Manuscript-grade vector output, matching scripts/make_manuscript_pdfs.py.
    matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,
                                "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
    main()
