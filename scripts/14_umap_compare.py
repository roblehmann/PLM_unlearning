"""Base A vs task-blocked B in the same embedding space, per anchor.

Companion to `scripts/13_umap_plot.py`. Stage-1 caches come from
`scripts/12_embed_corpora.py --variant base` and `--variant B`.

Why a JOINT fit is legitimate here. A and B are the same architecture at the same layer
and B is a finetune of A, so the hidden-state basis is shared; the question is only
whether the two clouds have drifted into disjoint regions, which would make a joint UMAP
degenerate into "two blobs, one per model". They have not: measured per sequence,
cos(A,B) is 0.985-0.995 for the large majority of corpora, with centroid shifts ~0.5
against vector norms of ~5. The exceptions are the point of the figure — B rewrites the
representation of the blocked family and its nearest relatives specifically, driving
cos(A,B) to ~0 or negative there while leaving everything else in place. A joint fit
therefore puts unaffected corpora on top of themselves and shows the rewritten ones
moving, which is exactly the comparison asked for.

One fit PER ANCHOR (not one global fit as in 13_): under variant B each anchor has its
OWN trained checkpoint, so embeddings from different anchors are not commensurable and
must not share a frame. Rows within a column do share coordinates.

    python scripts/14_umap_compare.py
    python scripts/14_umap_compare.py --retain-cap 40000     # lighter fit
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import _bootstrap  # noqa: F401

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = _bootstrap.ROOT

C_F = "#D55E00"
C_FPRIME = "#0072B2"
RAMP = {
    8: ["#45C396", "#26AD81", "#00966D", "#007F5C", "#00694B", "#00533B", "#003E2B", "#002B1C"],
    6: ["#45C396", "#18A57A", "#008661", "#00684B", "#004B35", "#003021"],
}
ANCHOR_TITLE = {
    "mgnify": "MGnify MGYF0000018690",
    "PF00106": "Pfam PF00106 / adh_short",
    "PF00107": "Pfam PF00107 / ADH_zinc_N",
}
ANCHOR_ORDER = ["mgnify", "PF00106", "PF00107"]
ROW_TITLE = {"A": "base A (pretrained)", "B": "task-blocked B"}


def ramp_for(n: int) -> List[str]:
    if n in RAMP:
        return RAMP[n]
    base = RAMP[8]
    return [base[i] for i in np.linspace(0, len(base) - 1, n).round().astype(int)]


def _read_group(root: str, anchor: str, label: str) -> Tuple[np.ndarray, List[str]]:
    X = np.load(os.path.join(root, anchor, f"{label}.npy")).astype(np.float32)
    ids = [ln.strip() for ln in open(os.path.join(root, anchor, f"{label}.tsv"))][1:]
    return X, ids


def load_anchor(emb_a: str, emb_b: str, anchor: str,
                retain_cap: Optional[int], seed: int) -> Optional[dict]:
    """Load one anchor's groups from both caches, keeping only groups present in both.

    Retain lives under `shared/` in the base cache (one model, so one copy) but under
    `<anchor>/` in the B cache (per-anchor model). Both are the same FASTA read in file
    order, so the id lists must match — that is asserted, not assumed.
    """
    man_a = json.load(open(os.path.join(emb_a, "manifest.json")))
    man_b = json.load(open(os.path.join(emb_b, "manifest.json")))

    labels = sorted({k.split("/", 1)[1] for k in man_a if k.startswith(f"{anchor}/")} &
                    {k.split("/", 1)[1] for k in man_b if k.startswith(f"{anchor}/")})
    if not labels:
        return None

    Xa_parts, Xb_parts, meta_rows = [], [], []
    for label in labels:
        Xa, ida = _read_group(emb_a, anchor, label)
        Xb, idb = _read_group(emb_b, anchor, label)
        if ida != idb:
            raise SystemExit(f"{anchor}/{label}: id lists differ between A and B caches")
        m = man_a[f"{anchor}/{label}"]
        Xa_parts.append(Xa)
        Xb_parts.append(Xb)
        meta_rows.append({"label": label, "role": m["role"], "homology": m.get("homology"),
                          "band": m.get("band"), "n": len(Xa)})

    # Retain: base cache keeps it under shared/, B cache under the anchor.
    if "retain" not in labels and "shared/retain" in man_a and f"{anchor}/retain" in man_b:
        Xa, ida = _read_group(emb_a, "shared", "retain")
        Xb, idb = _read_group(emb_b, anchor, "retain")
        if ida != idb:
            raise SystemExit(f"{anchor}: retain id lists differ between A and B caches")
        if retain_cap is not None and len(Xa) > retain_cap:
            keep = np.random.default_rng(seed).choice(len(Xa), retain_cap, replace=False)
            keep.sort()
            Xa, Xb = Xa[keep], Xb[keep]
        Xa_parts.append(Xa)
        Xb_parts.append(Xb)
        meta_rows.append({"label": "retain", "role": "retain", "homology": None,
                          "band": None, "n": len(Xa)})

    XA = np.concatenate(Xa_parts, axis=0)
    XB = np.concatenate(Xb_parts, axis=0)
    group_of = np.concatenate([[m["label"]] * m["n"] for m in meta_rows])
    role_of = np.concatenate([[m["role"]] * m["n"] for m in meta_rows])
    return {"anchor": anchor, "XA": XA, "XB": XB, "group": group_of,
            "role": role_of, "meta": {m["label"]: m for m in meta_rows}}


def fit_joint(XA: np.ndarray, XB: np.ndarray, pca_dim: int, n_neighbors: int,
              min_dist: float, metric: str, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    from sklearn.decomposition import PCA
    import umap

    X = np.concatenate([XA, XB], axis=0)
    n_comp = min(pca_dim, X.shape[1], X.shape[0])
    Xp = PCA(n_components=n_comp, random_state=seed).fit_transform(X)
    coords = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric,
                       random_state=seed, verbose=True).fit_transform(Xp)
    return coords[: len(XA)], coords[len(XA):]


def _deoverlap(labels: List[dict], xlim, ylim, min_sep: float = 0.06, iters: int = 400) -> None:
    if len(labels) < 2:
        for l in labels:
            l["lx"], l["ly"] = l["x"], l["y"]
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


def draw_panel(ax, coords: np.ndarray, data: dict, xlim, ylim,
               ghost: Optional[np.ndarray] = None) -> None:
    """One panel. `ghost` (the other model's coords) draws displacement arrows."""
    group_of, role_of = data["group"], data["role"]

    m = role_of == "retain"
    if m.any():
        ax.hexbin(coords[m, 0], coords[m, 1], gridsize=170, bins="log", cmap="Greys",
                  mincnt=1, linewidths=0, alpha=0.55, extent=(*xlim, *ylim), zorder=1)

    rungs = sorted({g for g in np.unique(group_of) if g.startswith("L")},
                   key=lambda g: int(g.split("_")[0][1:]))
    ramp = ramp_for(len(rungs))
    labels: List[dict] = []

    def centroid(mask, arr):
        return float(np.median(arr[mask, 0])), float(np.median(arr[mask, 1]))

    for j, g in enumerate(rungs):
        mk = group_of == g
        if not mk.any():
            continue
        ax.scatter(coords[mk, 0], coords[mk, 1], s=7, c=ramp[j], marker="^",
                   linewidths=0, alpha=0.75, zorder=3)
        cx, cy = centroid(mk, coords)
        labels.append({"x": cx, "y": cy, "text": g.split("_")[0], "color": ramp[j],
                       "size": 8, "mask": mk})

    mk = role_of == "F_prime"
    if mk.any():
        ax.scatter(coords[mk, 0], coords[mk, 1], s=7, c=C_FPRIME, marker="s",
                   linewidths=0, alpha=0.70, zorder=4)
        cx, cy = centroid(mk, coords)
        labels.append({"x": cx, "y": cy, "text": "F'", "color": C_FPRIME, "size": 10,
                       "mask": mk})

    mtr = role_of == "F_train"
    if mtr.any():
        ax.scatter(coords[mtr, 0], coords[mtr, 1], s=8, c=C_F, marker="o",
                   linewidths=0, alpha=0.70, zorder=5)
    mte = role_of == "F_test"
    if mte.any():
        ax.scatter(coords[mte, 0], coords[mte, 1], s=14, facecolors="none", edgecolors=C_F,
                   marker="o", linewidths=0.6, alpha=0.95, zorder=5)
    if mtr.any() or mte.any():
        mm = mtr | mte
        cx, cy = centroid(mm, coords)
        labels.append({"x": cx, "y": cy, "text": "F", "color": C_F, "size": 11, "mask": mm})

    # Displacement arrows: near-zero for corpora B leaves alone, long for the rewritten
    # ones. Drawn only where the move is big enough to read.
    if ghost is not None:
        span = max(xlim[1] - xlim[0], ylim[1] - ylim[0])
        for l in labels:
            gx, gy = centroid(l["mask"], ghost)
            if np.hypot(l["x"] - gx, l["y"] - gy) > 0.06 * span:
                ax.annotate("", xy=(l["x"], l["y"]), xytext=(gx, gy),
                            arrowprops=dict(arrowstyle="-|>", color=l["color"], lw=1.1,
                                            alpha=0.85, shrinkA=2, shrinkB=2),
                            zorder=6)

    _deoverlap(labels, xlim, ylim)
    lead = 0.012 * max(xlim[1] - xlim[0], ylim[1] - ylim[0])
    fx = [pe.withStroke(linewidth=2.4, foreground="white")]
    for l in labels:
        if np.hypot(l["lx"] - l["x"], l["ly"] - l["y"]) > lead:
            ax.plot([l["x"], l["lx"]], [l["y"], l["ly"]], color=l["color"], lw=0.6,
                    alpha=0.55, zorder=7, solid_capstyle="round")
        ax.text(l["lx"], l["ly"], l["text"], fontsize=l["size"], fontweight="bold",
                color=l["color"], ha="center", va="center", zorder=8, path_effects=fx)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#cccccc")


def write_table(results: List[dict], out_tsv: str) -> None:
    """Per-group summary. `cos_A_vs_B` is the same sequence under the two models and is
    the sharpest number here: ~0.99 means B left that corpus alone, ~0 means B rewrote
    its representation. cos_to_F columns are within-model, so they stay comparable
    across A and B even though the two spaces are not identical."""
    lines = ["anchor\tgroup\trole\tn\thomology\tband\t"
             "cos_A_to_F\tcos_B_to_F\tcos_A_vs_B\tumap_shift"]
    for r in results:
        XA, XB = r["XA"], r["XB"]
        An = XA / np.clip(np.linalg.norm(XA, axis=1, keepdims=True), 1e-8, None)
        Bn = XB / np.clip(np.linalg.norm(XB, axis=1, keepdims=True), 1e-8, None)
        fm = r["role"] == "F_train"
        fa = An[fm].mean(0); fa /= np.clip(np.linalg.norm(fa), 1e-8, None)
        fb = Bn[fm].mean(0); fb /= np.clip(np.linalg.norm(fb), 1e-8, None)

        for label, meta in r["meta"].items():
            m = r["group"] == label
            ca = An[m].mean(0); ca /= np.clip(np.linalg.norm(ca), 1e-8, None)
            cb = Bn[m].mean(0); cb /= np.clip(np.linalg.norm(cb), 1e-8, None)
            per_seq = float((An[m] * Bn[m]).sum(1).mean())
            shift = float(np.linalg.norm(np.median(r["cA"][m], axis=0) -
                                         np.median(r["cB"][m], axis=0)))
            hom = "" if meta["homology"] is None else f"{meta['homology']:.3f}"
            lines.append(f"{r['anchor']}\t{label}\t{meta['role']}\t{int(m.sum())}\t{hom}\t"
                         f"{meta['band'] or ''}\t{float(ca @ fa):.4f}\t{float(cb @ fb):.4f}\t"
                         f"{per_seq:.4f}\t{shift:.2f}")
    os.makedirs(os.path.dirname(out_tsv), exist_ok=True)
    open(out_tsv, "w").write("\n".join(lines) + "\n")
    print(f"wrote {out_tsv}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emb-a", default="results/umap/embeddings")
    ap.add_argument("--emb-b", default="results/umap/embeddings_B")
    ap.add_argument("--out", default="results/umap_compare")
    ap.add_argument("--retain-cap", type=int, default=0,
                    help="0 = use the whole retain corpus (default).")
    ap.add_argument("--pca-dim", type=int, default=50)
    ap.add_argument("--n-neighbors", type=int, default=30)
    ap.add_argument("--min-dist", type=float, default=0.1)
    ap.add_argument("--metric", default="cosine")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    emb_a = os.path.join(ROOT, args.emb_a)
    emb_b = os.path.join(ROOT, args.emb_b)
    out_root = os.path.join(ROOT, args.out)
    os.makedirs(out_root, exist_ok=True)
    cap = None if args.retain_cap <= 0 else args.retain_cap

    _bootstrap.banner(f"14_umap_compare | A={args.emb_a} B={args.emb_b} | retain_cap={cap}")

    results = []
    for anchor in ANCHOR_ORDER:
        data = load_anchor(emb_a, emb_b, anchor, cap, args.seed)
        if data is None:
            print(f"  {anchor}: not present in both caches — skipping")
            continue
        print(f"\n=== {anchor}: {len(data['XA']):,} sequences x2 models ===")
        cA, cB = fit_joint(data["XA"], data["XB"], args.pca_dim, args.n_neighbors,
                           args.min_dist, args.metric, args.seed)
        data["cA"], data["cB"] = cA, cB
        results.append(data)

    if not results:
        raise SystemExit("nothing to plot — check both embedding caches exist")

    fig, axes = plt.subplots(2, len(results), figsize=(5.0 * len(results), 10.0),
                             squeeze=False)
    for col, r in enumerate(results):
        both = np.concatenate([r["cA"], r["cB"]], axis=0)
        lo = np.percentile(both, 0.2, axis=0)
        hi = np.percentile(both, 99.8, axis=0)
        pad = 0.04 * (hi - lo)
        xlim = (lo[0] - pad[0], hi[0] + pad[0])
        ylim = (lo[1] - pad[1], hi[1] + pad[1])

        draw_panel(axes[0][col], r["cA"], r, xlim, ylim)
        draw_panel(axes[1][col], r["cB"], r, xlim, ylim, ghost=r["cA"])
        axes[0][col].set_title(ANCHOR_TITLE.get(r["anchor"], r["anchor"]), fontsize=10)

    for row, tag in enumerate(["A", "B"]):
        axes[row][0].set_ylabel(ROW_TITLE[tag], fontsize=10, fontweight="bold")

    handles = [
        Line2D([], [], marker="o", ls="", ms=6, mfc=C_F, mec=C_F, label="F (train)"),
        Line2D([], [], marker="o", ls="", ms=6, mfc="none", mec=C_F, label="F (held-out test)"),
        Line2D([], [], marker="s", ls="", ms=6, mfc=C_FPRIME, mec=C_FPRIME, label="F' (near family)"),
        Line2D([], [], marker="^", ls="", ms=6, mfc=RAMP[8][1], mec=RAMP[8][1],
               label="ladder rungs L0→Ln (light→dark)"),
        Line2D([], [], marker="h", ls="", ms=6, mfc="#999999", mec="#999999",
               label="retain corpus (density)"),
        Line2D([], [], marker=">", ls="-", color="#555555", ms=5, label="shift from A to B"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=8.5, frameon=False,
               bbox_to_anchor=(0.5, -0.004))
    fig.suptitle("Task-blocking rewrites the forget family's representation and little else\n"
                 "one joint UMAP per anchor; rows share coordinates within a column",
                 fontsize=11.5, y=0.985)
    fig.tight_layout(rect=(0, 0.035, 1, 0.955))
    png = os.path.join(out_root, "plots", "umap_A_vs_B.png")
    pdf = os.path.join(out_root, "plots", "umap_A_vs_B.pdf")
    os.makedirs(os.path.dirname(png), exist_ok=True)
    for p in (png, pdf):
        fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"\nwrote {png}\nwrote {pdf}")

    write_table(results, os.path.join(out_root, "group_shift.tsv"))


if __name__ == "__main__":
    matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,
                                "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
    main()
