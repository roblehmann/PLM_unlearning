#!/usr/bin/env python3
"""T3 — one joint PCA-50 + UMAP over both corpora, and the separation statistics.

The fit is GLOBAL, over the union of reps and retain, so the two corpora share one
coordinate frame — the same recipe (PCA-50, n_neighbors 30, min_dist 0.1, cosine, seeded)
as scripts/13_umap_plot.py, so these coordinates are comparable with the project's existing
UMAP figures.

The picture alone cannot answer "same region or separate", because inter-cluster distance
in a UMAP embedding is not metric — a caveat scripts/13_umap_plot.py already carries. So
this phase also computes the numbers that can: kNN label mixing in PCA-50 space against the
chance rate implied by the corpus sizes, and the centroid cosine in the ORIGINAL 480-d
space.

CPU only, ~5 min at 229,523 points (the 258k-point fit of the same recipe took 3m20s at
3.9 GB). Depends on t1 alone, so it runs alongside t2.

  python experiments/6_mgnify_reps_vs_retain/t3_umap.py
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

import common as C


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pca-dim", type=int, default=C.PCA_DIM)
    ap.add_argument("--n-neighbors", type=int, default=C.UMAP_N_NEIGHBORS)
    ap.add_argument("--min-dist", type=float, default=C.UMAP_MIN_DIST)
    ap.add_argument("--metric", default=C.UMAP_METRIC)
    ap.add_argument("--knn-k", type=int, default=C.KNN_K)
    ap.add_argument("--retain-npy", default=None,
                    help="override the retain embedding source (default: this experiment's "
                         "results/embeddings/retain.npy if t1 wrote one, else the shared cache)")
    ap.add_argument("--force", action="store_true")
    C.common_args(ap)
    return ap.parse_args()


def _read_ids(tsv: Path):
    rows = [ln.strip() for ln in open(tsv) if ln.strip()]
    return rows[1:] if rows and rows[0] == "id" else rows


def _work_ids(corpus: str) -> set:
    """The ids t0 planned for this corpus — the authoritative set for this run."""
    import csv

    with open(C.RESULTS / "work_table.tsv") as fh:
        return {r["id"] for r in csv.DictReader(fh, delimiter="\t") if r["corpus"] == corpus}


def _subset_to_work(X: np.ndarray, ids: list, corpus: str):
    """Keep only the rows t0 planned, in their existing order.

    A no-op on the real run (the work table is the whole corpus). It matters for smoke and
    partial runs, where t0 caps a corpus but the SHARED retain embedding cache still holds
    all 194,064 rows — without this the smoke would quietly fit a 194k-point UMAP against
    200 scored sequences.
    """
    want = _work_ids(corpus)
    keep = [i for i, cid in enumerate(ids) if cid in want]
    if len(keep) == len(ids):
        return X, ids
    print(f"  {corpus}: work table covers {len(keep):,} of {len(ids):,} cached rows — subsetting",
          flush=True)
    if not keep:
        raise SystemExit(f"FAIL: no {corpus} embedding rows match the work table ids")
    return X[keep], [ids[i] for i in keep]


def main() -> int:
    args = parse_args()
    C.apply_results_override(args.results)
    C.banner("T3 — joint PCA-50 + UMAP")
    out_dir = C.results_dir("umap")

    emb_dir = C.RESULTS / "embeddings"
    reps_npy, reps_tsv = emb_dir / "reps.npy", emb_dir / "reps.tsv"
    if not reps_npy.exists():
        raise SystemExit(f"FAIL: {reps_npy} missing — run t1 first.")

    local_retain = emb_dir / "retain.npy"
    if args.retain_npy:
        retain_npy = Path(args.retain_npy)
        retain_tsv = retain_npy.with_suffix(".tsv")
    elif local_retain.exists():
        retain_npy, retain_tsv = local_retain, emb_dir / "retain.tsv"
        print("  using this experiment's freshly embedded retain (t1 --reembed-retain)")
    else:
        retain_npy, retain_tsv = C.CACHED_RETAIN_NPY, C.CACHED_RETAIN_TSV
        print("  using the shared cached retain embeddings (verified by t0 and t1)")

    Xr, reps_ids = _subset_to_work(np.load(reps_npy).astype(np.float32),
                                   _read_ids(reps_tsv), "reps")
    Xt, retain_ids = _subset_to_work(np.load(retain_npy).astype(np.float32),
                                     _read_ids(retain_tsv), "retain")
    ids = reps_ids + retain_ids
    labels = np.array(["reps"] * Xr.shape[0] + ["retain"] * Xt.shape[0])
    X = np.concatenate([Xr, Xt], axis=0)
    print(f"  {X.shape[0]:,} points x {X.shape[1]} dims "
          f"({Xr.shape[0]:,} reps + {Xt.shape[0]:,} retain), {X.nbytes / 1e6:.0f} MB", flush=True)
    if len(ids) != X.shape[0]:
        raise SystemExit(f"FAIL: {len(ids)} ids for {X.shape[0]} embedding rows")

    with open(out_dir / "rows.tsv", "w") as fh:
        fh.write("row\tcorpus\tid\n")
        for i, (lab, cid) in enumerate(zip(labels, ids)):
            fh.write(f"{i}\t{lab}\t{cid}\n")

    # ------------------------------------------------------------------ PCA
    from sklearn.decomposition import PCA

    t0 = time.time()
    n_comp = int(min(args.pca_dim, X.shape[0], X.shape[1]))
    pca = PCA(n_components=n_comp, random_state=args.seed)
    Xp = pca.fit_transform(X)
    evr = float(pca.explained_variance_ratio_.sum())
    t_pca = time.time() - t0
    print(f"  PCA-{n_comp}: {evr:.4f} explained variance in {t_pca:.1f}s", flush=True)
    np.save(out_dir / "pca50.npy", Xp.astype(np.float32))

    # ------------------------------------------------------ separation stats
    # These, not the picture, are what the claim rests on.
    t0 = time.time()
    k = int(min(args.knn_k, X.shape[0] - 2))
    sep = C.knn_mixing(Xp, labels, k=k, seed=args.seed, metric=args.metric)
    sep["centroid_cosine_reps_vs_retain"] = C.centroid_cosine(Xr, Xt)
    sep["space"] = f"PCA-{n_comp} (kNN) / original {X.shape[1]}-d (centroid cosine)"
    sep["seconds"] = time.time() - t0
    print(f"  kNN mixing (k={sep['knn_k']}): reps see {sep['mean_frac_retain_neighbours_of_reps']:.3f} "
          f"retain neighbours vs chance {sep['null_frac_retain']:.3f}; "
          f"centroid cosine {sep['centroid_cosine_reps_vs_retain']:.4f}", flush=True)

    # ----------------------------------------------------------------- UMAP
    import umap

    n_neighbors = int(min(args.n_neighbors, max(2, X.shape[0] - 1)))
    clamped = n_neighbors != args.n_neighbors
    t0 = time.time()
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=args.min_dist, metric=args.metric,
                        random_state=args.seed, verbose=True)
    coords = reducer.fit_transform(Xp)
    t_umap = time.time() - t0
    np.save(out_dir / "coords.npy", np.asarray(coords, dtype=np.float32))
    print(f"  UMAP fit in {t_umap / 60:.1f} min", flush=True)

    cen = {c: [float(v) for v in coords[labels == c].mean(axis=0)] for c in C.CORPORA}
    with open(out_dir / "group_centroids.tsv", "w") as fh:
        fh.write("corpus\tn\tumap_x\tumap_y\tcentroid_cosine_to_retain\n")
        for c, X_c in (("reps", Xr), ("retain", Xt)):
            fh.write(f"{c}\t{X_c.shape[0]}\t{cen[c][0]:.4f}\t{cen[c][1]:.4f}\t"
                     f"{C.centroid_cosine(X_c, Xt):.6f}\n")

    C.save_json({
        "protocol": C.protocol_block(),
        "pca_dim": n_comp, "explained_variance_ratio": evr,
        "n_neighbors": n_neighbors, "n_neighbors_clamped": bool(clamped),
        "min_dist": args.min_dist, "metric": args.metric, "seed": int(args.seed),
        "n_points": int(X.shape[0]),
        "n_reps": int(Xr.shape[0]), "n_retain": int(Xt.shape[0]),
        "retain_source": str(retain_npy),
        "seconds": {"pca": t_pca, "umap": t_umap},
        "centroids_umap": cen,
        "caveat": "UMAP inter-cluster distance is not metric — read separation from "
                  "knn_mixing.json and the centroid cosine, not from the picture.",
    }, out_dir / "umap_params.json")
    C.save_json(sep, out_dir / "knn_mixing.json")
    print("\nT3 OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
