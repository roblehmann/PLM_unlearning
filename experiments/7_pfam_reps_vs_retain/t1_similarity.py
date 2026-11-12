#!/usr/bin/env python3
"""t1 — the 21,763 x 21,763 Pfam representative similarity matrix and its isolation metrics.

One blocked matmul per embedding space produces, in a single pass:

  * the full similarity matrix, stored fp16 for VISUALISATION ONLY;
  * every statistic that decides anything, accumulated from the fp32 block BEFORE the
    downcast — top-k neighbours (with and without same-clan families), row sums, the
    exact off-diagonal histogram, and the condensed distance vector for linkage;
  * the anisotropy numbers that say whether the space is usable at all.

Isolation is measured against the bulk of Pfam itself. Distance to the retain corpus R is
computed here too, but only as a reported column — `cluster_families.py` already ranks by
that axis, and this stage exists because it is not the same question.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

import common as C


def blocked_similarity(unit: np.ndarray, clan_members, matrix_path, *, block: int,
                       topk: int, hist_bins: int):
    """One pass over the matrix. Returns every accumulator the later stages need.

    The fp16 file is written per block and never read back for a metric: fp16 spacing near
    1.0 is ~5e-4 and these decisions live at the 1e-2 scale, so it is honest for a picture
    and not for a number.
    """
    n = unit.shape[0]
    k = int(min(topk, n - 1))
    mm = None
    if matrix_path is not None:
        mm = np.lib.format.open_memmap(matrix_path, mode='w+', dtype=np.float16, shape=(n, n))
    condensed = np.empty(n * (n - 1) // 2, dtype=np.float64)
    hist = np.zeros(int(hist_bins), dtype=np.int64)
    edges = np.linspace(-1.0, 1.0, int(hist_bins) + 1)
    rowsum = np.empty(n, dtype=np.float64)
    top_sim = np.empty((n, k), dtype=np.float32)
    top_idx = np.empty((n, k), dtype=np.int32)
    xclan_sim = np.empty((n, k), dtype=np.float32)
    xclan_idx = np.empty((n, k), dtype=np.int32)
    started = time.time()

    for start in range(0, n, block):
        stop = min(start + block, n)
        sim = np.asarray(unit[start:stop] @ unit.T, dtype=np.float32)
        np.clip(sim, -1.0, 1.0, out=sim)
        rows = np.arange(start, stop)
        sim[np.arange(stop - start), rows] = 1.0            # exact diagonal
        if mm is not None:
            mm[start:stop] = sim.astype(np.float16)
        rowsum[start:stop] = sim.sum(axis=1, dtype=np.float64) - 1.0
        for local, i in enumerate(rows):
            if i + 1 < n:
                base = n * i - i * (i + 1) // 2
                condensed[base:base + n - i - 1] = 1.0 - sim[local, i + 1:]
        sim[np.arange(stop - start), rows] = -np.inf        # exclude self everywhere below
        hist += np.histogram(sim, bins=edges)[0]
        part = np.argpartition(sim, n - k, axis=1)[:, n - k:]
        vals = np.take_along_axis(sim, part, axis=1)
        order = np.argsort(-vals, axis=1, kind='stable')
        top_sim[start:stop] = np.take_along_axis(vals, order, axis=1)
        top_idx[start:stop] = np.take_along_axis(part, order, axis=1)
        for local, i in enumerate(rows):                    # mask the family's own clan
            members = clan_members.get(i)
            if members is not None:
                sim[local, members] = -np.inf
        part = np.argpartition(sim, n - k, axis=1)[:, n - k:]
        vals = np.take_along_axis(sim, part, axis=1)
        order = np.argsort(-vals, axis=1, kind='stable')
        xclan_sim[start:stop] = np.take_along_axis(vals, order, axis=1)
        xclan_idx[start:stop] = np.take_along_axis(part, order, axis=1)
        del sim
        if start % (block * 4) == 0 or stop == n:
            print(f'  rows {stop:,}/{n:,} ({time.time()-started:.0f}s)', flush=True)

    if mm is not None:
        mm.flush()
        del mm
    return dict(condensed=condensed, hist=hist, hist_edges=edges, rowsum=rowsum,
                top_sim=top_sim, top_idx=top_idx, xclan_sim=xclan_sim, xclan_idx=xclan_idx, k=k)


def retain_neighbourhood(unit_reps: np.ndarray, unit_retain: np.ndarray, *, topk: int,
                         block: int = 256):
    """Exact top-k cosine of every representative against the retain corpus."""
    n = unit_reps.shape[0]
    k = int(min(topk, unit_retain.shape[0]))
    sims = np.empty((n, k), dtype=np.float32)
    ids = np.empty(n, dtype=np.int64)
    for start in range(0, n, block):
        stop = min(start + block, n)
        sim = np.asarray(unit_reps[start:stop] @ unit_retain.T, dtype=np.float32)
        np.clip(sim, -1.0, 1.0, out=sim)
        part = np.argpartition(sim, sim.shape[1] - k, axis=1)[:, sim.shape[1] - k:]
        vals = np.take_along_axis(sim, part, axis=1)
        order = np.argsort(-vals, axis=1, kind='stable')
        sims[start:stop] = np.take_along_axis(vals, order, axis=1)
        ids[start:stop] = np.take_along_axis(part, order, axis=1)[:, 0]
        del sim
        if start % (block * 20) == 0:
            print(f'  retain rows {stop:,}/{n:,}', flush=True)
    return sims, ids


def frac_retain_in_topk(rep_sims: np.ndarray, retain_sims: np.ndarray, k: int) -> np.ndarray:
    """Fraction of a representative's overall k nearest neighbours that are retain rows.

    Exact without a joint search: the global top-k over reps u retain is contained in the
    union of the rep top-k and the retain top-k, so merging the two stored lists and
    taking the k largest gives the same answer as scoring both corpora at once.
    """
    k = int(min(k, rep_sims.shape[1], retain_sims.shape[1]))
    merged = np.concatenate([rep_sims[:, :k], retain_sims[:, :k]], axis=1)
    is_retain = np.concatenate([np.zeros(k, dtype=bool), np.ones(k, dtype=bool)])
    order = np.argsort(-merged, axis=1, kind='stable')[:, :k]
    return is_retain[order].mean(axis=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    C.common_args(ap)
    ap.add_argument('--spaces', default='raw,cen,whi')
    ap.add_argument('--store-matrix', default='raw,cen',
                    help='which spaces persist their full fp16 matrix (947 MB each)')
    ap.add_argument('--block', type=int, default=2048)
    ap.add_argument('--topk', type=int, default=50)
    ap.add_argument('--hist-bins', type=int, default=4096)
    ap.add_argument('--calibration-size', type=int, default=4096,
                    help='seeded retain rows for the leave-one-out distance calibration')
    ap.add_argument('--whiten-components', type=int, default=128)
    ap.add_argument('--linkage-method', default='average',
                    choices=['average', 'ward', 'complete'])
    ap.add_argument('--optimal-leaf-order', action='store_true',
                    help='needs the 1.9 GB condensed vector resident; off by default')
    ap.add_argument('--clusters', default='30,100,300')
    ap.add_argument('--limit-families', type=int, default=0)
    ap.add_argument('--limit-retain', type=int, default=0)
    args = ap.parse_args()
    C.apply_results_override(args.results)
    C.guard_limits(args)
    started = time.time()
    verify = not args.no_verify
    spaces = [s for s in args.spaces.split(',') if s]
    store = {s for s in args.store_matrix.split(',') if s}
    cuts = [int(c) for c in args.clusters.split(',') if c]

    C.banner('t1 — inputs')
    x, rows, meta = C.load_reps(limit=args.limit_families, verify=verify)
    n = len(rows)
    print(f'{n:,} representatives x {x.shape[1]} dims', flush=True)
    accs = [r['id'] for r in rows]
    clan = np.array([m['clan_acc'] or '' for m in meta])
    clan_members = {}
    for value in set(clan) - {''}:
        idx = np.flatnonzero(clan == value)
        for i in idx:
            clan_members[int(i)] = idx
    retain_x, retain_rows = C.load_group('retain', verify=verify)
    if args.limit_retain and args.limit_retain < len(retain_rows):
        keep = np.linspace(0, len(retain_rows) - 1, args.limit_retain).astype(int)
        keep = np.unique(keep)
        retain_x, retain_rows = retain_x[keep], [retain_rows[i] for i in keep]
    print(f'{len(retain_rows):,} retain sequences', flush=True)
    anchor_raw = {}
    for group in C.EXISTING_ANCHOR_GROUPS:
        try:
            ax, _ = C.load_group(group, verify=False)
        except SystemExit:
            print(f'  note: {group} embeddings absent; skipping that anchor column', flush=True)
            continue
        anchor_raw[group] = np.asarray(ax, dtype=np.float32).mean(axis=0, keepdims=True)

    matrix_dir = C.results_dir('matrix')
    columns = {}
    summary = {'protocol': C.protocol_block(stage='t1_similarity'), 'spaces': {},
               'n_reps': n, 'n_retain': len(retain_rows),
               'topk': int(args.topk), 'block': int(args.block),
               'linkage_method': args.linkage_method,
               'optimal_leaf_order': bool(args.optimal_leaf_order),
               'inputs': {str(p.relative_to(C.ROOT)): C.sha(p) for p in
                          [C.EMBED_DIR / 'pfam_reps.npy', C.EMBED_DIR / 'retain.npy', C.META_TSV]}}
    validation = {'checks': {}}

    for key in spaces:
        C.banner(f't1 — space "{key}"')
        space = C.fit_space(x, key, n_components=args.whiten_components, seed=args.seed)
        space.save(matrix_dir / f'space_{key}.npz')
        unit = np.ascontiguousarray(space.apply(x))
        info = {'transform': space.as_dict(), 'anisotropy': C.anisotropy(unit)}
        print(f"  mean resultant length {info['anisotropy']['mean_resultant_length']:.4f}, "
              f"effective rank {info['anisotropy']['effective_rank']:.1f}", flush=True)

        path = matrix_dir / f'sim_{key}.f16.npy' if key in store else None
        acc = blocked_similarity(unit, clan_members, path, block=args.block,
                                 topk=args.topk, hist_bins=args.hist_bins)
        np.savez(matrix_dir / f'topk_{key}.npz', top_sim=acc['top_sim'], top_idx=acc['top_idx'],
                 xclan_sim=acc['xclan_sim'], xclan_idx=acc['xclan_idx'])
        np.savez(matrix_dir / f'hist_{key}.npz', counts=acc['hist'], edges=acc['hist_edges'])
        centres = 0.5 * (acc['hist_edges'][1:] + acc['hist_edges'][:-1])
        total = float(acc['hist'].sum())
        mean_off = float((acc['hist'] * centres).sum() / total)
        info['off_diagonal'] = {
            'n_pairs_counted': int(total),
            'mean_similarity_binned': mean_off,
            'mean_similarity_exact': float(acc['rowsum'].sum() / (n * (n - 1))),
            'median_similarity_binned': float(centres[min(
                int(np.searchsorted(np.cumsum(acc['hist']), total / 2)), centres.size - 1)]),
            'note': 'both (i,j) and (j,i) are counted; the diagonal never is',
        }

        # --- isolation metrics, all from the fp32 accumulators ---------------- #
        iso_nn = 1.0 - acc['top_sim'][:, 0].astype(np.float64)
        iso_top10 = 1.0 - acc['top_sim'][:, :10].astype(np.float64).mean(axis=1)
        iso_nn_x = 1.0 - acc['xclan_sim'][:, 0].astype(np.float64)
        iso_top10_x = 1.0 - acc['xclan_sim'][:, :10].astype(np.float64).mean(axis=1)
        dist = 1.0 - acc['top_sim'].astype(np.float64)
        lof20 = C.lof_from_topk(dist, acc['top_idx'], 20)
        lof50 = C.lof_from_topk(dist, acc['top_idx'], min(50, acc['k']))
        centroid = C.normalize(unit.mean(axis=0, keepdims=True))[0]
        iso_centroid = 1.0 - (unit @ centroid).astype(np.float64)
        medoid = int(np.argmax(acc['rowsum']))
        medoid_sim = (unit @ unit[medoid]).astype(np.float64)
        same_clan_frac = np.array([
            float(np.mean([clan[j] != '' and clan[j] == clan[i] for j in acc['top_idx'][i, :10]]))
            for i in range(n)])

        print('  linkage', flush=True)
        from scipy.cluster.hierarchy import fcluster, leaves_list, linkage, optimal_leaf_ordering

        np.maximum(acc['condensed'], 0, out=acc['condensed'])
        z = linkage(acc['condensed'], method=args.linkage_method)
        if args.optimal_leaf_order:
            z = optimal_leaf_ordering(z, acc['condensed'])
        np.save(matrix_dir / f'linkage_{key}.npy', z)
        leaf_order = leaves_list(z)
        np.save(matrix_dir / f'leaf_order_{key}.npy', leaf_order)
        heights = C.linkage_leaf_heights(z, n, thresholds=(20, 100))
        assignments = {c: fcluster(z, t=c, criterion='maxclust') for c in cuts}
        sizes = {c: np.bincount(assignments[c]).max() for c in cuts}
        info['clusters'] = {str(c): {'k': int(len(set(assignments[c]))),
                                     'largest_cluster_families': int(sizes[c]),
                                     'largest_cluster_frac': float(sizes[c] / n)} for c in cuts}
        if key == 'raw' and not args.limit_families:
            reference = C.CLUSTERING_DIR / 'linkage.npy'
            if reference.exists() and args.linkage_method == 'average':
                old = np.load(reference)
                delta = float(np.abs(np.sort(old[:, 2]) - np.sort(z[:, 2])).max())
                validation['checks']['raw_linkage_matches_cluster_families'] = {
                    'max_abs_height_difference': delta, 'passed': bool(delta < 1e-6),
                    'reference': str(reference.relative_to(C.ROOT))}
                print(f'  raw linkage vs results/clustering: max |dh| = {delta:.2e}', flush=True)

        print('  retain neighbourhood', flush=True)
        unit_retain = np.ascontiguousarray(space.apply(retain_x))
        retain_sims, retain_ids = retain_neighbourhood(unit, unit_retain, topk=args.topk)
        mixing = frac_retain_in_topk(acc['top_sim'], retain_sims, k=min(50, acc['k']))
        chance = len(retain_rows) / (len(retain_rows) + n - 1)
        # Retain's own leave-one-out nearest-neighbour distance, in THIS space: the only
        # honest yardstick for "far from R" once the geometry has been transformed.
        rng = np.random.default_rng(args.seed)
        sample = np.sort(rng.choice(len(retain_rows),
                                    size=min(args.calibration_size, len(retain_rows)),
                                    replace=False))
        cal_near, _, _ = C.neighbors(unit_retain[sample], unit_retain,
                                     exclude=[[int(i)] for i in sample])
        cal_dist = 1.0 - cal_near
        np.savez(matrix_dir / f'retain_loo_{key}.npz', rows=sample, nearest_cosine=cal_near)
        info['retain'] = {'chance_frac_retain_in_top50': float(chance),
                          'mean_frac_retain_in_top50': float(mixing.mean()),
                          'loo_calibration_size': int(sample.size),
                          'loo_distance_p95': float(np.quantile(cal_dist, 0.95)),
                          'loo_distance_p99': float(np.quantile(cal_dist, 0.99))}
        print(f"  retain leave-one-out p95={info['retain']['loo_distance_p95']:.5f} "
              f"p99={info['retain']['loo_distance_p99']:.5f}", flush=True)
        anchor_cos = np.full(n, np.nan)
        if anchor_raw:
            anchor_units = np.vstack([space.apply(v) for v in anchor_raw.values()])
            anchor_cos = (unit @ anchor_units.T).max(axis=1).astype(np.float64)
            info['existing_anchor_groups'] = list(anchor_raw)
        del unit_retain

        prefix = f'{key}_'
        columns.update({
            prefix + 'iso_nn': iso_nn, prefix + 'iso_top10': iso_top10,
            prefix + 'iso_nn_xclan': iso_nn_x, prefix + 'iso_top10_xclan': iso_top10_x,
            prefix + 'lof20': lof20, prefix + 'lof50': lof50,
            prefix + 'iso_centroid': iso_centroid, prefix + 'medoid_sim': medoid_sim,
            prefix + 'same_clan_frac': same_clan_frac,
            prefix + 'first_merge_height': heights['first_merge_height'],
            prefix + 'merge_order_frac': heights['merge_order_frac'],
            prefix + 'height_join20': heights['height_join20'],
            prefix + 'height_join100': heights['height_join100'],
            prefix + 'nearest_retain_cos': retain_sims[:, 0].astype(np.float64),
            prefix + 'top10_retain_cos': retain_sims[:, :10].astype(np.float64).mean(axis=1),
            prefix + 'frac_retain_in_top50': mixing,
            prefix + 'max_cos_to_existing_anchor': anchor_cos,
        })
        for c in cuts:
            columns[f'{prefix}cluster_k{c}'] = assignments[c]
        columns[prefix + 'nn1_acc'] = np.array([accs[j] for j in acc['top_idx'][:, 0]])
        columns[prefix + 'nn1_acc_xclan'] = np.array([accs[j] for j in acc['xclan_idx'][:, 0]])
        columns[prefix + 'nearest_retain_id'] = np.array(
            [retain_rows[j]['id'] for j in retain_ids])
        info['medoid_family'] = accs[medoid]

        if path is not None:
            info['matrix'] = {'path': str(path.relative_to(C.ROOT)), 'dtype': 'float16',
                              'shape': [n, n], 'sha256': C.sha(path),
                              'use': 'visualisation and inspection only; no metric is read back from it'}
            rng = np.random.default_rng(args.seed)
            probe = rng.choice(n, size=min(64, n), replace=False)
            stored = np.load(path, mmap_mode='r')
            recomputed = (unit[probe] @ unit.T).astype(np.float32)
            recomputed[np.arange(probe.size), probe] = 1.0
            delta = float(np.abs(np.asarray(stored[probe], dtype=np.float32) - recomputed).max())
            sym = float(np.abs(np.asarray(stored[probe][:, probe], dtype=np.float32)
                               - np.asarray(stored[probe][:, probe], dtype=np.float32).T).max())
            diag = float(np.abs(np.asarray(stored[probe, probe], dtype=np.float32) - 1.0).max())
            # fp16 spacing near 1.0 is ~5e-4, and (i,j)/(j,i) are accumulated in different
            # blocks, so both checks are to fp16 precision rather than bitwise.
            validation['checks'][f'matrix_{key}'] = {
                'rows_reprobed': int(probe.size), 'max_abs_difference': delta,
                'max_symmetry_difference': sym, 'max_diagonal_difference': diag,
                'tolerance': 2e-3,
                'passed': bool(delta < 2e-3 and sym < 2e-3 and diag == 0.0)}
            del stored

        summary['spaces'][key] = info
        del acc, unit
    del x

    C.banner('t1 — writing metrics')
    order_spaces = [s for s in spaces]
    table = []
    for i, (row, m) in enumerate(zip(rows, meta)):
        entry = {
            'pfam_acc': row['id'], 'pfam_id': m['pfam_id'], 'description': m['desc'],
            'type': m['type'], 'clan_acc': m['clan_acc'], 'in_clan': m['in_clan'],
            'clan_id': m['clan_id'], 'clan_n_families': m['clan_n_families'],
            'members_nr90': int(m['n_proteins_nr90']), 'regions_nr90': int(m['n_regions_nr90']),
            'swissprot_nr90': m['n_swissprot_nr90'], 'model_length': int(m['model_length']),
            'hazard_flag': m['hazard_flag'], 'hazard_keyword': m['hazard_keyword'],
            'rep_uniprot': m['rep_uniprot_acc'], 'rep_length': int(m['rep_length']),
            'rep_domain_ranges': m['rep_domain_ranges'],
            'rep_family_coverage': m['rep_family_coverage'],
            'rep_reviewed': m['rep_reviewed'], 'truncated': row['truncated'],
            'sequence_sha256': row['sequence_sha256'],
        }
        for name, values in columns.items():
            value = values[i]
            entry[name] = value.item() if isinstance(value, np.generic) else value
        table.append(entry)
    C.write_tsv(C.results_dir('matrix') / 'family_metrics.tsv', table)

    if len(order_spaces) > 1:
        from scipy.stats import spearmanr

        pairs = {}
        for a in order_spaces:
            for b in order_spaces:
                if a < b:
                    va = np.mean([C.pctrank(columns[f'{a}_{c}']) for c in C.ISOLATION_COMPONENTS], axis=0)
                    vb = np.mean([C.pctrank(columns[f'{b}_{c}']) for c in C.ISOLATION_COMPONENTS], axis=0)
                    pairs[f'{a}_vs_{b}'] = float(spearmanr(va, vb).statistic)
        summary['isolation_rank_spearman_between_spaces'] = pairs
        print(f'  Spearman between spaces: {pairs}', flush=True)

    validation['checks']['family_metrics_rows'] = {
        'expected': n, 'actual': len(table), 'passed': len(table) == n}
    validation['checks']['accession_order_matches_fasta'] = {
        'passed': [r['pfam_acc'] for r in table] == accs}
    validation['status'] = 'passed' if all(
        v.get('passed', True) for v in validation['checks'].values()) else 'FAILED'
    summary['seconds'] = time.time() - started
    C.save_json(summary, C.results_dir('matrix') / 'summary_similarity.json')
    C.save_json(validation, C.results_dir('matrix') / 'validation.json')
    print(f"\nDONE t1 [{validation['status']}] in {summary['seconds']:.0f}s -> "
          f"{C.results_dir('matrix')}", flush=True)
    if validation['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
