#!/usr/bin/env python3
"""t4 — visualise the similarity matrix and the candidate screen.

The full matrix cannot be drawn honestly at one pixel per pair (21,763^2 = 474M pairs), so
it is block-aggregated and the caption says by how much — and because a mean over 256 pairs
hides the single near-1 pair that decides whether a family is isolated, the block MAX is
drawn beside the block mean, and a 2,000-family exact submatrix ships as the
no-aggregation control.
"""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np

import common as C


def block_pass(mat, leaf_order, clusters, factor: int):
    """One read of the matrix: block mean, block max, and cluster x cluster means.

    Rows are read in leaf order strips; the fp16 values are widened to float32 for the
    aggregation. Peak memory is one strip: factor x n floats.
    """
    n = mat.shape[0]
    nb = math.ceil(n / factor)
    pad = nb * factor - n
    mean = np.full((nb, nb), np.nan, dtype=np.float32)
    top = np.full((nb, nb), np.nan, dtype=np.float32)
    labels = np.asarray(clusters)
    k = int(labels.max()) + 1
    onehot = np.zeros((n, k), dtype=np.float32)
    onehot[np.arange(n), labels] = 1.0
    csum = np.zeros((k, k), dtype=np.float64)
    ccount = np.zeros((k, k), dtype=np.float64)
    counts = np.bincount(labels, minlength=k).astype(np.float64)
    import warnings

    for b in range(nb):
        rows = leaf_order[b * factor:(b + 1) * factor]
        order = np.argsort(rows)
        strip = np.asarray(mat[np.sort(rows)], dtype=np.float32)
        strip = strip[np.argsort(order)]                       # back to leaf order
        # Cluster means first, while the diagonal is still present (it contributes one
        # self-pair per family and the denominator counts it).
        csum += (labels[rows][:, None] == np.arange(k)[None, :]).T.astype(np.float64) @ (strip @ onehot)
        ccount += np.bincount(labels[rows], minlength=k).astype(np.float64)[:, None] * counts[None, :]
        strip[np.arange(rows.size), rows] = np.nan             # the diagonal is not a pair
        strip = strip[:, leaf_order]
        if pad:
            strip = np.concatenate([strip, np.full((strip.shape[0], pad), np.nan, np.float32)], axis=1)
        tile = strip.reshape(strip.shape[0], nb, factor)
        with warnings.catch_warnings():                        # all-NaN blocks are expected
            warnings.simplefilter('ignore', RuntimeWarning)
            mean[b] = np.nanmean(np.nanmean(tile, axis=2), axis=0)
            top[b] = np.nanmax(np.nanmax(tile, axis=2), axis=0)
        if b % 200 == 0:
            print(f'  block row {b:,}/{nb:,}', flush=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        cluster_mean = csum / np.maximum(ccount, 1.0)
    return mean, top, cluster_mean, nb, pad


def annotation_bars(rows, leaf_order, factor, project, shortlist):
    """Per-leaf annotation strips, block-reduced to the matrix figure's resolution."""
    n = len(rows)
    nb = math.ceil(n / factor)

    def reduce_numeric(values):
        v = np.asarray(values, dtype=np.float64)[leaf_order]
        if nb * factor > n:
            v = np.concatenate([v, np.full(nb * factor - n, np.nan)])
        return np.nanmean(v.reshape(nb, factor), axis=1)

    def reduce_flag(mask):
        v = np.asarray(mask, dtype=np.float64)[leaf_order]
        if nb * factor > n:
            v = np.concatenate([v, np.zeros(nb * factor - n)])
        return v.reshape(nb, factor).max(axis=1)

    def reduce_mode(codes):
        """Categorical strips take the block's most common value; a mean of category
        codes would invent categories that are not there."""
        v = np.asarray(codes, dtype=np.int64)[leaf_order]
        if nb * factor > n:
            v = np.concatenate([v, np.full(nb * factor - n, v[-1])])
        v = v.reshape(nb, factor)
        return np.array([np.bincount(block).argmax() for block in v], dtype=np.float64)

    types = sorted({r['type'] for r in rows})
    type_code = {t: i for i, t in enumerate(types)}
    clans = [r['clan_acc'] for r in rows]
    common_clans = sorted({c for c in clans if c},
                          key=lambda c: -sum(1 for x in clans if x == c))[:12]
    clan_code = {c: i + 2 for i, c in enumerate(common_clans)}
    clan_values = np.array([0 if not c else clan_code.get(c, 1) for c in clans], dtype=np.float64)
    return {
        'types': types,
        'common_clans': common_clans,
        'type': reduce_mode([type_code[r['type']] for r in rows]),
        'clan': reduce_mode(clan_values.astype(np.int64)),
        'log_members': reduce_numeric([math.log10(max(1, int(r['members_nr90']))) for r in rows]),
        'isolation': reduce_numeric([float(r['isolation_score_pct']) for r in rows]),
        'project': reduce_flag([r['pfam_acc'] in project for r in rows]),
        'shortlist': reduce_flag([r['pfam_acc'] in shortlist for r in rows]),
    }


def fig_matrix(mean, top, bars, space, factor, pad, n, shortlist_n):
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(13.5, 7.6))
    gs = GridSpec(2, 3, height_ratios=[0.17, 1], width_ratios=[1, 1, 0.04],
                  hspace=0.18, wspace=0.12)
    vmin = float(np.nanpercentile(mean, 1))
    handles = []
    for col, (data, title) in enumerate(((mean, 'block mean'), (top, 'block max'))):
        ax = fig.add_subplot(gs[1, col])
        im = ax.imshow(data, cmap=C.MATRIX_CMAP, interpolation='nearest', origin='upper',
                       vmin=vmin, vmax=1.0)
        ax.set_xlabel('families, dendrogram order')
        if col == 0:
            ax.set_ylabel('families, dendrogram order')
        ax.set_xticks([]); ax.set_yticks([])
        bar = fig.add_subplot(gs[0, col], sharex=ax)
        strips = np.vstack([bars['clan'], bars['type'], bars['log_members'], bars['isolation']])
        norm = (strips - np.nanmin(strips, axis=1, keepdims=True))
        norm /= np.maximum(np.nanmax(norm, axis=1, keepdims=True), 1e-9)
        bar.imshow(norm, aspect='auto', cmap='viridis', interpolation='nearest')
        bar.set_yticks(range(4))
        bar.set_yticklabels(['clan', 'type', 'log10 members', 'isolation pct'], fontsize=7)
        bar.set_xticks([])
        # The title belongs to the pair of panels, above the annotation strip.
        bar.set_title(title, fontsize=10, pad=4)
        for name, colour in (('project', C.C_PROJECT), ('shortlist', C.C_CANDIDATE)):
            marks = np.flatnonzero(bars[name] > 0)
            handle = bar.scatter(marks, np.full(marks.size, 3.9), s=3, c=colour, marker='|',
                                 linewidths=0.6,
                                 label=f'{name}: {marks.size} of {bars["clan"].size} blocks')
            if col == 0:
                handles.append(handle)
        if col == 1:
            cax = fig.add_subplot(gs[1, 2])
            fig.colorbar(im, cax=cax, label='cosine similarity')
    fig.legend(handles=handles, loc='lower center', ncol=2, fontsize=7,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(
        f'Pfam representative similarity, {space} space — {n:,} families, aggregated '
        f'{factor}x{factor} ({factor*factor} family pairs per pixel'
        + (f'; last block padded by {pad}' if pad else '') + f'); {shortlist_n:,} shortlisted',
        fontsize=11, y=0.98)
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    C.common_args(ap)
    ap.add_argument('--space', default=C.HEADLINE_SPACE, choices=list(C.SPACES))
    ap.add_argument('--block-factor', type=int, default=16)
    ap.add_argument('--exact-subsample', type=int, default=2000)
    ap.add_argument('--profile-sample', type=int, default=2000)
    ap.add_argument('--top', type=int, default=25)
    ap.add_argument('--cluster-cut', type=int, default=100)
    ap.add_argument('--no-interactive', action='store_true')
    args = ap.parse_args()
    C.apply_results_override(args.results)
    C.set_rc()
    import matplotlib.pyplot as plt

    started = time.time()
    sp = args.space
    matrix_dir = C.results_dir('matrix')
    sel_dir = C.results_dir('selection')

    C.banner('t4 — inputs')
    scored = C.read_tsv(sel_dir / 'families_scored.tsv')
    order_by_acc = {r['pfam_acc']: r for r in scored}
    metrics = C.read_tsv(matrix_dir / 'family_metrics.tsv')          # FASTA order
    rows = [order_by_acc[r['pfam_acc']] for r in metrics]            # scored, FASTA order
    n = len(rows)
    summary_sim = C.load_json(matrix_dir / 'summary_similarity.json')
    summary_sel = C.load_json(sel_dir / 'summary_selection.json')
    project = set(C.load_project_families())
    shortlist_rows = C.read_tsv(sel_dir / 'shortlist.tsv')
    shortlist = {r['pfam_acc'] for r in shortlist_rows}
    top_rows = shortlist_rows[:args.top]
    iso = np.array([float(r['isolation_rank_score']) for r in rows])
    iso_pct = np.array([float(r['isolation_score_pct']) for r in rows])
    members = np.array([int(r['members_nr90']) for r in rows])
    leaf_order = np.load(matrix_dir / f'leaf_order_{sp}.npy')
    clusters = np.array([int(r[f'{sp}_cluster_k{args.cluster_cut}']) for r in rows]) - 1
    matrix_path = matrix_dir / f'sim_{sp}.f16.npy'
    written = []

    # --- F1 / F4: one pass over the stored matrix -------------------------- #
    if matrix_path.exists():
        C.banner(f't4 — F1 block matrix ({args.block_factor}x)')
        mat = np.load(matrix_path, mmap_mode='r')
        mean, top, cluster_mean, nb, pad = block_pass(mat, leaf_order, clusters, args.block_factor)
        bars = annotation_bars(rows, leaf_order, args.block_factor, project, shortlist)
        written += C.save_fig(fig_matrix(mean, top, bars, sp, args.block_factor, pad, n,
                                         len(shortlist)), f'similarity_matrix_{sp}')
        np.savez(matrix_dir / f'block_matrix_{sp}.npz', mean=mean, max=top,
                 factor=args.block_factor, leaf_order=leaf_order)

        C.banner('t4 — F2 exact subsample')
        rng = np.random.default_rng(args.seed)
        pick = np.sort(rng.choice(n, size=min(args.exact_subsample, n), replace=False))
        sub = np.asarray(mat[pick][:, pick], dtype=np.float32)
        from scipy.cluster.hierarchy import leaves_list, linkage
        from scipy.spatial.distance import squareform

        d = np.clip(1.0 - sub, 0, None)
        np.fill_diagonal(d, 0.0)
        sub_order = leaves_list(linkage(squareform(d, checks=False), method='average'))
        fig, ax = plt.subplots(figsize=(7.4, 6.6))
        im = ax.imshow(sub[np.ix_(sub_order, sub_order)], cmap=C.MATRIX_CMAP,
                       interpolation='nearest', vmin=float(np.percentile(sub, 1)), vmax=1.0)
        ax.set(xticks=[], yticks=[], xlabel='families, dendrogram order',
               ylabel='families, dendrogram order',
               title=f'Exact submatrix — {pick.size:,} seeded families, one pixel per pair\n'
                     f'(no aggregation; the control for the {args.block_factor}x figure)')
        fig.colorbar(im, ax=ax, label='cosine similarity')
        written += C.save_fig(fig, f'similarity_matrix_exact_subsample_{sp}')

        C.banner('t4 — F4 cluster matrix')
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4))
        sizes = np.bincount(clusters)
        cl_order = np.argsort(-sizes)
        im = axes[0].imshow(cluster_mean[np.ix_(cl_order, cl_order)], cmap=C.MATRIX_CMAP,
                            interpolation='nearest')
        axes[0].set(title=f'Mean similarity between k={args.cluster_cut} clusters ({sp})',
                    xlabel='cluster (largest first)', ylabel='cluster (largest first)')
        fig.colorbar(im, ax=axes[0], label='mean cosine similarity')
        axes[1].bar(np.arange(sizes.size), sizes[cl_order], color=C.C_BULK)
        axes[1].set(yscale='log', xlabel='cluster (largest first)', ylabel='families in cluster',
                    title=f'Cluster sizes — largest holds {sizes.max():,} of {n:,} '
                          f'({sizes.max()/n:.1%})')
        C.style_axis(axes[1])
        written += C.save_fig(fig, f'cluster_matrix_{sp}')

        C.banner('t4 — F7/F8 candidate views')
        if top_rows:
            idx_by_acc = {r['pfam_acc']: i for i, r in enumerate(metrics)}
            topk = np.load(matrix_dir / f'topk_{sp}.npz')
            cols = min(5, len(top_rows))
            grid = math.ceil(len(top_rows) / cols)
            fig, axes = plt.subplots(grid, cols, figsize=(2.5 * cols, 2.6 * grid))
            axes = np.atleast_1d(axes).ravel()
            for ax, r in zip(axes, top_rows):
                i = idx_by_acc[r['pfam_acc']]
                nbrs = np.concatenate([[i], topk['top_idx'][i, :60]])
                block = np.asarray(mat[np.sort(nbrs)][:, np.sort(nbrs)], dtype=np.float32)
                keep = np.argsort(np.argsort(nbrs))
                block = block[np.ix_(keep, keep)]
                ax.imshow(block, cmap=C.MATRIX_CMAP, interpolation='nearest', vmin=0, vmax=1)
                ax.set(xticks=[], yticks=[], title=f"{r['pfam_acc']}\n{r['pfam_id'][:18]}", )
                ax.title.set_fontsize(7)
            for ax in axes[len(top_rows):]:
                ax.axis('off')
            fig.suptitle('Exact 61x61 neighbourhood of each candidate '
                         '(itself plus its 60 nearest families)', fontsize=10)
            written += C.save_fig(fig, f'candidate_neighbourhoods_{sp}')

            profile_idx = np.sort(rng.choice(n, size=min(args.profile_sample, n), replace=False))
            band = -np.sort(-np.asarray(mat[profile_idx], dtype=np.float32), axis=1)[:, 1:]
            xs = np.unique(np.geomspace(1, band.shape[1], 400).astype(int)) - 1
            fig, ax = plt.subplots(figsize=(8.2, 5.4))
            ax.fill_between(xs + 1, np.percentile(band[:, xs], 5, axis=0),
                            np.percentile(band[:, xs], 95, axis=0), color=C.C_BULK, alpha=0.35,
                            label=f'5-95% band ({band.shape[0]:,} seeded families)')
            ax.plot(xs + 1, np.median(band[:, xs], axis=0), color='#444', lw=1.4, label='median family')
            for r in top_rows[:8]:
                row = -np.sort(-np.asarray(mat[idx_by_acc[r['pfam_acc']]], dtype=np.float32))[1:]
                ax.plot(xs + 1, row[xs], lw=1.0, alpha=0.9, label=r['pfam_acc'])
            for acc, style in (('PF00106', '--'), ('PF00107', ':')):
                if acc in idx_by_acc:
                    row = -np.sort(-np.asarray(mat[idx_by_acc[acc]], dtype=np.float32))[1:]
                    ax.plot(xs + 1, row[xs], lw=1.6, ls=style, color=C.C_PROJECT, alpha=0.9,
                            label=f'{acc} (current anchor)')
            ax.set(xscale='log', xlabel='neighbour rank (log)', ylabel='cosine similarity',
                   title='Similarity to every other family, sorted — "far from the bulk" at every scale')
            ax.legend(fontsize=7, ncol=2)
            C.style_axis(ax)
            written += C.save_fig(fig, f'bulk_profile_candidates_{sp}')
        del mat
    else:
        print(f'note: {matrix_path} absent — matrix figures skipped', flush=True)

    # --- F3: similarity distributions -------------------------------------- #
    C.banner('t4 — F3 distributions')
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    for key in summary_sim['spaces']:
        path = matrix_dir / f'hist_{key}.npz'
        if not path.exists():
            continue
        z = np.load(path)
        centres = 0.5 * (z['edges'][1:] + z['edges'][:-1])
        counts = z['counts'].astype(np.float64)
        axes[0].plot(centres, counts / counts.sum(), lw=1.2, label=key)
        topk = np.load(matrix_dir / f'topk_{key}.npz')
        axes[1].hist(topk['top_sim'][:, 0], bins=120, histtype='step', lw=1.2, density=True, label=key)
    axes[0].set(yscale='log', xlabel='cosine similarity (all off-diagonal pairs)',
                ylabel='fraction of pairs', title='Off-diagonal similarity, exact')
    axes[1].set(xlabel='similarity to nearest other family', ylabel='density',
                title='Nearest-neighbour similarity')
    for ax in axes:
        ax.legend(fontsize=8)
        C.style_axis(ax)
    aniso = {k: v['anisotropy'] for k, v in summary_sim['spaces'].items()}
    note = '  |  '.join(f"{k}: |mean unit vector| = {v['mean_resultant_length']:.3f}, "
                        f"eff. rank {v['effective_rank']:.0f}" for k, v in aniso.items())
    fig.suptitle('Why the headline space is centered — ' + note, fontsize=9)
    written += C.save_fig(fig, 'similarity_distributions')

    # --- F5: components ----------------------------------------------------- #
    C.banner('t4 — F5 components')
    comp = list(C.ISOLATION_COMPONENTS)
    corr = summary_sel['component_spearman']
    mat_corr = np.array([[corr[a][b] for b in comp] for a in comp])
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
    im = axes[0].imshow(mat_corr, cmap='PuOr', vmin=-1, vmax=1)
    axes[0].set(xticks=range(len(comp)), yticks=range(len(comp)),
                title='Spearman between isolation components')
    axes[0].set_xticklabels(comp, rotation=45, ha='right', fontsize=7)
    axes[0].set_yticklabels(comp, fontsize=7)
    for i in range(len(comp)):
        for j in range(len(comp)):
            axes[0].text(j, i, f'{mat_corr[i, j]:.2f}', ha='center', va='center', fontsize=6)
    fig.colorbar(im, ax=axes[0], label='Spearman rho')
    other = 'raw' if sp != 'raw' else 'cen'
    if f'{other}_iso_nn_xclan' in metrics[0]:
        alt = np.mean([C.pctrank(np.array([float(r[f'{other}_{c}']) for r in metrics]))
                       for c in comp], axis=0)
        changed = np.array([r['pfam_acc'] in shortlist for r in rows])
        axes[1].scatter(alt, iso, s=5, c=C.C_BULK, alpha=0.4, linewidths=0)
        axes[1].scatter(alt[changed], iso[changed], s=14, c=C.C_CANDIDATE, linewidths=0,
                        label='shortlisted')
        from scipy.stats import spearmanr

        axes[1].set(xlabel=f'isolation score, {other} space', ylabel=f'isolation score, {sp} space',
                    title=f'Space sensitivity — Spearman {spearmanr(alt, iso).statistic:.3f}')
        axes[1].legend(fontsize=8)
        C.style_axis(axes[1])
    written += C.save_fig(fig, 'isolation_components')

    # --- F6: size vs isolation ---------------------------------------------- #
    C.banner('t4 — F6 size vs isolation')
    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    failed = np.array([int(r['n_gates_failed']) for r in rows])
    sc = ax.scatter(members, iso, c=failed, cmap='viridis_r', s=9, alpha=0.6, linewidths=0)
    short = np.array([r['pfam_acc'] in shortlist for r in rows])
    ax.scatter(members[short], iso[short], s=26, facecolors='none', edgecolors=C.C_CANDIDATE,
               linewidths=0.8, label=f'shortlist ({short.sum():,})')
    proj = np.array([r['pfam_acc'] in project for r in rows])
    ax.scatter(members[proj], iso[proj], marker='*', s=90, c=C.C_PROJECT,
               label='already used by this project')
    for r in top_rows[:8]:
        i = [j for j, q in enumerate(rows) if q['pfam_acc'] == r['pfam_acc']][0]
        ax.annotate(r['pfam_acc'], (members[i], iso[i]), fontsize=7,
                    xytext=(5, 4), textcoords='offset points')
    ax.axvline(summary_sel['thresholds']['min_members'], ls='--', c='#777', lw=0.8)
    ax.axhline(summary_sel['thresholds']['isolation_score_threshold'], ls='--', c=C.C_CANDIDATE, lw=0.8)
    ax.set(xscale='log', xlabel='NR90 member proteins (log)', ylabel='isolation score (mean percentile rank)',
           title=f'Family size versus isolation from the Pfam bulk ({sp} space)')
    fig.colorbar(sc, ax=ax, label='gates failed')
    ax.legend(fontsize=8, loc='lower left')
    C.style_axis(ax)
    written += C.save_fig(fig, f'size_vs_isolation_{sp}')

    # --- F9: waterfall ------------------------------------------------------ #
    water = summary_sel['gate_waterfall']
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ax.bar([w['gate'] for w in water], [w['survivors'] for w in water], color=C.C_BULK)
    ax.bar([w['gate'] for w in water], [w['survivors_clanless'] for w in water], color=C.C_OK,
           label='clan-less')
    for i, w in enumerate(water):
        ax.text(i, w['survivors'], f"{w['survivors']:,}", ha='center', va='bottom', fontsize=7)
    ax.set(yscale='log', ylabel='families surviving', title='Cumulative gate waterfall')
    ax.set_xticklabels([w['gate'].replace('gate_', '') for w in water], rotation=30, ha='right')
    ax.legend(fontsize=8)
    C.style_axis(ax)
    written += C.save_fig(fig, 'gate_waterfall')

    # --- F10: repeat calibration -------------------------------------------- #
    cal_path = C.results_dir('seqstats') / 'retain_calibration.npz'
    stats = {r['pfam_acc']: r for r in C.read_tsv(C.results_dir('seqstats') / 'rep_sequence_stats.tsv')}
    if cal_path.exists():
        C.banner('t4 — F10 repeat calibration')
        cal = np.load(cal_path)
        names = [m.name for m in C.SEQ_METRICS]
        fig, axes = plt.subplots(1, len(names), figsize=(3.0 * len(names), 3.6))
        for ax, name in zip(np.atleast_1d(axes).ravel(), names):
            ax.hist(cal[f'sample_{name}'], bins=60, density=True, color=C.C_BULK, alpha=0.7,
                    label='retain')
            # Full-chain values: that is the scope where the control types separate, and the
            # scope an anchor's training corpus would actually consist of.
            for label, colour in (('Family', C.C_OK), ('Coiled-coil', C.C_CANDIDATE),
                                  ('Disordered', C.C_REPEAT)):
                vals = [float(stats[r['pfam_acc']][f'full_{name}']) for r in rows
                        if r['type'] == label and np.isfinite(float(stats[r['pfam_acc']][f'full_{name}']))]
                if vals:
                    ax.hist(vals, bins=60, density=True, histtype='step', lw=1.2, color=colour,
                            label=f'{label} ({len(vals):,})')
            ax.set(title=name, xlabel='value')
            ax.set_yticks([])
            C.style_axis(ax, ygrid=False)
        np.atleast_1d(axes).ravel()[0].set_ylabel('density')
        np.atleast_1d(axes).ravel()[0].legend(fontsize=6)
        fig.suptitle('Low-complexity metrics (full chain) — retain reference vs Pfam types. '
                     'Coiled-coil and Disordered families are the positive control; Repeat-type '
                     'families do NOT separate here (t2), which is why Pfam type is reported for '
                     'every family and never used as a gate.', fontsize=9)
        written += C.save_fig(fig, 'repeat_calibration')

    # --- interactive explorer ------------------------------------------------ #
    interactive = None
    if not args.no_interactive:
        interactive = write_interactive(rows, sp, summary_sel, shortlist, project)

    C.save_json({'protocol': C.protocol_block(stage='t4_figures', space=sp),
                 'block_factor': args.block_factor,
                 'pairs_per_pixel': args.block_factor ** 2,
                 'exact_subsample': int(args.exact_subsample),
                 'profile_sample': int(args.profile_sample),
                 'figures': [str(p.relative_to(C.ROOT)) for p in written],
                 'interactive': str(interactive.relative_to(C.ROOT)) if interactive else None,
                 'seconds': time.time() - started},
                C.results_dir('plots') / 'summary_figures.json')
    print(f'\nDONE t4 in {time.time()-started:.0f}s — {len(written)} files -> '
          f'{C.results_dir("plots")}', flush=True)


def write_interactive(rows, space, summary_sel, shortlist, project):
    """Self-contained hover/filter explorer over the existing UMAP coordinates.

    Same canvas pattern as family_counts.py: no network, no libraries. Coordinates are the
    UMAP fit from `analyze.py` and are display only — every number in the tooltip comes
    from the 480-d matrix.
    """
    coords_tsv = C.UMAP_DIR / 'coordinates.tsv'
    coords_npy = C.UMAP_DIR / 'coords.npy'
    if not (coords_tsv.exists() and coords_npy.exists()):
        print('note: UMAP coordinates absent — interactive explorer skipped', flush=True)
        return None
    coords = np.load(coords_npy)
    meta = C.read_tsv(coords_tsv)
    if len(meta) != len(coords):
        print('note: UMAP coordinate/metadata mismatch — interactive explorer skipped', flush=True)
        return None
    xy = {r['id']: (round(float(coords[i][0]), 3), round(float(coords[i][1]), 3))
          for i, r in enumerate(meta) if r['group'] == 'pfam_reps'}
    retain_xy = [[round(float(coords[i][0]), 2), round(float(coords[i][1]), 2)]
                 for i, r in enumerate(meta) if r['group'] == 'retain'][::7]
    payload = []
    for r in rows:
        if r['pfam_acc'] not in xy:
            continue
        x, y = xy[r['pfam_acc']]
        payload.append([
            x, y, r['pfam_acc'], r['pfam_id'], r['description'][:90], r['type'],
            int(r['members_nr90']), round(float(r['isolation_rank_score']), 1),
            round(float(r[f'{space}_iso_nn_xclan']), 4), round(float(r[f'{space}_lof50']), 2),
            r['clan_acc'] or '', int(r['model_length']),
            round(float(r['dom_max_periodicity']), 3), round(float(r['distance_to_retain']), 4),
            int(r['n_gates_failed']),
            (2 if r['pfam_acc'] in shortlist else 1 if r['pfam_acc'] in project else 0),
            ''.join('1' if r[g] == 'True' or r[g] is True else '0' for g in C.GATE_ORDER),
        ])
    data = json.dumps({'reps': payload, 'retain': retain_xy, 'gates': list(C.GATE_ORDER),
                       'space': space}, separators=(',', ':')).replace('<', '\\u003c')
    html = INTERACTIVE_TEMPLATE.replace('DATA_PLACEHOLDER', data)
    path = C.results_dir('plots') / 'anchors_interactive.html'
    path.write_text(html)
    print(f'  wrote {path} ({path.stat().st_size/1e6:.1f} MB)', flush=True)
    return path


INTERACTIVE_TEMPLATE = '''<!doctype html><meta charset="utf-8">
<title>Isolated Pfam families — anchor screen</title>
<style>body{margin:0;font:14px system-ui;color:#17212b;background:#fafafa}
header{padding:12px 20px}h2{margin:0 0 4px}p{margin:4px 0;max-width:70em}
#wrap{position:relative;margin:0 18px}canvas{display:block;background:#fff;border:1px solid #ccc;touch-action:none}
#tip{display:none;position:absolute;pointer-events:none;white-space:pre-line;background:#fffffff5;
border:1px solid #888;border-radius:5px;padding:9px;max-width:380px;font-size:12px}
label{margin-right:14px;font-size:13px}input,button{font:inherit}</style>
<header><h2>Pfam families by isolation from the bulk</h2>
<p>Colour is the isolation score (mean percentile rank over six components, measured in the
480-d embedding space). Positions are the existing UMAP fit and are <b>display only</b> —
UMAP distances are not metric. Hover a point for its metrics and gate outcomes; drag to pan,
scroll to zoom.</p>
<div><label><input type="checkbox" id="fShort"> shortlist only</label>
<label><input type="checkbox" id="fClan"> clan-less only</label>
<label><input type="checkbox" id="fSize" checked> &ge;1000 members</label>
<label><input type="checkbox" id="fRetain" checked> show retain</label>
<label>find <input id="search" placeholder="PF00106 or adh_short" size="18"></label>
<button id="reset">reset view</button></div>
<div id="status" style="font-size:12px;min-height:18px"></div></header>
<div id="wrap"><canvas id="plot"></canvas><div id="tip"></div></div>
<script>const D=DATA_PLACEHOLDER;
const F={ACC:2,ID:3,DESC:4,TYPE:5,MEM:6,ISO:7,NN:8,LOF:9,CLAN:10,MLEN:11,PER:12,RET:13,FAIL:14,ROLE:15,GATES:16};
const cv=document.getElementById('plot'),ctx=cv.getContext('2d'),tip=document.getElementById('tip'),
status=document.getElementById('status');
let view,initial,drag=null,pts=[],raf=false;
const b=[Infinity,-Infinity,Infinity,-Infinity];
for(const p of D.reps){b[0]=Math.min(b[0],p[0]);b[1]=Math.max(b[1],p[0]);b[2]=Math.min(b[2],p[1]);b[3]=Math.max(b[3],p[1]);}
function colour(v){const t=Math.max(0,Math.min(1,v/100));
const r=Math.round(20+t*220),g=Math.round(90-t*40+t*t*60),bl=Math.round(160-t*140);return `rgb(${r},${g},${bl})`;}
function visible(p){if(document.getElementById('fShort').checked&&p[F.ROLE]!==2)return false;
if(document.getElementById('fClan').checked&&p[F.CLAN])return false;
if(document.getElementById('fSize').checked&&p[F.MEM]<1000)return false;return true;}
function resize(){cv.width=Math.max(600,window.innerWidth-60);cv.height=Math.max(420,window.innerHeight-250);
initial={x:b[0],y:b[2],w:b[1]-b[0],h:b[3]-b[2]};if(!view)view={...initial};draw();}
function toScreen(p){return [(p[0]-view.x)/view.w*cv.width,cv.height-(p[1]-view.y)/view.h*cv.height];}
function draw(){ctx.clearRect(0,0,cv.width,cv.height);
if(document.getElementById('fRetain').checked){ctx.fillStyle='rgba(160,168,175,0.25)';
for(const q of D.retain){const[s,t]=toScreen(q);if(s>=0&&s<=cv.width&&t>=0&&t<=cv.height)ctx.fillRect(s,t,1,1);}}
pts=[];let shown=0;
for(const p of D.reps){if(!visible(p))continue;const[s,t]=toScreen(p);
if(s<-5||s>cv.width+5||t<-5||t>cv.height+5)continue;shown++;
const r=p[F.ROLE]===2?3.4:p[F.ROLE]===1?3.0:1.8;
ctx.beginPath();ctx.arc(s,t,r,0,6.283);ctx.fillStyle=colour(p[F.ISO]);ctx.fill();
if(p[F.ROLE]===2){ctx.strokeStyle='#D55E00';ctx.lineWidth=1.1;ctx.stroke();}
else if(p[F.ROLE]===1){ctx.strokeStyle='#0072B2';ctx.lineWidth=1.1;ctx.stroke();}
pts.push([s,t,p]);}
status.textContent=`${shown.toLocaleString()} families drawn · orange ring = shortlist · blue ring = already used by this project`;}
function schedule(){if(!raf){raf=true;requestAnimationFrame(()=>{raf=false;draw();});}}
cv.addEventListener('mousemove',e=>{const r=cv.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top;
if(drag){view.x-=(mx-drag.x)/cv.width*view.w;view.y+=(my-drag.y)/cv.height*view.h;drag={x:mx,y:my};schedule();return;}
let best=null,bd=100;for(const[s,t,p]of pts){const d=(s-mx)**2+(t-my)**2;if(d<bd){bd=d;best=p;}}
if(!best){tip.style.display='none';return;}
const g=best[F.GATES].split('').map((c,i)=>`${c==='1'?'✓':'✗'} ${D.gates[i].replace('gate_','')}`).join('\\n');
tip.style.display='block';tip.style.left=(mx+14)+'px';tip.style.top=(my+12)+'px';
tip.textContent=`${best[F.ACC]} ${best[F.ID]} (${best[F.TYPE]})\\n${best[F.DESC]}\\n`+
`members ${best[F.MEM].toLocaleString()} · HMM ${best[F.MLEN]} aa · clan ${best[F.CLAN]||'none'}\\n`+
`isolation ${best[F.ISO]} · nearest non-clan family ${best[F.NN]} · LOF50 ${best[F.LOF]}\\n`+
`max periodicity ${best[F.PER]} · distance to retain ${best[F.RET]}\\n${g}`;});
cv.addEventListener('mousedown',e=>{const r=cv.getBoundingClientRect();drag={x:e.clientX-r.left,y:e.clientY-r.top};});
window.addEventListener('mouseup',()=>drag=null);
cv.addEventListener('wheel',e=>{e.preventDefault();const r=cv.getBoundingClientRect(),
mx=(e.clientX-r.left)/cv.width,my=1-(e.clientY-r.top)/cv.height,k=e.deltaY>0?1.12:0.89;
view.x+=view.w*mx*(1-k);view.y+=view.h*my*(1-k);view.w*=k;view.h*=k;schedule();},{passive:false});
for(const id of ['fShort','fClan','fSize','fRetain'])document.getElementById(id).addEventListener('change',draw);
document.getElementById('reset').addEventListener('click',()=>{view={...initial};draw();});
document.getElementById('search').addEventListener('change',e=>{const q=e.target.value.trim().toLowerCase();
const hit=D.reps.find(p=>p[F.ACC].toLowerCase()===q||p[F.ID].toLowerCase().includes(q));
if(!hit){status.textContent='no family matches '+q;return;}
view={x:hit[0]-initial.w/14,y:hit[1]-initial.h/14,w:initial.w/7,h:initial.h/7};draw();
status.textContent=`${hit[F.ACC]} ${hit[F.ID]} — isolation ${hit[F.ISO]}, members ${hit[F.MEM].toLocaleString()}`;});
window.addEventListener('resize',resize);resize();
</script>
'''


if __name__ == '__main__':
    main()
