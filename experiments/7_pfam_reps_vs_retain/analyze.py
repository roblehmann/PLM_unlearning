#!/usr/bin/env python3
"""Embed complete Pfam representatives, R, and F corpora; fit one joint PCA/UMAP."""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import numpy as np

MODEL = 'facebook/esm2_t12_35M_UR50D'
GROUPS = {
    'pfam_reps': 'data/Pfam_family_representatives/family_representatives.fasta',
    'retain': 'data/splits/retain_train.fasta',
    'MGYF0000018690_F_train': 'runs/MGYF18690_ladder_decon/data/splits/F_train.fasta',
    'MGYF0000018690_F_test': 'runs/MGYF18690_ladder_decon/data/splits/F_test.fasta',
    'PF00106_F_train': 'data/ladder/PF00106/fullchain/F_train_full.fasta',
    'PF00106_F_test': 'data/ladder/PF00106/fullchain/F_test_full.fasta',
    'PF00107_F_train': 'data/ladder/PF00107/fullchain/F_train_full.fasta',
    'PF00107_F_test': 'data/ladder/PF00107/fullchain/F_test_full.fasta',
}
COLORS = {'pfam_reps': '#D55E00', 'retain': '#9ba5ad',
          'MGYF0000018690': '#009E73', 'PF00106': '#0072B2', 'PF00107': '#CC79A7'}


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(path)


def records(path):
    from Bio import SeqIO
    result = list(SeqIO.parse(path, 'fasta'))
    if not result or any(not len(r.seq) for r in result):
        raise ValueError(f'Empty corpus or sequence: {path}')
    return result


def embed(args):
    import torch
    from src.models.load import load_mlm, load_tokenizer
    from src.eval.membership import embed_sequences
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; submit embed.sbatch on a GPU node.')
    torch.manual_seed(args.seed)
    emb = args.out / 'embeddings'
    emb.mkdir(parents=True, exist_ok=True)
    model = tokenizer = None
    protocol = dict(model=MODEL, layer=12, hidden_dim=480, max_length=1024,
                    max_residues=1022, pooling='mean over non-special non-padding tokens',
                    autocast='float16', storage='float16', seed=args.seed,
                    batch_size=args.batch_size, limit_per_group=args.limit,
                    versions={p: importlib.metadata.version(p) for p in
                              ('torch', 'transformers', 'numpy', 'biopython')})
    manifest = {'protocol': protocol, 'groups': {}}
    for group, relative in GROUPS.items():
        path = ROOT / relative
        rs = records(path)
        total = len(rs)
        if args.limit:
            rs = rs[:args.limit]
        fingerprint = dict(fasta=relative, sha256=digest(path), n_total=total,
                           n=len(rs), protocol=protocol)
        npy, tsv, meta = (emb / f'{group}{ext}' for ext in ('.npy', '.tsv', '.json'))
        if not args.force and all(p.exists() for p in (npy, tsv, meta)):
            old = json.loads(meta.read_text())
            x = np.load(npy, mmap_mode='r')
            with tsv.open() as f:
                ids = [r['id'] for r in csv.DictReader(f, delimiter='\t')]
            if (old['fingerprint'] == fingerprint and x.shape == (len(rs), 480)
                    and ids == [r.id for r in rs] and np.isfinite(x).all()
                    and digest(npy) == old['embeddings_sha256']
                    and digest(tsv) == old['metadata_sha256']):
                manifest['groups'][group] = old
                print(f'{group}: verified local cache {x.shape}', flush=True)
                continue
        if model is None:
            tokenizer = load_tokenizer(MODEL)
            model = load_mlm(MODEL, device='cuda').eval()
            manifest['model_commit'] = getattr(model.config, '_commit_hash', None)
        start = time.time()
        seqs = [str(r.seq) for r in rs]
        order = np.argsort([len(s) for s in seqs], kind='stable')
        x = np.empty((len(rs), 480), dtype=np.float16)
        # Bound both padding and attention memory; restore original FASTA row order.
        done = 0
        while done < len(order):
            batch = min(args.batch_size, len(order) - done)
            while batch > 1 and batch * (min(len(seqs[order[done+batch-1]]), 1022)+2)**2 > 16_777_216:
                batch = max(1, batch // 2)
            idx = order[done:done+batch]
            with torch.autocast('cuda', dtype=torch.float16):
                values = embed_sequences(model, tokenizer, [seqs[i] for i in idx],
                                         device='cuda', batch_size=batch, max_length=1024)
            if values.shape != (len(idx), 480) or not np.isfinite(values).all():
                raise ValueError(f'Invalid embeddings: {group} at {done}')
            x[idx] = values
            done += batch
            if done == len(order) or done % (args.batch_size * 50) == 0:
                print(f'{group}: {done:,}/{len(order):,} ({time.time()-start:.1f}s)', flush=True)
        tmp = npy.with_suffix('.tmp.npy')
        np.save(tmp, x)
        tmp.replace(npy)
        with tsv.open('w') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['row', 'id', 'length', 'embedded_residues', 'truncated', 'sequence_sha256', 'description'])
            for i, r in enumerate(rs):
                writer.writerow([i, r.id, len(r.seq), min(len(r.seq), 1022), len(r.seq)>1022,
                                 hashlib.sha256(str(r.seq).encode()).hexdigest(), r.description])
        info = dict(fingerprint=fingerprint, seconds=time.time()-start,
                    model_commit=getattr(model.config, '_commit_hash', None),
                    truncated=sum(len(s)>1022 for s in seqs),
                    embeddings_sha256=digest(npy), metadata_sha256=digest(tsv))
        save_json(meta, info)
        manifest['groups'][group] = info
        save_json(emb / 'manifest.json', manifest)
    save_json(emb / 'manifest.json', manifest)


def plot(coords, labels, rows, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.colors import LogNorm, LinearSegmentedColormap
    from family_counts import load_counts, export
    family_table, member_counts = load_counts(rows)
    norm = LogNorm(vmin=member_counts.min(), vmax=member_counts.max())
    cmap = LinearSegmentedColormap.from_list('family_members', plt.get_cmap('YlOrBr')(np.linspace(.3, 1, 256)))
    color_map = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    anchors = ['MGYF0000018690', 'PF00106', 'PF00107']
    def draw(ax, selected):
        m = labels == 'retain'
        ax.scatter(*coords[m].T, s=0.5, c=COLORS['retain'], alpha=.16, linewidths=0, rasterized=True)
        m = labels == 'pfam_reps'
        ax.scatter(*coords[m].T, s=12, c=member_counts, norm=norm, cmap=cmap, alpha=.95,
                   edgecolors='#56432a', linewidths=.25, zorder=4, rasterized=True)
        handles = [Line2D([], [], marker='o', linestyle='', color=COLORS[g], markersize=5,
                          label=f'{name} (n={sum(labels == g):,})')
                   for g, name in [('retain', 'Retain R'), ('pfam_reps', 'Pfam representatives')]]
        for anchor in selected:
            for split, marker, size, alpha in [('train', 'o', 2, .22), ('test', '^', 9, .8)]:
                m = labels == f'{anchor}_F_{split}'
                ax.scatter(*coords[m].T, s=size, c=COLORS[anchor], marker=marker,
                           alpha=alpha, linewidths=0, rasterized=True)
                handles.append(Line2D([], [], marker=marker, linestyle='', color=COLORS[anchor],
                                      markersize=5, label=f'{anchor} F {split} (n={m.sum():,})'))
        for i, row in enumerate(rows):
            if row['group'] == 'pfam_reps' and row['id'] in ('PF00106', 'PF00107'):
                ax.scatter(*coords[i], marker='*', s=140, c=COLORS[row['id']], edgecolors='black', linewidths=.8, zorder=10)
                ax.annotate(row['id']+' representative\n'+f"{int(family_table[row['id']]['n_proteins_nr90']):,} member proteins (NR90)", coords[i], xytext=(-8 if row['id'] == 'PF00107' else 8, 8),
                            ha='right' if row['id'] == 'PF00107' else 'left',
                            bbox=dict(facecolor='white', alpha=.8, edgecolor='none', pad=2),
                            textcoords='offset points', fontsize=8, zorder=11)
        ax.set(xlabel='UMAP 1', ylabel='UMAP 2')
        ax.legend(handles=handles, loc='upper left', fontsize=8, framealpha=.94, markerscale=1)
    fig, ax = plt.subplots(figsize=(13, 10))
    draw(ax, anchors)
    ax.set_title('Pretrained ESM2-35M: Pfam representatives, retain R, and anchor sets F\n'
                 'All sequences · final-layer residue mean · joint PCA-50 → UMAP', pad=12)
    fig.colorbar(color_map, ax=ax, fraction=.035, pad=.03, label='Pfam member proteins (NR90; logarithmic scale)')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(out / f'landscape.{ext}', dpi=250)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(21, 7), sharex=True, sharey=True)
    for ax, anchor in zip(axes, anchors):
        draw(ax, [anchor])
        ax.set_title(anchor)
    fig.suptitle('Anchor views in the same joint UMAP coordinates', fontsize=15)
    fig.tight_layout(rect=(0, .12, 1, 1))
    cax = fig.add_axes([.35, .055, .3, .022])
    fig.colorbar(color_map, cax=cax, orientation='horizontal', label='Pfam member proteins (NR90; logarithmic scale)')
    for ext in ('png', 'pdf'):
        fig.savefig(out / f'landscape_by_anchor.{ext}', dpi=250)
    plt.close(fig)

    export(coords, labels, rows, out, family_table, member_counts, cmap, norm)


def replot(args):
    with (args.out / 'umap/coordinates.tsv').open() as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    coords = np.load(args.out / 'umap/coords.npy')
    if coords.shape != (len(rows), 2):
        raise ValueError('Coordinate row mismatch')
    if not np.allclose(coords, [[float(r['umap_x']), float(r['umap_y'])] for r in rows]):
        raise ValueError('Coordinate metadata mismatch')
    plot(coords, np.array([r['group'] for r in rows]), rows, args.out / 'plots')
    print('Updated plots and family counts from existing coordinates', flush=True)


def project(args):
    from sklearn.decomposition import PCA
    import umap
    import joblib
    out = args.out / 'umap'
    plots = args.out / 'plots'
    out.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)
    arrays, rows, counts, hashes = [], [], {}, {}
    manifest = json.loads((args.out / 'embeddings/manifest.json').read_text())
    for group in GROUPS:
        stem = args.out / 'embeddings' / group
        info = manifest['groups'][group]
        hashes[group] = digest(stem.with_suffix('.npy'))
        if hashes[group] != info['embeddings_sha256'] or digest(stem.with_suffix('.tsv')) != info['metadata_sha256']:
            raise ValueError(f'Embedding or metadata checksum mismatch: {group}')
        x = np.load(stem.with_suffix('.npy')).astype(np.float32)
        with stem.with_suffix('.tsv').open() as f:
            group_rows = [dict(r, group=group) for r in csv.DictReader(f, delimiter='\t')]
        if x.shape != (len(group_rows), 480) or not np.isfinite(x).all():
            raise ValueError(f'Invalid embedding rows: {group}')
        arrays.append(x)
        rows.extend(group_rows)
        counts[group] = len(group_rows)
    x = np.concatenate(arrays)
    del arrays
    labels = np.array([r['group'] for r in rows])
    print(f'Joint fit: {x.shape}; groups={counts}', flush=True)
    start = time.time()
    pca = PCA(n_components=50, svd_solver='randomized', random_state=args.seed)
    xp = pca.fit_transform(x)
    del x
    np.save(out / 'pca50.npy', xp)
    joblib.dump(pca, out / 'pca.joblib')
    pca_seconds = time.time()-start
    print(f'PCA: {pca.explained_variance_ratio_.sum():.5f} variance; {pca_seconds:.1f}s', flush=True)
    start = time.time()
    reducer = umap.UMAP(n_neighbors=30, min_dist=.1, metric='cosine',
                        random_state=args.seed, n_jobs=1, verbose=True)
    coords = reducer.fit_transform(xp)
    if coords.shape != (len(rows), 2) or not np.isfinite(coords).all():
        raise ValueError('Invalid UMAP coordinates')
    np.save(out / 'coords.npy', coords)
    joblib.dump(reducer, out / 'umap.joblib')
    with (out / 'coordinates.tsv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=['joint_row', 'group', 'row', 'id', 'length',
            'embedded_residues', 'truncated', 'sequence_sha256', 'description', 'umap_x', 'umap_y'], delimiter='\t')
        writer.writeheader()
        for i, r in enumerate(rows):
            writer.writerow(dict(r, joint_row=i, umap_x=float(coords[i, 0]), umap_y=float(coords[i, 1])))
    summary = dict(n_points=len(rows), counts=counts, model=MODEL,
                   pca_dim=50, pca_explained_variance=float(pca.explained_variance_ratio_.sum()),
                   n_neighbors=30, min_dist=.1, metric='cosine', seed=args.seed,
                   pca_seconds=pca_seconds, umap_seconds=time.time()-start,
                   embedding_hashes=hashes,
                   versions={p: importlib.metadata.version(p) for p in ('numpy', 'scikit-learn', 'umap-learn', 'matplotlib')},
                   unique_sequences=len({r['sequence_sha256'] for r in rows}),
                   caveat='UMAP preserves local neighborhoods; inter-cluster distances and area are not quantitative measures of biological similarity.')
    save_json(args.out / 'summary.json', summary)
    plot(coords, labels, rows, plots)
    coverage = ('All input records are included; there is no subsampling in the full run.'
                if not manifest['protocol']['limit_per_group'] else
                f"SMOKE RUN: first {manifest['protocol']['limit_per_group']} records per group only.")
    table = '\n'.join(f'| {g} | {n:,} | {manifest["groups"][g]["truncated"]:,} |' for g, n in counts.items())
    (args.out / 'RESULTS.md').write_text(f'''# Pfam representatives versus retain and anchor sets

Computed **{len(rows):,}** embeddings with the unmodified pretrained `{MODEL}` model.
{coverage}

| Set | Sequences | Truncated (>1,022 residues) |
| --- | ---: | ---: |
{table}

## Method

Final layer (12), 480 dimensions, mean pooling excluding padding and special tokens.
All sets freshly embedded using the same model, float16 CUDA autocast and maximum
1,024 tokens (1,022 residues); embeddings stored as float16 in FASTA order.
Sequences are used as provided, including full-chain Pfam representatives; domain
coordinates in headers are retained as metadata, not used to crop sequences.
Anchor sets F comprise train and test for MGYF0000018690, PF00106 and PF00107.
R is `data/splits/retain_train.fasta`. F-prime and ladder families are outside this analysis.
Repeated sequences are retained as input records ({summary['unique_sequences']:,} unique sequence hashes).

One joint randomized PCA-50 explained **{summary['pca_explained_variance']:.2%}** of variance.
UMAP fitted jointly to all PCA rows: cosine metric, 30 neighbors, min_dist=0.1,
seed={args.seed}. All panels use the same fitted coordinates. Coordinates from other
independent UMAP fits are not directly comparable. UMAP is exploratory: distances
between clusters and plotted areas do not quantify biological similarity. Corpus
size imbalance and sequence truncation can influence the landscape.

## Family membership counts

Pfam representatives are colored by `n_proteins_nr90` from
`data/Pfam_family_representatives/family_representatives.tsv`, joined by Pfam accession.
This is the number of member proteins in the supplied NR90 metadata, rather than the
number of domain regions (`n_regions_nr90`) or the number of sequences plotted.
The shared logarithmic color scale covers 1–591,968 proteins across all 21,763 families.
PF00106 and PF00107 have 189,045 and 99,059 member proteins, respectively.

[Interactive UMAP](plots/landscape_interactive.html): open in a browser, hover for exact
counts, search by Pfam accession/name, drag to pan, and scroll to zoom. The HTML is
self-contained and requires no network connection. Every Pfam representative is
included, together with all R/F points. Coordinates are unchanged from the original fit.
Exact counts and coordinates are also in [pfam_family_counts.tsv](plots/pfam_family_counts.tsv);
source and coordinate checksums are in `plots/family_counts_manifest.json`.

## Artifacts

- [Overview](plots/landscape.png) ([PDF](plots/landscape.pdf))
- [Per-anchor views](plots/landscape_by_anchor.png) ([PDF](plots/landscape_by_anchor.pdf))
- `embeddings/`: per-set `.npy`, metadata `.tsv`, and provenance `.json`.
- `umap/coordinates.tsv`: all labeled coordinates and sequence metadata.
- `umap/coords.npy`, `umap/pca50.npy`, fitted PCA and UMAP `.joblib` files.
- `summary.json`: counts, parameters, software versions, checksums, and timings.
''')
    print(f'DONE: {args.out / "RESULTS.md"}', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage', choices=['embed', 'umap', 'plot'])
    ap.add_argument('--out', type=Path, default=HERE / 'results')
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--limit', type=int, default=0, help='Smoke only: first N records per group')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    if args.batch_size < 1 or args.limit < 0:
        ap.error('batch-size must be positive and limit nonnegative')
    if args.limit and args.out.resolve() == (HERE / 'results').resolve():
        ap.error('--limit requires a separate --out directory')
    args.out.mkdir(parents=True, exist_ok=True)
    {'embed': embed, 'umap': project, 'plot': replot}[args.stage](args)


if __name__ == '__main__':
    main()
