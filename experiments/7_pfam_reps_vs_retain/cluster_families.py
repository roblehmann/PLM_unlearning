#!/usr/bin/env python3
"""Hierarchical cosine clustering and exact retain-neighbor screening of Pfam reps."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import time
import numpy as np

# These five helpers now live in common.py, shared with the anchor-screen stages
# (t1..t5) so both halves of this experiment read the embeddings, verify the
# checksums and search neighbours through one implementation.
from common import neighbors, normalize, read_tsv, sha, write_tsv

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--min-members', type=int, default=1000)
    ap.add_argument('--clusters', type=int, default=100)
    ap.add_argument('--calibration-size', type=int, default=4096)
    ap.add_argument('--seed', type=int, default=0)
    args=ap.parse_args()
    if args.min_members < 1 or args.clusters < 2 or args.calibration_size < 10:
        ap.error('Positive member threshold, at least 2 clusters and 10 calibration rows required')
    start=time.time()
    out=HERE/'results/clustering'; out.mkdir(parents=True, exist_ok=True)
    emb=HERE/'results/embeddings'
    manifest=json.loads((emb/'manifest.json').read_text())
    for group in ['pfam_reps','retain']:
        for suffix,key in [('.npy','embeddings_sha256'),('.tsv','metadata_sha256')]:
            if sha(emb/f'{group}{suffix}')!=manifest['groups'][group][key]:
                raise ValueError(f'Input checksum mismatch: {group}{suffix}')
    reps=read_tsv(emb/'pfam_reps.tsv'); retain_rows=read_tsv(emb/'retain.tsv')
    x=normalize(np.load(emb/'pfam_reps.npy')); r=normalize(np.load(emb/'retain.npy'))
    assert x.shape==(len(reps),480) and r.shape==(len(retain_rows),480)
    meta_rows=read_tsv(ROOT/'data/Pfam_family_representatives/family_representatives.tsv')
    meta={row['pfam_acc']:row for row in meta_rows}
    assert len(meta)==len(meta_rows)
    counts=np.array([int(meta[row['id']]['n_proteins_nr90']) for row in reps])
    assert np.all(counts>0)
    # Equal sequence hashes yield the same full sequence; retain one reference row per sequence.
    first={}
    for i,row in enumerate(retain_rows):
        first.setdefault(row['sequence_sha256'],i)
    unique_indices=np.array(list(first.values()))
    r=r[unique_indices]
    retain_rows=[retain_rows[i] for i in unique_indices]
    retained_hashes={row['sequence_sha256']:i for i,row in enumerate(retain_rows)}
    nearest, top10, nearest_idx=neighbors(x,r)
    rng=np.random.default_rng(args.seed)
    sample=np.sort(rng.choice(len(r),min(args.calibration_size,len(r)),replace=False))
    cal_near,cal_top,_=neighbors(r[sample],r,exclude=[[int(i)] for i in sample])
    cal_dist=1-cal_near
    np.savez(out/'retain_calibration.npz',retain_unique_row=sample,
             original_row=unique_indices[sample],nearest_cosine=cal_near,mean_top10_cosine=cal_top)
    distance=1-nearest
    percentile=np.searchsorted(np.sort(cal_dist),distance,side='right')/len(cal_dist)*100
    p95=float(np.quantile(cal_dist,.95)); p99=float(np.quantile(cal_dist,.99))
    print(f'Calibration R leave-one-out nearest distance: p95={p95:.5f}; p99={p99:.5f}',flush=True)
    # SciPy average linkage: hierarchy over all original normalized 480-d vectors.
    from scipy.spatial.distance import pdist
    from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
    print('Computing all pairwise cosine distances and average-linkage hierarchy',flush=True)
    condensed=pdist(x, metric='cosine')
    np.maximum(condensed,0,out=condensed)
    z=linkage(condensed,method='average')
    del condensed
    np.save(out/'linkage.npy',z)
    resolutions=sorted(set([30,args.clusters,300]))
    assignments={n:fcluster(z,t=n,criterion='maxclust') for n in resolutions}
    labels=assignments[args.clusters]
    assert len(labels)==len(reps)
    rows=[]
    for i,row in enumerate(reps):
        entry=meta[row['id']]
        d=dict(pfam_acc=row['id'],pfam_id=entry['pfam_id'],description=entry['desc'],
               members_nr90=int(counts[i]),regions_nr90=int(entry['n_regions_nr90']),
               cluster=int(labels[i]),nearest_retain_cosine=float(nearest[i]),
               distance_to_retain=float(distance[i]),mean_top10_retain_cosine=float(top10[i]),
               retain_isolation_percentile=float(percentile[i]),
               beyond_retain_p95=bool(distance[i]>p95),beyond_retain_p99=bool(distance[i]>p99),
               meets_min_members=bool(counts[i]>=args.min_members),
               candidate=bool(counts[i]>=args.min_members and distance[i]>p95),
               nearest_retain_id=retain_rows[nearest_idx[i]]['id'],
               exact_sequence_in_retain=row['sequence_sha256'] in retained_hashes,
               sequence_sha256=row['sequence_sha256'],
               representative_length=int(row['length']),truncated=row['truncated'],
               family_domain_coverage=entry['rep_family_coverage'],
               representative_reviewed=entry['rep_reviewed'],representative_uniprot=entry['rep_uniprot_acc'],
               clan=entry['clan_acc'])
        for n in resolutions:
            d[f'cluster_k{n}']=int(assignments[n][i])
        rows.append(d)
    ranked=sorted(rows,key=lambda row:(-row['distance_to_retain'],-row['members_nr90'],row['pfam_acc']))
    for rank,row in enumerate(ranked,1):row['novelty_rank']=rank
    write_tsv(out/'families_ranked.tsv',ranked)
    candidates=[row for row in ranked if row['candidate']]
    write_tsv(out/'candidates.tsv',candidates)
    # Also retain every size-eligible family even if the calibrated cutoff selects none.
    size_eligible=[row for row in ranked if row['meets_min_members']]
    write_tsv(out/'size_eligible_ranked.tsv',size_eligible)
    summaries=[]
    centroids=[]
    for cluster in sorted(set(labels)):
        idx=np.flatnonzero(labels==cluster)
        centroid=normalize(x[idx].mean(axis=0,keepdims=True))[0]
        representative=idx[np.argmax(x[idx]@centroid)]
        group=[rows[i] for i in idx]
        summaries.append(dict(cluster=int(cluster),n_families=len(idx),
            summed_members_nr90_not_deduplicated=int(counts[idx].sum()),
            max_family_members=int(counts[idx].max()),median_family_members=float(np.median(counts[idx])),
            median_distance_to_retain=float(np.median(distance[idx])),
            max_distance_to_retain=float(distance[idx].max()),
            n_size_eligible=sum(d['meets_min_members'] for d in group),
            n_candidates=sum(d['candidate'] for d in group),
            centroid_nearest_family=reps[representative]['id']))
        centroids.append(centroid)
    write_tsv(out/'clusters.tsv',summaries)
    # Pareto frontier: neither larger nor more distant alternatives exist.
    frontier=[]; max_size=-1
    for row in ranked:
        if row['members_nr90']>max_size:
            frontier.append(row); max_size=row['members_nr90']
    write_tsv(out/'size_novelty_frontier.tsv',frontier)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    plots=out/'plots';plots.mkdir(exist_ok=True)
    def save(fig,name):
        for ext in ['png','pdf']:fig.savefig(plots/f'{name}.{ext}',dpi=220,bbox_inches='tight')
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,8))
    pts=ax.scatter(counts,distance,c=percentile,cmap='viridis_r',s=16,alpha=.85,linewidths=.2,edgecolors='#555')
    ax.axvline(args.min_members,c='#777',ls='--',label=f'Minimum members: {args.min_members:,}')
    ax.axhline(p95,c='#D55E00',ls='--',label='Retain leave-one-out 95th percentile')
    ax.axhline(p99,c='#AA3377',ls=':',label='Retain leave-one-out 99th percentile')
    for j,row in enumerate(size_eligible[:6]):
        ax.annotate(row['pfam_acc'],(row['members_nr90'],row['distance_to_retain']),
                    xytext=[(6,6),(6,-12),(12,15),(-100,0),(12,-16),(8,8)][j],
                    textcoords='offset points',fontsize=8,
                    bbox=dict(facecolor='white',alpha=.8,edgecolor='none',pad=1),
                    arrowprops=dict(arrowstyle='-',color='#777',lw=.5))
    ax.set(xscale='log',xlabel='Family member proteins (NR90; log scale)',
           ylabel='Cosine distance to nearest retain sequence (higher = more isolated)',
           title='Pfam family size versus embedding isolation from retain R')
    ax.legend(loc='upper left',fontsize=8)
    fig.colorbar(pts,ax=ax,label='Isolation percentile versus retain leave-one-out sample')
    save(fig,'size_vs_novelty')
    coords_rows=read_tsv(HERE/'results/umap/coordinates.tsv')
    coords=np.load(HERE/'results/umap/coords.npy')
    mask=np.array([row['group']=='pfam_reps' for row in coords_rows])
    assert [row['id'] for row in coords_rows if row['group']=='pfam_reps']==[row['id'] for row in reps]
    xy=coords[mask];rmask=np.array([row['group']=='retain' for row in coords_rows])
    fig,axes=plt.subplots(1,2,figsize=(18,8),sharex=True,sharey=True)
    for ax in axes:
        ax.scatter(*coords[rmask].T,s=.5,c='#b0b0b0',alpha=.15,linewidths=0,rasterized=True)
        ax.set(xlabel='UMAP 1',ylabel='UMAP 2')
    pts=axes[0].scatter(*xy.T,c=labels,cmap='turbo',s=10,alpha=.85,linewidths=.15,edgecolors='#555',rasterized=True)
    axes[0].set_title(f'Average-linkage cosine clusters (k={len(set(labels))})')
    fig.colorbar(pts,ax=axes[0],label='Cluster ID (arbitrary category)')
    sizes=5+5*np.log10(counts)
    pts=axes[1].scatter(*xy.T,c=distance,cmap='magma_r',s=sizes,alpha=.85,linewidths=.15,edgecolors='#555',rasterized=True)
    axes[1].set_title('Isolation from retain R; marker area grows with log family size')
    fig.colorbar(pts,ax=axes[1],label='Cosine distance to nearest retain sequence')
    for n in [10,1000,100000]:
        axes[1].scatter([],[],s=5+5*np.log10(n),c='#777',label=f'{n:,} members')
    axes[1].legend(loc='upper left',fontsize=8)
    fig.tight_layout();save(fig,'clusters_and_novelty_umap')
    # Compact dendrogram, all 21,763 leaves retained in the saved hierarchy.
    node_n=np.ones(2*len(reps)-1,dtype=np.int64)
    node_members=np.zeros(2*len(reps)-1,dtype=np.int64);node_members[:len(reps)]=counts
    for i,(a,b,_,_) in enumerate(z):
        node_n[len(reps)+i]=node_n[int(a)]+node_n[int(b)]
        node_members[len(reps)+i]=node_members[int(a)]+node_members[int(b)]
    fig,ax=plt.subplots(figsize=(16,10))
    dendrogram(z,truncate_mode='lastp',p=30,orientation='right',
               leaf_label_func=lambda i:f'{reps[i]["id"] if i<len(reps) else "branch"} | {node_n[i]:,} families | Σ {node_members[i]:,} members',
               leaf_font_size=8,ax=ax)
    ax.set(xlabel='Average cosine linkage distance',title='Pfam embedding hierarchy — last 30 branches\nMember sums may count the same protein in multiple families')
    save(fig,'hierarchy')
    dominant=max(summaries,key=lambda d:d['n_families'])
    summary=dict(largest_cluster=dominant,model=manifest['protocol']['model'],n_families=len(reps),
        reference='retain R: data/splits/retain_train.fasta',reference_records=len(unique_indices),
        reference_original_records=len(read_tsv(emb/'retain.tsv')),
        space='Original final-layer 480-dimensional mean-pooled embeddings; L2 normalized',
        clustering='Exact cosine pairwise distances; average linkage',clusters=args.clusters,
        additional_resolutions=resolutions,min_members=args.min_members,
        shortlist_rule='members >= minimum and nearest-retain cosine distance > retain leave-one-out p95',
        ranking='Nearest-retain cosine distance descending, then family size descending',
        n_size_eligible=len(size_eligible),n_candidates=len(candidates),
        calibration_size=len(sample),retain_distance_p95=p95,retain_distance_p99=p99,
        candidate_counts_by_minimum={str(n):sum(row['members_nr90']>=n and row['distance_to_retain']>p95 for row in rows) for n in [100,1000,10000]},
        truncated_candidates=sum(row['truncated']=='True' for row in candidates),
        seed=args.seed,seconds=time.time()-start,
        input_checksums={str(p.relative_to(ROOT)):sha(p) for p in [emb/'pfam_reps.npy',emb/'retain.npy',ROOT/'data/Pfam_family_representatives/family_representatives.tsv']},
        versions={p:importlib.metadata.version(p) for p in ['numpy','scipy','scikit-learn','matplotlib']})
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    table='\n'.join(f"| {r['pfam_acc']} | {r['pfam_id']} | {r['members_nr90']:,} | {r['distance_to_retain']:.4f} | {r['retain_isolation_percentile']:.1f} | {r['cluster']} | {r['truncated']} |" for r in size_eligible[:25])
    (out/'RESULTS.md').write_text(f'''# Pfam embedding clusters and candidate-family screen

Clustered all **{len(reps):,}** representatives; **{len(size_eligible):,}** families have at least
**{args.min_members:,}** NR90 member proteins. **{len(candidates):,}** also exceed the retain
leave-one-out 95th-percentile isolation threshold ({p95:.4f} cosine distance).

## Ranking and clustering

Similarity is measured in the original 480-dimensional ESM2-35M embeddings, not UMAP.
For each representative, the exact nearest and mean top-10 cosine similarity to R are
computed against {len(r):,} unique retain sequences. Higher `distance_to_retain = 1 - max cosine`
means that even its closest retain sequence is relatively distant.
The percentile compares this distance with {len(sample):,} seeded retain sequences queried
against the remaining retain set, excluding themselves. It is a descriptive calibration,
not a significance test or a claim about all of ESM2's original pretraining corpus.

Exact all-pairs cosine distances feed average-linkage hierarchical clustering. The main
cut has {args.clusters} clusters, with 30 and 300 cluster assignments also provided.
The cut is a descriptive resolution choice; cluster IDs are arbitrary, not biological families.
UMAP panels reuse the previous coordinates purely for visualization.
The hierarchy is highly imbalanced: cluster {dominant['cluster']} contains
{dominant['n_families']:,} families ({dominant['n_families']/len(reps):.1%}). This cut primarily
separates peripheral families; it does not imply 100 equally distinct biological groups.
The per-family distance ranking is independent of this cluster cut.

## Top size-eligible families, sorted by isolation

This table includes the top size-eligible families whether or not they cross the threshold;
`candidates.tsv` contains only the families satisfying both criteria.

| Pfam | Name | NR90 proteins | Nearest-R distance | R isolation percentile | Cluster | Truncated |
| --- | --- | ---: | ---: | ---: | ---: | --- |
{table}

## Existing project family

PF13561, the most isolated size-eligible representative in this run, is already the
rung-0 family in `data/ladder/PF00106/ladder_selected.json`. Its score describes
distance from R, not novelty relative to all existing project corpora.

## Interpretation for continued pretraining

These are representation-based screening candidates. Each family is represented by one
sequence; a distant representative does not establish that all its members are distant.
The counts are the supplied `n_proteins_nr90`, not verified downloadable, nonredundant
training sequences. Family memberships can overlap, so cluster member sums are not unique
protein counts. Adequate training data depends on post-filtering diversity and token count.
Full-chain representatives may contain other domains; domain coverage and truncation
flags are included in the ranking for review. The model only embedded the first 1,022
residues of longer sequences. Before choosing a family, sample or retrieve its members,
check domain consistency, remove overlap with R and evaluation data, and measure diversity
and usable tokens. This analysis does not establish that a family was absent from ESM2 pretraining.

## Files

- [Size versus isolation](plots/size_vs_novelty.png)
- [Clusters and isolation on UMAP](plots/clusters_and_novelty_umap.png)
- [Family hierarchy](plots/hierarchy.png) (PDFs available for every figure)
- `families_ranked.tsv`: all families, sizes, cluster assignments, nearest R, quality flags.
- `size_eligible_ranked.tsv`: all families meeting the minimum size.
- `candidates.tsv`: minimum size plus above-calibration isolation.
- `clusters.tsv`: cluster family counts, member sums and isolation summaries.
- `size_novelty_frontier.tsv`: families on the size/isolation tradeoff frontier.
- `linkage.npy`: full hierarchy in FASTA order; `retain_calibration.npz`: calibration sample.
- `summary.json`: parameters, source checksums, sensitivity counts and timings.
''')
    print(json.dumps(summary,indent=2),flush=True)
    print(f'DONE: {out}',flush=True)


if __name__=='__main__':main()
