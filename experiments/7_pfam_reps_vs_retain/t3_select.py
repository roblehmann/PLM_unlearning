#!/usr/bin/env python3
"""t3 — combine isolation, size, length and repetitiveness into a ranked anchor shortlist.

Nothing is dropped. `families_scored.tsv` carries all 21,763 families with every metric and
every gate outcome; the gates define a *view* over that table, so a reader can always see
what a threshold excluded and re-derive the ranking under other weights.

Pfam `type` is tracked, never gating: a Repeat-type family whose sequences are empirically
non-repetitive stays eligible, and a Family-type one whose sequences are repetitive does
not. The clan criterion is a soft preference: reported, used to break ties, and
never a filter.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

import common as C

HAZARD_FALLBACK = ('toxin', 'toxic', 'virulence', 'capsid', 'effector', 'hemolysin',
                   'enterotoxin', 'neurotoxin', 'anthrax', 'botulinum', 'ricin', 'spike',
                   'envelope glycoprotein', 'pathogenesis')


def hazard_keywords(rows) -> list:
    """Keywords Pfam's own hazard screen used, re-derived from the metadata column."""
    found = set()
    for r in rows:
        for token in str(r.get('hazard_keyword') or '').replace(';', ',').split(','):
            token = token.strip().lower()
            if token:
                found.add(token)
    return sorted(found) or list(HAZARD_FALLBACK)


def as_float(rows, key) -> np.ndarray:
    return np.array([float(r[key]) if r.get(key) not in (None, '', 'None') else np.nan
                     for r in rows], dtype=np.float64)


# TSV rounds every value through str(); these columns are read back as numbers so the
# ranking, the sorts and the report formatting all operate on numbers, not text.
_INT_COLUMNS = ('members_nr90', 'regions_nr90', 'model_length', 'rep_length',
                'clan_n_families', 'swissprot_nr90')
_FLOAT_SUFFIXES = ('iso_nn', 'iso_top10', 'iso_nn_xclan', 'iso_top10_xclan', 'lof20', 'lof50',
                   'iso_centroid', 'medoid_sim', 'same_clan_frac', 'first_merge_height',
                   'merge_order_frac', 'height_join20', 'height_join100',
                   'nearest_retain_cos', 'top10_retain_cos', 'frac_retain_in_top50',
                   'max_cos_to_existing_anchor')


def coerce(row: dict, space: str) -> dict:
    """Numeric columns back to numbers; unparseable or empty cells become NaN/0."""
    out = dict(row)
    for key in _INT_COLUMNS:
        if key in out:
            try:
                out[key] = int(float(out[key]))
            except (TypeError, ValueError):
                out[key] = 0
    for key in list(out):
        numeric = (key == 'rep_family_coverage'
                   or key.startswith('pct_')
                   or (key.startswith(f'{space}_') and key.split(f'{space}_', 1)[1] in _FLOAT_SUFFIXES))
        if numeric:
            try:
                out[key] = float(out[key])
            except (TypeError, ValueError):
                out[key] = float('nan')
        elif key.startswith(f'{space}_cluster_k'):
            try:
                out[key] = int(float(out[key]))
            except (TypeError, ValueError):
                out[key] = -1
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    C.common_args(ap)
    ap.add_argument('--space', default=C.HEADLINE_SPACE, choices=list(C.SPACES))
    ap.add_argument('--min-members', type=int, default=1000)
    ap.add_argument('--min-model-length', type=int, default=80)
    ap.add_argument('--min-domain-length', type=int, default=60)
    ap.add_argument('--min-iso-pct', type=float, default=99.0)
    ap.add_argument('--retain-iso-pct', type=float, default=95.0,
                    help='retain leave-one-out percentile for the reported gate_retain column')
    ap.add_argument('--max-anchor-similarity', type=float, default=0.95)
    ap.add_argument('--require-no-clan', action='store_true',
                    help='off by default: clan membership is a preference, not a filter')
    ap.add_argument('--weights', default='',
                    help='component=weight,... overriding the equal-weight composite')
    ap.add_argument('--top', type=int, default=25)
    args = ap.parse_args()
    C.apply_results_override(args.results)
    started = time.time()
    sp = args.space
    out = C.results_dir('selection')

    C.banner('t3 — inputs')
    metrics = C.read_tsv(C.results_dir('matrix') / 'family_metrics.tsv')
    stats = {r['pfam_acc']: r for r in C.read_tsv(C.results_dir('seqstats') / 'rep_sequence_stats.tsv')}
    similarity = C.load_json(C.results_dir('matrix') / 'summary_similarity.json')
    if sp not in similarity['spaces']:
        raise SystemExit(f'space "{sp}" was not computed by t1 (have: {list(similarity["spaces"])})')
    missing = [r['pfam_acc'] for r in metrics if r['pfam_acc'] not in stats]
    if missing:
        raise SystemExit(f'{len(missing)} families have no sequence statistics, e.g. {missing[:3]}')
    project = C.load_project_families()
    n = len(metrics)
    print(f'{n:,} families; {len(project)} already used by this project; space "{sp}"', flush=True)

    # --- composite isolation score ---------------------------------------- #
    weights = {c: 1.0 for c in C.ISOLATION_COMPONENTS}
    for item in [w for w in args.weights.split(',') if w]:
        name, _, value = item.partition('=')
        if name not in weights:
            raise SystemExit(f'unknown isolation component: {name}')
        weights[name] = float(value)
    components = {c: as_float(metrics, f'{sp}_{c}') for c in C.ISOLATION_COMPONENTS}
    # A missing component must never look isolated: fill it with the least-isolated value.
    filled = {c: np.where(np.isfinite(v), v, np.nanmin(v) if np.isfinite(v).any() else 0.0)
              for c, v in components.items()}
    ranks = {c: C.pctrank(v) for c, v in filled.items()}
    total_w = sum(weights.values())
    score = sum(weights[c] * ranks[c] for c in components) / total_w
    zscore = sum(weights[c] * C.robust_z(filled[c]) for c in components) / total_w
    score_pct = C.pctrank(score)

    from scipy.stats import spearmanr

    names = list(components)
    corr = {a: {b: float(spearmanr(components[a], components[b]).statistic) for b in names}
            for a in names}

    # --- per-family rows --------------------------------------------------- #
    keywords = hazard_keywords(metrics)
    hazard_clans = {r['clan_acc'] for r in metrics if r['hazard_flag'] == 'True' and r['clan_acc']}
    iso_threshold = float(np.percentile(score, args.min_iso_pct))
    retain_p = similarity['spaces'][sp]['retain'][
        'loo_distance_p95' if args.retain_iso_pct == 95.0 else 'loo_distance_p99']

    rows = []
    for i, m in enumerate(metrics):
        s = stats[m['pfam_acc']]
        members = int(m['members_nr90'])
        desc = f"{m['description']} {m['pfam_id']}".lower()
        keyword_hit = next((k for k in keywords if k and k in desc), '')
        retain_distance = 1.0 - float(m[f'{sp}_nearest_retain_cos'])
        row = coerce(m, sp)
        anchor_sim = row.get(f'{sp}_max_cos_to_existing_anchor', np.nan)
        row.update({
            'rep_domain_length': int(s['rep_domain_length']),
            'domain_coverage_measured': float(s['domain_coverage']),
            'dom_lowcomp_frac_w12': float(s['dom_lowcomp_frac_w12']),
            'dom_max_periodicity': float(s['dom_max_periodicity']),
            'dom_best_period': int(s['dom_best_period']),
            'dom_max_aa_frac': float(s['dom_max_aa_frac']),
            'dom_kmer_dup_frac_k3': float(s['dom_kmer_dup_frac_k3']),
            'dom_zlib_ratio': float(s['dom_zlib_ratio']),
            'dom_longest_homopolymer': int(s['dom_longest_homopolymer']),
            'full_lowcomp_frac_w12': float(s['full_lowcomp_frac_w12']),
            'full_max_periodicity': float(s['full_max_periodicity']),
            'copies_per_protein': float(s['copies_per_protein']),
            'flag_tandem_copies': s['flag_tandem_copies'] == 'True',
            'n_repetitive_flags': int(s['n_repetitive_flags']),
            'repetitive': s['repetitive'] == 'True',
            'isolation_rank_score': float(score[i]),
            'isolation_score_pct': float(score_pct[i]),
            'isolation_robust_z': float(zscore[i]),
            'distance_to_retain': retain_distance,
            'project_family': m['pfam_acc'] in project,
            'project_family_reason': project.get(m['pfam_acc'], ''),
            'in_project_clan': m['clan_acc'] == C.PROJECT_CLAN,
            'clan_hazard_free': m['clan_acc'] not in hazard_clans if m['clan_acc'] else True,
            'hazard_keyword_hit': keyword_hit,
        })
        for c in C.ISOLATION_COMPONENTS:
            row[f'pct_{c}'] = float(ranks[c][i])
        row['gate_size'] = members >= args.min_members
        row['gate_length'] = (int(m['model_length']) >= args.min_model_length
                              and int(s['rep_domain_length']) >= args.min_domain_length)
        row['gate_complexity'] = not row['repetitive']
        row['gate_hazard'] = (m['hazard_flag'] != 'True' and not keyword_hit
                              and row['clan_hazard_free'])
        row['gate_project'] = (not row['project_family'] and not row['in_project_clan']
                               and not (np.isfinite(anchor_sim) and anchor_sim >= args.max_anchor_similarity))
        row['gate_isolation'] = float(score[i]) >= iso_threshold
        row['gate_retain'] = retain_distance > retain_p          # reported, not required
        row['gate_clan'] = m['in_clan'] != 'True'                 # preference, not required
        row['n_gates_failed'] = int(sum(not row[g] for g in C.GATE_ORDER))
        row['shortlisted'] = bool(all(row[g] for g in C.GATE_ORDER)
                                  and (row['gate_clan'] or not args.require_no_clan))
        rows.append(row)

    ranked = sorted(rows, key=lambda r: (-r['isolation_rank_score'], -r['members_nr90'], r['pfam_acc']))
    for rank, r in enumerate(ranked, 1):
        r['isolation_rank'] = rank
    C.write_tsv(out / 'families_scored.tsv', ranked)

    shortlist = [r for r in ranked if r['shortlisted']]
    # Clan-less families first at comparable isolation: the soft preference, made explicit.
    shortlist_sorted = sorted(shortlist, key=lambda r: (not r['gate_clan'], -r['isolation_rank_score']))
    C.write_tsv(out / 'shortlist.tsv', shortlist_sorted)
    C.write_tsv(out / 'shortlist_top25.tsv', shortlist_sorted[:args.top])

    # --- waterfall and sensitivity ---------------------------------------- #
    waterfall, alive = [], np.ones(len(rows), dtype=bool)
    waterfall.append({'gate': '(all families)', 'survivors': int(alive.sum()),
                      'removed': 0, 'survivors_clanless': int(sum(r['gate_clan'] for r in rows))})
    for gate in C.GATE_ORDER:
        passed = np.array([r[gate] for r in rows])
        before = int(alive.sum())
        alive = alive & passed
        waterfall.append({'gate': gate, 'survivors': int(alive.sum()),
                          'removed': before - int(alive.sum()),
                          'survivors_clanless': int(sum(r['gate_clan'] for r, a in zip(rows, alive) if a))})
    C.write_tsv(out / 'gate_waterfall.tsv', waterfall)

    sensitivity = []
    for pct in (90.0, 95.0, 99.0, 99.5):
        threshold = float(np.percentile(score, pct))
        for min_members in (args.min_members, 5000, 10000):
            sel = [r for r in rows
                   if r['isolation_rank_score'] >= threshold and r['members_nr90'] >= min_members
                   and all(r[g] for g in C.GATE_ORDER if g != 'gate_isolation')]
            sensitivity.append({'isolation_pct': pct, 'min_members': min_members,
                                'n_shortlist': len(sel),
                                'n_clanless': sum(r['gate_clan'] for r in sel),
                                'n_repeat_type': sum(r['type'] == 'Repeat' for r in sel)})
    C.write_tsv(out / 'sensitivity.tsv', sensitivity)

    by_type = {}
    for label in sorted({r['type'] for r in rows}):
        sel = [r for r in rows if r['type'] == label]
        by_type[label] = {'n': len(sel), 'n_shortlisted': sum(r['shortlisted'] for r in sel),
                          'median_isolation_pct': float(np.median([r['isolation_score_pct'] for r in sel])),
                          'frac_repetitive': float(np.mean([r['repetitive'] for r in sel]))}

    summary = {
        'protocol': C.protocol_block(stage='t3_select', space=sp),
        'n_families': n, 'n_shortlisted': len(shortlist),
        'thresholds': {'min_members': args.min_members, 'min_model_length': args.min_model_length,
                       'min_domain_length': args.min_domain_length,
                       'isolation_percentile': args.min_iso_pct,
                       'isolation_score_threshold': iso_threshold,
                       'retain_loo_percentile': args.retain_iso_pct,
                       'retain_distance_threshold': retain_p,
                       'max_anchor_similarity': args.max_anchor_similarity,
                       'require_no_clan': bool(args.require_no_clan)},
        'weights': weights,
        'component_spearman': corr,
        'gate_waterfall': waterfall,
        'by_type': by_type,
        'shortlist_by_type': {k: v['n_shortlisted'] for k, v in by_type.items()},
        'n_project_families_seen': sum(r['project_family'] for r in rows),
        'hazard_keywords_used': keywords[:40],
        'n_hazard_clans': len(hazard_clans),
        'seconds': time.time() - started,
    }
    C.save_json(summary, out / 'summary_selection.json')

    validation = {'checks': {
        'rows': {'expected': n, 'actual': len(ranked), 'passed': len(ranked) == n},
        'waterfall_monotone': {'passed': all(
            waterfall[i + 1]['survivors'] <= waterfall[i]['survivors']
            for i in range(len(waterfall) - 1))},
        'shortlist_satisfies_every_gate': {'passed': all(
            all(r[g] for g in C.GATE_ORDER) for r in shortlist)},
        'project_families_present_and_flagged': {
            'n_seen': sum(r['project_family'] for r in rows),
            'passed': not any(r['project_family'] and r['shortlisted'] for r in rows)},
        'repeat_type_families_retained': {
            'n': sum(r['type'] == 'Repeat' for r in rows),
            'passed': any(r['type'] == 'Repeat' for r in rows),
            'note': 'repetitive-by-construction families are tracked, never removed'},
    }}
    validation['status'] = 'passed' if all(
        v.get('passed', True) for v in validation['checks'].values()) else 'FAILED'
    C.save_json(validation, out / 'validation.json')

    write_results_md(out, args, summary, shortlist_sorted, similarity, by_type, waterfall)
    print(f"\nDONE t3 [{validation['status']}] — {len(shortlist):,} shortlisted -> {out}", flush=True)
    if validation['status'] != 'passed':
        raise SystemExit(1)


def write_results_md(out, args, summary, shortlist, similarity, by_type, waterfall):
    sp = args.space
    space_info = similarity['spaces'][sp]
    raw_info = similarity['spaces'].get('raw', {})
    head = shortlist[:args.top]
    table = '\n'.join(
        f"| {r['pfam_acc']} | {r['pfam_id']} | {r['type']} | {r['members_nr90']:,} | "
        f"{r['isolation_rank_score']:.1f} | {r[f'{sp}_iso_nn_xclan']:.4f} | {r[f'{sp}_lof50']:.2f} | "
        f"{'—' if r['gate_clan'] else r['clan_acc']} | {r['model_length']} | "
        f"{r['full_lowcomp_frac_w12']:.3f} | {r['copies_per_protein']:.2f} | "
        f"{r['distance_to_retain']:.4f} |"
        for r in head)
    water = '\n'.join(f"| {w['gate']} | {w['survivors']:,} | {w['removed']:,} | "
                      f"{w['survivors_clanless']:,} |" for w in waterfall)
    types = '\n'.join(f"| {k} | {v['n']:,} | {v['n_shortlisted']:,} | {v['frac_repetitive']:.1%} |"
                      for k, v in by_type.items())
    aniso = space_info['anisotropy']
    raw_aniso = raw_info.get('anisotropy', {})
    (out / 'RESULTS.md').write_text(f"""# Isolated Pfam families — candidate anchors

Screened **{summary['n_families']:,}** family representatives; **{summary['n_shortlisted']:,}**
pass every gate. Isolation is measured against the bulk of Pfam itself, in the
**{sp}** space; distance to the retain corpus R is reported beside it, not required.

## Why the space is centered

Mean-pooled ESM-2 embeddings are anisotropic: in the raw space the L2-normalised rows have
a mean resultant length of **{raw_aniso.get('mean_resultant_length', float('nan')):.4f}**
(1.0 would mean every family points the same way) and a mean off-diagonal cosine of
**{raw_info.get('off_diagonal', {}).get('mean_similarity_exact', float('nan')):.4f}**. That
is the geometry in which `cluster_families.py` found one cluster holding 95.7% of families.
After centering, the mean resultant length is **{aniso['mean_resultant_length']:.4f}**, the
mean off-diagonal cosine **{space_info['off_diagonal']['mean_similarity_exact']:.4f}**, and
the effective rank of the covariance **{aniso['effective_rank']:.1f}** of
{aniso['n_dim']} dimensions. Centering removes a shared component; it does not make the
space metrically meaningful, and the raw-vs-centered rank correlation is in
`summary_similarity.json`.

## Shortlist (top {len(head)})

| Pfam | Name | Type | NR90 members | Isolation score | NN dist (x-clan) | LOF50 | Clan | HMM len | Low-complexity (chain) | Copies/protein | Dist to R |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
{table}

`Isolation score` is the mean percentile rank over {', '.join(C.ISOLATION_COMPONENTS)}.
Averaging six correlated components is a choice, not a derivation — the per-component
percentiles are columns in `families_scored.tsv` and their rank correlations are in
`summary_selection.json`, so the ranking can be re-derived under other weights.

## Gate waterfall

| Gate | Survivors | Removed | of which clan-less |
| --- | ---: | ---: | ---: |
{water}

Thresholds: >= {args.min_members:,} NR90 members; HMM model length >= {args.min_model_length}
and representative domain >= {args.min_domain_length} aa; no repetitiveness flag; benign
(Pfam hazard flag, a keyword re-screen, and no hazard-flagged family anywhere in the same
clan); not already used by this project; isolation above the p{args.min_iso_pct:g} of the
21,763-family population. Clan membership is a **preference**: clan-less families sort first
at comparable isolation, and `--require-no-clan` is off.

## Pfam type is tracked, not gating

| Type | Families | Shortlisted | Flagged repetitive |
| --- | ---: | ---: | ---: |
{types}

Repeat, Coiled-coil, Motif and Disordered families are repetitive *by construction*, and they
stay in every table. The repetitiveness decision is made by measured statistics — window
entropy, maximum periodicity, k-mer duplication, compositional bias and compression ratio,
each as a percentile of retain within the family's own length decile, on the domain region
and on the full chain — plus Pfam's own regions-per-protein ratio.

**What those statistics do and do not detect** (measured over every representative, see
`seqstats/summary_seqstats.json`): they detect low complexity — Coiled-coil and Disordered
representatives sit far above the Family/Domain baseline — but they do **not** detect
degenerate tandem repeats. Repeat-type families score at or below baseline on periodicity,
k-mer duplication and compression, because protein repeat copies share a profile rather than
residues. The signal that does see them is `copies_per_protein` (Pfam regions per member
protein), where Repeat families are strongly enriched in the multi-copy tail, and for the
shortlisted families the member stage counts HMM domain copies per member directly. The type
label is therefore reported beside every family and never used as a gate.

## Caveats

- Each family is represented by **one sequence**. A distant representative does not make the
  family distant; `t5_members.py` is what upgrades this to a family-level claim, and any
  family whose members do not follow its representative must be dropped.
- `n_proteins_nr90` is supplied metadata, not downloadable, deduplicated, leakage-filtered
  training sequences. The usable count comes from the member stage and will be smaller.
- "Isolated" means isolated among these {summary['n_families']:,} curated representatives
  under ESM-2 35M — not isolated in protein space, and **not** absent from ESM-2
  pretraining. Experiment 6 showed representational separation does not imply the model
  models a family worse.
- Every threshold here is a descriptive calibration against a non-random reference
  population, not a significance test.
- The hazard screen is keyword-based over Pfam descriptions and is **not** a biosecurity
  control. A human signs off on the final family.

## Files

- `families_scored.tsv` — all {summary['n_families']:,} families, every metric and gate.
- `shortlist.tsv`, `shortlist_top25.tsv` — the gated view, clan-less first.
- `gate_waterfall.tsv`, `sensitivity.tsv` — what each gate cost, and shortlist size at
  p90/p95/p99 isolation and 1k/5k/10k members.
- `summary_selection.json`, `validation.json`.
""")
    print(f'  wrote {out / "RESULTS.md"}', flush=True)


if __name__ == '__main__':
    main()
