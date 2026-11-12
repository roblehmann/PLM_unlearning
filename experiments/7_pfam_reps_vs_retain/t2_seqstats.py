#!/usr/bin/env python3
"""t2 — length, low-complexity and repetitiveness statistics, calibrated against retain.

"Not only very short sequences" and "not only repetitive sequence" are measured, never
inferred from Pfam's `type` label. Repeat / Coiled-coil / Motif / Disordered families stay
in every table; they are the built-in positive control that these metrics detect what they
claim to, and a family whose sequences are empirically fine stays eligible whatever its
label says.

Every threshold is a percentile of the 194,064-sequence retain corpus, computed here rather
than hardcoded, and **stratified by retain length decile** — `zlib_ratio` and
`kmer_dup_frac_k3` both vary with length, so a global percentile would flag short families
for being short.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import time

import numpy as np

import common as C

# Metrics whose percentile is taken against retain within a length stratum.
CALIBRATED = tuple(m.name for m in C.SEQ_METRICS)


def _stats_for(seq: str) -> dict:
    return C.seq_stats(seq)


def profile(seqs, workers: int, chunk: int = 256, label: str = '') -> list:
    started = time.time()
    if workers <= 1:
        out = []
        for i, s in enumerate(seqs):
            out.append(_stats_for(s))
            if label and i and i % 20000 == 0:
                print(f'  {label}: {i:,}/{len(seqs):,} ({time.time()-started:.0f}s)', flush=True)
        return out
    with mp.Pool(workers) as pool:
        out = []
        for i, row in enumerate(pool.imap(_stats_for, seqs, chunksize=chunk)):
            out.append(row)
            if label and i and i % 20000 == 0:
                print(f'  {label}: {i:,}/{len(seqs):,} ({time.time()-started:.0f}s)', flush=True)
    return out


def calibrate(rows, edges, grid) -> dict:
    """Per-length-decile quantile grids of every calibrated metric over the reference."""
    lengths = np.array([r['length'] for r in rows])
    bins = np.array([C.length_bin(int(L), edges) for L in lengths])
    tables = {}
    for name in CALIBRATED:
        values = np.array([r[name] for r in rows], dtype=np.float64)
        per_bin = np.full((len(edges) + 1, grid.size), np.nan)
        for b in range(len(edges) + 1):
            v = values[(bins == b) & np.isfinite(values)]
            if v.size:
                per_bin[b] = np.quantile(v, grid / 100.0)
        tables[name] = per_bin
    return {'edges': edges, 'grid': grid, 'tables': tables, 'bins': bins, 'lengths': lengths}


def percentile_in_stratum(value: float, table_row: np.ndarray, grid: np.ndarray) -> float:
    """Where ``value`` falls in a stratum's quantile grid, in [0, 100]."""
    if not np.isfinite(value) or not np.isfinite(table_row).any():
        return float('nan')
    row = table_row[np.isfinite(table_row)]
    g = grid[np.isfinite(table_row)]
    idx = int(np.searchsorted(row, float(value), side='right'))
    if idx <= 0:
        return float(g[0])
    if idx >= g.size:
        return float(g[-1])
    return float(g[idx - 1])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    C.common_args(ap)
    ap.add_argument('--workers', type=int, default=0, help='0 = SLURM_CPUS_PER_TASK')
    ap.add_argument('--limit-families', type=int, default=0)
    ap.add_argument('--limit-retain', type=int, default=0)
    ap.add_argument('--flag-pct', type=float, default=99.0,
                    help='retain percentile at which a repetitiveness flag fires')
    ap.add_argument('--sample-rows', type=int, default=8192,
                    help='seeded raw retain rows kept for the calibration figure')
    ap.add_argument('--min-copies', type=float, default=2.0,
                    help='Pfam regions per member protein at which a family counts as tandem-repeating')
    args = ap.parse_args()
    C.apply_results_override(args.results)
    C.guard_limits(args)
    started = time.time()
    import os

    workers = args.workers or int(os.environ.get('SLURM_CPUS_PER_TASK', '1'))
    out = C.results_dir('seqstats')

    C.banner('t2 — sequences')
    from src.utils.io import read_fasta

    rep_records = read_fasta(str(C.REPS_FASTA))
    meta = C.load_metadata()
    if args.limit_families and args.limit_families < len(rep_records):
        keep = np.unique(np.linspace(0, len(rep_records) - 1, args.limit_families).astype(int))
        rep_records = [rep_records[i] for i in keep]
    retain_records = read_fasta(str(C.RETAIN_FASTA))
    if args.limit_retain and args.limit_retain < len(retain_records):
        keep = np.unique(np.linspace(0, len(retain_records) - 1, args.limit_retain).astype(int))
        retain_records = [retain_records[i] for i in keep]
    print(f'{len(rep_records):,} representatives, {len(retain_records):,} retain sequences, '
          f'{workers} workers', flush=True)

    C.banner('t2 — retain calibration')
    retain_stats = profile([str(r.seq) for r in retain_records], workers, label='retain')
    grid = np.arange(0, 100.5, 0.5)
    edges = C.length_decile_edges([r['length'] for r in retain_stats])
    cal = calibrate(retain_stats, edges, grid)
    rng = np.random.default_rng(args.seed)
    sample = np.sort(rng.choice(len(retain_stats), size=min(args.sample_rows, len(retain_stats)),
                                replace=False))
    np.savez(out / 'retain_calibration.npz',
             edges=edges, grid=grid,
             sample_rows=sample,
             sample_lengths=np.array([retain_stats[i]['length'] for i in sample]),
             **{f'table_{k}': v for k, v in cal['tables'].items()},
             **{f'sample_{k}': np.array([retain_stats[i][k] for i in sample], dtype=np.float64)
                for k in CALIBRATED})
    print(f'  length decile edges: {np.round(edges, 1).tolist()}', flush=True)

    C.banner('t2 — representatives')
    full_stats = profile([str(r.seq) for r in rep_records], workers, label='reps (full chain)')
    domain_seqs = [C.domain_sequence(str(r.seq), meta[r.id]['rep_domain_ranges'])
                   for r in rep_records]
    dom_stats = profile(domain_seqs, workers, label='reps (domain region)')

    rows = []
    for rec, full, dom in zip(rep_records, full_stats, dom_stats):
        m = meta[rec.id]
        proteins = int(m['n_proteins_nr90'])
        entry = {'pfam_acc': rec.id, 'pfam_id': m['pfam_id'], 'type': m['type'],
                 'model_length': int(m['model_length']),
                 'rep_domain_ranges': m['rep_domain_ranges'],
                 'rep_domain_length': int(dom['length']),
                 'rep_domain_spans': len(C.domain_spans(m['rep_domain_ranges'], full['length'])),
                 'domain_coverage': float(dom['length'] / full['length']) if full['length'] else float('nan'),
                 # Pfam's own count of domain regions per member protein: the only signal
                 # here that sees TANDEM repeats (see the positive control below).
                 'copies_per_protein': float(int(m['n_regions_nr90']) / proteins) if proteins else float('nan')}
        for scope, stats in (('full', full), ('dom', dom)):
            for k, v in stats.items():
                if k == 'length' and scope == 'dom':
                    continue
                entry[f'{scope}_{k}'] = v
        b_dom = C.length_bin(int(dom['length']), edges)
        b_full = C.length_bin(int(full['length']), edges)
        entry['retain_length_decile'] = b_dom
        entry['retain_length_decile_full'] = b_full
        for metric in C.SEQ_METRICS:
            threshold = args.flag_pct if metric.higher_is_worse else 100.0 - args.flag_pct
            fired = False
            for scope, stats, b in (('dom', dom, b_dom), ('full', full, b_full)):
                pct = percentile_in_stratum(stats[metric.name], cal['tables'][metric.name][b], grid)
                entry[f'{metric.name}_retain_pct_{scope}'] = pct
                fired = fired or (pct >= threshold if metric.higher_is_worse else pct <= threshold)
            # Both scopes matter: a clean domain can sit in a low-complexity chain, and the
            # training corpus for an anchor is full chains.
            entry[f'flag_{metric.name}'] = bool(fired)
        entry['flag_tandem_copies'] = bool(entry['copies_per_protein'] >= args.min_copies)
        entry['n_repetitive_flags'] = int(sum(entry[f'flag_{m.name}'] for m in C.SEQ_METRICS)
                                          + entry['flag_tandem_copies'])
        entry['repetitive'] = bool(entry['n_repetitive_flags'] > 0)
        rows.append(entry)
    C.write_tsv(out / 'rep_sequence_stats.tsv', rows)

    C.banner('t2 — positive control')
    # What these metrics do and do not detect, measured rather than assumed. The
    # repetitive-by-construction Pfam types are the control sample; they are reported,
    # never removed, so they stay available as that control.
    kinds = {}
    for label in sorted({r['type'] for r in rows}):
        sel = [r for r in rows if r['type'] == label]
        kinds[label] = {
            'n': len(sel),
            'median_lowcomp_frac_full': float(np.nanmedian([r['full_lowcomp_frac_w12'] for r in sel])),
            'median_lowcomp_frac_dom': float(np.nanmedian([r['dom_lowcomp_frac_w12'] for r in sel])),
            'median_max_periodicity_full': float(np.nanmedian([r['full_max_periodicity'] for r in sel])),
            'median_zlib_ratio_full': float(np.nanmedian([r['full_zlib_ratio'] for r in sel])),
            'median_copies_per_protein': float(np.nanmedian([r['copies_per_protein'] for r in sel])),
            'frac_multi_copy': float(np.mean([r['copies_per_protein'] >= args.min_copies for r in sel])),
            'frac_flagged_repetitive': float(np.mean([r['repetitive'] for r in sel])),
        }
    baseline = [r for r in rows if r['type'] in ('Family', 'Domain')]
    n_base = len(baseline)
    base_lc = float(np.nanmedian([r['full_lowcomp_frac_w12'] for r in baseline])) if baseline else float('nan')
    base_per = float(np.nanmedian([r['full_max_periodicity'] for r in baseline])) if baseline else float('nan')
    base_multi = float(np.mean([r['copies_per_protein'] >= args.min_copies for r in baseline])) if baseline else float('nan')
    MIN_N = 30                       # a 512-family smoke slice cannot test a rate

    def ratio(label, key, base):
        return (kinds[label][key] / base) if label in kinds and base else float('nan')

    control = {
        'baseline_family_domain': {'n': n_base, 'median_lowcomp_frac_full': base_lc,
                                   'median_max_periodicity_full': base_per,
                                   'frac_multi_copy': base_multi},
        'low_complexity_is_detected': {
            'claim': 'Coiled-coil and Disordered representatives are more low-complexity '
                     'than Family/Domain ones — the metrics detect what they claim to',
            'coiled_coil_ratio': ratio('Coiled-coil', 'median_lowcomp_frac_full', base_lc),
            'disordered_ratio': ratio('Disordered', 'median_lowcomp_frac_full', base_lc),
            'testable': all(kinds.get(k, {}).get('n', 0) >= MIN_N for k in ('Coiled-coil', 'Disordered')),
        },
        'tandem_repeats_are_NOT_detected_by_sequence_self_similarity': {
            'finding': 'measured over every representative: Repeat-type families are at or '
                       'below the Family/Domain baseline on periodicity, k-mer duplication and '
                       'compression — degenerate protein repeats share a profile, not residues',
            'repeat_periodicity_ratio': ratio('Repeat', 'median_max_periodicity_full', base_per),
            'consequence': 'Pfam type is reported for every family and the member stage counts '
                           'HMM domain copies per member; the type label is never used as a gate',
        },
        'repeat_families_are_enriched_in_the_multi_copy_tail': {
            'claim': f'>= {args.min_copies} Pfam regions per member protein is rare in '
                     'Family/Domain and common in Repeat',
            'repeat_frac_multi_copy': kinds.get('Repeat', {}).get('frac_multi_copy', float('nan')),
            'baseline_frac_multi_copy': base_multi,
            'enrichment': (kinds['Repeat']['frac_multi_copy'] / base_multi)
                          if 'Repeat' in kinds and base_multi else float('nan'),
            'testable': kinds.get('Repeat', {}).get('n', 0) >= MIN_N,
        },
        'by_type': kinds,
    }
    lc = control['low_complexity_is_detected']
    tail = control['repeat_families_are_enriched_in_the_multi_copy_tail']
    passed = True
    if lc['testable']:
        passed &= bool(lc['coiled_coil_ratio'] > 1.5 and lc['disordered_ratio'] > 1.5)
    if tail['testable']:
        passed &= bool(np.isfinite(tail['enrichment']) and tail['enrichment'] > 3.0)
    for label, stats in kinds.items():
        print(f"  {label:<12} n={stats['n']:>6,}  lowcomp(full)={stats['median_lowcomp_frac_full']:.4f}"
              f"  periodicity(full)={stats['median_max_periodicity_full']:.3f}"
              f"  copies/protein={stats['median_copies_per_protein']:.3f}"
              f"  multi-copy={stats['frac_multi_copy']:.1%}"
              f"  flagged={stats['frac_flagged_repetitive']:.1%}", flush=True)
    print(f"  low-complexity detection: Coiled-coil {lc['coiled_coil_ratio']:.2f}x, "
          f"Disordered {lc['disordered_ratio']:.2f}x baseline "
          f"({'tested' if lc['testable'] else 'sample too small to test'})", flush=True)
    print(f"  tandem repeats: Repeat periodicity is "
          f"{control['tandem_repeats_are_NOT_detected_by_sequence_self_similarity']['repeat_periodicity_ratio']:.2f}x "
          f"baseline — sequence self-similarity does NOT detect them; multi-copy enrichment "
          f"{tail['enrichment']:.1f}x", flush=True)

    summary = {'protocol': C.protocol_block(stage='t2_seqstats'),
               'n_reps': len(rows), 'n_retain': len(retain_stats),
               'workers': workers, 'flag_pct': args.flag_pct, 'min_copies': args.min_copies,
               'flag_rule': 'a metric fires when EITHER the domain region or the full chain '
                            'crosses the retain percentile within its own length decile',
               'lowcomp_window': C.LOWCOMP_WINDOW, 'lowcomp_bits': C.LOWCOMP_BITS,
               'max_period': C.MAX_PERIOD,
               'calibration': 'retain quantile grid, stratified by retain length decile',
               'length_decile_edges': edges.tolist(),
               'metrics': {m.name: ('high values are repetitive' if m.higher_is_worse
                                    else 'low values are repetitive') for m in C.SEQ_METRICS},
               'positive_control': control,
               'n_flagged_repetitive': int(sum(r['repetitive'] for r in rows)),
               'inputs': {str(p.relative_to(C.ROOT)): C.sha(p)
                          for p in (C.REPS_FASTA, C.RETAIN_FASTA, C.META_TSV)},
               'seconds': time.time() - started}
    C.save_json(summary, out / 'summary_seqstats.json')
    validation = {'checks': {
        'rows': {'expected': len(rep_records), 'actual': len(rows),
                 'passed': len(rows) == len(rep_records)},
        'positive_control': {
            'passed': bool(passed),
            'note': 'low-complexity metrics must separate Coiled-coil/Disordered from '
                    'Family/Domain, and Pfam multi-copy families must be enriched among '
                    'Repeat-type families; both are skipped when the sample is too small'},
        'domain_lengths_positive': {
            'passed': bool(all(r['rep_domain_length'] > 0 for r in rows))},
    }}
    validation['status'] = 'passed' if all(
        v.get('passed', True) for v in validation['checks'].values()) else 'FAILED'
    C.save_json(validation, out / 'validation.json')
    print(f"\nDONE t2 [{validation['status']}] in {time.time()-started:.0f}s -> {out}", flush=True)
    if validation['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
