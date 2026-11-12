#!/usr/bin/env python3
"""t5 — verify the shortlisted families at MEMBER level.

Stages t1-t4 describe one representative sequence per family. That is a screen, not a
finding: a distant representative does not make its family distant, and "not only very
short / not only repetitive sequences" is a statement about a distribution. This stage
fetches real members and answers, per family:

  fetch    members + domain coordinates + the family HMM (InterPro/UniProt)
  profile  member length and repetitiveness distributions -> n_usable
  cluster  redundancy (mmseqs 50% clustering) and overlap with retain R, BOTH directions
  embed    does the family CENTROID sit where its representative sits?
  report   one table, one verdict column, one RESULTS.md

Subcommands run as separate SLURM jobs; each caches its outputs and re-runs cheaply.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

import common as C

INTERPRO = 'https://www.ebi.ac.uk/interpro/api/protein/UniProt/entry/pfam'
M8_FIELDS = 'query,target,fident,alnlen,evalue,bits,qcov,tcov'
LEAKAGE_IDENTITY = 0.30          # configs/data/pfam_default.yaml leakage_filter
LEAKAGE_COVERAGE = 0.50


def family_dir(acc: str) -> Path:
    return C.results_dir('members', acc)


def shortlist(args) -> list:
    if args.families:
        wanted = [a.strip() for a in args.families.split(',') if a.strip()]
        rows = {r['pfam_acc']: r for r in C.read_tsv(C.results_dir('selection') / 'families_scored.tsv')}
        return [rows[a] for a in wanted if a in rows]
    path = C.results_dir('selection') / 'shortlist.tsv'
    if not path.exists():
        raise SystemExit(f'{path} is missing — run t3_select.py first')
    return C.read_tsv(path)[:args.top]


# --------------------------------------------------------------------------- #
# fetch
# --------------------------------------------------------------------------- #
def stage_fetch(args):
    from src.data.acquire import _http_get, download_family_hmm

    rows = shortlist(args)
    C.banner(f't5 fetch — {len(rows)} families, up to {args.max_members:,} members each')
    index = []
    for r in rows:
        acc = r['pfam_acc']
        out = family_dir(acc)
        fasta, tsv, meta = out / 'members.fasta', out / 'members.tsv', out / 'fetch.json'
        if meta.exists() and fasta.exists() and not args.force:
            cached = C.load_json(meta)
            if cached.get('max_members') == args.max_members and C.sha(fasta) == cached.get('fasta_sha256'):
                print(f'  {acc}: cached ({cached["n_members"]:,} members)', flush=True)
                index.append(cached)
                continue
        started = time.time()
        url = f'{INTERPRO}/{acc}/?page_size={args.page_size}&extra_fields=sequence'
        members, urls, total = [], [], None
        while url and len(members) < args.max_members:
            payload = _http_get(url, timeout=args.timeout).json()
            urls.append(url)
            total = payload.get('count', total)
            for item in payload.get('results', []):
                md = item.get('metadata', {})
                seq = (item.get('extra_fields', {}) or {}).get('sequence') or md.get('sequence')
                if not seq:
                    continue
                spans = []
                for entry in item.get('entries', []) or []:
                    for loc in entry.get('entry_protein_locations', []) or []:
                        for frag in loc.get('fragments', []) or []:
                            try:
                                spans.append((int(frag['start']), int(frag['end'])))
                            except (KeyError, TypeError, ValueError):
                                continue
                members.append({'accession': md.get('accession', ''), 'name': md.get('name', ''),
                                'length': len(seq), 'sequence': seq,
                                'source': md.get('source_database', ''),
                                'domain_ranges': ','.join(f'{a}-{b}' for a, b in sorted(spans))})
                if len(members) >= args.max_members:
                    break
            url = payload.get('next')
            time.sleep(args.sleep)
        if not members:
            raise SystemExit(f'{acc}: InterPro returned no member sequences')
        with fasta.open('w') as fh:
            for m in members:
                fh.write(f'>{m["accession"]} {m["domain_ranges"]}\n')
                for i in range(0, len(m['sequence']), 60):
                    fh.write(m['sequence'][i:i+60] + '\n')
        C.write_tsv(tsv, [{k: v for k, v in m.items() if k != 'sequence'} for m in members])
        hmm = out / 'family.hmm'
        try:
            download_family_hmm(acc, str(hmm))
        except Exception as exc:                        # noqa: BLE001 — profile falls back
            print(f'  {acc}: HMM download failed ({exc}); profile will use InterPro coordinates only',
                  flush=True)
        record = {'pfam_acc': acc, 'n_members': len(members),
                  'interpro_count': int(total) if total is not None else None,
                  'members_nr90_metadata': int(r['members_nr90']),
                  'max_members': args.max_members, 'urls': urls[:5], 'n_pages': len(urls),
                  'fasta_sha256': C.sha(fasta), 'hmm': hmm.exists(),
                  'seconds': time.time() - started,
                  'sampling': 'first N pages of the InterPro cursor, NOT a random sample'}
        C.save_json(record, meta)
        index.append(record)
        print(f'  {acc}: {len(members):,} of {total:,} members in {time.time()-started:.0f}s'
              if total else f'  {acc}: {len(members):,} members', flush=True)
    C.save_json({'protocol': C.protocol_block(stage='t5_fetch'), 'families': index},
                C.results_dir('members') / 'fetch_index.json')
    print(f'\nDONE t5 fetch — {len(index)} families', flush=True)


# --------------------------------------------------------------------------- #
# profile
# --------------------------------------------------------------------------- #
def parse_domtblout(path):
    """Best per-target HMM envelope, plus the number of envelopes on that target.

    The copy count is the direct tandem-repeat measurement: t2 established that sequence
    self-similarity does NOT detect degenerate protein repeats, so this — how many times
    the family's own HMM hits one member — is what actually answers "is this family a
    repeat array?" for the shortlisted families.
    """
    best = {}
    with open(path) as fh:
        for line in fh:
            if line.startswith('#'):
                continue
            f = line.split()
            if len(f) < 23:
                continue
            target, bits = f[0], float(f[13])
            start, end = int(f[19]), int(f[20])          # env from / env to
            row = best.get(target)
            if row is None:
                best[target] = {'bits': bits, 'env_start': start, 'env_end': end,
                                'ievalue': float(f[12]), 'env_len': end - start + 1,
                                'n_copies': 1}
            else:
                row['n_copies'] += 1
                if bits > row['bits']:
                    row.update({'bits': bits, 'env_start': start, 'env_end': end,
                                'ievalue': float(f[12]), 'env_len': end - start + 1})
    return best


def stage_profile(args):
    from src.utils.io import read_fasta

    rows = shortlist(args)
    cal = np.load(C.results_dir('seqstats') / 'retain_calibration.npz')
    grid, edges = cal['grid'], cal['edges']
    import t2_seqstats as T2

    C.banner(f't5 profile — {len(rows)} families')
    summaries = []
    for r in rows:
        acc = r['pfam_acc']
        out = family_dir(acc)
        fasta = out / 'members.fasta'
        if not fasta.exists():
            print(f'  {acc}: no members.fasta — run `fetch` first', flush=True)
            continue
        started = time.time()
        records = read_fasta(str(fasta))
        coords = {m['accession']: m['domain_ranges'] for m in C.read_tsv(out / 'members.tsv')}
        hmm_best = {}
        hmm = out / 'family.hmm'
        if hmm.exists() and shutil.which('hmmsearch'):
            dom = out / 'hmmsearch.domtblout'
            subprocess.run(['hmmsearch', '--cut_ga', '--domtblout', str(dom), '--cpu',
                            str(args.threads), str(hmm), str(fasta)],
                           check=True, stdout=subprocess.DEVNULL)
            hmm_best = parse_domtblout(dom)
        member_rows = []
        for rec in records:
            seq = str(rec.seq)
            dom_seq = C.domain_sequence(seq, coords.get(rec.id, ''))
            full_stats, dom_stats = C.seq_stats(seq), C.seq_stats(dom_seq)
            b = C.length_bin(int(dom_stats['length']), edges)
            b_full = C.length_bin(int(len(seq)), edges)
            entry = {'accession': rec.id, 'full_length': len(seq),
                     'domain_length': dom_stats['length'],
                     'domain_source': 'interpro' if coords.get(rec.id) else 'full-chain',
                     'hmm_bits': hmm_best.get(rec.id, {}).get('bits', float('nan')),
                     'hmm_env_len': hmm_best.get(rec.id, {}).get('env_len', 0),
                     'hmm_copies': hmm_best.get(rec.id, {}).get('n_copies', 0),
                     'aa_entropy_bits': dom_stats['aa_entropy_bits']}
            flags = 0
            for metric in C.SEQ_METRICS:
                entry[metric.name] = dom_stats[metric.name]
                fired = False
                for scope, stats, bin_ in (('dom', dom_stats, b), ('full', full_stats, b_full)):
                    pct = T2.percentile_in_stratum(stats[metric.name],
                                                   cal[f'table_{metric.name}'][bin_], grid)
                    entry[f'{metric.name}_retain_pct_{scope}'] = pct
                    fired = fired or (pct >= args.flag_pct if metric.higher_is_worse
                                      else pct <= 100.0 - args.flag_pct)
                flags += int(fired)
            entry['repetitive'] = flags > 0
            entry['usable'] = bool(not entry['repetitive']
                                   and dom_stats['length'] >= args.min_domain_length)
            entry['full_entropy_bits'] = full_stats['aa_entropy_bits']
            member_rows.append(entry)
        C.write_tsv(out / 'members_profile.tsv', member_rows)
        dom_len = np.array([m['domain_length'] for m in member_rows], dtype=np.float64)
        full_len = np.array([m['full_length'] for m in member_rows], dtype=np.float64)
        n_usable = int(sum(m['usable'] for m in member_rows))
        scale = (int(r['members_nr90']) / len(member_rows)) if member_rows else 0.0
        summaries.append({
            'pfam_acc': acc, 'pfam_id': r['pfam_id'], 'n_members_profiled': len(member_rows),
            'median_domain_length': float(np.median(dom_len)),
            'p05_domain_length': float(np.percentile(dom_len, 5)),
            'p25_domain_length': float(np.percentile(dom_len, 25)),
            'p75_domain_length': float(np.percentile(dom_len, 75)),
            'median_full_length': float(np.median(full_len)),
            'frac_domain_len_ge_min': float(np.mean(dom_len >= args.min_domain_length)),
            'frac_members_repetitive': float(np.mean([m['repetitive'] for m in member_rows])),
            'frac_members_usable': float(n_usable / max(1, len(member_rows))),
            'n_usable_profiled': n_usable,
            'n_usable_extrapolated': int(round(n_usable * scale)),
            'frac_with_interpro_domain': float(np.mean([m['domain_source'] == 'interpro'
                                                        for m in member_rows])),
            'frac_with_ga_hit': float(np.mean([np.isfinite(m['hmm_bits']) for m in member_rows]))
            if hmm_best else float('nan'),
            'median_hmm_env_len': float(np.median([m['hmm_env_len'] for m in member_rows]))
            if hmm_best else float('nan'),
            # The direct tandem-repeat measurement for this family (t2 showed sequence
            # self-similarity cannot supply it): how many times the family's own HMM hits
            # a typical member, and how often a member carries an array.
            'median_hmm_copies': float(np.median([m['hmm_copies'] for m in member_rows]))
            if hmm_best else float('nan'),
            'frac_members_multi_copy': float(np.mean([m['hmm_copies'] >= 2 for m in member_rows]))
            if hmm_best else float('nan'),
            'seconds': time.time() - started,
        })
        s = summaries[-1]
        print(f"  {acc}: median domain {s['median_domain_length']:.0f} aa, "
              f"{s['frac_members_repetitive']:.1%} repetitive, "
              f"{n_usable:,}/{len(member_rows):,} usable", flush=True)
    C.write_tsv(C.results_dir('members') / 'families_profile.tsv', summaries)
    C.save_json({'protocol': C.protocol_block(stage='t5_profile'),
                 'min_domain_length': args.min_domain_length, 'flag_pct': args.flag_pct,
                 'n_families': len(summaries),
                 'note': 'n_usable_extrapolated scales the profiled pass rate to n_proteins_nr90; '
                         'the profiled members are the first InterPro pages, not a random sample'},
                C.results_dir('members') / 'summary_profile.json')
    print(f'\nDONE t5 profile — {len(summaries)} families', flush=True)


# --------------------------------------------------------------------------- #
# cluster / leakage
# --------------------------------------------------------------------------- #
def mmseqs_search(query, target, out_m8, threads, sensitivity=7.5, coverage=LEAKAGE_COVERAGE,
                  max_seqs=300):
    tmp_root = os.environ.get('TMPDIR') or tempfile.gettempdir()
    with tempfile.TemporaryDirectory(dir=tmp_root, prefix='anchor_mmseqs_') as tmp:
        subprocess.run(['mmseqs', 'easy-search', str(query), str(target), str(out_m8), tmp,
                        '-s', str(sensitivity), '-c', str(coverage), '--cov-mode', '0',
                        '--max-seqs', str(max_seqs), '--format-output', M8_FIELDS,
                        '--threads', str(threads), '-v', '1'], check=True)
    hits = {}
    with open(out_m8) as fh:
        for line in fh:
            f = line.rstrip('\n').split('\t')
            if len(f) < 8:
                continue
            q, fident, qcov, tcov = f[0], float(f[2]), float(f[6]), float(f[7])
            if fident >= LEAKAGE_IDENTITY and max(qcov, tcov) >= LEAKAGE_COVERAGE:
                hits[q] = max(hits.get(q, 0.0), fident)
    return hits


def stage_cluster(args):
    if shutil.which('mmseqs') is None:
        raise SystemExit('mmseqs not on PATH — slurm/env_setup.sh loads mmseqs2/14.7')
    rows = shortlist(args)
    if args.shard is not None:
        rows = [r for i, r in enumerate(rows) if i % args.n_shards == args.shard]
    C.banner(f't5 cluster — {len(rows)} families (shard {args.shard})')
    for r in rows:
        acc = r['pfam_acc']
        out = family_dir(acc)
        fasta = out / 'members.fasta'
        if not fasta.exists():
            print(f'  {acc}: no members.fasta — run `fetch` first', flush=True)
            continue
        started = time.time()
        n_members = sum(1 for line in open(fasta) if line.startswith('>'))
        tmp_root = os.environ.get('TMPDIR') or tempfile.gettempdir()
        with tempfile.TemporaryDirectory(dir=tmp_root, prefix='anchor_cluster_') as tmp:
            prefix = Path(tmp) / 'clu'
            subprocess.run(['mmseqs', 'easy-cluster', str(fasta), str(prefix), tmp,
                            '--min-seq-id', str(args.cluster_identity), '-c', '0.8',
                            '--cov-mode', '0', '--threads', str(args.threads), '-v', '1'],
                           check=True)
            reps = sum(1 for line in open(f'{prefix}_rep_seq.fasta') if line.startswith('>'))
            sizes = {}
            with open(f'{prefix}_cluster.tsv') as fh:
                for line in fh:
                    head = line.split('\t')[0]
                    sizes[head] = sizes.get(head, 0) + 1
        # Direction 1: how much of the family does R already contain?
        forward = mmseqs_search(fasta, C.RETAIN_FASTA, out / 'members_vs_retain.m8', args.threads)
        # Direction 2: what would adopting this family DELETE from R? This is the cost the
        # project actually pays — the leakage filter drops retain sequences, not family ones.
        reverse = mmseqs_search(C.RETAIN_FASTA, fasta, out / 'retain_vs_members.m8', args.threads)
        n_retain = sum(1 for line in open(C.RETAIN_FASTA) if line.startswith('>'))
        record = {
            'pfam_acc': acc, 'n_members': n_members,
            'n_clusters_at_identity': reps, 'cluster_identity': args.cluster_identity,
            'largest_cluster_frac': float(max(sizes.values()) / n_members) if sizes else float('nan'),
            'redundancy': float(1.0 - reps / n_members) if n_members else float('nan'),
            'n_members_with_retain_homolog': len(forward),
            'frac_members_with_retain_homolog': float(len(forward) / max(1, n_members)),
            'n_retain_removed': len(reverse),
            'frac_retain_removed': float(len(reverse) / max(1, n_retain)),
            'n_retain_total': n_retain,
            'leakage_rule': f'fident >= {LEAKAGE_IDENTITY} and coverage >= {LEAKAGE_COVERAGE} '
                            '(configs/data/pfam_default.yaml leakage_filter)',
            'seconds': time.time() - started,
        }
        C.save_json(record, out / 'cluster.json')
        print(f"  {acc}: {reps:,}/{n_members:,} clusters, "
              f"{record['frac_members_with_retain_homolog']:.1%} of members have an R homolog, "
              f"{len(reverse):,} R sequences would be removed", flush=True)
    print('\nDONE t5 cluster', flush=True)


# --------------------------------------------------------------------------- #
# embed
# --------------------------------------------------------------------------- #
def stratified_sample(lengths, n, seed):
    """Seeded sample spread over length deciles, so the embedded members are not the
    short (or the long) ones by accident."""
    lengths = np.asarray(lengths, dtype=np.float64)
    if n >= lengths.size:
        return np.arange(lengths.size)
    rng = np.random.default_rng(seed)
    edges = np.quantile(lengths, np.linspace(0, 1, 11)[1:-1])
    bins = np.searchsorted(edges, lengths, side='right')
    picked = []
    per = max(1, n // 10)
    for b in range(11):
        idx = np.flatnonzero(bins == b)
        if idx.size:
            picked.append(rng.choice(idx, size=min(per, idx.size), replace=False))
    picked = np.concatenate(picked) if picked else np.arange(0)
    if picked.size < n:
        rest = np.setdiff1d(np.arange(lengths.size), picked)
        picked = np.concatenate([picked, rng.choice(rest, size=min(n - picked.size, rest.size),
                                                    replace=False)])
    return np.sort(picked[:n])


def stage_embed(args):
    import torch

    from src.models.load import load_mlm, load_tokenizer
    from src.utils.io import read_fasta

    if not torch.cuda.is_available():
        raise SystemExit('CUDA required — submit t5_embed.sbatch on a GPU node')
    rows = shortlist(args)
    space = C.Space.load(C.results_dir('matrix') / f'space_{args.space}.npz')
    scored = {r['pfam_acc']: r for r in C.read_tsv(C.results_dir('selection') / 'families_scored.tsv')}
    reps_x, reps_meta = C.load_group('pfam_reps', verify=False)
    # The bulk is exactly the population t1 measured — which is a subset of the cached
    # embeddings whenever the run was limited (a smoke slice), and all of it otherwise.
    keep = [i for i, r in enumerate(reps_meta) if r['id'] in scored]
    if len(keep) != len(reps_meta):
        print(f'  bulk restricted to the {len(keep):,} scored families '
              f'(of {len(reps_meta):,} cached representatives)', flush=True)
    reps_x = reps_x[keep]
    accs = [reps_meta[i]['id'] for i in keep]
    unit_reps = np.ascontiguousarray(space.apply(reps_x))
    population_nn = np.array([float(scored[a][f'{args.space}_iso_nn_xclan']) for a in accs])
    clan_of = {r['pfam_acc']: r['clan_acc'] for r in scored.values()}
    C.banner(f't5 embed — {len(rows)} families, up to {args.sample:,} members each')
    model = load_mlm(C.MODEL_ID, device='cuda').eval()
    tokenizer = load_tokenizer(C.MODEL_ID)
    out_rows = []
    for r in rows:
        acc = r['pfam_acc']
        out = family_dir(acc)
        fasta = out / 'members.fasta'
        if not fasta.exists():
            print(f'  {acc}: no members.fasta — run `fetch` first', flush=True)
            continue
        started = time.time()
        records = read_fasta(str(fasta))
        seqs = [str(rec.seq) for rec in records]
        pick = stratified_sample([len(s) for s in seqs], args.sample, args.seed)
        chosen = [seqs[i] for i in pick]
        feats = C.embed_length_sorted(model, tokenizer, chosen, 'cuda', args.batch_size, 1024)
        np.save(out / 'member_embeddings.npy', feats.astype(np.float16))
        unit_members = space.apply(np.asarray(feats, dtype=np.float32))
        centroid = C.normalize(unit_members.mean(axis=0, keepdims=True))
        i = accs.index(acc) if acc in accs else -1
        # Exactly the exclusion the representative-level metric uses: never score a family
        # against its own representative or against its clan-mates. Without this, every
        # member's nearest representative is its OWN family's and the number is ~0 for all.
        clan = clan_of.get(acc)
        exclude = np.array([bool(clan) and clan_of.get(a) == clan for a in accs])
        if i >= 0:
            exclude[i] = True
        sims = (centroid @ unit_reps.T)[0]
        sims_xclan = sims.copy()
        sims_xclan[exclude] = -np.inf
        top10 = np.sort(sims_xclan)[-10:]
        rep_unit = unit_reps[i] if i >= 0 else None
        rep_iso = float(scored[acc][f'{args.space}_iso_nn_xclan']) if acc in scored else float('nan')
        # Per-MEMBER isolation: single sequences against single representatives, the only
        # apples-to-apples comparison with the representative-level screen. The centroid is
        # reported too, but averaging shrinks a family's idiosyncratic directions and pulls
        # the mean towards the corpus, so it is biased downwards by construction.
        member_sims = np.asarray(unit_members @ unit_reps.T, dtype=np.float32)
        member_sims[:, exclude] = -np.inf
        member_iso = 1.0 - member_sims.max(axis=1).astype(np.float64)
        del member_sims
        population = np.sort(population_nn)
        record = {
            'pfam_acc': acc, 'pfam_id': r['pfam_id'], 'n_members_embedded': len(chosen),
            'compactness': float((unit_members @ centroid.T).mean()),
            'rep_to_centroid_cos': float(rep_unit @ centroid[0]) if rep_unit is not None else float('nan'),
            'rep_iso_nn_xclan': rep_iso,
            'member_iso_median': float(np.median(member_iso)),
            'member_iso_p25': float(np.percentile(member_iso, 25)),
            'member_iso_p75': float(np.percentile(member_iso, 75)),
            'member_iso_percentile_vs_population': float(C.percentile_against(
                float(np.median(member_iso)), population)),
            'frac_members_above_rep_iso': float(np.mean(member_iso >= rep_iso))
            if np.isfinite(rep_iso) else float('nan'),
            'frac_members_above_population_p99': float(np.mean(
                member_iso >= np.quantile(population, 0.99))),
            'centroid_iso_nn_xclan': float(1.0 - sims_xclan.max()),
            'centroid_iso_top10': float(1.0 - top10.mean()),
            'centroid_nn_family': accs[int(np.argmax(sims_xclan))],
            'centroid_iso_percentile': float(C.percentile_against(
                1.0 - sims_xclan.max(), population)),
            'seconds': time.time() - started,
        }
        out_rows.append(record)
        C.save_json(record, out / 'embed.json')
        print(f"  {acc}: member isolation median {record['member_iso_median']:.4f} "
              f"(rep {record['rep_iso_nn_xclan']:.4f}, population pct "
              f"{record['member_iso_percentile_vs_population']:.1f}); centroid "
              f"{record['centroid_iso_nn_xclan']:.4f}; rep-to-centroid cos "
              f"{record['rep_to_centroid_cos']:.3f}, compactness {record['compactness']:.3f}",
              flush=True)
    C.write_tsv(C.results_dir('members') / 'families_embed.tsv', out_rows)
    print(f'\nDONE t5 embed — {len(out_rows)} families', flush=True)


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def stage_report(args):
    import matplotlib.pyplot as plt

    C.set_rc()
    rows = shortlist(args)
    base = C.results_dir('members')
    profile = {r['pfam_acc']: r for r in C.read_tsv(base / 'families_profile.tsv')} \
        if (base / 'families_profile.tsv').exists() else {}
    embed = {r['pfam_acc']: r for r in C.read_tsv(base / 'families_embed.tsv')} \
        if (base / 'families_embed.tsv').exists() else {}
    merged = []
    for r in rows:
        acc = r['pfam_acc']
        cluster_path = family_dir(acc) / 'cluster.json'
        fetch_path = family_dir(acc) / 'fetch.json'
        entry = {'pfam_acc': acc, 'pfam_id': r['pfam_id'], 'type': r['type'],
                 'description': r['description'], 'clan_acc': r['clan_acc'],
                 'members_nr90': int(r['members_nr90']),
                 'isolation_rank_score': float(r['isolation_rank_score']),
                 'model_length': int(r['model_length'])}
        for source in (fetch_path, cluster_path):
            if source.exists():
                payload = C.load_json(source)
                entry.update({k: v for k, v in payload.items()
                              if k not in ('pfam_acc', 'urls', 'protocol')})
        entry.update({k: v for k, v in profile.get(acc, {}).items() if k != 'pfam_acc'})
        entry.update({k: v for k, v in embed.get(acc, {}).items()
                      if k not in ('pfam_acc', 'pfam_id')})
        usable = float(entry.get('n_usable_extrapolated', 0) or 0)
        rep_cos = float(entry.get('rep_to_centroid_cos', float('nan')) or float('nan'))
        rep_iso = float(entry.get('rep_iso_nn_xclan', float('nan')) or float('nan'))
        member_iso = float(entry.get('member_iso_median', float('nan')) or float('nan'))
        member_pct = float(entry.get('member_iso_percentile_vs_population', float('nan')) or float('nan'))
        reasons = []
        if usable < args.min_usable:
            reasons.append(f'only ~{usable:,.0f} usable members')
        if np.isfinite(rep_cos) and rep_cos < args.min_rep_centroid_cos:
            reasons.append(f'representative is not typical of its family (cos {rep_cos:.2f})')
        # Judged on the per-member statistic, not on the centroid: averaging biases a
        # centroid towards the corpus mean, so a centroid/representative ratio would reject
        # families for a property of the arithmetic rather than of the family.
        if np.isfinite(member_iso) and np.isfinite(rep_iso) and member_iso < args.min_member_iso_frac * rep_iso:
            reasons.append(f'typical member far less isolated than the representative '
                           f'({member_iso:.2f} vs {rep_iso:.2f})')
        if np.isfinite(member_pct) and member_pct < args.min_member_iso_pct:
            reasons.append(f'typical member only at population percentile {member_pct:.0f}')
        if float(entry.get('frac_members_repetitive', 0) or 0) > args.max_repetitive_frac:
            reasons.append('members are largely repetitive')
        entry['verdict'] = 'verified' if not reasons else 'rejected'
        entry['verdict_reasons'] = '; '.join(reasons)
        merged.append(entry)
    merged.sort(key=lambda e: (e['verdict'] != 'verified', -e['isolation_rank_score']))
    C.write_tsv(base / 'verified_shortlist.tsv', merged)

    verified = [m for m in merged if m['verdict'] == 'verified']
    if merged:
        fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))
        names = [m['pfam_acc'] for m in merged]
        axes[0].barh(names, [float(m.get('median_domain_length', 0) or 0) for m in merged],
                     color=C.C_BULK)
        axes[0].axvline(args.min_domain_length, ls='--', c=C.C_CANDIDATE, lw=0.8)
        axes[0].set(xlabel='median member domain length (aa)', title='Length distribution')
        axes[1].barh(names, [float(m.get('frac_members_repetitive', 0) or 0) for m in merged],
                     color=C.C_REPEAT)
        axes[1].set(xlabel='fraction of members flagged repetitive', title='Repetitiveness')
        axes[2].barh(names, [float(m.get('frac_retain_removed', 0) or 0) for m in merged],
                     color=C.C_PROJECT)
        axes[2].set(xlabel='fraction of retain R that would be removed', title='Leakage cost to R')
        for ax in axes:
            ax.tick_params(labelsize=6)
            C.style_axis(ax, ygrid=False)
        fig.suptitle('Member-level verification of the shortlisted families', fontsize=10)
        C.save_fig(fig, 'member_verification')

        pairs = [(float(m.get('rep_iso_nn_xclan', np.nan) or np.nan),
                  float(m.get('member_iso_median', np.nan) or np.nan), m['pfam_acc'])
                 for m in merged]
        pairs = [p for p in pairs if np.isfinite(p[0]) and np.isfinite(p[1])]
        if pairs:
            fig, ax = plt.subplots(figsize=(6.2, 5.6))
            xs, ys, labels = zip(*pairs)
            lim = max(max(xs), max(ys)) * 1.1
            ax.plot([0, lim], [0, lim], ls='--', c='#999', lw=0.8)
            ax.scatter(xs, ys, c=C.C_CANDIDATE, s=28)
            for x, y, label in pairs:
                ax.annotate(label, (x, y), fontsize=6, xytext=(4, 3), textcoords='offset points')
            ax.set(xlabel='isolation of the representative (1 - nearest non-clan cosine)',
                   ylabel='median isolation of a member (same metric, same exclusions)',
                   title='Does the family sit where its representative sits?\n'
                         'Below the diagonal: the one-sequence screen overstated isolation')
            C.style_axis(ax)
            C.save_fig(fig, 'member_vs_rep_isolation')

    summary = {'protocol': C.protocol_block(stage='t5_report'),
               'n_families': len(merged), 'n_verified': len(verified),
               'thresholds': {'min_usable': args.min_usable,
                              'min_rep_centroid_cos': args.min_rep_centroid_cos,
                              'min_member_iso_frac': args.min_member_iso_frac,
                              'min_member_iso_pct': args.min_member_iso_pct,
                              'max_repetitive_frac': args.max_repetitive_frac,
                              'min_domain_length': args.min_domain_length}}
    C.save_json(summary, base / 'summary_members.json')
    write_member_results(base, merged, verified, args)
    print(f'\nDONE t5 report — {len(verified)}/{len(merged)} verified -> {base}', flush=True)


def write_member_results(base, merged, verified, args):
    def g(m, key, default=float('nan')):
        value = m.get(key, default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    table = '\n'.join(
        f"| {m['pfam_acc']} | {m['pfam_id']} | {m['members_nr90']:,} | "
        f"{int(g(m, 'n_members_profiled', 0)):,} | {g(m, 'median_domain_length'):.0f} | "
        f"{g(m, 'frac_members_repetitive'):.1%} | {int(g(m, 'n_usable_extrapolated', 0)):,} | "
        f"{g(m, 'median_hmm_copies'):.1f} | {g(m, 'rep_to_centroid_cos'):.2f} | "
        f"{g(m, 'rep_iso_nn_xclan'):.3f} | {g(m, 'member_iso_median'):.3f} | "
        f"{g(m, 'member_iso_percentile_vs_population'):.0f} | "
        f"{g(m, 'centroid_iso_nn_xclan'):.3f} | "
        f"{g(m, 'frac_members_with_retain_homolog'):.1%} | {int(g(m, 'n_retain_removed', 0)):,} | "
        f"{m['verdict']} |" for m in merged)
    picks = '\n'.join(
        f"{i}. **{m['pfam_acc']} ({m['pfam_id']})** — {m['description'][:70]}. "
        f"~{int(g(m, 'n_usable_extrapolated', 0)):,} usable members, median domain "
        f"{g(m, 'median_domain_length'):.0f} aa, {g(m, 'frac_members_repetitive'):.1%} repetitive, "
        f"clan {m['clan_acc'] or 'none'}, typical member isolation {g(m, 'member_iso_median'):.3f} "
        f"(representative {g(m, 'rep_iso_nn_xclan'):.3f}, percentile "
        f"{g(m, 'member_iso_percentile_vs_population'):.0f} of all representatives), adopting it "
        f"would remove {int(g(m, 'n_retain_removed', 0)):,} retain sequences."
        for i, m in enumerate(verified[:5], 1)) or '_No family passed every member-level check._'
    (base / 'RESULTS.md').write_text(f"""# Member-level verification of the anchor shortlist

{len(verified)} of {len(merged)} shortlisted families pass every member-level check.

## Per-family results

| Pfam | Name | NR90 | Profiled | Median domain (aa) | Repetitive | ~Usable | HMM copies/member | Rep↔centroid cos | Rep isolation | Member isolation (median) | Member pct vs all reps | Centroid isolation | Members with R homolog | R sequences removed | Verdict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
{table}

## Recommended anchors

{picks}

## How to read this

- **Profiled** members are the first InterPro cursor pages, capped at {args.top} families ×
  the fetch limit. Accession-ordered pagination correlates with taxonomy, so every member
  statistic is biased unless the cap exceeds the family size. `~Usable` scales the profiled
  pass rate up to `n_proteins_nr90` and is an estimate, not a count.
- **Rep↔centroid cos** is the check the representative-level screen cannot make: a low value
  means the one sequence that carried the family through stages t1–t4 is not typical of it.
- **Member isolation** is the verdict-bearing number: the median over sampled members of
  `1 − (cosine to the nearest OTHER family's representative)`, excluding the family's own
  representative and its clan-mates — exactly the exclusion the representative-level screen
  uses, so single sequences are compared with single sequences. `Member pct vs all reps`
  places that median in the distribution of all 21,763 representative isolations.
- **Centroid isolation** is reported but is **not** used for the verdict. Averaging members
  shrinks a family's idiosyncratic directions and pulls the mean towards the corpus, so a
  centroid is biased towards the bulk by construction; rejecting a family on a
  centroid-versus-representative ratio would penalise it for a property of the arithmetic.
  It is here because a centroid that stays isolated (PF07972, PF10436) is genuinely strong
  evidence, while a low value on its own is not evidence of anything.
- **Two leakage directions.** "Members with R homolog" is how much of the family the retain
  corpus already contains. "R sequences removed" is what adopting the family would delete
  from R under the project's leakage filter (identity ≥ {LEAKAGE_IDENTITY:.2f}, coverage ≥
  {LEAKAGE_COVERAGE:.2f}) — that is the cost actually paid, and the one the headroom-collapse
  post-mortem says to check first.
- Isolation in ESM-2 35M representation space is **not** evidence of absence from ESM-2
  pretraining, and per experiment 6 it does not imply the model models the family worse.
- The hazard screen upstream is keyword-based and is not a biosecurity control. A human signs
  off on the final family.
""")
    print(f'  wrote {base / "RESULTS.md"}', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', choices=['fetch', 'profile', 'cluster', 'embed', 'report'])
    C.common_args(ap)
    ap.add_argument('--top', type=int, default=25)
    ap.add_argument('--families', default='', help='explicit comma-separated accessions')
    ap.add_argument('--space', default=C.HEADLINE_SPACE, choices=list(C.SPACES))
    ap.add_argument('--threads', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', '8')))
    ap.add_argument('--force', action='store_true')
    # fetch
    ap.add_argument('--max-members', type=int, default=2000)
    ap.add_argument('--page-size', type=int, default=200)
    ap.add_argument('--sleep', type=float, default=0.34, help='seconds between InterPro pages')
    ap.add_argument('--timeout', type=int, default=120)
    # profile
    ap.add_argument('--min-domain-length', type=int, default=60)
    ap.add_argument('--flag-pct', type=float, default=99.0)
    # cluster
    ap.add_argument('--cluster-identity', type=float, default=0.5)
    ap.add_argument('--shard', type=int, default=None)
    ap.add_argument('--n-shards', type=int, default=1)
    # embed
    ap.add_argument('--sample', type=int, default=1000)
    ap.add_argument('--batch-size', type=int, default=C.EMBED_BATCH)
    # report
    ap.add_argument('--min-usable', type=int, default=1000)
    ap.add_argument('--min-rep-centroid-cos', type=float, default=0.80)
    ap.add_argument('--min-member-iso-frac', type=float, default=0.5,
                    help='median member isolation as a fraction of the representative’s')
    ap.add_argument('--min-member-iso-pct', type=float, default=90.0,
                    help='percentile of the median member isolation against all 21,763 reps')
    ap.add_argument('--max-repetitive-frac', type=float, default=0.5)
    args = ap.parse_args()
    C.apply_results_override(args.results)
    {'fetch': stage_fetch, 'profile': stage_profile, 'cluster': stage_cluster,
     'embed': stage_embed, 'report': stage_report}[args.stage](args)


if __name__ == '__main__':
    main()
