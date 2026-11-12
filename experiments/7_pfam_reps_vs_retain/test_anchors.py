"""Offline unit tests for the anchor screen. No GPU, no network, no torch.

Each test pins a decision the pipeline makes, not an implementation detail:
the premise that centering is necessary, the LOF shortcut that avoids a 3.8 GB matrix,
the blocked top-k with clan masking, the repetitiveness metrics, and the gate logic.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import common as C                                             # noqa: E402
import t1_similarity as T1                                     # noqa: E402
import t2_seqstats as T2                                       # noqa: E402


# --------------------------------------------------------------------------- #
# The premise: mean-pooled embeddings are anisotropic and centering fixes it
# --------------------------------------------------------------------------- #
def test_centering_removes_the_shared_direction():
    rng = np.random.default_rng(0)
    shared = np.ones(32) * 3.0
    x = shared + rng.normal(scale=0.2, size=(400, 32))
    raw = C.fit_space(x, 'raw').apply(x)
    cen = C.fit_space(x, 'cen').apply(x)
    off_raw = (raw @ raw.T)[~np.eye(400, dtype=bool)]
    off_cen = (cen @ cen.T)[~np.eye(400, dtype=bool)]
    assert off_raw.mean() > 0.95, 'the synthetic cloud must reproduce the anisotropy problem'
    assert abs(off_cen.mean()) < 0.1
    assert C.anisotropy(raw)['mean_resultant_length'] > 0.95
    assert C.anisotropy(cen)['mean_resultant_length'] < 0.2


def test_whitening_projects_and_renormalises():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(200, 16)) @ np.diag(np.linspace(5, 0.1, 16))
    whi = C.fit_space(x, 'whi', n_components=8).apply(x)
    assert whi.shape == (200, 8)
    assert np.allclose(np.linalg.norm(whi, axis=1), 1.0, atol=1e-5)


# --------------------------------------------------------------------------- #
# LOF: the shortcut that licenses not calling sklearn at n=21,763
# --------------------------------------------------------------------------- #
def test_lof_from_topk_matches_sklearn():
    from sklearn.neighbors import LocalOutlierFactor

    rng = np.random.default_rng(2)
    x = np.vstack([rng.normal(size=(280, 5)), rng.normal(loc=8, scale=0.2, size=(20, 5))])
    k = 20
    d = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    idx = np.argsort(d, axis=1)[:, :k]
    dist = np.take_along_axis(d, idx, axis=1)
    mine = C.lof_from_topk(dist, idx, k)
    reference = -LocalOutlierFactor(n_neighbors=k).fit(x).negative_outlier_factor_
    assert np.allclose(mine, reference, atol=1e-6)
    assert mine[-20:].mean() > mine[:280].mean(), 'the planted outliers must score higher'


def test_lof_rejects_k_beyond_the_stored_neighbourhood():
    dist = np.linspace(0.1, 1.0, 20).reshape(4, 5)
    idx = np.tile(np.arange(5) % 4, (4, 1))
    with pytest.raises(ValueError):
        C.lof_from_topk(dist, idx, 9)


# --------------------------------------------------------------------------- #
# Blocked matrix pass
# --------------------------------------------------------------------------- #
def _reference_topk(sim, k, mask=None):
    s = sim.copy()
    np.fill_diagonal(s, -np.inf)
    if mask is not None:
        s[mask] = -np.inf
    idx = np.argsort(-s, axis=1, kind='stable')[:, :k]
    return np.take_along_axis(s, idx, axis=1), idx


def test_blocked_pass_matches_a_full_argsort(tmp_path):
    rng = np.random.default_rng(3)
    unit = C.normalize(rng.normal(size=(120, 8)))
    clans = np.array([f'CL{i % 5:02d}' if i % 3 else '' for i in range(120)])
    members = {}
    for value in set(clans) - {''}:
        ids = np.flatnonzero(clans == value)
        for i in ids:
            members[int(i)] = ids
    acc = T1.blocked_similarity(unit, members, tmp_path / 'sim.npy', block=17, topk=6,
                                hist_bins=64)
    sim = unit @ unit.T
    ref_sim, ref_idx = _reference_topk(sim, 6)
    assert np.allclose(acc['top_sim'], ref_sim, atol=1e-6)
    assert (acc['top_idx'] == ref_idx).all()

    mask = np.zeros_like(sim, dtype=bool)
    for i, ids in members.items():
        mask[i, ids] = True
    ref_sim_x, ref_idx_x = _reference_topk(sim, 6, mask)
    assert np.allclose(acc['xclan_sim'], ref_sim_x, atol=1e-6)
    same_clan = [(i, int(j)) for i in range(120) for j in acc['xclan_idx'][i]
                 if clans[i] and clans[i] == clans[j]]
    assert not same_clan, 'no clan-mate may survive the masked search'

    stored = np.load(tmp_path / 'sim.npy')
    assert stored.shape == (120, 120)
    assert np.allclose(np.diag(stored).astype(np.float32), 1.0), 'the diagonal is written exactly'
    # Symmetric to fp16 precision, not bitwise: (i,j) and (j,i) are accumulated in
    # different blocks, so BLAS may order the sums differently before the downcast.
    wide = stored.astype(np.float32)
    assert np.abs(wide - wide.T).max() < 1e-3
    assert acc['hist'].sum() == 120 * 119, 'every off-diagonal pair is counted exactly once'
    from scipy.spatial.distance import squareform

    assert np.allclose(squareform(acc['condensed']), 1.0 - sim, atol=1e-6)


def test_global_topk_union_trick():
    rng = np.random.default_rng(4)
    reps = C.normalize(rng.normal(size=(40, 6)))
    retain = C.normalize(rng.normal(size=(90, 6)))
    k = 10
    rep_sim = -np.sort(-(reps @ reps.T + np.diag(np.full(40, -np.inf))), axis=1)[:, :k]
    ret_sim = -np.sort(-(reps @ retain.T), axis=1)[:, :k]
    mine = T1.frac_retain_in_topk(rep_sim, ret_sim, k)
    joint = np.hstack([reps @ reps.T + np.diag(np.full(40, -np.inf)), reps @ retain.T])
    is_retain = np.concatenate([np.zeros(40, bool), np.ones(90, bool)])
    reference = np.array([is_retain[np.argsort(-row)[:k]].mean() for row in joint])
    assert np.allclose(mine, reference)


# --------------------------------------------------------------------------- #
# Linkage-derived heights
# --------------------------------------------------------------------------- #
def test_linkage_leaf_heights_on_a_hand_built_tree():
    # 6 leaves: (0,1) at 0.1, (2,3) at 0.2, those two at 0.5, then 4 at 0.7, then 5 at 0.9
    z = np.array([[0, 1, 0.1, 2], [2, 3, 0.2, 2], [6, 7, 0.5, 4],
                  [4, 8, 0.7, 5], [5, 9, 0.9, 6]])
    out = C.linkage_leaf_heights(z, 6, thresholds=(4, 6))
    assert np.allclose(out['first_merge_height'], [0.1, 0.1, 0.2, 0.2, 0.7, 0.9])
    assert np.allclose(out['height_join4'], [0.5, 0.5, 0.5, 0.5, 0.7, 0.9])
    assert np.allclose(out['height_join6'], 0.9)
    assert out['merge_order_frac'][5] == 1.0


def test_linkage_leaf_heights_rejects_a_mismatched_tree():
    with pytest.raises(ValueError):
        C.linkage_leaf_heights(np.zeros((3, 4)), 6)


# --------------------------------------------------------------------------- #
# Sequence quality
# --------------------------------------------------------------------------- #
def test_window_entropy_matches_a_naive_loop():
    rng = np.random.default_rng(5)
    seq = ''.join(rng.choice(list(C.STD_AA), size=200))
    codes = C.encode(seq)
    fast = C.window_entropy(codes, 12)
    slow = []
    for i in range(len(seq) - 11):
        window = seq[i:i + 12]
        p = np.array([window.count(a) for a in set(window)], dtype=float) / 12
        slow.append(-np.sum(p * np.log2(p)))
    assert np.allclose(fast, slow, atol=1e-9)


def test_sequence_metrics_detect_low_complexity_not_degenerate_repeats():
    """Pins the measured scope of these metrics (probe over all 21,763 representatives).

    A residue-level tandem repeat is caught; a degenerate one — copies sharing a profile,
    which is what a Pfam Repeat family actually is — is not. That is why t2's positive
    control asserts low-complexity separation and records the tandem-repeat negative, and
    why the Pfam `type` label is reported rather than used as a gate.
    """
    rng = np.random.default_rng(11)
    alphabet = list('ACDEFGHIKLMNPQRSTVWY')

    def mutate(seq, p):
        return ''.join(ch if rng.random() > p else rng.choice(alphabet) for ch in seq)

    unit = ''.join(rng.choice(alphabet, size=30))
    natural = ''.join(rng.choice(alphabet, size=300))
    exact = unit * 10
    # What a Pfam Repeat family really looks like: ~25% identity between copies, indels
    # shifting the register, and the array covering only part of a longer chain.
    array = ''
    for _ in range(8):
        copy = mutate(unit, 0.75)
        cut = int(rng.integers(0, 3))
        array += copy[:len(copy) - cut] + ''.join(rng.choice(alphabet, size=int(rng.integers(0, 3))))
    realistic = (''.join(rng.choice(alphabet, size=180)) + array
                 + ''.join(rng.choice(alphabet, size=180)))
    s_exact, s_real, s_nat = (C.seq_stats(s) for s in (exact, realistic, natural))
    assert s_exact['max_periodicity'] > 0.95, 'an exact tandem repeat must be detected'
    assert s_real['max_periodicity'] <= 1.5 * s_nat['max_periodicity'], (
        'a degenerate, indel-shifted repeat array is NOT detected by residue-identity '
        'periodicity — the measured limitation the pipeline documents (t2 positive control), '
        'not a bug to fix silently')


def test_repetitiveness_ordering_on_known_sequences():
    rng = np.random.default_rng(6)
    natural = ''.join(rng.choice(list(C.STD_AA), size=300))
    tandem = 'ACDEF' * 60                             # a 5-periodic tandem repeat
    poly = 'A' * 300
    s_nat, s_tan, s_poly = (C.seq_stats(s) for s in (natural, tandem, poly))
    assert s_poly['max_aa_frac'] == 1.0
    assert s_poly['longest_homopolymer'] == 300
    assert s_tan['max_periodicity'] > 0.99 and s_tan['best_period'] == 5
    assert s_nat['max_periodicity'] < 0.3
    assert s_poly['zlib_ratio'] < s_tan['zlib_ratio'] < s_nat['zlib_ratio']
    assert s_poly['lowcomp_frac_w12'] == 1.0 and s_nat['lowcomp_frac_w12'] < 0.2
    assert s_nat['aa_entropy_bits'] > s_tan['aa_entropy_bits'] > s_poly['aa_entropy_bits']


def test_seq_stats_handles_degenerate_input():
    assert C.seq_stats('')['length'] == 0
    short = C.seq_stats('ACD')
    assert short['length'] == 3 and np.isfinite(short['zlib_ratio'])
    assert np.isfinite(C.seq_stats('AC')['max_aa_frac'])


def test_domain_span_parsing():
    assert C.domain_spans('58-425') == [(57, 425)]
    assert C.domain_spans('21-81,97-166') == [(20, 81), (96, 166)]
    assert C.domain_spans('') == []
    assert C.domain_spans('garbage') == []
    assert C.domain_spans('10-5') == []
    assert C.domain_spans('0-5') == []
    assert C.domain_spans('1-500', length=100) == [(0, 100)]
    seq = 'ABCDEFGHIJ'
    assert C.domain_sequence(seq, '2-4,9-10') == 'BCDIJ'
    assert C.domain_sequence(seq, '') == seq, 'no coordinates means the whole chain'


def test_stratified_percentile_is_taken_within_the_length_bin():
    grid = np.arange(0, 100.5, 0.5)
    low = np.linspace(0.0, 0.5, grid.size)          # short sequences: low values
    high = np.linspace(0.5, 1.0, grid.size)         # long sequences: high values
    assert T2.percentile_in_stratum(0.5, low, grid) >= 99.0
    assert T2.percentile_in_stratum(0.5, high, grid) <= 1.0


def test_length_deciles_and_bins():
    edges = C.length_decile_edges(np.arange(1, 1001))
    assert edges.size == 9
    assert C.length_bin(1, edges) == 0
    assert C.length_bin(10_000, edges) == 9


# --------------------------------------------------------------------------- #
# Ranking and gate logic
# --------------------------------------------------------------------------- #
def test_pctrank_and_robust_z():
    x = np.array([1.0, 1.0, 2.0, 3.0])
    r = C.pctrank(x)
    assert r[0] == r[1] and r[3] == 100.0
    assert np.all(np.diff(r[1:]) > 0)
    z = C.robust_z(np.array([5.0, 5.0, 5.0]))
    assert np.allclose(z, 0.0), 'a zero MAD must not produce infinities'


def test_gate_waterfall_is_monotone():
    import t3_select as T3                                     # noqa: F401 (import check)

    rng = np.random.default_rng(7)
    flags = rng.random((500, len(C.GATE_ORDER))) > 0.3
    alive = np.ones(500, dtype=bool)
    counts = [int(alive.sum())]
    for j in range(flags.shape[1]):
        alive &= flags[:, j]
        counts.append(int(alive.sum()))
    assert all(counts[i + 1] <= counts[i] for i in range(len(counts) - 1))


def test_project_families_are_read_from_disk():
    if not any(C.LADDER_DIR.glob('*/ladder_selected.json')):
        pytest.skip('needs the ladder files written by the pipeline '
                    '(data/ladder/*/ladder_selected.json)')
    used = C.load_project_families()
    for acc in ('PF00106', 'PF00107', 'PF13561', 'PF01593'):
        assert acc in used, f'{acc} must be recognised as already in use'
    assert len(used) >= 10
    assert all(v for v in used.values()), 'every entry records why the family is in use'


def test_ladder_files_are_loud_on_corruption(tmp_path, monkeypatch):
    bad = tmp_path / 'PFXXXX'
    bad.mkdir()
    (bad / 'ladder_selected.json').write_text('{not json')
    monkeypatch.setattr(C, 'LADDER_DIR', tmp_path)
    with pytest.raises(json.JSONDecodeError):
        C.load_project_families()


def test_repetitive_by_construction_families_are_tracked_not_dropped():
    """The standing rule: Repeat/Coiled-coil/Motif/Disordered families stay in the table.

    Repetitiveness is decided by the measured statistics, so no gate may read Pfam's own
    `type` label — the label is reported, and those families are the positive control for
    the metrics that do decide.
    """
    import t3_select as T3

    assert 'gate_type' not in C.GATE_ORDER, 'Pfam type must never gate the shortlist'
    source = (HERE / 't3_select.py').read_text()
    gate_block = source.split("row['gate_size']")[1].split("row['n_gates_failed']")[0]
    assert "'type'" not in gate_block, 'no gate may be computed from the Pfam type label'
    assert T3.HAZARD_FALLBACK, 'the hazard re-screen keeps a fallback keyword list'


def test_guard_rejects_a_limited_run_into_the_real_results_dir(monkeypatch):
    class Args:
        limit_families = 512
        limit_retain = 0
        results = None

    monkeypatch.setattr(C, 'TAG', '')
    with pytest.raises(SystemExit):
        C.guard_limits(Args())
    Args.results = '/tmp/elsewhere'
    C.guard_limits(Args())


def test_percentile_against_reference():
    ref = np.sort(np.linspace(0, 1, 101))
    assert C.percentile_against(-1.0, ref) == 0.0
    assert C.percentile_against(2.0, ref) == 100.0
    assert 45 <= C.percentile_against(0.5, ref) <= 55


# --------------------------------------------------------------------------- #
# Figure aggregation
# --------------------------------------------------------------------------- #
def test_block_aggregation_shape_and_values(tmp_path):
    import t4_figures as T4

    n, factor = 37, 8
    rng = np.random.default_rng(8)
    sim = rng.random((n, n)).astype(np.float32)
    sim = (sim + sim.T) / 2
    np.fill_diagonal(sim, 1.0)
    path = tmp_path / 'sim.npy'
    np.save(path, sim.astype(np.float16))
    mat = np.load(path, mmap_mode='r')
    leaf_order = rng.permutation(n)
    clusters = rng.integers(0, 3, size=n)
    mean, top, cluster_mean, nb, pad = T4.block_pass(mat, leaf_order, clusters, factor)
    assert nb == 5 and pad == 3
    assert mean.shape == (nb, nb) and top.shape == (nb, nb)
    assert np.nanmax(top) <= 1.0 and np.nanmin(mean) >= 0.0
    # The first block's value must be the mean of exactly those off-diagonal pairs.
    rows = leaf_order[:factor]
    block = np.asarray(sim[np.ix_(rows, rows)], dtype=np.float64)
    np.fill_diagonal(block, np.nan)
    assert abs(np.nanmean(block) - mean[0, 0]) < 2e-3
    assert cluster_mean.shape == (3, 3)
    assert np.all(np.isfinite(cluster_mean))
