"""Offline unit tests for experiment 6 — no GPU, no network, no torch.

Run explicitly (this file is outside `testpaths`, like exp-5's test_splice.py):

    pytest experiments/6_mgnify_reps_vs_retain/test_mgreps.py -q

The scope is the decision logic that a smoke run cannot catch cheaply: header parsing, the
shard partition that protects the wall-time budget, the length matching the headline rests
on, and the agreement between this experiment's aggregation and the project's.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import common as C  # noqa: E402


# --------------------------------------------------------------------------- #
# Header parsing
# --------------------------------------------------------------------------- #
def test_parse_rep_header_real_format():
    h = "MGYF0000000001 rep=MGYP006279162177/1-95 len=95 full_size=143098 source=tm_blob"
    d = C.parse_rep_header(h)
    assert d["id"] == "MGYF0000000001"
    assert d["rep"] == "MGYP006279162177/1-95"
    assert d["len"] == "95"
    assert d["full_size"] == "143098"
    assert d["source"] == "tm_blob"


@pytest.mark.parametrize("header", [
    "MGYF0000000001",                                   # id only
    "MGYF0000000001 len=95",                            # partial
    "MGYF0000000001 len=95 extra=thing more=stuff",     # unknown extras
    "  MGYF0000000001   source=s4pred_blob  ",          # ragged whitespace
    "",                                                 # empty
])
def test_parse_rep_header_never_raises(header):
    d = C.parse_rep_header(header)
    assert isinstance(d, dict)


def test_parse_rep_header_ignores_valueless_tokens():
    d = C.parse_rep_header("MGYF1 junk len=10")
    assert d["id"] == "MGYF1" and d["len"] == "10" and "junk" not in d


# --------------------------------------------------------------------------- #
# Cost model and sharding — the test that protects the wall-time budget
# --------------------------------------------------------------------------- #
def test_scored_positions_truncates_at_max_length():
    assert C.scored_positions(100) == 100
    assert C.scored_positions(2000) == C.SCORED_MAX == 1022


def test_predicted_cost_is_quadratic_ish():
    assert C.predicted_cost(0) == 0
    assert C.predicted_cost(8, mpp=8) == 8 * 1
    assert C.predicted_cost(16, mpp=8) == 16 * 2
    # doubling the length roughly quadruples the cost
    assert 3.5 < C.predicted_cost(400) / C.predicted_cost(200) < 4.5
    # and it saturates past truncation
    assert C.predicted_cost(2000) == C.predicted_cost(1022)


def test_shard_assign_is_an_exact_partition_and_cost_balanced():
    rng = np.random.default_rng(0)
    # heavy-tailed, like the real corpora: median ~200, a long tail to 2000
    lengths = np.clip((rng.lognormal(mean=5.3, sigma=0.8, size=5000)).astype(int), 75, 2000)
    n_shards = 24
    shards = C.shard_assign(lengths, n_shards)

    assert shards.shape == lengths.shape
    assert set(np.unique(shards)) <= set(range(n_shards))
    assert len(np.unique(shards)) == n_shards, "every shard must get work"

    rows = C.shard_cost_table(lengths, shards, n_shards)
    assert sum(r["n_seqs"] for r in rows) == lengths.size
    costs = np.array([r["token_forwards"] for r in rows], dtype=float)
    assert costs.max() / costs.mean() < 1.05, "cost imbalance would time out the slow shard"


def test_shard_assign_beats_a_naive_contiguous_split():
    """The reason sharding is cost-based at all — on sorted input a contiguous split is
    catastrophically unbalanced, and that is exactly the order the work table is in."""
    lengths = np.sort(np.clip((np.random.default_rng(1).lognormal(5.3, 0.8, 3000)).astype(int),
                              75, 2000))
    n = 12
    smart = np.array([r["token_forwards"] for r in
                      C.shard_cost_table(lengths, C.shard_assign(lengths, n), n)], dtype=float)
    naive_shards = np.repeat(np.arange(n), math.ceil(lengths.size / n))[:lengths.size]
    naive = np.array([r["token_forwards"] for r in
                      C.shard_cost_table(lengths, naive_shards, n)], dtype=float)
    assert smart.max() / smart.mean() < naive.max() / naive.mean() / 2


def test_shard_assign_is_deterministic():
    lengths = np.random.default_rng(2).integers(75, 1500, size=500)
    assert np.array_equal(C.shard_assign(lengths, 7), C.shard_assign(lengths, 7))


def test_shard_assign_rejects_bad_n():
    with pytest.raises(ValueError):
        C.shard_assign([100, 200], 0)


def test_length_bin_edges():
    assert C.length_bin(1) == 0            # (0, 50]
    assert C.length_bin(50) == 0
    assert C.length_bin(51) == 1
    assert C.length_bin(10_000) == len(C.LEN_EDGES) - 1


# --------------------------------------------------------------------------- #
# Length matching — the headline arm
# --------------------------------------------------------------------------- #
def test_length_match_without_replacement_and_within_tolerance():
    rng = np.random.default_rng(3)
    target = rng.integers(80, 900, size=400)
    pool = rng.integers(80, 900, size=4000)
    idx, diag = C.length_match(target, pool, seed=0)

    used = [i for i in idx if i >= 0]
    assert len(set(used)) == len(used), "a pool sequence must not be matched twice"
    for t, i in zip(target, idx):
        if i >= 0:
            assert abs(int(pool[i]) - int(t)) <= C.MATCH_MAX_DELTA
    assert diag["n_matched"] + diag["n_unmatched"] == len(target)
    assert diag["n_matched"] > 0.95 * len(target)


def test_length_match_is_deterministic():
    rng = np.random.default_rng(4)
    target, pool = rng.integers(80, 900, 200), rng.integers(80, 900, 2000)
    a, _ = C.length_match(target, pool, seed=7)
    b, _ = C.length_match(target, pool, seed=7)
    assert a == b
    c, _ = C.length_match(target, pool, seed=8)
    assert a != c, "a different seed should draw a different matched sample"


def test_length_match_degrades_gracefully_when_the_pool_is_too_small():
    target = np.array([100, 200, 300, 400])
    pool = np.array([100, 205])
    idx, diag = C.length_match(target, pool, seed=0)
    assert diag["n_matched"] == 2 and diag["n_unmatched"] == 2
    assert sorted(i for i in idx if i >= 0) == [0, 1]


def test_length_match_marks_out_of_range_targets_unmatched():
    target = np.array([5000])
    pool = np.array([100, 200, 300])
    idx, diag = C.length_match(target, pool, seed=0)
    assert idx == [-1] and diag["n_unmatched"] == 1


def test_length_match_reproduces_the_target_histogram():
    rng = np.random.default_rng(5)
    target = rng.integers(100, 600, size=1000)
    pool = rng.integers(80, 900, size=20000)
    idx, _ = C.length_match(target, pool, seed=0)
    matched = np.array([pool[i] for i in idx if i >= 0])
    scipy_stats = pytest.importorskip("scipy.stats")
    assert scipy_stats.ks_2samp(matched, target).statistic < 0.02


# --------------------------------------------------------------------------- #
# Aggregation — must BE the project's aggregation, not merely resemble it
# --------------------------------------------------------------------------- #
def test_micro_ppl_matches_the_project_aggregation():
    from src.eval.pseudo_perplexity import aggregate_nll

    rng = np.random.default_rng(6)
    pos = rng.integers(50, 900, size=200)
    nll = rng.uniform(1.0, 3.0, size=200) * pos
    assert C.micro_ppl(nll, pos) == pytest.approx(
        aggregate_nll(list(nll), list(pos)).ppl, rel=1e-12)


def test_micro_ppl_handles_empty():
    assert math.isnan(C.micro_ppl([], []))


# --------------------------------------------------------------------------- #
# Effect sizes and adjustments
# --------------------------------------------------------------------------- #
def test_cliffs_delta_sign_convention():
    pytest.importorskip("scipy")
    assert C.cliffs_delta([10, 11, 12], [1, 2, 3]) == pytest.approx(1.0)
    assert C.cliffs_delta([1, 2, 3], [10, 11, 12]) == pytest.approx(-1.0)
    assert C.cliffs_delta([1, 2, 3], [1, 2, 3]) == pytest.approx(0.0)


def test_bootstrap_ci_is_deterministic_and_brackets_the_estimate():
    x = np.random.default_rng(7).normal(5.0, 1.0, size=500)
    a = C.bootstrap_ci(lambda v: float(np.mean(v)), x, n_boot=200, seed=0)
    b = C.bootstrap_ci(lambda v: float(np.mean(v)), x, n_boot=200, seed=0)
    assert a == b
    assert a[0] < float(x.mean()) < a[1]


def test_bootstrap_ci_on_constant_input_is_degenerate_not_broken():
    x = np.full(50, 3.0)
    lo, hi = C.bootstrap_ci(lambda v: float(np.mean(v)), x, n_boot=50, seed=0)
    assert lo == pytest.approx(3.0) and hi == pytest.approx(3.0)


def test_standardize_by_length_is_identity_when_distributions_match():
    lengths = np.random.default_rng(8).integers(100, 800, size=2000)
    vals = np.random.default_rng(9).normal(2.0, 0.3, size=2000)
    adj, w = C.standardize_by_length(lengths, lengths, vals)
    assert w.sum() == pytest.approx(1.0)
    assert adj == pytest.approx(float(vals.mean()), abs=0.05)


def test_standardize_by_length_reweights_towards_the_reference_mix():
    # `other` is dominated by short sequences that score low; `ref` is all long.
    ref = np.full(100, 700)
    other_len = np.concatenate([np.full(900, 100), np.full(100, 700)])
    other_val = np.concatenate([np.zeros(900), np.ones(100)])
    adj, _ = C.standardize_by_length(ref, other_len, other_val)
    assert adj == pytest.approx(1.0)          # the long stratum alone, not the 0.1 raw mean


def test_ols_bootstrap_recovers_known_coefficients():
    rng = np.random.default_rng(10)
    n = 800
    x1 = rng.normal(size=n)
    x2 = rng.integers(0, 2, size=n).astype(float)
    y = 1.5 + 0.3 * x1 - 0.8 * x2 + rng.normal(scale=0.1, size=n)
    X = np.column_stack([np.ones(n), x1, x2])
    res = C.ols_bootstrap(X, y, n_boot=100, seed=0)
    assert res["coef"][2] == pytest.approx(-0.8, abs=0.05)
    lo, hi = res["coef_ci95"][2]
    assert lo < -0.8 < hi
    assert res["r2"] > 0.9


# --------------------------------------------------------------------------- #
# Composition
# --------------------------------------------------------------------------- #
def test_js_divergence_properties():
    p = {"A": 0.5, "C": 0.5}
    q = {"A": 0.5, "C": 0.5}
    r = {"A": 1.0}
    assert C.js_divergence(p, q) == pytest.approx(0.0, abs=1e-12)
    assert C.js_divergence(p, r) == pytest.approx(C.js_divergence(r, p))
    assert 0.0 < C.js_divergence(p, r) <= math.log(2) + 1e-12


def test_seq_entropy():
    assert C.seq_entropy("AAAA") == pytest.approx(0.0)
    assert C.seq_entropy("ACGT") == pytest.approx(math.log(4))
    assert math.isnan(C.seq_entropy(""))


def test_aa_composition_sums_to_one():
    comp = C.aa_composition(["ACDE", "AAGG"])
    assert sum(comp.values()) == pytest.approx(1.0)
    assert comp["A"] == pytest.approx(3 / 8)


# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #
def test_save_json_scrubs_non_finite(tmp_path):
    p = tmp_path / "x.json"
    C.save_json({"a": float("nan"), "b": [1.0, float("inf")], "c": {"d": float("-inf")},
                 "e": np.float64(2.5), "f": np.array([1, 2])}, p)
    got = C.load_json(p)
    assert got["a"] is None and got["b"] == [1.0, None] and got["c"]["d"] is None
    assert got["e"] == 2.5 and got["f"] == [1, 2]


def test_protocol_block_pins_the_eval_settings():
    b = C.protocol_block()
    assert b["model_id"] == "facebook/esm2_t12_35M_UR50D"
    assert b["max_length"] == 1024 and b["masks_per_pass"] == 8 and b["eval_batch"] == 64
    assert "no task-blocked checkpoint" in b["variant"]


def test_read_fasta_headers_keeps_the_full_description(tmp_path):
    p = tmp_path / "t.fasta"
    p.write_text(">MGYF1 rep=X/1-9 len=9 source=tm_blob\nACDEFGHIK\n>MGYF2 len=4\nACDE\n")
    heads = C.read_fasta_headers(p)
    assert heads == ["MGYF1 rep=X/1-9 len=9 source=tm_blob", "MGYF2 len=4"]
    assert C.parse_rep_header(heads[0])["source"] == "tm_blob"


def test_load_reps_pairs_metadata_with_sequences(tmp_path):
    p = tmp_path / "t.fasta"
    p.write_text(">MGYF1 rep=X/1-9 len=9 full_size=12 source=tm_blob\nACDEFGHIK\n"
                 ">MGYF2 len=4 source=s4pred_blob\nACDE\n")
    records, meta = C.load_reps(p)
    assert [r.id for r in records] == ["MGYF1", "MGYF2"]
    assert meta[0].length == 9 and meta[0].declared_len == 9 and meta[0].full_size == 12
    assert meta[1].source == "s4pred_blob" and meta[1].full_size is None
    row = meta[0].as_row()
    assert row["is_anchor"] is False and row["truncated"] is False
