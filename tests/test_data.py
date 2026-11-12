"""Tests for the data pipeline: synthesis, leakage filter, homologs, datamodule."""
from src.data.datamodule import (
    attacker_pool,
    attacker_trainset,
    random_split,
    subsample,
    target_test_seqs,
)
from src.data.homologs import mutate_seeds
from src.utils.io import count_fasta, fasta_seqs, read_fasta


def test_synthesized_files_exist(built_data):
    p = built_data.paths
    for f in (p.F_train, p.F_test, p.Fp_train, p.Fp_test, p.retain_train, p.retain_holdout,
              p.background, p.homologs("F"), p.homologs("F_prime")):
        assert count_fasta(f) > 0


def test_leakage_filter_removes_contaminants(built_data):
    report = built_data.report
    contaminants = set(built_data.raw["contaminant_ids"])
    removed = set(report["removed_ids"])
    # every injected near-duplicate of F must be removed
    assert contaminants.issubset(removed)
    assert report["n_removed"] >= len(contaminants)
    assert report["n_out"] == report["n_in"] - report["n_removed"]
    # most unrelated random retain sequences are kept
    assert report["n_out"] > report["n_removed"]


def test_random_split_proportions():
    recs = [("id%d" % i, "ACDE") for i in range(100)]
    tr, te = random_split(recs, 0.7, seed=0)
    assert len(tr) == 70 and len(te) == 30
    # deterministic
    tr2, _ = random_split(recs, 0.7, seed=0)
    assert [r.id if hasattr(r, "id") else r[0] for r in tr] == [r.id if hasattr(r, "id") else r[0] for r in tr2]


def test_subsample_deterministic_and_capped():
    seqs = [f"S{i}" for i in range(10)]
    assert len(subsample(seqs, 4, seed=0)) == 4
    assert subsample(seqs, 4, seed=0) == subsample(seqs, 4, seed=0)
    assert len(subsample(seqs, 999, seed=0)) == 10  # capped at len


def test_attacker_pool_homolog_augmented_is_larger(built_data):
    p = built_data.paths
    clean = attacker_pool(p, "F", "clean")
    aug = attacker_pool(p, "F", "homolog_augmented")
    assert len(aug) > len(clean)
    assert len(clean) == count_fasta(p.F_train)
    assert len(target_test_seqs(p, "F")) == count_fasta(p.F_test)


def test_retain_only_trains_on_R_not_F(built_data):
    """The relearning probe: retain_only draws from R, contains zero F examples, and
    ignores target_family + homolog augmentation (only the *scored* family differs)."""
    p = built_data.paths
    retain = set(fasta_seqs(p.retain_train))
    f_train = set(fasta_seqs(p.F_train))

    seqs = attacker_trainset(p, "F", "retain_only", 32, seed=0)
    assert len(seqs) == min(32, count_fasta(p.retain_train))
    assert set(seqs).issubset(retain)          # drawn from R
    assert set(seqs).isdisjoint(f_train)        # zero F examples

    # target_family only selects what gets scored, not the training set: F and F_prime
    # yield an identical retain-only trainset, and homolog augmentation is a no-op here.
    same_tgt = attacker_trainset(p, "F_prime", "retain_only", 32, seed=0)
    aug = attacker_trainset(p, "F", "retain_only", 32, seed=0, homolog_ratio=1.0)
    assert seqs == same_tgt == aug

    # attacker_pool exposes the full retain pool for retain_only (not the F pool).
    pool = attacker_pool(p, "F", "retain_only")
    assert len(pool) == count_fasta(p.retain_train)
    assert set(pool).isdisjoint(f_train)


def test_mutate_seeds_count(tmp_path):
    src = tmp_path / "seeds.fasta"
    from src.utils.io import write_fasta

    write_fasta([("a", "ACDEFGHIK"), ("b", "MNPQRSTVW")], src)
    out = tmp_path / "hom.fasta"
    n = mutate_seeds(str(src), n_per_seed_seq=3, out_fasta=str(out), mutation_rate=0.2, seed=0)
    assert n == 6 and count_fasta(out) == 6
    # mutated sequences keep the original length
    assert all(len(s) == 9 for s in fasta_seqs(out))
