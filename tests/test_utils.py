"""Tests for src.utils (seeding, flops, io, tracking)."""
import json

from src.utils.flops import FlopCounter, forward_flops, relative_gap, steps_to_match, train_flops
from src.utils.io import SeqRecord, count_fasta, read_fasta, write_fasta
from src.utils.seeding import set_seed
from src.utils.tracking import JSONLTracker, init_tracker


def test_flops_basic():
    assert train_flops(100, 10) == 6 * 100 * 10
    assert forward_flops(100, 10) == 2 * 100 * 10
    assert steps_to_match(1000, 100) == 10
    assert steps_to_match(1050, 100) == 11  # ceil
    assert relative_gap(110, 100) == 0.1
    assert relative_gap(0, 0) == 0.0


def test_flop_counter():
    c = FlopCounter(n_params=1000)
    c.record(10, fwd_bwd=True, is_optimizer_step=True)
    c.record(5, fwd_bwd=False)
    assert c.total_tokens == 15
    assert c.optimizer_steps == 1
    assert c.total_flops == train_flops(1000, 10) + forward_flops(1000, 5)
    d = c.as_dict()
    assert "wall_clock_s" in d and "_t0" not in d


def test_fasta_roundtrip(tmp_path):
    recs = [SeqRecord("a", "ACDEFG"), ("b", "MMMM"), SeqRecord("c", "WYWY")]
    p = tmp_path / "x.fasta"
    n = write_fasta(recs, p)
    assert n == 3 and count_fasta(p) == 3
    back = read_fasta(p)
    assert [r.id for r in back] == ["a", "b", "c"]
    assert back[0].seq == "ACDEFG" and back[1].seq == "MMMM"


def test_set_seed_deterministic():
    import random

    set_seed(123)
    a = [random.random() for _ in range(5)]
    set_seed(123)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_jsonl_tracker(tmp_path):
    t = init_tracker("none", "proj", "run1", {"k": 1}, str(tmp_path))
    assert isinstance(t, JSONLTracker)
    t.log({"loss": 0.5}, step=1)
    t.log_summary({"final": 1.0})
    t.finish()
    lines = (tmp_path / "runs" / "run1.jsonl").read_text().strip().splitlines()
    assert any(json.loads(ln).get("loss") == 0.5 for ln in lines)
    assert json.loads((tmp_path / "runs" / "run1.summary.json").read_text())["final"] == 1.0
