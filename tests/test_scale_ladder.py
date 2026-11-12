"""Tests for the scale-ladder helpers (experiments/3_scale_ladder/common.py).

Offline and torch-free: these cover the pure decision logic the 650M spot-check hangs
on — ladder ordering, per-scale run roots, manifest round-tripping, the bootstrap SE and
plateau test the headroom gate gates on, and relative-depth normalization (the only place
where 6/12/30/33-layer models are reconciled).
"""
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LADDER = os.path.join(ROOT, "experiments", "3_scale_ladder")
if LADDER not in sys.path:
    sys.path.insert(0, LADDER)

import common  # noqa: E402


def test_ladder_order_is_by_parameter_count_not_lexicographic():
    # '150M' < '35M' as text; the ladder must not report 150M before 35M.
    assert common.sort_scales(["650M", "35M", "150M", "8M"]) == ["8M", "35M", "150M", "650M"]


def test_scales_table_matches_the_shipped_hydra_model_configs():
    from omegaconf import OmegaConf

    for scale in common.SCALES.values():
        cfg = OmegaConf.load(os.path.join(ROOT, "configs", "model", f"{scale['hydra_model']}.yaml"))
        assert cfg.name == scale["model_id"]
        assert int(cfg.num_layers) == scale["num_layers"]
        assert int(cfg.embed_dim) == scale["embed_dim"]


def test_relative_depth_spans_zero_to_one_at_every_scale():
    for scale in common.SCALES.values():
        n = scale["num_layers"]
        assert common.relative_depth(0, n) == 0.0
        assert common.relative_depth(n - 1, n) == 1.0
    # A 6-layer and a 33-layer model put their middle layers at the same place.
    assert common.relative_depth(3, 7) == common.relative_depth(15, 31) == 0.5


def test_prepare_run_root_links_data_and_is_idempotent(tmp_path):
    data = tmp_path / "anchor" / "data"
    data.mkdir(parents=True)
    root = common.prepare_run_root(tmp_path / "ladder", "650M", data)
    assert (root / "checkpoints").is_dir() and (root / "results").is_dir()
    assert (root / "data").is_symlink() and (root / "data").resolve() == data.resolve()
    assert common.prepare_run_root(tmp_path / "ladder", "650M", data) == root  # re-run is a no-op

    other = tmp_path / "anchor2" / "data"          # re-selection repoints the link
    other.mkdir(parents=True)
    root = common.prepare_run_root(tmp_path / "ladder", "650M", other)
    assert (root / "data").resolve() == other.resolve()


def test_prepare_run_root_refuses_to_clobber_a_real_data_dir(tmp_path):
    data = tmp_path / "anchor" / "data"
    data.mkdir(parents=True)
    (tmp_path / "ladder" / "650M" / "data").mkdir(parents=True)
    with pytest.raises(SystemExit):
        common.prepare_run_root(tmp_path / "ladder", "650M", data)


def test_upsert_row_replaces_and_orders_rows(tmp_path):
    p = tmp_path / "headroom_ladder.json"
    common.upsert_row(p, {"scale": "650M", "gate_pass": False})
    common.upsert_row(p, {"scale": "35M", "gate_pass": True})
    common.upsert_row(p, {"scale": "650M", "gate_pass": True})   # re-run must not duplicate
    rows = common.load_manifest(p)["rows"]
    assert [r["scale"] for r in rows] == ["35M", "650M"]
    assert rows[-1]["gate_pass"] is True
    assert common.row_for(p, "35M")["gate_pass"] is True
    assert common.row_for(p, "8M") is None


def test_manifests_are_strict_json_and_byte_stable(tmp_path):
    p = tmp_path / "m.json"
    common.upsert_row(p, {"scale": "650M", "tau_floor_margin_ses": float("nan"), "M": 1.08})
    text = p.read_text()
    assert "NaN" not in text                       # jq / LaTeX consumers would choke
    assert json.loads(text)["rows"][0]["tau_floor_margin_ses"] is None
    common.save_manifest(p, common.load_manifest(p))
    assert p.read_text() == text                   # re-writing changes nothing


def test_bootstrap_se_shrinks_as_the_eval_set_grows():
    nll, pos = [200.0, 240.0, 180.0, 220.0], [100] * 4
    se_small = common.bootstrap_ppl_se(nll, pos, n_boot=200, seed=42)
    se_large = common.bootstrap_ppl_se(nll * 25, pos * 25, n_boot=200, seed=42)
    assert se_small > se_large > 0
    assert common.bootstrap_ppl_se(nll, pos, n_boot=200, seed=42) == se_small  # seeded


def test_corpus_ppl_weights_by_scored_positions():
    # A long sequence dominates the corpus aggregate; the macro mean would not.
    long_seq_dominated = common.corpus_ppl([1000.0, 10.0], [1000, 10])
    assert abs(long_seq_dominated - common.corpus_ppl([1010.0], [1010])) < 1e-9


def test_plateaued_distinguishes_a_converged_floor_from_a_still_falling_one():
    flat = [{"target_ppl": x} for x in [13, 11, 10.3, 10.05, 10.04, 10.03, 10.03]]
    falling = [{"target_ppl": x} for x in [13, 11, 10.3, 9.0, 8.0, 7.0]]
    assert common.plateaued(flat) is True
    assert common.plateaued(falling) is False
    assert common.plateaued([{"target_ppl": 13}]) is False   # too short to judge


def test_log_grid_brackets_the_requested_range():
    g = common.log_grid(16, 4096, 6)
    assert g[0] == 16 and g[-1] == 4096 and len(g) == 6
    assert g == sorted(set(g))


def test_random_ceiling_is_twenty():
    # ESM-2's random-prediction perplexity ceiling: 20 residue types.
    assert common.RANDOM_CEILING == 20.0
