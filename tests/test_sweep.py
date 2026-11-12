"""Tests for the sweep grid enumeration (pure)."""
from src.sweep.run_sweep import build_cells

SWEEP = {
    "budgets_n": [16, 64, 256, 1024, 4096],
    "seeds": [0, 1, 2],
    "recipes": ["lora", "full_ft"],
    "lora_ranks": [8, 32],
    "models": ["A", "B", "A_prime"],
    "adversary_data": ["clean", "homolog_augmented"],
    "target_family": ["F", "F_prime"],
}


def test_build_cells_count():
    cells = build_cells(SWEEP)
    # lora: 3 models * 5 budgets * 3 seeds * 2 adv * 2 tgt * 2 ranks = 360
    # full_ft: same but 1 rank = 180  -> 540
    assert len(cells) == 540


def test_build_cells_ranks():
    cells = build_cells(SWEEP)
    lora = [c for c in cells if c["recipe"] == "lora"]
    full = [c for c in cells if c["recipe"] == "full_ft"]
    assert all(c["lora_rank"] in (8, 32) for c in lora)
    assert all(c["lora_rank"] is None for c in full)
    assert len(lora) == 360 and len(full) == 180


def test_build_cells_retain_only_composes():
    """The relearning probe rides the adversary_data axis: retain_only crosses every
    other axis (incl. both target families, since target_family only picks what is
    scored)."""
    sweep = {**SWEEP, "adversary_data": ["clean", "homolog_augmented", "retain_only"]}
    cells = build_cells(sweep)
    # adding a 3rd adversary_data value scales the grid by 3/2.
    assert len(cells) == 540 // 2 * 3
    ro = [c for c in cells if c["adversary_data"] == "retain_only"]
    assert len(ro) == 540 // 2
    assert {c["target_family"] for c in ro} == {"F", "F_prime"}
    assert {c["model"] for c in ro} == {"A", "B", "A_prime"}


def test_sharding_partitions_cells():
    cells = build_cells(SWEEP)
    n_shards = 4
    shards = [cells[i::n_shards] for i in range(n_shards)]
    assert sum(len(s) for s in shards) == len(cells)
    # disjoint + complete
    seen = [tuple(sorted(c.items(), key=lambda kv: kv[0])) for s in shards for c in s]
    assert len(seen) == len(cells)
