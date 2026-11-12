"""Tests for the metric math in src.eval (the model-free parts)."""
import math

import pytest

from src.eval.pseudo_perplexity import (
    _strided_groups,
    acquisition_threshold,
    aggregate_nll,
    is_acquired,
)
from src.eval.utility import utility_retention, within_tolerance


def test_aggregate_nll_single():
    res = aggregate_nll([2.0], [2])  # mean nll 1.0
    assert res.ppl == pytest.approx(math.exp(1.0))
    assert res.ppl_macro == pytest.approx(math.exp(1.0))
    assert res.n_seqs == 1 and res.n_positions == 2


def test_aggregate_nll_micro_vs_macro():
    res = aggregate_nll([2.0, 4.0], [2, 2])
    assert res.ppl == pytest.approx(math.exp(6.0 / 4.0))                      # micro
    assert res.ppl_macro == pytest.approx((math.exp(1.0) + math.exp(2.0)) / 2)  # macro


def test_acquisition_threshold():
    assert acquisition_threshold(10.0, 2.0, 0.8) == pytest.approx(3.6)
    assert is_acquired(3.5, 3.6) and not is_acquired(3.7, 3.6)
    with pytest.raises(ValueError):
        acquisition_threshold(10, 2, 1.5)


def test_strided_groups_cover_all_positions_once():
    positions = [0, 1, 2, 3, 4]
    groups = _strided_groups(positions, masks_per_pass=2)
    assert len(groups) == 3  # ceil(5/2)
    flat = [p for g in groups for p in g]
    assert sorted(flat) == positions          # every position covered exactly once
    assert _strided_groups(positions, 1) == [[0], [1], [2], [3], [4]]


def test_utility_retention():
    assert utility_retention(10.2, 10.0) == pytest.approx(0.02)
    assert within_tolerance(10.2, 10.0, 0.02)
    assert not within_tolerance(10.5, 10.0, 0.02)
