"""Tests for src.analysis.weightspace (pure logic; needs torch tensors only)."""
import pytest

torch = pytest.importorskip("torch")

from src.analysis.weightspace import (  # noqa: E402
    _barrier_nats,
    interpolate_state_dicts,
    l2_by_group,
    param_layer_group,
)


def test_param_layer_group():
    assert param_layer_group("esm.encoder.layer.3.attention.self.query.weight") == "layer_03"
    assert param_layer_group("esm.encoder.layer.11.output.dense.bias") == "layer_11"
    assert param_layer_group("esm.embeddings.word_embeddings.weight") == "embeddings"
    assert param_layer_group("lm_head.decoder.weight") == "lm_head"
    assert param_layer_group("esm.encoder.emb_layer_norm_after.weight") == "other"


def test_l2_identical_is_zero():
    sd = {"esm.encoder.layer.0.w": torch.ones(4), "lm_head.w": torch.ones(2)}
    out = l2_by_group(sd, {k: v.clone() for k, v in sd.items()})
    assert out["__total__"]["l2"] == pytest.approx(0.0)
    assert out["layer_00"]["l2"] == pytest.approx(0.0)


def test_l2_is_localized_to_the_changed_layer():
    sd0 = {"esm.encoder.layer.0.w": torch.zeros(3), "esm.encoder.layer.3.w": torch.zeros(3)}
    sd1 = {"esm.encoder.layer.0.w": torch.zeros(3), "esm.encoder.layer.3.w": torch.ones(3)}
    out = l2_by_group(sd0, sd1)
    assert out["layer_00"]["l2"] == pytest.approx(0.0)
    assert out["layer_03"]["l2"] == pytest.approx(3 ** 0.5)            # ‖[1,1,1]‖
    assert out["__total__"]["l2"] == pytest.approx(3 ** 0.5)


def test_l2_skips_non_float_keys():
    sd0 = {"w": torch.zeros(2), "buf": torch.tensor([1, 2])}
    sd1 = {"w": torch.ones(2), "buf": torch.tensor([9, 9])}
    out = l2_by_group(sd0, sd1)
    # only the float "w" contributes; the int buffer is ignored
    assert out["__total__"]["l2"] == pytest.approx(2 ** 0.5)


def test_interpolate_endpoints_midpoint_and_buffers():
    sd0 = {"a": torch.zeros(3), "i": torch.tensor([1, 2, 3])}
    sd1 = {"a": torch.full((3,), 2.0), "i": torch.tensor([9, 9, 9])}
    assert torch.allclose(interpolate_state_dicts(sd0, sd1, 0.0)["a"], sd0["a"])
    assert torch.allclose(interpolate_state_dicts(sd0, sd1, 1.0)["a"], sd1["a"])
    assert torch.allclose(interpolate_state_dicts(sd0, sd1, 0.5)["a"], torch.ones(3))
    # non-float buffer copied from sd0 unchanged
    assert torch.equal(interpolate_state_dicts(sd0, sd1, 0.5)["i"], sd0["i"])


def test_barrier_zero_when_monotone_no_barrier_when_bump():
    a = [0.0, 0.5, 1.0]
    b_mono, mono = _barrier_nats(a, [2.0, 5.0, 20.0])      # monotone increasing
    assert mono and b_mono == pytest.approx(0.0, abs=1e-9)
    b_bump, bumpy = _barrier_nats(a, [2.0, 50.0, 20.0])    # interior spike above endpoints
    assert b_bump > 0.5 and not bumpy
