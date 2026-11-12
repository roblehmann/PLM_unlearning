"""Tests for src.train. cfg_get is pure; the bilevel/attack integration tests are
gated on transformers + a modern torch (they run in the conda env, skip otherwise)."""
from types import SimpleNamespace

import pytest

from src.train.mlm import cfg_get


def test_cfg_get_dict_and_namespace_and_default():
    assert cfg_get({"a": 1}, "a", 99) == 1
    assert cfg_get({"a": 1}, "missing", 99) == 99
    assert cfg_get(SimpleNamespace(b=2), "b", 99) == 2
    assert cfg_get(SimpleNamespace(b=2), "missing", 99) == 99
    assert cfg_get(None, "x", 7) == 7


def _maybe_skip_env():
    pytest.importorskip("transformers")
    import torch

    if tuple(int(x) for x in torch.__version__.split(".")[:2]) < (2, 0):
        pytest.skip("needs a modern torch for ESM-2")


@pytest.mark.slow
def test_taskblock_micro_run(built_data):
    """Two outer steps of task blocking on the 8M model over toy sequences. Asserts a
    model + a defender-side FLOP record come back. Skips unless the env supports it."""
    _maybe_skip_env()
    from omegaconf import OmegaConf

    from src.models.load import load_mlm, load_tokenizer
    from src.train.taskblock import train_taskblock
    from src.utils.io import fasta_seqs

    cfg = OmegaConf.create({
        "inner_steps_K": 2, "eta_inner": 1e-4, "eta_outer": 5e-5, "lambda_tr": 1.0,
        "forget_margin": 6.0, "outer_steps": 2, "retain_batch": 4, "forget_batch": 4,
        "warmup_steps": 1, "mlm_probability": 0.15, "grad_clip": 1.0, "log_every": 1,
    })
    tok = load_tokenizer("facebook/esm2_t6_8M_UR50D")
    base = load_mlm("facebook/esm2_t6_8M_UR50D", device="cpu")
    out = train_taskblock(
        base, tok,
        retain_seqs=fasta_seqs(built_data.paths.retain_train)[:8],
        forget_seqs=fasta_seqs(built_data.paths.F_train)[:8],
        cfg=cfg, device="cpu", seed=0, max_length=64,
    )
    assert "model" in out
    assert out["flops"]["total_flops"] > 0
    assert out["flops"]["overhead_factor"] > 1.0  # K inner steps cost more than a plain step
