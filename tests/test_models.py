"""Tests for src.models.load. The full model build needs transformers + a modern
torch + network, so it is gated; the config-parsing logic is tested unconditionally."""
from types import SimpleNamespace

import pytest

from src.models.load import LoraSpec


def test_lora_spec_from_cfg():
    cfg = SimpleNamespace(lora_rank=8, lora_alpha=16, lora_dropout=0.0,
                          target_modules=["query", "key", "value", "dense"])
    spec = LoraSpec.from_cfg(cfg)
    assert spec.rank == 8 and spec.alpha == 16 and spec.dropout == 0.0
    assert spec.target_modules == ["query", "key", "value", "dense"]


@pytest.mark.slow
def test_build_8m_model_and_lora():
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    import torch

    if tuple(int(x) for x in torch.__version__.split(".")[:2]) < (2, 0):
        pytest.skip("ESM-2 load needs a modern torch")
    from src.models.load import LoraSpec, build_model

    cfg = SimpleNamespace(name="facebook/esm2_t6_8M_UR50D", max_length=64)
    model, tok = build_model(cfg, device="cpu", lora=LoraSpec(rank=8, alpha=16))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    assert 0 < trainable < total  # LoRA freezes the base
